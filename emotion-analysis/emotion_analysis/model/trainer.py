from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal

import evaluate
import flax
import flax.linen as nn
import jax.numpy as jnp
import jax.random as jrm
import mlflow
import optax
from flax.core import FrozenDict, freeze
from flax.training import train_state as ts
from flax.traverse_util import path_aware_map
from jax import Array, grad, jit, value_and_grad, vmap
from jax.typing import ArrayLike
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from .. import EmotionAnalysisConfig
from .metrics import MeanMetric, F1Metric
from ..data.types import EmotionCauseEncoding
from .finetune import FineTune, FineTuneStrategy, ParamLabel
from .model import EmotionCauseTextModel
from .pretrained import PretrainedTextModel, load_text_model


@dataclass()
class TrainerModule(object):
    # PRNG Key
    key: Any
    # Mini-batch size
    batch_size: int
    # Maximum number of utterances in a conversation
    max_conv_len: int
    # Maximum number of tokens a utterance may have
    max_uttr_len: int
    # Text Encoder
    text_model_repo: str
    # Learning rate
    learning_rate: float
    # Finetune method
    finetune: FineTune

    def __post_init__(self):
        # Create and init the model using a pretrained architecture
        self.text_encoder = load_text_model(self.text_model_repo)
        self.model = EmotionCauseTextModel(text_encoder=self.text_encoder.module, num_emotions=7)
        self.init_model()
        self.jit_func()

    def init_model(self):
        # Generate fake data
        fake_data = self.fake_input()
        self.key, init_key, drop_key = jrm.split(self.key, 3)

        # Perform a forward pass to init the model with random weights and state
        rng_keys = { 'params': init_key, 'dropout': drop_key }
        params = self.model.init(rng_keys, **fake_data, train=True)['params']
        params['text_encoder'] = self.text_encoder.params

        # Create optimizers for finetuning
        opt = self.init_optim(params)

        # Create train state
        self.state = ts.TrainState.create(apply_fn=self.model.apply, params=params, tx=opt)

        # Create reusable metrics
        self.init_metrics()

    def init_metrics(self):
        self.train_loss = MeanMetric()
        self.valid_loss = MeanMetric()
        self.train_f1 = F1Metric('weighted')
        self.valid_f1 = F1Metric('weighted')

    def init_optim(self, params: FrozenDict[str, Any]):
        # Create trainable & no-op optimizers
        train_sch =  optax.constant_schedule(value=self.learning_rate)
        frozen_opt = optax.set_to_zero()
        train_opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=train_sch),
        )

        # Create param labels for trainable and frozem params
        param_labels = FineTuneStrategy.from_enum(self.finetune).param_labels(params)
        opt = optax.multi_transform({
            ParamLabel.TRAINABLE: train_opt,
            ParamLabel.FROZEN: frozen_opt,
        }, param_labels=param_labels)
        return opt

    def fake_input(self) -> Dict[str, Any]:
        # Create fake data found in a batch
        fake_input_ids = jnp.zeros((self.batch_size, self.max_conv_len, self.max_uttr_len))
        fake_conv_attn_mask = jnp.zeros((self.batch_size, self.max_conv_len))
        fake_uttr_attn_mask = jnp.zeros_like(fake_input_ids)
        return dict(
            input_ids=fake_input_ids,
            uttr_attn_mask=fake_uttr_attn_mask,
            conv_attn_mask=fake_conv_attn_mask,
        )

    def jit_func(self):
        def compute_loss(
            key: Any,
            params: FrozenDict,
            state: ts.TrainState,
            batch: EmotionCauseEncoding,
            train: bool,
        ):
            # Data Input
            data: Dict[str, Array] = {}
            data['input_ids'] = batch['input_ids']
            data['uttr_attn_mask'] = batch['uttr_attn_mask']
            data['conv_attn_mask'] = batch['conv_attn_mask']

            # Forward pass
            key, drop_key = jrm.split(key, 2)
            output = state.apply_fn({ 'params': params }, **data, train=train, rngs={ 'dropout': drop_key })
            logits = output['emotion']['out']

            # Compute loss
            assert 'emotion_labels' in batch and 'emotion_weight' in batch
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['emotion_labels'])

            # Normalized loss weight for unmasked entries
            weight: Array = batch['conv_attn_mask'] # * batch['emotion_weight']

            # Per-batch loss
            loss = jnp.mean(loss * weight)
            return loss, (logits, key)

        def train_step(
            key: Any,
            state: ts.TrainState,
            batch: EmotionCauseEncoding
        ):
            # Wrap loss function
            loss_fn = lambda params: compute_loss(key, params, state, batch, True)

            # Compute loss & gradients
            val, grads = value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, logits, key = val[0], *val[1]

            # Update the model and optimizers
            state = state.apply_gradients(grads=grads)

            # Return the new state
            return loss, logits, key, state
        self.train_step = jit(train_step)

        def eval_step(
            key: Any,
            state: ts.TrainState,
            batch: EmotionCauseEncoding,
        ):
            # Wrap loss function
            loss_fn = lambda params: compute_loss(key, params, state, batch, False)

            # Compute loss & gradients
            loss, aux = loss_fn(state.params)
            logits, key = aux[0], aux[1] 

            # Return the new state
            return loss, logits, key, state
        self.eval_step = jit(eval_step)

    def train_epoch(self, epoch: int, dataloader: DataLoader[EmotionCauseEncoding]) -> None:
        for batch in tqdm(dataloader, desc=f'[train][epoch: {epoch}]'):
            batch: EmotionCauseEncoding
            assert 'emotion_labels' in batch, 'no emotion labels were found during validation'

            # Forward and backward pass for one batch
            loss, logits, self.key, self.state = self.train_step(self.key, self.state, batch)

            # Compute emotion predictions
            conv_mask = batch['conv_attn_mask'].astype(bool)
            pred = logits[conv_mask, :].argmax(axis=1)
            refs = batch['emotion_labels'][conv_mask]

            # Update metrics
            self.train_loss.update(loss)
            self.train_f1.update(predictions=pred, references=refs)
        mlflow.log_metric(key='train_loss', value=self.train_loss.compute(), step=epoch)
        mlflow.log_metric(key='train_f1', value=self.train_f1.compute(), step=epoch)

    def valid_epoch(self, epoch: int, dataloader: DataLoader[EmotionCauseEncoding]) -> None:
        for batch in tqdm(dataloader, desc=f'[eval]'):
            batch: EmotionCauseEncoding
            assert 'emotion_labels' in batch, 'no emotion labels were found during validation'

            # Forward pass for one batch
            loss, logits, self.key, _ = self.eval_step(self.key, self.state, batch)

            # Compute emotion predictions
            conv_mask = batch['conv_attn_mask'].astype(bool)
            pred = logits[conv_mask, :].argmax(axis=1)
            refs = batch['emotion_labels'][conv_mask]

            # Update metrics
            self.valid_loss.update(loss)
            self.valid_f1.update(predictions=pred, references=refs)
        mlflow.log_metric(key='valid_loss', value=self.valid_loss.compute(), step=epoch)
        mlflow.log_metric(key='valid_f1', value=self.valid_f1.compute(), step=epoch)

    def train(
        self,
        *,
        train_dataloader: DataLoader[EmotionCauseEncoding],
        valid_dataloader: DataLoader[EmotionCauseEncoding],
        tags: Dict[str, Any] = {},
        num_epochs: int = 10,
    ) -> None:
        with mlflow.start_run():
            # Add tags
            mlflow.set_tags(tags)

            # Perform train & valid at each step
            for epoch in trange(num_epochs, desc='[training]'):
                self.train_epoch(epoch, train_dataloader)
                self.valid_epoch(epoch, valid_dataloader)
