from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, cast

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
from optax import softmax_cross_entropy_with_integer_labels as cross_entropy
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from .. import EmotionAnalysisConfig
from ..data.types import EmotionCauseEncoding
from .finetune import FineTune, FineTuneStrategy, ParamLabel
from .metrics import F1Metric, MeanMetric
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
    # Apply weighted loss for emotions
    wloss: bool = False

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
        # Average Loss
        self.train_loss = MeanMetric()
        self.valid_loss = MeanMetric()

        # F1Score for causes
        self.train_f1_cause = F1Metric('weighted')
        self.valid_f1_cause = F1Metric('weighted')

        # F1Score for emotions
        self.train_f1_emotion = F1Metric('weighted')
        self.valid_f1_emotion = F1Metric('weighted')

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
        def forward(
            state: ts.TrainState,
            batch: EmotionCauseEncoding,
        ):
            # Data Input
            data: Dict[str, Array] = {}
            data['input_ids'] = batch['input_ids']
            data['uttr_attn_mask'] = batch['uttr_attn_mask']
            data['conv_attn_mask'] = batch['conv_attn_mask']
            output = state.apply_fn({ 'params': state.params }, **data, train=False)

            # Infer the emotion and whether it's a cause
            span_start = output['ec_table']['span_start'].argmax(axis=-1)
            span_stop = output['ec_table']['span_stop'].argmax(axis=-1)
            emotion_pred = output['emotion']['out'].argmax(axis=-1)
            cause_pred = output['cause']['out'].argmax(axis=-1)

            # Return predicted labels
            return {
                'span_start': span_start,
                'span_stop': span_stop,
                'emotion': emotion_pred,
                'cause': cause_pred,
            }
        self.forward = jit(forward)

        def compute_loss(
            key: Any,
            params: FrozenDict,
            state: ts.TrainState,
            batch: EmotionCauseEncoding,
            train: bool,
            wloss: bool,
        ):
            # Ensure all labels are present
            assert 'cause_span' in batch, 'missing cause_span'
            assert 'emotion_weight' in batch, 'missing emotion weights'
            assert 'span_mask' in batch and 'cause_span' in batch, 'missing cause_span or span_mask'
            assert 'cause_labels' in batch and 'emotion_labels' in batch, 'missing emotion or cause labels'

            # Data Input
            data: Dict[str, Array] = {}
            data['input_ids'] = batch['input_ids']
            data['uttr_attn_mask'] = batch['uttr_attn_mask']
            data['conv_attn_mask'] = batch['conv_attn_mask']

            # Forward pass
            key, drop_key = jrm.split(key, 2)
            output = state.apply_fn({ 'params': params }, **data, train=train, rngs={ 'dropout': drop_key })

            # Compute span weight only
            conv_attn_mask = nn.make_attention_mask(batch['conv_attn_mask'], batch['conv_attn_mask']).squeeze(axis=1)
            span_start_loss = conv_attn_mask * cross_entropy(output['ec_table']['span_start'], batch['cause_span'][..., 0])
            span_stop_loss = conv_attn_mask * cross_entropy(output['ec_table']['span_stop'], batch['cause_span'][..., 1])

            # Padded Utterances * Imbalanced Weight
            weight_emotion = jnp.where(wloss, batch['emotion_weight'], jnp.ones_like(batch['emotion_weight']))
            weight_pad = batch['conv_attn_mask']

            # Compute losses
            loss_emotion = cast(Array, cross_entropy(output['emotion']['out'], batch['emotion_labels']))
            loss_cause = cast(Array, cross_entropy(output['cause']['out'], batch['cause_labels']))
            loss_class = weight_pad * loss_cause + weight_pad * weight_emotion * loss_emotion
            loss_span = span_start_loss + span_stop_loss
            loss = jnp.mean(loss_class) + jnp.mean(loss_span)
            return loss, (output, key)

        def train_step(
            key: Any,
            state: ts.TrainState,
            batch: EmotionCauseEncoding,
            wloss: bool,
        ):
            # Wrap loss function
            loss_fn = lambda params: compute_loss(key, params, state, batch, True, wloss)

            # Compute loss & gradients
            val, grads = value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, output, key = val[0], *val[1]

            # Update the model and optimizers
            state = state.apply_gradients(grads=grads)

            # Return the new state
            return loss, output, key, state
        self.train_step = jit(train_step)

        def eval_step(
            key: Any,
            state: ts.TrainState,
            batch: EmotionCauseEncoding,
            wloss: bool,
        ):
            # Wrap loss function
            loss_fn = lambda params: compute_loss(key, params, state, batch, False, wloss)

            # Compute loss & gradients
            loss, aux = loss_fn(state.params)
            output, key = aux[0], aux[1] 

            # Return the new state
            return loss, output, key, state
        self.eval_step = jit(eval_step)

    def train_epoch(self, epoch: int, dataloader: DataLoader[EmotionCauseEncoding]) -> None:
        for batch in tqdm(dataloader, desc=f'[train][epoch: {epoch}]'):
            batch: EmotionCauseEncoding
            assert 'cause_labels' in batch, 'no cause labels were found during training'
            assert 'emotion_labels' in batch, 'no emotion labels were found during training'
            conv_mask = batch['conv_attn_mask'].astype(bool)

            # Forward and backward pass for one batch
            loss, output, self.key, self.state = self.train_step(self.key, self.state, batch, self.wloss)
            self.train_loss.update(loss)

            # Compute cause predictions
            pred = output['cause']['out'][conv_mask, :].argmax(axis=1)
            refs = batch['cause_labels'][conv_mask]
            self.train_f1_cause.update(predictions=pred, references=refs)

            # Compute emotion predictions
            pred = output['emotion']['out'][conv_mask, :].argmax(axis=1)
            refs = batch['emotion_labels'][conv_mask]
            self.train_f1_emotion.update(predictions=pred, references=refs)
        mlflow.log_metric(key='train_loss', value=self.train_loss.compute(), step=epoch)
        mlflow.log_metric(key='train_f1_cause', value=self.train_f1_cause.compute(), step=epoch)
        mlflow.log_metric(key='train_f1_emotion', value=self.train_f1_emotion.compute(), step=epoch)

    def valid_epoch(self, epoch: int, dataloader: DataLoader[EmotionCauseEncoding]) -> None:
        for batch in tqdm(dataloader, desc=f'[eval]'):
            batch: EmotionCauseEncoding
            assert 'cause_labels' in batch, 'no cause labels were found during validation'
            assert 'emotion_labels' in batch, 'no emotion labels were found during validation'
            conv_mask = batch['conv_attn_mask'].astype(bool)

            # Forward pass for one batch
            loss, output, self.key, _ = self.eval_step(self.key, self.state, batch, self.wloss)
            self.valid_loss.update(loss)

            # Compute cause predictions
            pred = output['cause']['out'][conv_mask, :].argmax(axis=1)
            refs = batch['cause_labels'][conv_mask]
            self.valid_f1_cause.update(predictions=pred, references=refs)

            # Compute emotion predictions
            pred = output['emotion']['out'][conv_mask, :].argmax(axis=1)
            refs = batch['emotion_labels'][conv_mask]
            self.valid_f1_emotion.update(predictions=pred, references=refs)
        mlflow.log_metric(key='valid_loss', value=self.valid_loss.compute(), step=epoch)
        mlflow.log_metric(key='valid_f1_cause', value=self.valid_f1_cause.compute(), step=epoch)
        mlflow.log_metric(key='valid_f1_emotion', value=self.valid_f1_emotion.compute(), step=epoch)

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

    def predict(
        self,
        dataloader,
    ):
        results = []
        for batch in tqdm(dataloader, desc='inference'):
            output = self.forward(self.state, batch)
            results.append(output)
        return results
