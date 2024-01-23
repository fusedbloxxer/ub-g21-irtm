from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal

import flax
import flax.linen as nn
import jax.numpy as jnp
import jax.random as jrm
import optax
from jax import jit, vmap, grad, value_and_grad
from flax.training import train_state as ts
from jax import Array
from jax.typing import ArrayLike
from flax.core import FrozenDict, freeze
from flax.traverse_util import path_aware_map

from .. import EmotionAnalysisConfig
from .model import EmotionCauseTextModel
from ..data.types import EmotionCauseEncoding
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
    finetune: Literal['full', 'freeze']

    def __post_init__(self):
        # Create and init the model using a pretrained architecture
        self.text_encoder = load_text_model(self.text_model_repo)
        self.model = EmotionCauseTextModel(text_encoder=self.text_encoder.module, num_classes=7)
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
        opt = self.create_optim(params)

        # Create train state
        self.state = ts.TrainState.create(apply_fn=self.model.apply, params=params, tx=opt)

    def create_optim(self, params: FrozenDict[str, Any]):
        # Create optimizer for trainable params
        train_sch =  optax.constant_schedule(value=self.learning_rate)
        train_opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=train_sch),
        )

        # Create dummy optimizer for no-op frozen params
        frozen_opt = optax.set_to_zero()

        # Create param labels
        match self.finetune:
            case   'full':
                is_trainable = lambda *_: 'trainable'
            case 'frozen':
                is_trainable = lambda p, _: 'frozen' if 'text_encoder' in p else 'trainable'
            case _:
                raise ValueError('invalid finetune option: {}'.format(self.finetune))
        param_labels = path_aware_map(is_trainable, params)

        # Optimize only the trainable params
        opt = optax.multi_transform({
            'trainable': train_opt,
            'frozen': frozen_opt,
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
        ):
            # Data Input
            data: Dict[str, Array] = {}
            data['input_ids'] = batch['input_ids']
            data['uttr_attn_mask'] = batch['uttr_attn_mask']
            data['conv_attn_mask'] = batch['conv_attn_mask']

            # Forward pass
            key, drop_key = jrm.split(key, 2)
            logits = state.apply_fn({ 'params': params }, **data, train=True, rngs={ 'dropout': drop_key })

            # Compute loss
            assert 'emotion_labels' in batch
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['emotion_labels'])
            loss = loss * data['conv_attn_mask']
            loss = jnp.mean(loss)
            return loss, (logits, key)

        def train_step(
            key: Any,
            state: ts.TrainState,
            batch: EmotionCauseEncoding
        ):
            # Compute gradient using backprop
            loss_fn = lambda params: compute_loss(key, params, state, batch)
            ret, grads = value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, logits, key = ret[0], *ret[1]
            state = state.apply_gradients(grads=grads)
            return loss, logits, key, state
        self.train_step = jit(train_step)

        def eval_step(
            key: Any,
            state: ts.TrainState,
            batch: EmotionCauseEncoding,
        ):
            loss_fn = lambda params: compute_loss(key, params, state, batch)
            loss, aux = loss_fn(state.params)
            logits, key = aux[0], aux[1] 
            return loss, logits, key, state
        self.eval_step = jit(eval_step)
