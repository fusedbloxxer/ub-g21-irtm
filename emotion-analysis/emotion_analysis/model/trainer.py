from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal

import flax
import flax.linen as nn
import jax.numpy as jnp
import jax.random as jrm
import optax
from flax.training import train_state as ts
from jax import Array
from jax.typing import ArrayLike
from flax.core import FrozenDict
from flax.traverse_util import path_aware_map

from .. import EmotionAnalysisConfig
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
    finetune: Literal['full', 'freeze']

    def __post_init__(self):
        # Create and init the model using a pretrained architecture
        self.text_encoder = load_text_model(self.text_model_repo)
        self.model = EmotionCauseTextModel(text_encoder=self.text_encoder.module, num_classes=7)
        self.init_model()

    def init_model(self):
        # Generate fake data
        fake_data = self.fake_input()
        self.key, init_key, drop_key = jrm.split(self.key, 3)

        # Perform a forward pass to init the model with random weights and state
        rng_keys = { 'params': init_key, 'dropout': drop_key }
        params = self.model.init(rng_keys, **fake_data, train=True)['params']
        params['text_encoder'] = self.text_encoder.params

        # Create optimizers
        opt = self.create_optim(params)

        # Create train state
        self.state = ts.TrainState.create(apply_fn=self.model.apply, params=params, tx=opt)

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
