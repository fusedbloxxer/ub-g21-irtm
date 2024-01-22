import typing as t
from dataclasses import dataclass
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.activation import gelu
from jax import Array
from jax.typing import ArrayLike
from transformers import (AutoConfig, AutoTokenizer, FlaxAutoModel,
                          FlaxPreTrainedModel, PretrainedConfig,
                          PreTrainedTokenizerFast)


class PositionEmbeddings(nn.Module):
    hidden_dim:  int
    max_seq_len: int
    n: int = 10_000

    def setup(self) -> None:
        # (H // 2,)
        h_term: np.ndarray = np.exp(np.arange(0, self.hidden_dim, 2) * (-np.log(self.n) / self.hidden_dim))

        # (S, 1)
        s_indx: np.ndarray = np.arange(0, self.max_seq_len)[:, np.newaxis] 

        # (S, 1) * (H // 2,) => (S, H // 2)
        e_term: np.ndarray = s_indx * h_term 

        # (S, H // 2) => (S, H)
        pos_embeddings = np.zeros((self.max_seq_len, self.hidden_dim))
        pos_embeddings[:, 0::2] = np.sin(e_term)
        pos_embeddings[:, 1::2] = np.cos(e_term)
        pos_embeddings = jnp.asarray(pos_embeddings)

        # Init & retrieve pos_embeddings from a variable collection
        self.variable('state', 'embeddings', lambda x: x, pos_embeddings)

    def __call__(self, x: Array) -> Array:
        pos_embeddings: Array = self.get_variable('state', 'embeddings')
        x = x + pos_embeddings[:x.shape[1], :]
        return x


class TransformerLayer(nn.Module):
    num_heads: int
    embed_dim: int
    input_dim: int
    dense_dim: int 
    drop_p: float
    drop_a: float
    norm_eps: float = 1e-6
    activ_fn: Callable[[ArrayLike], Array] = gelu

    def setup(self) -> None:
        # Self-Attention
        self.attn_layer = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim * self.num_heads,
            kernel_init=nn.initializers.glorot_normal(),
            bias_init=nn.initializers.zeros_init(),
            dropout_rate=self.drop_a,
            use_bias=True,
        )

        # In-between layers
        self.norm_layer_1 = nn.LayerNorm(epsilon=self.norm_eps)
        self.norm_layer_2 = nn.LayerNorm(epsilon=self.norm_eps)
        self.drop_layer = nn.Dropout(self.drop_p)

        # Feedforward MLP
        self.mlp_layer = [
            nn.Dense(self.dense_dim, True, kernel_init=nn.initializers.glorot_normal()),
            nn.Dropout(self.drop_p),
            self.activ_fn,
            nn.Dense(self.input_dim, True, kernel_init=nn.initializers.glorot_normal())
        ]

    def __call__(
        self,
        x: Array,
        attn_mask: Array,
        train: bool,
    ) -> Array:
        # Self-Attention
        attn_out: Array = self.attn_layer(x, x, x, deterministic=not train, mask=attn_mask)

        # Add & Norm
        x = x + self.drop_layer(attn_out, deterministic=not train)
        x = self.norm_layer_1(x)

        # Feedforward MLP
        mlp_out: Array = x
        for layer in self.mlp_layer:
            if isinstance(layer, nn.Dropout):
                mlp_out = layer(mlp_out, deterministic=not train)
            else:
                mlp_out = layer(mlp_out)

        # Add & Norm
        x = x + self.drop_layer(mlp_out, deterministic=not train)
        x = self.norm_layer_2(x)
        return x


class TransformerEncoder(nn.Module):
    num_layers: int
    

    def setup(self) -> None:
        
