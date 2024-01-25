import typing as t
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Tuple, Dict, Literal

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
    hidden_dim:  int=768
    max_seq_len: int=33
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
        pos_embeddings[:, 1::2] = np.cos(e_term[:, :self.hidden_dim // 2])
        pos_embeddings = jnp.asarray(pos_embeddings)

        # Init & retrieve pos_embeddings from a variable collection
        self.param('embeddings', lambda _, x: x, pos_embeddings)

    def __call__(self, layer_in: Array) -> Array:
        pos_embeddings: Array = self.get_variable('params', 'embeddings')
        x = layer_in + pos_embeddings[:layer_in.shape[1], :]
        return x


class TransformerLayer(nn.Module):
    num_heads: int=2
    embed_dim: int=768
    input_dim: int=768
    dense_dim: int=768
    drop_p: float=0.1
    drop_a: float=0.1
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
        layer_in: Array,
        attn_mask: Array,
        train: bool,
    ) -> Array:
        # Self-Attention
        attn_out: Array = self.attn_layer(layer_in, layer_in, layer_in, deterministic=not train, mask=attn_mask)

        # Add & Norm
        x = layer_in + self.drop_layer(attn_out, deterministic=not train)
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
    num_layers: int=2
    num_heads: int=2
    embed_dim: int=768
    input_dim: int=768
    dense_dim: int=768
    drop_p: float=0.1
    drop_a: float=0.1
    max_con_len: int=33 
    norm_eps: float=1e-6
    activ_fn: Callable[[ArrayLike], Array] = gelu

    def setup(self) -> None:
        # Position embeddings used to encode the relative position of conversations
        self.pos_embeddings = PositionEmbeddings(
            hidden_dim=self.input_dim,
            max_seq_len=self.max_con_len,
        )

        # All layers follow the same configuration
        transformer_layer = partial(
            TransformerLayer,
            num_heads=self.num_heads,
            embed_dim=self.embed_dim,
            input_dim=self.input_dim,
            dense_dim=self.dense_dim,
            drop_p=self.drop_p,
            drop_a=self.drop_a,
            norm_eps=self.norm_eps,
            activ_fn=self.activ_fn,
        )

        # Stack multiple transformer layers
        self.layers: List[TransformerLayer] = [transformer_layer() for _ in range(self.num_layers)]

    def __call__(
        self,
        layer_in: Array,
        attn_mask: Array,
        train: bool,
    ) -> Array:
        # Add Positional Embeddings
        x = self.pos_embeddings(layer_in)        

        # Apply multiple Transformer Layers
        for layer in self.layers:
            x = layer(x, attn_mask, train)
        return x


class TransformerClassifier(nn.Module):
    num_classes: int
    num_layers: int=2
    num_heads: int=2
    embed_dim: int=768
    input_dim: int=768
    dense_dim: int=768
    drop_p: float=0.1
    drop_a: float=0.1
    max_con_len: int=33 
    norm_eps: float=1e-6
    activ_fn: Callable[[ArrayLike], Array] = gelu

    def setup(self) -> None:
        self.transformer = TransformerEncoder(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            embed_dim=self.embed_dim,
            input_dim=self.input_dim,
            dense_dim=self.dense_dim,
            drop_a=self.drop_a,
            drop_p=self.drop_p,
            norm_eps=self.norm_eps,
            activ_fn=self.activ_fn,
            max_con_len=self.max_con_len,
        )

        self.classifier = nn.Dense(
            features=self.num_classes,
            kernel_init=nn.initializers.glorot_normal(),
            bias_init=nn.initializers.zeros_init(),
            name='classifier',
            use_bias=True,
        )

    def __call__(
        self,
        layer_in: Array,
        attn_mask: Array,
        train: bool,
    ) -> Dict[Literal['out', 'hidden'], Array]:
        hidden = self.transformer(layer_in, attn_mask, train)
        out = self.classifier(hidden)
        return { 'out': out, 'hidden': hidden }


class EmotionCausality(nn.Module):
    features: int=256
    drop_p: float=0.2
    max_seq_len: int=93
    max_con_len: int=33
    num_classes: int=2
    activ_fn: Callable[[ArrayLike], Array] = gelu

    def setup(self) -> None:
        self.conv_features = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            padding='SAME',
            strides=1,
            kernel_init=nn.initializers.glorot_normal(),
            bias_init=nn.initializers.zeros_init(),
            use_bias=True,
        )

        self.conv_span_start = nn.Conv(
            features=self.max_seq_len,
            kernel_size=(3, 3),
            padding='SAME',
            strides=1,
            kernel_init=nn.initializers.glorot_normal(),
            bias_init=nn.initializers.zeros_init(),
            use_bias=True,
        )

        self.conv_span_stop = nn.Conv(
            features=self.max_seq_len,
            kernel_size=(3, 3),
            padding='SAME',
            strides=1,
            kernel_init=nn.initializers.glorot_normal(),
            bias_init=nn.initializers.zeros_init(),
            use_bias=True,
        )

        self.dropout = nn.Dropout(self.drop_p)

    def __call__(
        self,
        *,
        emotion_hidden: Array,
        emotion_probs: Array,
        cause_hidden: Array,
        cause_probs: Array,
        train: bool,
    ) -> Dict[str, Array]:
        # Create individual emotion-cause features
        emotion = jnp.concatenate((emotion_probs, emotion_hidden), axis=2)
        cause = jnp.concatenate((cause_probs, cause_hidden), axis=2)
        
        # Allocate joined emotion-cause table
        batch_size = emotion.shape[0]
        ec_table = []

        # Fill in joined emotion-cause table
        for i in range(self.max_con_len):
            for j in range(self.max_con_len):
                ec_table.append(jnp.concatenate((emotion[:, i, :], cause[:, j, :]), axis=1))

        # Reshape
        ec_table = jnp.array(ec_table).reshape((batch_size, self.max_con_len, self.max_con_len, -1))

        # Apply convolutional layers
        x = self.activ_fn(self.conv_features(self.dropout(ec_table, deterministic=not train)))
        span_start = self.conv_span_start(x)
        span_stop = self.conv_span_stop(x)

        # Aggregate results
        return { 'span_start': span_start, 'span_stop': span_stop }
