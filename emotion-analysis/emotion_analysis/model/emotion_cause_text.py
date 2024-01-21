import jax
import typing as t
import flax.linen as nn
import jax.numpy as jnp
from jax import Array
from flax.linen.activation import gelu
from typing import Any
from jax.typing import ArrayLike
from dataclasses import dataclass
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers import FlaxAutoModel, FlaxPreTrainedModel
from transformers import AutoConfig, PretrainedConfig
from typing import Callable


@dataclass(frozen=True)
class PretrainedTextModel(object):
    module: nn.Module
    params: Any
    tokenizer: PreTrainedTokenizerFast


def load_text_model(model_repo: str) -> PretrainedTextModel:
    # Load tokenizer
    llm_tokenizer = t.cast(t.Any, AutoTokenizer.from_pretrained(model_repo, add_prefix_space=False))
    llm_tokenizer: PreTrainedTokenizerFast = llm_tokenizer

    # Load base configuration
    llm_config = t.cast(t.Any, AutoConfig.from_pretrained(model_repo))
    llm_config: PretrainedConfig = llm_config

    # Load base model using standard config
    llm = t.cast(t.Any, FlaxAutoModel.from_pretrained(model_repo, config=llm_config, add_pooling_layer=False))
    llm: FlaxPreTrainedModel = llm

    # Aggregate all elements
    return PretrainedTextModel(llm.module, llm.params, llm_tokenizer)


class PositionEmbeddings(nn.Module):
    def setup(self) -> None:
        pass

    def __call__(self) -> Any:
        pass


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


class EmotionCauseTextModel(nn.Module):
    # Pretrained BERT Architecture
    text_encoder: nn.Module
    # The number of emotions
    num_classes: int

    def setup(self) -> None:
        """Model Architecture"""
        self.transformer_layer = TransformerLayer(
            num_heads=4,
            embed_dim=768,
            input_dim=768,
            dense_dim=768,
            drop_a=0.1,
            drop_p=0.1,
            norm_eps=1e-6,
            activ_fn=gelu,
        )

        self.classifier = nn.Dense(
            features=self.num_classes,
            kernel_init=nn.initializers.glorot_normal(),
            bias_init=nn.initializers.zeros_init(),
            name='classifier',
            use_bias=True,
        )

    def encode(
        self,
        *,
        input_ids: Array,
        attn_mask: Array,
        train: bool,
    ) -> Array:
        """Encode each utterance individually from a batch of conversations.

        Args:
            input_ids (Array): (B, C, S)
            attn_mask (Array): Attention mask of size (BxC, S) containing 1/0 markings.
            training (bool): Changes internals such as dropout or normalization behavior.

        Returns:
            Array: An embedding for each token/utterance/conversation: (B, C, S, H)
        """
        # (B, C, S) -> (BxC, S)
        B, C, S = input_ids.shape
        input_ids = input_ids.reshape((-1, input_ids.shape[-1]))
        attn_mask = attn_mask.reshape((-1, attn_mask.shape[-1]))

        # Compute embedding for each token using a pretrained text encoder
        x = self.text_encoder(input_ids, attn_mask, deterministic=not train).last_hidden_state

        # (BxC, S) -> (B, C, S)
        return jnp.reshape(x, (B, C, S, -1))

    def __call__(
        self,
        *,
        input_ids: Array,
        attn_mask: Array,
        train: bool,
    ) -> Array:
        """Classify utterances from a conversation into 7 emotion categories.

        Args:
            input_ids (Array): token_ids of shape (B, C, S)
            attn_mask (Array): attention mask consisting of ones and zeros of shape (B, C, S)

        Returns:
            Array: An array of probabilities for each utterance of shape (B, C, S, E)
        """
        # (B, C, S) -> (B, C, S, H)
        x = self.encode(input_ids=input_ids, attn_mask=attn_mask, train=train)

        # (B, C, S) -> (B, C, 1, S, S)
        attn_mask = nn.make_attention_mask(attn_mask, attn_mask)

        # Apply Transformer Layers
        x = self.transformer_layer(x, attn_mask, train=train)

        # Apply classification layer
        x = self.classifier(x)

        # Logits
        return x
