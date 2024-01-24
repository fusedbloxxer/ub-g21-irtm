from typing import Callable, Dict, Literal, Tuple

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.activation import relu
from jax import Array
from jax.typing import ArrayLike

from .modules import TransformerClassifier, TransformerEncoder


class EmotionCauseTextModel(nn.Module):
    # Pretrained Transformer Architecture
    text_encoder: nn.Module
    # The number of emotions
    num_emotions: int=7
    # Labels for whether a clause is a cause
    num_causes: int=2
    # The number of Transformer layers
    num_layers: int=2
    # The number of attention heads in each layer
    num_heads: int=4
    # The embedding dim used per AttentionHead
    embed_dim: int=768
    # The input dim that a Transformer layer receives
    input_dim: int=768
    # The hidden dim of the inner MLP layer
    dense_dim: int=768
    # The dropout rate used in attention heads
    drop_a: float=0.2
    # The dropout rate used in-between layers
    drop_p: float=0.2
    # The eps value used by normalization layers
    norm_eps: float=1e-6
    # The activation function used in MLP layers
    activ_fn: Callable[[ArrayLike], Array]=relu
    # The maximum conversation length to be processed
    max_con_len: int=33

    def setup(self) -> None:
        """Model Architecture"""
        self.classifier_emotion = TransformerClassifier(
            num_classes=self.num_emotions,
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

        self.classifier_cause = TransformerClassifier(
            num_classes=self.num_causes,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            embed_dim=self.embed_dim,
            input_dim=self.input_dim,
            dense_dim=self.dense_dim,
            drop_a=self.drop_a,
            drop_p=self.drop_p,
            norm_eps=self.norm_eps,
            activ_fn=self.activ_fn,
            max_con_len=self.max_con_len
        )

        self.classifier = nn.Dense(
            features=self.num_emotions,
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
            attn_mask (Array): Attention mask of size (B, C, S) containing 1/0 markings.
            train (bool): Changes internals such as dropout or normalization behavior.

        Returns:
            Array: An embedding for each utterance: (B, C, H)
        """
        # (B, C, S) -> (BxC, S)
        B, C, _ = input_ids.shape
        input_ids = input_ids.reshape((-1, input_ids.shape[-1]))
        attn_mask = attn_mask.reshape((-1, attn_mask.shape[-1]))

        # Extract pretrained embedding for each utterance
        x = self.text_encoder(input_ids, attn_mask, deterministic=not train).last_hidden_state.mean(axis=1)

        # (BxC, H) -> (B, C, H)
        return jnp.reshape(x, (B, C, -1))

    def __call__(
        self,
        *,
        input_ids: Array,
        uttr_attn_mask: Array,
        conv_attn_mask: Array,
        train: bool,
    ) -> Dict[Literal['emotion', 'cause'], Dict[Literal['out', 'hidden'], Array]]:
        """Classify utterances from a conversation into 7 emotion categories.

        Args:
            input_ids (Array): token_ids of shape (B, C, S)
            attn_mask (Array): attention mask consisting of ones and zeros of shape (B, C, S)

        Returns:
            Array: An array of probabilities for each utterance of shape (B, C, S, E)
        """
        # Encode utterances individually: (B, C, S) -> (B, C, H)
        x = self.encode(input_ids=input_ids, attn_mask=uttr_attn_mask, train=train)

        # Ignore padded conversations: (B, C) -> (B, 1, C, C)
        attn_mask = nn.make_attention_mask(conv_attn_mask, conv_attn_mask)

        # Apply Transformers to determine cause and emotion for each utterance
        emotion = self.classifier_emotion(x, attn_mask, train)
        cause = self.classifier_cause(x, attn_mask, train)

        # Logits
        return { 'emotion': emotion, 'cause': cause }
