import flax.linen as nn
import jax.numpy as jnp
from flax.linen.activation import gelu
from jax import Array

from .modules import PositionEmbeddings, TransformerLayer


class EmotionCauseTextModel(nn.Module):
    # Pretrained BERT Architecture
    text_encoder: nn.Module
    # The number of emotions
    num_classes: int

    def setup(self) -> None:
        """Model Architecture"""
        self.pos_embeddings = PositionEmbeddings(
            hidden_dim=768,
            max_seq_len=93,
            n=10_000,
        )

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
            attn_mask (Array): Attention mask of size (B, C, S) containing 1/0 markings.
            train (bool): Changes internals such as dropout or normalization behavior.

        Returns:
            Array: An embedding for each utterance: (B, C, H)
        """
        # (B, C, S) -> (BxC, S)
        B, C, _ = input_ids.shape
        input_ids = input_ids.reshape((-1, input_ids.shape[-1]))
        attn_mask = attn_mask.reshape((-1, attn_mask.shape[-1]))

        # Extract pretrained embedding for [CLS] token for each utterance
        x = self.text_encoder(input_ids, attn_mask, deterministic=not train).last_hidden_state[:, 0, :]

        # (BxC, H) -> (B, C, H)
        return jnp.reshape(x, (B, C, -1))

    def __call__(
        self,
        *,
        input_ids: Array,
        uttr_attn_mask: Array,
        conv_attn_mask: Array,
        train: bool,
    ) -> Array:
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

        # Apply Transformer Layers
        x = self.pos_embeddings(x)
        x = self.transformer_layer(x, attn_mask, train=train)

        # Apply classification layer
        x = self.classifier(x)

        # Logits
        return x
