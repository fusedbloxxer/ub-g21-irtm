import typing as t
from typing import Any
from dataclasses import dataclass
import jax
from jax import Array
from jax.typing import ArrayLike
import flax.linen as nn
from transformers import RobertaTokenizerFast, RobertaConfig, FlaxRobertaModel, FlaxPreTrainedModel, PreTrainedTokenizerFast


@dataclass(frozen=True)
class PretrainedTextModel(object):
    module: nn.Module
    params: Any
    tokenizer: PreTrainedTokenizerFast


def load_text_model() -> PretrainedTextModel:
    # Load tokenizer
    llm_tokenizer = t.cast(t.Any, RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=False))
    llm_tokenizer: RobertaTokenizerFast = llm_tokenizer

    # Load base configuration
    llm_config = t.cast(t.Any, RobertaConfig.from_pretrained('roberta-base'))
    llm_config: RobertaConfig = llm_config

    # Load base model using standard config
    llm = t.cast(t.Any, FlaxRobertaModel.from_pretrained('roberta-base', config=llm_config, add_pooling_layer=False))
    llm: FlaxRobertaModel = llm

    # Aggregate all elements
    return PretrainedTextModel(llm.module, llm.params, llm_tokenizer)


class EmotionCauseTextModel(nn.Module):
    # Pretrained MLM Architecture
    text_encoder: nn.Module
    # The number of emotions
    num_classes: int

    def setup(self) -> None:
        """Model Architecture"""
        self.classifier = nn.Dense(features=self.num_classes,
                                   name='classifier',
                                   kernel_init=nn.initializers.kaiming_normal())

    def __call__(self,
                 input_ids: Array,
                 attention_mask: Array) -> Array:
        x = self.text_encoder(input_ids, attention_mask, output_hidden_states=True).last_hidden_state
        x = self.classifier(x)
        return x
