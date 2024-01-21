import jax
import typing as t
import flax.linen as nn
from jax import Array
from typing import Any
from jax.typing import ArrayLike
from dataclasses import dataclass
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers import FlaxAutoModel, FlaxPreTrainedModel
from transformers import AutoConfig, PretrainedConfig


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
