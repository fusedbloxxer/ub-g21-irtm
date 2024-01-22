from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from flax.linen import Module
from transformers import (AutoConfig, AutoTokenizer, FlaxAutoModel,
                          FlaxPreTrainedModel, PretrainedConfig,
                          PreTrainedTokenizerFast)
from .. import config


@dataclass(frozen=True)
class PretrainedTextModel(object):
    module: Module
    params: Any
    tokenizer: PreTrainedTokenizerFast


def load_text_model(model_repo: str, cache_dir: Path = config.cache_dir) -> PretrainedTextModel:
    # Load tokenizer
    llm_tokenizer = cast(Any, AutoTokenizer.from_pretrained(model_repo, add_prefix_space=False, cache_dir=str(cache_dir)))
    llm_tokenizer: PreTrainedTokenizerFast = llm_tokenizer

    # Load base configuration
    llm_config = cast(Any, AutoConfig.from_pretrained(model_repo, cache_dir=str(cache_dir)))
    llm_config: PretrainedConfig = llm_config

    # Load base model using standard config
    llm = cast(Any, FlaxAutoModel.from_pretrained(model_repo, config=llm_config, add_pooling_layer=False, cache_dir=str(cache_dir)))
    llm: FlaxPreTrainedModel = llm

    # Aggregate all elements
    return PretrainedTextModel(llm.module, llm.params, llm_tokenizer)
