from flax.traverse_util import path_aware_map
from dataclasses import dataclass, field
from flax.core.scope import VariableDict
from abc import ABC, abstractmethod
from typing import List, Any, Tuple, Literal
from enum import StrEnum
from jax import Array


class ParamLabel(StrEnum):
    TRAINABLE = 'trainable'
    FROZEN = 'frozen'


class FineTune(StrEnum):
    HEAD = 'head'
    FULL = 'full'


@dataclass
class FineTuneStrategy(ABC):
    def param_labels(self, params: VariableDict) -> VariableDict:
        return path_aware_map(self.param_mapping, params)

    @abstractmethod
    def param_mapping(self, path: Tuple[str, ...], param: Array) -> ParamLabel:
        raise NotImplementedError()

    @classmethod
    def from_enum(cls, enum: FineTune | str):
        match enum:
            case FineTune.FULL:
                return FullFineTuneStrategy()
            case FineTune.HEAD:
                return HeadFineTuneStrategy()
            case _:
                raise ValueError('no finetune strategy exists for {}'.format(enum))


@dataclass
class FullFineTuneStrategy(FineTuneStrategy):
    def param_mapping(self, path: Tuple[str, ...], param: Array) -> ParamLabel:
        if 'pos_embeddings' in path:
            return ParamLabel.FROZEN
        else:
            return ParamLabel.TRAINABLE


@dataclass
class HeadFineTuneStrategy(FineTuneStrategy):
    def param_mapping(self, path: Tuple[str, ...], param: Array) -> ParamLabel:
        if 'text_encoder' in path:
            return ParamLabel.FROZEN
        if 'pos_embeddings' in path:
            return ParamLabel.FROZEN
        return ParamLabel.TRAINABLE
