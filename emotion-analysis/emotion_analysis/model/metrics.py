from abc import ABC, abstractmethod, abstractstaticmethod
from dataclasses import dataclass, field
from typing import Any, List, Tuple, overload, Literal

import evaluate
import jax.numpy as jnp
import numpy as np
from jax import Array


@dataclass
class Metric:
    @abstractmethod
    def update(self, value: Any):
        raise NotImplementedError()

    @abstractmethod
    def compute(self, values: Any | None=None):
        raise NotImplementedError()

    @abstractmethod
    def reset():
        raise NotImplementedError()


@dataclass
class MeanMetric(Metric):
    values: List[float] = field(init=False, default_factory=list)

    def update(self, value: float) -> None:
        self.values.append(value)

    def compute(self, values: Array | None=None) -> float:
        mean: float
        if   isinstance(values, Array):
            mean = np.mean(values).item()
        elif len(self.values) == 0:
            mean = 0.0
        else:
            mean = np.mean(self.values).item()
        self.reset()
        return mean

    def reset(self) -> None:
        self.values = []


@dataclass
class F1Metric(Metric):
    average: Literal['binary', 'micro', 'macro', 'weighted'] = 'weighted'

    def __post_init__(self):
        self.f1_score = evaluate.load('f1')

    def update(self, *, predictions: Array, references: Array) -> None:
        self.f1_score.add_batch(predictions=predictions, references=references)

    def compute(self, *, predictions: Array | None=None, references: Array | None=None) -> float:
        score = self.f1_score.compute(predictions=predictions, references=references, average=self.average)
        assert score is not None, 'got none in f1 metric'
        return score['f1']

    def reset(self) -> None:
        pass
