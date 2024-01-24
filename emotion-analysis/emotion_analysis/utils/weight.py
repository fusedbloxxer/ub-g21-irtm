from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from jax import Array
from numpy import ndarray, vectorize
from sklearn.utils.class_weight import compute_class_weight

ArrayLike = Array | ndarray


@dataclass
class ClassWeight(ABC):
    num_classes: int
    labels: Sequence[int]

    def __post_init__(self) -> None:
        assert np.unique(self.labels).shape[0] == self.num_classes, 'samples_per_class does not match num_classes'

    @abstractmethod
    def __call__(self, labels: ArrayLike) -> ArrayLike:
        raise NotImplementedError()


    @property
    @abstractmethod
    def weight_per_class(self) -> ArrayLike:
        raise NotImplementedError()


@dataclass
class INSWeight(ClassWeight):
    def __post_init__(self) -> None:
        super(INSWeight, self).__post_init__()
        self.__getitem__ = vectorize(lambda x: self.weight_per_class[x])

    def __call__(self, labels: ArrayLike) -> ArrayLike:
        return self.__getitem__(labels)

    @property
    def weight_per_class(self) -> ArrayLike:
        if not hasattr(self, '_weight_per_class'):
            self._weight_per_class = compute_class_weight('balanced', classes=np.arange(self.num_classes), y=self.labels)
        return self._weight_per_class

