import abc
from typing import Tuple

import numpy as np


class IndexWrapper(metaclass=abc.ABCMeta):
    @staticmethod
    def load_index(filename, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def from_index(index, **kwargs):
        raise NotImplementedError()

    @property
    def dimension(self):
        raise NotImplementedError()

    @property
    def maximum_id(self):
        raise NotImplementedError()

    def search(
        self, vector: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def search_by_id(
        self, request_id: int, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()
