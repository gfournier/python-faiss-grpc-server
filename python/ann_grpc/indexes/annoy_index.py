from typing import Tuple

from annoy import AnnoyIndex
import numpy as np

from ann_grpc.indexes.base import IndexWrapper


class AnnoyIndexWrapper(IndexWrapper):

    def __init__(self, index: AnnoyIndex, dimension: int):
        self.index = index
        self._dimension = dimension

    @staticmethod
    def load_index(filename, **kwargs):
        if 'dimension' not in kwargs:
            raise ValueError("'dimension' size of vectors must be specified.")
        if 'metric' not in kwargs:
            raise ValueError("'metric' for distance computation must be specified.")
        index = AnnoyIndex(kwargs.get('dimension'), kwargs.get('metric'))
        index.load(filename)
        return AnnoyIndexWrapper.from_index(index)

    @staticmethod
    def from_index(index: AnnoyIndex, **kwargs):
        return AnnoyIndexWrapper(index, kwargs.get('dimension'))

    @property
    def dimension(self):
        return self._dimension

    @property
    def maximum_id(self):
        return self.index.get_n_items()

    def search(self, vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        ids, distances = self.index.get_nns_by_vector(vector, k, include_distances=True)
        return distances, ids

    def search_by_id(self, request_id: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
        ids, distances = self.index.get_nns_by_item(request_id, k, include_distances=True)
        return distances, ids
