from typing import Tuple

import faiss
import numpy as np
from faiss import Index

from faiss_grpc.indexes.base import IndexWrapper


class FaissIndexWrapper(IndexWrapper):
    def __init__(self, index: Index):
        self.index = index

    @staticmethod
    def load_index(filename, **kwargs):
        index = faiss.read_index(filename)
        return FaissIndexWrapper.from_index(index)

    @staticmethod
    def from_index(index: Index, **kwargs):
        if 'nprobe' in kwargs.keys():
            index.nprobe = kwargs.get('nprobe')
        return FaissIndexWrapper(index)

    @property
    def dimension(self):
        return self.index.d

    @property
    def maximum_id(self):
        return self.index.ntotal - 1

    def search(
        self, vector: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        distances, ids = self.index.search(vector, k)
        return distances, ids

    def search_by_id(
        self, request_id: int, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        query = self.index.reconstruct_n(request_id, 1)
        distances, ids = self.index.search(query, k + 1)
        return distances, ids
