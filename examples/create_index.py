import random

import numpy as np

import faiss
from faiss import Index

from annoy import AnnoyIndex


def create_faiss_index(dim: int, n_samples: int, n_list: int) -> Index:
    # following code is same as sample of faiss github wiki
    #   https://github.com/facebookresearch/faiss/wiki/Getting-started
    #   https://github.com/facebookresearch/faiss/wiki/Faster-search
    np.random.seed(1234)
    xb = np.random.random((n_samples, dim)).astype('float32')
    xb[:, 0] += np.arange(n_samples) / 1000.0

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, n_list)
    index.train(xb)
    index.add(xb)

    return index


def create_annoy_index(dim, db_size, n_trees, metric):
    # https://github.com/spotify/annoy
    index = AnnoyIndex(dim, metric)  # Length of item vector that will be indexed
    for i in range(db_size):
        v = [random.gauss(0, 1) for _ in range(dim)]
        index.add_item(i, v)
    index.build(n_trees)
    return index


def run() -> None:
    index = create_faiss_index(64, 100000, 100)
    faiss.write_index(index, 'index.faiss')

    index = create_annoy_index(64, 100000, 10, 'euclidean')
    index.save('index.ann')


if __name__ == "__main__":
    run()
