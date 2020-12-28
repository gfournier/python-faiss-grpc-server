import numpy as np
import pytest

from ann_grpc.indexes.annoy_index import AnnoyIndexWrapper
from ann_grpc.proto.ann_pb2 import Vector, SearchRequest
from tests.util import create_annoy_index


@pytest.fixture(scope='module')
def index():
    dim = 200
    return AnnoyIndexWrapper.from_index(
        create_annoy_index(dim, 1000, 10, 'angular'), **{'dimension': dim}
    )


def test_successful_search(grpc_sub_and_index):
    grpc_stub, index = grpc_sub_and_index

    k = 100
    val = np.ones(index.dimension, dtype=np.float32)
    vector = Vector(val=val)
    request = SearchRequest(query=vector, k=k)
    response = grpc_stub.search(request)
    expected_distances, expected_ids = index.search(val, k)
    distances, ids = zip(
        *list(map(lambda x: (x.score, x.id), response.neighbors))
    )
    assert np.array_equal(expected_ids, ids)
    assert np.array_equal(expected_distances, distances)
