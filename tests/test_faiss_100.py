import numpy as np
import pytest
from ann_grpc.indexes.faiss_index import FaissIndexWrapper
from ann_grpc.proto.ann_pb2 import SearchRequest, Vector

from tests.util import create_faiss_index


@pytest.fixture(scope='module')
def index():
    return FaissIndexWrapper.from_index(
        create_faiss_index(200, 100000, 100), **{'nprobe': 10}
    )


def test_successful_search(grpc_sub_and_index):
    grpc_stub, index = grpc_sub_and_index
    # k must be set large value,
    # because of avoiding to miss error case came from small nprobe value.
    # if both nprobe and k are set small, search result would be same, even
    # if failed to set nprobe on server side.
    k = 1000
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
