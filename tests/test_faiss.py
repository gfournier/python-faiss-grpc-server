import faiss
import grpc
import numpy as np
import pytest
from ann_grpc.indexes.faiss_index import FaissIndexWrapper
from ann_grpc.proto.ann_pb2 import (
    HeartbeatResponse,
    SearchByIdRequest,
    SearchRequest,
    Vector,
)
from google.protobuf.empty_pb2 import Empty

from tests.util import create_faiss_index


@pytest.fixture(scope='module')
def index():
    return FaissIndexWrapper.from_index(
        create_faiss_index(64, 100000, 100), **{'nprobe': 10}
    )


def test_successful_heartbeat(grpc_stub):
    response = grpc_stub.heartbeat(Empty())
    assert isinstance(response, HeartbeatResponse)
    assert response.message == "OK"


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


def test_failed_different_nprobe_search(grpc_sub_and_index):
    grpc_stub, index = grpc_sub_and_index

    k = 1000
    val = np.ones(index.dimension, dtype=np.float32)
    vector = Vector(val=val)
    request = SearchRequest(query=vector, k=k)
    response = grpc_stub.search(request)

    index.index = faiss.clone_index(index.index)
    index.index.nprobe = 1
    expected_distances, expected_ids = index.search(val, k)
    distances, ids = zip(
        *list(map(lambda x: (x.score, x.id), response.neighbors))
    )
    assert len(distances) == k
    assert len(ids) == k
    assert not np.array_equal(expected_ids, ids)
    assert not np.array_equal(expected_distances, distances)


def test_failed_illegal_query_dimension_search(grpc_sub_and_index):
    grpc_stub, index = grpc_sub_and_index

    k = 10
    val = np.ones(index.dimension * 2, dtype=np.float32)
    vector = Vector(val=val)
    request = SearchRequest(query=vector, k=k)
    try:
        grpc_stub.search(request)
        raise Exception("Search request should have failed.")
    except grpc.RpcError as e:
        assert (
            e.details()
            == f"query vector dimension mismatch expected {index.dimension} "
            f"but passed {index.dimension * 2}"
        )


def test_successful_search_by_id(grpc_sub_and_index):
    grpc_stub, index = grpc_sub_and_index

    request_id = 42
    k = 1000
    request = SearchByIdRequest(id=request_id, k=k)
    response = grpc_stub.search_by_id(request)

    query = index.index.reconstruct_n(request_id, 1)
    expected_distances, expected_ids = index.search(query, k + 1)
    distances, ids = zip(
        *list(map(lambda x: (x.score, x.id), response.neighbors))
    )
    assert np.array_equal(expected_ids[1:], ids)
    assert np.array_equal(expected_distances[1:], distances)


def test_failed_unknown_id_search_by_id(grpc_sub_and_index):
    grpc_stub, index = grpc_sub_and_index

    request_id = index.maximum_id * 2
    k = 1000
    request = SearchByIdRequest(id=request_id, k=k)
    try:
        grpc_stub.search_by_id(request)
        raise Exception("Search request should have failed.")
    except grpc.RpcError as e:
        assert (
            e.details() == f"request id must be 0 <= id <= {index.maximum_id}"
        )
