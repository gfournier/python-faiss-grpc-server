import grpc
import numpy as np
import pytest
from ann_grpc.indexes.faiss_index import FaissIndexWrapper
from ann_grpc.proto.ann_pb2 import SearchByIdRequest, SearchRequest, Vector

from tests.util import create_faiss_index


@pytest.fixture(scope='module')
def index():
    return {
        'index1': FaissIndexWrapper.from_index(
            create_faiss_index(64, 100000, 100), **{'nprobe': 10}
        ),
        'index2': FaissIndexWrapper.from_index(
            create_faiss_index(128, 100000, 100), **{'nprobe': 10}
        ),
    }


def test_successful_search(grpc_sub_and_index):
    grpc_stub, index = grpc_sub_and_index

    k = 1000
    val = np.ones(index['index1'].dimension, dtype=np.float32)
    vector = Vector(val=val)
    request = SearchRequest(query=vector, k=k, name='index1')
    response = grpc_stub.search(request)
    expected_distances, expected_ids = index['index1'].search(val, k)
    distances, ids = zip(
        *list(map(lambda x: (x.score, x.id), response.neighbors))
    )
    assert np.array_equal(expected_ids, ids)
    assert np.array_equal(expected_distances, distances)

    val = np.ones(index['index2'].dimension, dtype=np.float32)
    vector = Vector(val=val)
    request = SearchRequest(query=vector, k=k, name='index2')
    response = grpc_stub.search(request)
    expected_distances, expected_ids = index['index2'].search(val, k)
    distances, ids = zip(
        *list(map(lambda x: (x.score, x.id), response.neighbors))
    )
    assert np.array_equal(expected_ids, ids)
    assert np.array_equal(expected_distances, distances)


def test_failed_invalid_index_name(grpc_sub_and_index):
    grpc_stub, index = grpc_sub_and_index

    k = 1000
    val = np.ones(index['index1'].dimension, dtype=np.float32)
    vector = Vector(val=val)
    try:
        grpc_stub.search(SearchRequest(query=vector, k=k))
        raise ValueError("Search request should have failed.")
    except grpc.RpcError as e:
        assert e.details() == "Unknown index name: 'default'"
    try:
        grpc_stub.search(SearchRequest(query=vector, k=k, name='fake_index'))
        raise ValueError("Search request should have failed.")
    except grpc.RpcError as e:
        assert e.details() == "Unknown index name: 'fake_index'"

    request_id = 42
    try:
        grpc_stub.search_by_id(SearchByIdRequest(id=request_id, k=k))
        raise ValueError("Search request should have failed.")
    except grpc.RpcError as e:
        assert e.details() == "Unknown index name: 'default'"
    try:
        grpc_stub.search_by_id(
            SearchByIdRequest(id=request_id, k=k, name='fake_index')
        )
        raise ValueError("Search request should have failed.")
    except grpc.RpcError as e:
        assert e.details() == "Unknown index name: 'fake_index'"


def test_successful_search_by_id(grpc_sub_and_index):
    grpc_stub, index = grpc_sub_and_index

    k = 1000
    request_id = 42
    request = SearchByIdRequest(id=request_id, k=k, name='index1')
    response = grpc_stub.search_by_id(request)
    expected_distances, expected_ids = index['index1'].search_by_id(
        request_id, k + 1
    )
    distances, ids = zip(
        *list(map(lambda x: (x.score, x.id), response.neighbors))
    )
    assert np.array_equal(expected_ids[1:], ids)
    assert np.array_equal(expected_distances[1:], distances)

    request_id = 100
    request = SearchByIdRequest(id=request_id, k=k, name='index2')
    response = grpc_stub.search_by_id(request)
    expected_distances, expected_ids = index['index2'].search_by_id(
        request_id, k + 1
    )
    distances, ids = zip(
        *list(map(lambda x: (x.score, x.id), response.neighbors))
    )
    assert np.array_equal(expected_ids[1:], ids)
    assert np.array_equal(expected_distances[1:], distances)
