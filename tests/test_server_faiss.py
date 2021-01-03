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
def grpc_server():
    pass


@pytest.fixture(scope='module')
def grpc_channel():
    return grpc.insecure_channel('localhost:50051')


@pytest.fixture(scope='module')
def index_filename_and_kwargs():
    index = create_faiss_index(64, 100000, 100)
    index_kwargs = {'nprobe': 10}
    faiss.write_index(index, 'index.faiss')
    index_wrapper = FaissIndexWrapper.from_index(index, **index_kwargs)
    return 'index.faiss', index_kwargs, index_wrapper


def test_server_heartbeat(grpc_client, index_filename_and_kwargs):
    _, _, index = index_filename_and_kwargs
    response = grpc_client.heartbeat(Empty())
    assert isinstance(response, HeartbeatResponse)
    assert response.message == "OK"


def test_server_search(grpc_client, index_filename_and_kwargs):
    _, _, index = index_filename_and_kwargs

    k = 1000
    val = np.ones(index.dimension, dtype=np.float32)
    vector = Vector(val=val)
    request = SearchRequest(query=vector, k=k)
    response = grpc_client.search(request)
    expected_distances, expected_ids = index.search(val, k)
    distances, ids = zip(
        *list(map(lambda x: (x.score, x.id), response.neighbors))
    )
    assert np.array_equal(expected_ids, ids)
    assert np.array_equal(expected_distances, distances)


def test_server_search_by_id(grpc_client, index_filename_and_kwargs):
    _, _, index = index_filename_and_kwargs

    request_id = 42
    k = 1000
    request = SearchByIdRequest(id=request_id, k=k)
    response = grpc_client.search_by_id(request)

    query = index.index.reconstruct_n(request_id, 1)
    expected_distances, expected_ids = index.search(query, k + 1)
    distances, ids = zip(
        *list(map(lambda x: (x.score, x.id), response.neighbors))
    )
    assert np.array_equal(expected_ids[1:], ids)
    assert np.array_equal(expected_distances[1:], distances)
