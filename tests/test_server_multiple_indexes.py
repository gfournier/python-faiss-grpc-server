import json

import faiss
import grpc
import numpy as np
import pytest
from ann_grpc.indexes.annoy_index import AnnoyIndexWrapper
from ann_grpc.indexes.faiss_index import FaissIndexWrapper
from ann_grpc.proto.ann_pb2 import (
    HeartbeatResponse,
    SearchByIdRequest,
    SearchRequest,
    Vector,
)
from google.protobuf.empty_pb2 import Empty

from tests.util import create_annoy_index, create_faiss_index


@pytest.fixture(scope='module')
def grpc_server():
    pass


@pytest.fixture(scope='module')
def grpc_channel():
    return grpc.insecure_channel('localhost:50051')


@pytest.fixture(scope='module')
def index_filename_and_kwargs():
    faiss_index_kwargs = {'nprobe': 10}
    faiss_index = create_faiss_index(64, 100000, 100)
    faiss.write_index(faiss_index, 'index.faiss')

    dim = 200
    metric = 'angular'
    annoy_index_kwargs = {'dimension': dim, 'metric': metric}
    annoy_index = create_annoy_index(dim, 1000, 10, metric)
    annoy_index.save('index.ann')

    index = {
        'index1': FaissIndexWrapper.from_index(
            faiss_index, **faiss_index_kwargs
        ),
        'index2': AnnoyIndexWrapper.from_index(
            annoy_index, **annoy_index_kwargs
        ),
    }

    with open('index.json', 'wt') as f:
        json.dump(
            [
                {
                    'name': 'index1',
                    'type': 'faiss',
                    'url': 'index.faiss',
                    'params': faiss_index_kwargs,
                },
                {
                    'name': 'index2',
                    'type': 'ann',
                    'url': 'index.ann',
                    'params': annoy_index_kwargs,
                },
            ],
            f,
        )

    return 'index.json', {}, index


def test_server_heartbeat(grpc_client, index_filename_and_kwargs):
    _, _, index = index_filename_and_kwargs
    response = grpc_client.heartbeat(Empty())
    assert isinstance(response, HeartbeatResponse)
    assert response.message == "OK"


def test_server_search(grpc_client, index_filename_and_kwargs):
    _, _, index = index_filename_and_kwargs

    k = 1000
    val = np.ones(index['index1'].dimension, dtype=np.float32)
    vector = Vector(val=val)
    request = SearchRequest(query=vector, k=k, name='index1')
    response = grpc_client.search(request)
    expected_distances, expected_ids = index['index1'].search(val, k)
    distances, ids = zip(
        *list(map(lambda x: (x.score, x.id), response.neighbors))
    )
    assert np.array_equal(expected_ids, ids)
    assert np.array_equal(expected_distances, distances)

    val = np.ones(index['index2'].dimension, dtype=np.float32)
    vector = Vector(val=val)
    request = SearchRequest(query=vector, k=k, name='index2')
    response = grpc_client.search(request)
    expected_distances, expected_ids = index['index2'].search(val, k)
    distances, ids = zip(
        *list(map(lambda x: (x.score, x.id), response.neighbors))
    )
    assert np.array_equal(expected_ids, ids)
    assert np.array_equal(expected_distances, distances)


def test_server_search_by_id(grpc_client, index_filename_and_kwargs):
    _, _, index = index_filename_and_kwargs

    k = 1000
    request_id = 42
    request = SearchByIdRequest(id=request_id, k=k, name='index1')
    response = grpc_client.search_by_id(request)
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
    response = grpc_client.search_by_id(request)
    expected_distances, expected_ids = index['index2'].search_by_id(
        request_id, k + 1
    )
    distances, ids = zip(
        *list(map(lambda x: (x.score, x.id), response.neighbors))
    )
    assert np.array_equal(expected_ids[1:], ids)
    assert np.array_equal(expected_distances[1:], distances)
