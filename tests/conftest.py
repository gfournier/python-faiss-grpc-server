import os
from concurrent.futures.thread import ThreadPoolExecutor

import pytest
from ann_grpc.ann_server import AnnServiceConfig, AnnServiceServicerImpl
from ann_grpc.main import create_server
from ann_grpc.proto.ann_pb2_grpc import (
    AnnServiceStub,
    add_AnnServiceServicer_to_server,
)
from environs import Env


@pytest.fixture(scope='module')
def config():
    config = AnnServiceConfig(normalize_query=False)
    return config


@pytest.fixture(scope='module')
def grpc_add_to_server():
    return add_AnnServiceServicer_to_server


@pytest.fixture(scope='module')
def grpc_servicer(index, config):
    return AnnServiceServicerImpl(index, config)


@pytest.fixture(scope='module')
def grpc_stub_cls(grpc_channel):
    return AnnServiceStub


@pytest.fixture(scope='module')
def grpc_sub_and_index(grpc_stub, index):
    return grpc_stub, index


@pytest.fixture(scope='module')
def server_env_variables(index_filename_and_kwargs):
    os.environ['ANN_GRPC_INDEX_PATH'] = index_filename_and_kwargs[0]
    if 'nprobe' in index_filename_and_kwargs[1]:
        os.environ['ANN_GRPC_FAISS_NPROBE'] = str(
            index_filename_and_kwargs[1]['nprobe']
        )
    if 'dimension' in index_filename_and_kwargs[1]:
        os.environ['ANN_GRPC_ANNOY_DIMENSION'] = str(
            index_filename_and_kwargs[1]['dimension']
        )
    if 'metric' in index_filename_and_kwargs[1]:
        os.environ['ANN_GRPC_ANNOY_METRIC'] = str(
            index_filename_and_kwargs[1]['metric']
        )


@pytest.fixture(scope='module')
def grpc_client(server_env_variables, grpc_stub):
    env = Env()
    env.read_env()
    server = create_server()
    executor = ThreadPoolExecutor(max_workers=1)
    server_future = executor.submit(server.serve)
    yield grpc_stub
    server.stop()
    server_future.result()
