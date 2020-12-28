import pytest

from ann_grpc.ann_server import AnnServiceConfig, AnnServiceServicerImpl
from ann_grpc.proto.ann_pb2_grpc import add_AnnServiceServicer_to_server, AnnServiceStub


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
