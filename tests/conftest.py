import pytest

from faiss_grpc.faiss_server import FaissServiceConfig, FaissIndexServicer
from faiss_grpc.proto.faiss_pb2_grpc import add_FaissServiceServicer_to_server, FaissServiceStub


@pytest.fixture(scope='module')
def config():
    config = FaissServiceConfig(normalize_query=False)
    return config


@pytest.fixture(scope='module')
def grpc_add_to_server():
    return add_FaissServiceServicer_to_server


@pytest.fixture(scope='module')
def grpc_servicer(index, config):
    return FaissIndexServicer(index, config)


@pytest.fixture(scope='module')
def grpc_stub_cls(grpc_channel):
    return FaissServiceStub


@pytest.fixture(scope='module')
def grpc_sub_and_index(grpc_stub, index):
    return grpc_stub, index
