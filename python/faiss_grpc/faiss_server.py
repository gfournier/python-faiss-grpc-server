from concurrent import futures
from dataclasses import dataclass
from typing import List, Optional

import grpc
import numpy as np

from faiss_grpc.indexes.base import IndexWrapper
from faiss_grpc.indexes.faiss_index import FaissIndexWrapper
from faiss_grpc.proto.faiss_pb2 import (
    HeartbeatResponse,
    Neighbor,
    SearchByIdResponse,
    SearchResponse,
    SearchRequest,
    SearchByIdRequest,
)
from faiss_grpc.proto.faiss_pb2_grpc import (
    FaissServiceServicer,
    add_FaissServiceServicer_to_server,
)


@dataclass(eq=True, frozen=True)
class ServerConfig:
    host: str = '[::]'
    port: int = 50051
    max_workers: int = 10


@dataclass(eq=True, frozen=True)
class FaissServiceConfig:
    normalize_query: bool = False


@dataclass(eq=True, frozen=True)
class AnnoyIndexConfig:
    dim: int
    metric: str


@dataclass(eq=True, frozen=True)
class FaissIndexConfig:
    nprobe: Optional[int] = None


class FaissIndexServicer(FaissServiceServicer):
    def __init__(self, index: IndexWrapper, config: FaissServiceConfig) -> None:
        self.index = index
        self.config = config

    def Search(self, request: SearchRequest, context) -> SearchResponse:
        query = request.query.val
        if len(query) != self.index.dimension:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            msg = (
                'query vector dimension mismatch '
                f'expected {self.index.dimension} but passed {len(query)}'
            )
            context.set_details(msg)
            return SearchResponse()

        if self.config.normalize_query:
            query = self.normalize(query)

        distances, ids = self.index.search(query, request.k)

        neighbors: List[Neighbor] = []
        for d, i in zip(distances, ids):
            if i != -1:
                neighbors.append(Neighbor(id=i, score=d))

        return SearchResponse(neighbors=neighbors)

    def SearchById(
        self, request: SearchByIdRequest, context
    ) -> SearchByIdResponse:
        request_id = request.id
        maximum_id = self.index.maximum_id
        if not (0 <= request_id <= maximum_id):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            msg = f'request id must be 0 <= id <= {maximum_id}'
            context.set_details(msg)
            return SearchByIdResponse()

        distances, ids = self.index.search_by_id(request_id, request.k)

        neighbors: List[Neighbor] = []
        for d, i in zip(distances, ids):
            if i not in [request_id, -1]:
                neighbors.append(Neighbor(id=i, score=d))

        return SearchByIdResponse(request_id=request_id, neighbors=neighbors)

    def Heartbeat(self, request, context) -> HeartbeatResponse:
        return HeartbeatResponse(message='OK')

    @staticmethod
    def normalize(vec: np.ndarray) -> np.ndarray:
        return vec / np.linalg.norm(vec, axis=1, keepdims=True)


class Server:
    def __init__(
        self,
        index_path: str,
        server_config: ServerConfig,
        service_config: FaissServiceConfig,
        **kwargs,
    ) -> None:
        index = FaissIndexWrapper.load_index(index_path, **kwargs)
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=server_config.max_workers)
        )
        add_FaissServiceServicer_to_server(
            FaissIndexServicer(index, service_config), self.server
        )
        self.server.add_insecure_port(
            f'{server_config.host}:{server_config.port}'
        )

    def serve(self) -> None:
        self.server.start()
        self.server.wait_for_termination()
