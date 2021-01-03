from environs import Env

from ann_grpc.ann_server import AnnServiceConfig, Server, ServerConfig

env = Env()
env.read_env()


def create_server() -> Server:
    server_config = ServerConfig(
        host=env.str('ANN_GRPC_HOST', '[::]'),
        port=env.int("ANN_GRPC_PORT", 50051),
        max_workers=env.int("ANN_GRPC_MAX_WORKERS", 10),
    )
    service_config = AnnServiceConfig(
        normalize_query=env.bool("ANN_GRPC_NORMALIZE_QUERY", False),
    )

    index_kwargs = {
        'nprobe': env.int("ANN_GRPC_FAISS_NPROBE", None),
        'dimension': env.int("ANN_GRPC_ANNOY_DIMENSION", None),
        'metric': env.str("ANN_GRPC_ANNOY_METRIC", None),
    }
    if index_kwargs['nprobe'] is None:
        del index_kwargs['nprobe']
    if index_kwargs['dimension'] is None:
        del index_kwargs['dimension']
    if index_kwargs['metric'] is None:
        del index_kwargs['metric']

    server = Server(
        env.str("ANN_GRPC_INDEX_PATH"),
        server_config,
        service_config,
        **index_kwargs
    )
    return server


def main() -> None:
    server = create_server()
    server.serve()


if __name__ == "__main__":
    main()
