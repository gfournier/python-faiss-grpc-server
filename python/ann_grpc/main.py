from environs import Env

from ann_grpc.ann_server import AnnServiceConfig, Server, ServerConfig

env = Env()
env.read_env()


def main() -> None:
    server_config = ServerConfig(
        host=env.str('FAISS_GRPC_HOST', '[::]'),
        port=env.int("FAISS_GRPC_PORT", 50051),
        max_workers=env.int("FAISS_GRPC_MAX_WORKERS", 10),
    )
    service_config = AnnServiceConfig(
        normalize_query=env.bool("FAISS_GRPC_NORMALIZE_QUERY", False),
    )

    index_kwargs = {'nprobe': env.int("FAISS_GRPC_NPROBE", None)}

    server = Server(
        env.str("FAISS_GRPC_INDEX_PATH"),
        server_config,
        service_config,
        **index_kwargs
    )
    server.serve()


if __name__ == "__main__":
    main()
