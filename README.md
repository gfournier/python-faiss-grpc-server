# python-faiss-grpc-server

Python gRPC server for apploximate nearest neighbor by [Faiss](https://github.com/facebookresearch/faiss).

## Installation

You can install as package by pip install on repository root.

```sh
pip install .
```

## Usage

You can start gRPC server by running following command.

```sh
python python/faiss_grpc/main.py
```

### Environment variable

Python Faiss gRPC server has some environment variables starts with prefix `FAISS_GRPC_`.

| Variable                   | Default | Description                                                            | Required |
| :------------------------- | :------ | :--------------------------------------------------------------------- | -------- |
| FAISS_GRPC_INDEX_PATH      | -       | Path to Faiss index                                                    | o        |
| FAISS_GRPC_NORMALIZE_QUERY | False   | Normalize query for search (This is useful to cosine distance metrics) | x        |
| FAISS_GRPC_NPROBE          | None    | Faiss nprobe parameter                                                 | x        |
| FAISS_GRPC_HOST            | [::]    | gRPC server host                                                       | x        |
| FAISS_GRPC_PORT            | 50051   | gRPC server listening port                                             | x        |
| FAISS_GRPC_MAX_WORKERS     | 10      | Maximum number of gRPC server workers                                  | x        |

## Examples

Client side code is under the `examples/client.py`.
You can run following command or directory running on interpreter.

```sh
cd examples
python client.py
```

## Development

### Generate python code

Following command will generate two python grpc code `faiss_pb2.py` and `faiss_pb2_grpc.py` under the `python/faiss_grpc/proto`.
If you update grpc definition `proto/faiss.proto`, you should regenerate at first.

```sh
make codegen
```

Then you have to fix import path in `faiss_pb2_grpc.py`.

```py
# Before
import faiss_pb2 as faiss__pb2
# After
import faiss_grpc.proto.faiss_pb2 as faiss__pb2
```

## Cautionary points

- Avoid to use SearchById on the index built by add_with_ids (can use from IndexIDMap, IndexIVF etc.). These index does not keep id complicatedly so reconstruct_n method may do unexpected behavior.
- Support only CPU index.
