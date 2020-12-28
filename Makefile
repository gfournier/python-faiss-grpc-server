.PHONY: all
all:

.PHONY: codegen
codegen:
	@python -m grpc_tools.protoc --proto_path=proto \
		--python_out=python/ann_grpc/proto \
		--grpc_python_out=python/ann_grpc/proto \
		proto/ann.proto

.PHONY: documents
documents:
	@protoc --doc_out=./docs \
		--doc_opt=markdown,grpc_specification.md \
		proto/*.proto

.PHONY: examples
examples:
	@python -m grpc_tools.protoc --proto_path=proto \
		--python_out=examples \
		--grpc_python_out=examples \
		proto/ann.proto
