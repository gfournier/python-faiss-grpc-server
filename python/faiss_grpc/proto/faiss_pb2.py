# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: faiss.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor.FileDescriptor(
    name='faiss.proto',
    package='faiss',
    syntax='proto3',
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n\x0b\x66\x61iss.proto\x12\x05\x66\x61iss\x1a\x1bgoogle/protobuf/empty.proto\"%\n\x08Neighbor\x12\n\n\x02id\x18\x01 \x01(\x04\x12\r\n\x05score\x18\x02 \x01(\x02\"\x15\n\x06Vector\x12\x0b\n\x03val\x18\x01 \x03(\x02\"8\n\rSearchRequest\x12\x1c\n\x05query\x18\x01 \x01(\x0b\x32\r.faiss.Vector\x12\t\n\x01k\x18\x02 \x01(\x04\"4\n\x0eSearchResponse\x12\"\n\tneighbors\x18\x01 \x03(\x0b\x32\x0f.faiss.Neighbor\"*\n\x11SearchByIdRequest\x12\n\n\x02id\x18\x01 \x01(\x04\x12\t\n\x01k\x18\x02 \x01(\x04\"L\n\x12SearchByIdResponse\x12\x12\n\nrequest_id\x18\x01 \x01(\x04\x12\"\n\tneighbors\x18\x02 \x03(\x0b\x32\x0f.faiss.Neighbor\"#\n\x10HeatbeatResponse\x12\x0f\n\x07message\x18\x01 \x01(\t2\xc5\x01\n\x0c\x46\x61issService\x12;\n\x08Heatbeat\x12\x16.google.protobuf.Empty\x1a\x17.faiss.HeatbeatResponse\x12\x35\n\x06Search\x12\x14.faiss.SearchRequest\x1a\x15.faiss.SearchResponse\x12\x41\n\nSearchById\x12\x18.faiss.SearchByIdRequest\x1a\x19.faiss.SearchByIdResponseb\x06proto3',
    dependencies=[google_dot_protobuf_dot_empty__pb2.DESCRIPTOR,],
)


_NEIGHBOR = _descriptor.Descriptor(
    name='Neighbor',
    full_name='faiss.Neighbor',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='id',
            full_name='faiss.Neighbor.id',
            index=0,
            number=1,
            type=4,
            cpp_type=4,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name='score',
            full_name='faiss.Neighbor.score',
            index=1,
            number=2,
            type=2,
            cpp_type=6,
            label=1,
            has_default_value=False,
            default_value=float(0),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=51,
    serialized_end=88,
)


_VECTOR = _descriptor.Descriptor(
    name='Vector',
    full_name='faiss.Vector',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='val',
            full_name='faiss.Vector.val',
            index=0,
            number=1,
            type=2,
            cpp_type=6,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=90,
    serialized_end=111,
)


_SEARCHREQUEST = _descriptor.Descriptor(
    name='SearchRequest',
    full_name='faiss.SearchRequest',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='query',
            full_name='faiss.SearchRequest.query',
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name='k',
            full_name='faiss.SearchRequest.k',
            index=1,
            number=2,
            type=4,
            cpp_type=4,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=113,
    serialized_end=169,
)


_SEARCHRESPONSE = _descriptor.Descriptor(
    name='SearchResponse',
    full_name='faiss.SearchResponse',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='neighbors',
            full_name='faiss.SearchResponse.neighbors',
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=171,
    serialized_end=223,
)


_SEARCHBYIDREQUEST = _descriptor.Descriptor(
    name='SearchByIdRequest',
    full_name='faiss.SearchByIdRequest',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='id',
            full_name='faiss.SearchByIdRequest.id',
            index=0,
            number=1,
            type=4,
            cpp_type=4,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name='k',
            full_name='faiss.SearchByIdRequest.k',
            index=1,
            number=2,
            type=4,
            cpp_type=4,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=225,
    serialized_end=267,
)


_SEARCHBYIDRESPONSE = _descriptor.Descriptor(
    name='SearchByIdResponse',
    full_name='faiss.SearchByIdResponse',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='request_id',
            full_name='faiss.SearchByIdResponse.request_id',
            index=0,
            number=1,
            type=4,
            cpp_type=4,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name='neighbors',
            full_name='faiss.SearchByIdResponse.neighbors',
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=269,
    serialized_end=345,
)


_HEATBEATRESPONSE = _descriptor.Descriptor(
    name='HeatbeatResponse',
    full_name='faiss.HeatbeatResponse',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='message',
            full_name='faiss.HeatbeatResponse.message',
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode('utf-8'),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=347,
    serialized_end=382,
)

_SEARCHREQUEST.fields_by_name['query'].message_type = _VECTOR
_SEARCHRESPONSE.fields_by_name['neighbors'].message_type = _NEIGHBOR
_SEARCHBYIDRESPONSE.fields_by_name['neighbors'].message_type = _NEIGHBOR
DESCRIPTOR.message_types_by_name['Neighbor'] = _NEIGHBOR
DESCRIPTOR.message_types_by_name['Vector'] = _VECTOR
DESCRIPTOR.message_types_by_name['SearchRequest'] = _SEARCHREQUEST
DESCRIPTOR.message_types_by_name['SearchResponse'] = _SEARCHRESPONSE
DESCRIPTOR.message_types_by_name['SearchByIdRequest'] = _SEARCHBYIDREQUEST
DESCRIPTOR.message_types_by_name['SearchByIdResponse'] = _SEARCHBYIDRESPONSE
DESCRIPTOR.message_types_by_name['HeatbeatResponse'] = _HEATBEATRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Neighbor = _reflection.GeneratedProtocolMessageType(
    'Neighbor',
    (_message.Message,),
    {
        'DESCRIPTOR': _NEIGHBOR,
        '__module__': 'faiss_pb2'
        # @@protoc_insertion_point(class_scope:faiss.Neighbor)
    },
)
_sym_db.RegisterMessage(Neighbor)

Vector = _reflection.GeneratedProtocolMessageType(
    'Vector',
    (_message.Message,),
    {
        'DESCRIPTOR': _VECTOR,
        '__module__': 'faiss_pb2'
        # @@protoc_insertion_point(class_scope:faiss.Vector)
    },
)
_sym_db.RegisterMessage(Vector)

SearchRequest = _reflection.GeneratedProtocolMessageType(
    'SearchRequest',
    (_message.Message,),
    {
        'DESCRIPTOR': _SEARCHREQUEST,
        '__module__': 'faiss_pb2'
        # @@protoc_insertion_point(class_scope:faiss.SearchRequest)
    },
)
_sym_db.RegisterMessage(SearchRequest)

SearchResponse = _reflection.GeneratedProtocolMessageType(
    'SearchResponse',
    (_message.Message,),
    {
        'DESCRIPTOR': _SEARCHRESPONSE,
        '__module__': 'faiss_pb2'
        # @@protoc_insertion_point(class_scope:faiss.SearchResponse)
    },
)
_sym_db.RegisterMessage(SearchResponse)

SearchByIdRequest = _reflection.GeneratedProtocolMessageType(
    'SearchByIdRequest',
    (_message.Message,),
    {
        'DESCRIPTOR': _SEARCHBYIDREQUEST,
        '__module__': 'faiss_pb2'
        # @@protoc_insertion_point(class_scope:faiss.SearchByIdRequest)
    },
)
_sym_db.RegisterMessage(SearchByIdRequest)

SearchByIdResponse = _reflection.GeneratedProtocolMessageType(
    'SearchByIdResponse',
    (_message.Message,),
    {
        'DESCRIPTOR': _SEARCHBYIDRESPONSE,
        '__module__': 'faiss_pb2'
        # @@protoc_insertion_point(class_scope:faiss.SearchByIdResponse)
    },
)
_sym_db.RegisterMessage(SearchByIdResponse)

HeatbeatResponse = _reflection.GeneratedProtocolMessageType(
    'HeatbeatResponse',
    (_message.Message,),
    {
        'DESCRIPTOR': _HEATBEATRESPONSE,
        '__module__': 'faiss_pb2'
        # @@protoc_insertion_point(class_scope:faiss.HeatbeatResponse)
    },
)
_sym_db.RegisterMessage(HeatbeatResponse)


_FAISSSERVICE = _descriptor.ServiceDescriptor(
    name='FaissService',
    full_name='faiss.FaissService',
    file=DESCRIPTOR,
    index=0,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_start=385,
    serialized_end=582,
    methods=[
        _descriptor.MethodDescriptor(
            name='Heatbeat',
            full_name='faiss.FaissService.Heatbeat',
            index=0,
            containing_service=None,
            input_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
            output_type=_HEATBEATRESPONSE,
            serialized_options=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.MethodDescriptor(
            name='Search',
            full_name='faiss.FaissService.Search',
            index=1,
            containing_service=None,
            input_type=_SEARCHREQUEST,
            output_type=_SEARCHRESPONSE,
            serialized_options=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.MethodDescriptor(
            name='SearchById',
            full_name='faiss.FaissService.SearchById',
            index=2,
            containing_service=None,
            input_type=_SEARCHBYIDREQUEST,
            output_type=_SEARCHBYIDRESPONSE,
            serialized_options=None,
            create_key=_descriptor._internal_create_key,
        ),
    ],
)
_sym_db.RegisterServiceDescriptor(_FAISSSERVICE)

DESCRIPTOR.services_by_name['FaissService'] = _FAISSSERVICE

# @@protoc_insertion_point(module_scope)
