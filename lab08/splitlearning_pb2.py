# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: splitlearning.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x13splitlearning.proto\x12\rsplitlearning\"\\\n\x0e\x43lientToServer\x12\x13\n\x0b\x61\x63tivations\x18\x01 \x03(\x02\x12\x0e\n\x06labels\x18\x02 \x03(\x05\x12\x12\n\nbatch_size\x18\x03 \x01(\x05\x12\x11\n\tclient_id\x18\x04 \x01(\x05\">\n\x0eServerToClient\x12\x11\n\tgradients\x18\x01 \x03(\x02\x12\x0c\n\x04loss\x18\x02 \x01(\x02\x12\x0b\n\x03\x61\x63\x63\x18\x03 \x01(\x02\x32h\n\rSplitLearning\x12W\n\x15SendClientActivations\x12\x1d.splitlearning.ClientToServer\x1a\x1d.splitlearning.ServerToClient\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'splitlearning_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_CLIENTTOSERVER']._serialized_start=38
  _globals['_CLIENTTOSERVER']._serialized_end=130
  _globals['_SERVERTOCLIENT']._serialized_start=132
  _globals['_SERVERTOCLIENT']._serialized_end=194
  _globals['_SPLITLEARNING']._serialized_start=196
  _globals['_SPLITLEARNING']._serialized_end=300
# @@protoc_insertion_point(module_scope)