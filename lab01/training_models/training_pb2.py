# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: training.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0etraining.proto\"(\n\x03Row\x12\x12\n\nattributes\x18\x01 \x03(\x02\x12\r\n\x05label\x18\x02 \x01(\x05\" \n\nFitRequest\x12\x12\n\x04rows\x18\x01 \x03(\x0b\x32\x04.Row\"#\n\x0ePredictRequest\x12\x11\n\x03row\x18\x01 \x01(\x0b\x32\x04.Row\"\x1f\n\x0b\x46itResponse\x12\x10\n\x08\x61\x63\x63uracy\x18\x01 \x01(\x02\" \n\x0fPredictResponse\x12\r\n\x05\x63lass\x18\x01 \x01(\x05\x32v\n\x0e\x43\x65ntralizedMLP\x12.\n\x0fGetTrainedModel\x12\x0b.FitRequest\x1a\x0c.FitResponse\"\x00\x12\x34\n\rGetPrediction\x12\x0f.PredictRequest\x1a\x10.PredictResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'training_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_ROW']._serialized_start=18
  _globals['_ROW']._serialized_end=58
  _globals['_FITREQUEST']._serialized_start=60
  _globals['_FITREQUEST']._serialized_end=92
  _globals['_PREDICTREQUEST']._serialized_start=94
  _globals['_PREDICTREQUEST']._serialized_end=129
  _globals['_FITRESPONSE']._serialized_start=131
  _globals['_FITRESPONSE']._serialized_end=162
  _globals['_PREDICTRESPONSE']._serialized_start=164
  _globals['_PREDICTRESPONSE']._serialized_end=196
  _globals['_CENTRALIZEDMLP']._serialized_start=198
  _globals['_CENTRALIZEDMLP']._serialized_end=316
# @@protoc_insertion_point(module_scope)
