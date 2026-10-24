"""Bounded TFRecord / tf.train.Example reader, no TensorFlow execution.

Wire schema: tensorflow/core/example/{feature,example}.proto (Apache-2.0).
Framing: tensorflow.org/tutorials/load_data/tfrecord, masked CRC32C on both
length and payload. O(bytes), one bounded record resident. Unknown fields refuse.
"""
import hashlib
import math
import struct

import google_crc32c
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf.message import DecodeError


class RecordError(ValueError):
    def __init__(self, code, detail):
        self.code = code
        super().__init__(f"{code}: {detail}")


def require(ok, code, detail):
    if not ok:
        raise RecordError(code, detail)


def example_type():
    schema = descriptor_pb2.FileDescriptorProto(name="mnlm_tf_example.proto", package="tensorflow", syntax="proto3")

    def add_field(message, name, number, kind, repeated=False, typename=None, oneof=None):
        f = message.field.add(name=name, number=number, type=kind, label=3 if repeated else 1)
        if typename:
            f.type_name = ".tensorflow." + typename
        if oneof is not None:
            f.oneof_index = oneof
        return f

    for name, kind in (("BytesList", 12), ("FloatList", 2), ("Int64List", 3)):
        m = schema.message_type.add(name=name)
        f = add_field(m, "value", 1, kind, repeated=True)
        if kind != 12:
            f.options.packed = True
    feature = schema.message_type.add(name="Feature")
    feature.oneof_decl.add(name="kind")
    for i, (name, typename) in enumerate((("bytes_list", "BytesList"), ("float_list", "FloatList"),
                                         ("int64_list", "Int64List")), 1):
        add_field(feature, name, i, 11, typename=typename, oneof=0)
    features = schema.message_type.add(name="Features")
    entry = features.nested_type.add(name="FeatureEntry")
    entry.options.map_entry = True
    add_field(entry, "key", 1, 9)
    add_field(entry, "value", 2, 11, typename="Feature")
    add_field(features, "feature", 1, 11, repeated=True, typename="Features.FeatureEntry")
    example = schema.message_type.add(name="Example")
    add_field(example, "features", 1, 11, typename="Features")
    pool = descriptor_pool.DescriptorPool()
    pool.Add(schema)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("tensorflow.Example"))


Example = example_type()


def decode(payload):
    obj = Example()
    try:
        obj.ParseFromString(payload)
    except DecodeError as exc:
        raise RecordError("MNLM_TF_PROTOBUF", "invalid Example payload") from exc
    before = obj.SerializeToString(deterministic=True)
    obj.DiscardUnknownFields()
    require(before == obj.SerializeToString(deterministic=True), "MNLM_TF_UNKNOWN_FIELD", "unknown protobuf field")
    result = {}
    for name, value in obj.features.feature.items():
        kind = value.WhichOneof("kind")
        require(bool(name) and kind is not None, "MNLM_TF_FEATURE", "unnamed or untyped feature")
        values = list(getattr(value, kind).value)
        require(kind != "float_list" or all(math.isfinite(x) for x in values),
                "MNLM_TF_NONFINITE", name)
        result[name] = (kind, values)
    require(bool(result), "MNLM_TF_FEATURE", "empty Example")
    return result


def masked_crc(data):
    crc = google_crc32c.value(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def frames(path, expected_sha256=None, max_record_bytes=64 << 20):
    require(type(max_record_bytes) is int and max_record_bytes > 0, "MNLM_TF_BUDGET", "positive record cap required")
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            offset = f.tell()
            head = f.read(12)
            if not head:
                break
            require(len(head) == 12, "MNLM_TF_TRUNCATED", f"length header at {offset}")
            length, crc = struct.unpack("<QI", head)
            require(masked_crc(head[:8]) == crc, "MNLM_TF_LENGTH_CRC", f"offset {offset}")
            require(0 < length <= max_record_bytes, "MNLM_TF_BUDGET", f"record at {offset} declares {length} bytes")
            payload, tail = f.read(length), f.read(4)
            require(len(payload) == length and len(tail) == 4, "MNLM_TF_TRUNCATED", f"payload at {offset}")
            require(masked_crc(payload) == struct.unpack("<I", tail)[0], "MNLM_TF_DATA_CRC", f"offset {offset}")
            digest.update(head)
            digest.update(payload)
            digest.update(tail)
            yield offset, payload
    require(expected_sha256 is None or digest.hexdigest() == expected_sha256,
            "MNLM_TF_FILE_DIGEST", "source changed; derived output must not be accepted")
