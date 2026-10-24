"""Source-bound native memory -> separately addressed model observation streams."""
from dataclasses import dataclass
import json

import numpy as np
import torch

from claims.episodic_memory import canonical
from model.stream_attention import BLOCK_ROWS, NativeBlock
from senses.contract import require


@dataclass(frozen=True)
class ReadingStream:
    reference: str
    instrument: str
    identity: str
    metadata: bytes
    values: np.ndarray
    memory: object
    head: str

    @property
    def counts(self):
        return (self.values.shape[0],)

    def blocks(self, config, *, purpose="engineering_boundary"):
        require(purpose == "engineering_boundary", "MNLM_NATIVE_STREAM_ADMISSION",
                "candidate native readings cannot authorize learning or meaning")
        require(self.memory.retained()["head"] == self.head, "MNLM_NATIVE_STREAM_STALE",
                "memory changed; resolve support again before consuming")
        step = min(BLOCK_ROWS, config.max_native_tokens)
        for start in range(0, self.values.shape[0], step):
            # Separate immutable native source, bounded CPU tensor and GPU work.
            value = torch.from_numpy(self.values[start:start+step].copy())[None]
            yield NativeBlock(start, value, torch.zeros(value.shape[:2], dtype=torch.bool))
        require(self.memory.retained()["head"] == self.head, "MNLM_NATIVE_STREAM_STALE",
                "memory changed during observation; discard partial model state")


def from_memory(memory, moment, **scope):
    retained = memory.consume(moment, **scope)
    streams = []
    for reading in retained["readings"]:
        values = reading["values"]
        require(values.dtype == np.float32 and values.ndim == 2 and not values.flags.writeable,
                "MNLM_NATIVE_STREAM_VALUES", "retain the immutable native measurement")
        metadata = {k: v for k, v in reading.items() if k != "values"}
        metadata["memory_head"] = retained["head"]
        metadata["moment"] = moment
        metadata["scope"] = scope
        streams.append(ReadingStream(reading["reference"],
            reading["medium"]+"_"+reading["name"], reading["entry"]["instrument_id"],
            canonical(metadata), values, memory, retained["head"]))
    require(len({s.reference for s in streams}) == len(streams),
            "MNLM_NATIVE_STREAM_DUPLICATE", "duplicate native reading")
    return tuple(streams)


def descriptor(stream):
    return dict(reference=stream.reference, instrument=stream.instrument,
                identity=stream.identity, native_shape=list(stream.values.shape),
                provenance=json.loads(stream.metadata))
