"""Pinned timed event readings to native model rows; source support stays live."""
import json

import torch

from audit.timed_speech_witness import request
from model.contracts import Instrument
from model.event_temporal_input import TimedNativeInput, descriptor
from senses.contract import canonical, digest, require


class TimedWorldInputs:
    def __init__(self, bank, memory, config):
        self.bank, self.memory, self.config = bank, memory, config
        self.head = memory.retained()["head"]
        self.moments = bank.inputs.moments
        self.streams = tuple(bank.streams(memory, m["id"], **request(m)) for m in self.moments)
        self.temporal = tuple(tuple(descriptor(s) for s in streams) for streams in self.streams)
        self.temporal_catalogue = {m["id"]: {s.instrument: digest(json.loads(raw))
            for s, raw in zip(streams, times, strict=True)}
            for m, streams, times in zip(self.moments, self.streams, self.temporal, strict=True)}
        self.specs = []
        manifest = bank.manifest
        for stream in self.streams[0]:
            contract = manifest["instruments"][stream.instrument]
            definition = dict(schema="timed-model-instrument-v1", name=stream.instrument,
                channel=stream.instrument.split("_", 1)[0], layout="multi",
                dims=stream.values.shape[1], frozen=True,
                weights_sha256=contract.get("weights_sha256", contract.get("state_sha256")),
                measurement_identity=stream.identity, measurement_contract=contract,
                training_lineage={"status": "upstream_not_independently_attested"})
            self.specs.append(Instrument(stream.instrument, digest(definition), definition["channel"],
                "multi", definition["dims"], canonical(definition).decode()).validate())
        self.specs = tuple(self.specs)
        self.binding = digest(dict(bank=bank.pin, head=self.head,
            moments=[m["id"] for m in self.moments], instruments=[s.identity for s in self.specs],
            temporal=self.temporal_catalogue))

    def fresh(self):
        require(self.memory.retained()["head"] == self.head, "MNLM_TIMED_MODEL_STALE",
                "source memory changed; rebuild contexts and bank from checked support")

    def batch(self, indices, device, *, purpose="engineering_boundary"):
        require(purpose == "engineering_boundary", "MNLM_TIMED_MODEL_ADMISSION",
                "candidate native measurements do not authorize learning or meaning")
        self.fresh()
        require(indices and len(set(indices)) == len(indices)
                and all(type(i) is int and 0 <= i < len(self.moments) for i in indices),
                "MNLM_TIMED_MODEL_ROWS", "select distinct source moments")
        for i in indices:
            self.memory.consume(self.moments[i]["id"], purpose=purpose, **request(self.moments[i]))
        rows = tuple(self.moments[i]["id"] for i in indices)
        result = {}
        for lane, spec in enumerate(self.specs):
            readings = [self.streams[i][lane] for i in indices]
            require(all(s.identity == json.loads(spec.definition_json)["measurement_identity"]
                        and s.instrument == spec.name for s in readings),
                    "MNLM_TIMED_MODEL_IDENTITY", "measurement runtime must match the typed model binding")
            width = max(s.values.shape[0] for s in readings)
            values = torch.zeros(len(rows), width, spec.dims, dtype=torch.float32)
            padding = torch.ones(len(rows), width, dtype=torch.bool)
            for i, stream in enumerate(readings):
                values[i, :len(stream.values)] = torch.from_numpy(stream.values.copy())
                padding[i, :len(stream.values)] = False
            result[spec.name] = TimedNativeInput(spec.identity, rows, values, padding,
                torch.zeros(len(rows), dtype=torch.bool),
                temporal=tuple(self.temporal[i][lane] for i in indices)).to(device)
            result[spec.name].check(spec, self.config)
        self.fresh()
        return result
