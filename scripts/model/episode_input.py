"""Complete recorded episodes as typed, time-bound streams of native readings.

PC-07/16/35: index metadata once, move bounded blocks, retain native geometry.
This remains candidate measurement infrastructure, never admission or online
feedback. The enclosing utterance is available only after its source cutoff.
"""
from dataclasses import dataclass
import hashlib
import json
from types import SimpleNamespace

import torch

from audit.timed_speech_witness import request
from model.constellation import Constellation
from model.event_temporal_input import descriptor, features
from model.stream_attention import BLOCK_ROWS, NativeBlock
from senses.contract import canonical, digest, require


@dataclass(frozen=True)
class EpisodeLane:
    owner: object
    index: int
    identity: str
    name: str
    rows: tuple[str, ...]
    pieces: tuple[bytes, ...]
    temporal: tuple[bytes, ...]
    count: int
    dims: int
    temporal_pin: str

    def blocks(self, config):
        self.owner.fresh()
        cursor, byte_count = 0, 0
        h = hashlib.sha256()
        for raw, timing in zip(self.pieces, self.temporal, strict=True):
            expected = json.loads(raw)
            entry, values = self.owner.bank.read(self.name, expected["input"])
            require(entry == expected, "MNLM_EPISODE_READING", "restore the pinned native reading")
            n = len(values)
            require(n <= config.max_native_tokens and n*self.dims <= config.max_native_elements,
                    "MNLM_EPISODE_READING_BUDGET", "one native reading exceeds the original limits")
            coordinates = features((timing,), (n,), n, "cpu")
            for start in range(0, n, min(BLOCK_ROWS, config.max_native_tokens)):
                stop = min(n, start+BLOCK_ROWS, start+config.max_native_tokens)
                value = torch.from_numpy(values[start:stop].copy())[None]
                yield NativeBlock(cursor+start, value,
                    torch.zeros(value.shape[:2], dtype=torch.bool), coordinates[:, start:stop])
            h.update(canonical(dict(file=entry["file"], sha256=entry["sha256"],
                                    rows=[cursor, cursor+n], timing=digest(json.loads(timing)))))
            cursor += n
            byte_count += values.nbytes
            self.owner.fresh()
        require(cursor == self.count, "MNLM_EPISODE_COVERAGE", "reject an incomplete episode")
        if self.owner.trace is not None:
            self.owner.trace(dict(row=self.rows[0], lane=self.name, count=cursor,
                readings=len(self.pieces), bytes=byte_count, traversal=h.hexdigest()))


class EpisodeBatch(dict):
    def __init__(self, owner, index):
        super().__init__(owner.lanes[index])
        self.owner, self.index = owner, index

    def validate(self, specs, config):
        self.owner.fresh()
        require(set(self) == {s.name for s in specs}, "MNLM_EPISODE_ROSTER", "complete typed episode required")
        for spec in specs:
            lane = self[spec.name]
            require(lane is self.owner.lanes[self.index][spec.name]
                    and lane.identity == spec.identity and lane.dims == spec.dims
                    and lane.rows == (self.owner.moments[self.index]["id"],) and lane.count > 0,
                    "MNLM_EPISODE_BINDING", "rebuild changed source streams through the pinned adapter")
        return self

    def check_catalogue(self, catalogue):
        self.owner.fresh()
        for lane in self.values():
            require(catalogue.get(lane.rows[0], {}).get(lane.name) == lane.temporal_pin,
                    "MNLM_TEMPORAL_BINDING", "episode timing differs from the model source catalogue")


def validate_observation(native, specs, config):
    if isinstance(native, EpisodeBatch):
        return native.validate(specs, config)
    return Constellation(native, "observed").validate(specs, config)


class EpisodeWorldInputs:
    def __init__(self, selected, *, trace=None):
        self.selected, self.bank, self.memory = selected, selected.bank, selected.memory
        self.specs, self.config, self.head = selected.specs, selected.config, selected.head
        # PC-28: no accumulating runtime audit list; callers own any trace sink.
        self.moments, self.lanes, self.trace = selected.moments, [], trace
        self.temporal_catalogue, self.records = {}, {}
        manifest = self.bank.manifest
        by_segment = {}
        for lane, entries in manifest["readings"].items():
            groups = by_segment[lane] = {}
            for entry in entries:
                groups.setdefault(entry["input"]["segment"], []).append(entry)
        for index, moment in enumerate(self.moments):
            row, lanes, records = moment["id"], {}, {}
            for spec in self.specs:
                entries = tuple(by_segment[spec.name].get(moment["source"]["segment"], ()))
                require(entries, "MNLM_EPISODE_EMPTY", "no native readings for this recorded episode")
                times = []
                for entry in entries:
                    meta = canonical(dict(bank=self.bank.pin, moment=moment, entry=entry,
                        scope=request(moment), memory_head=self.head))
                    times.append(descriptor(SimpleNamespace(instrument=spec.name,
                        identity=entry["instrument_id"], metadata=meta)))
                record = dict(schema="complete-recorded-utterance-v1", row=row, lane=spec.name,
                    source=moment["source"], scope="retrospective_complete_utterance",
                    pieces=[json.loads(t) for t in times])
                records[spec.name] = record
                lanes[spec.name] = EpisodeLane(self, index, spec.identity, spec.name, (row,),
                    tuple(canonical(e) for e in entries), tuple(times),
                    sum(e["shape"][0] for e in entries), spec.dims, digest(record))
            self.lanes.append(lanes)
            self.records[row] = records
            self.temporal_catalogue[row] = {n: x.temporal_pin for n, x in lanes.items()}
        self.binding = digest(dict(schema="complete-episode-model-input-v1", bank=self.bank.pin,
            selected=selected.binding, temporal=self.temporal_catalogue, head=self.head))

    def fresh(self):
        self.selected.fresh()

    def batch(self, index, *, purpose="engineering_boundary"):
        require(purpose == "engineering_boundary", "MNLM_EPISODE_ADMISSION",
                "registered native instruments cannot authorize learning or serving")
        require(type(index) is int and 0 <= index < len(self.moments), "MNLM_EPISODE_INDEX",
                "select one complete source episode per context")
        self.fresh()
        moment = self.moments[index]
        self.memory.consume(moment["id"], purpose=purpose, **request(moment))
        return EpisodeBatch(self, index).validate(self.specs, self.config)
