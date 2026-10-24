"""Join exact legal surfaces to complete native windows for decoder inversion.

The same-atom target readings are legitimate decoder inputs, not predictor
inputs. Addresses, split/exposure labels and text hashes remain audit metadata.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import torch

from model.contracts import Instrument, validate_bank
from model.native import NativeInput
from senses.contract import canonical, digest
from text.constellation_corpus import file_sha, need
from text.constellation_input import stamp


def instruments(bank):
    """Adapt pinned native machinery without loading or changing any embedder."""
    ref = bank.config['runtime_proof']; root = Path(ref['path'])
    path = root/'report.json'; need(file_sha(path) == ref['pin'], 'LEGAL_SURFACE_RUNTIME_PIN')
    report = json.loads(path.read_bytes())
    need(report['complete'] and not report['errors'] and report['instruments'] == bank.config['instruments'],
         'LEGAL_SURFACE_RUNTIME_SCOPE')
    specs = []; stamps = {path:stamp(path)}
    for name in bank.config['roster']:
        path = root/name/'report.json'; ref = bank.config['instruments'][name]
        need(file_sha(path) == ref['pin'], 'LEGAL_SURFACE_INSTRUMENT_PIN')
        lane = json.loads(path.read_bytes()); c = lane['contract']
        need(lane['lens_id'] == ref['lens_id'] == digest(c) and lane['frozen_unchanged']
             and lane['all_windows_retained'] and c['name'] == name
             and c['kind'] == 'dense-window-multi' and c['modality'] == 'text', 'LEGAL_SURFACE_MACHINERY')
        match = re.fullmatch(r'MultiWindows\(([1-9][0-9]*)\)', c['shape'])
        need(match is not None, 'LEGAL_SURFACE_NATIVE_SHAPE'); dims = int(match[1])
        definition = dict(schema='legal-native-model-instrument-v1', name=name, channel='text',
            layout='multi', dims=dims, frozen=True, weights_sha256=c['weights_sha256'],
            instrument_lens_id=ref['lens_id'], native_contract=c,
            training_lineage=dict(declared_corpus_hash=c.get('corpus_hash'),
                                  status='upstream_training_data_not_attested'))
        specs.append(Instrument(name, digest(definition), 'text', 'multi', dims,
                                canonical(definition).decode()).validate())
        stamps[path] = stamp(path)
    validate_bank(specs)
    return tuple(specs), stamps


@dataclass(frozen=True)
class SurfaceExample:
    occurrence: int
    uid: int
    side: str
    raw: bytes
    native: dict


class LegalSurfaceInputs:
    """One resident byte reader and native reader; bounded tiles, no model loads."""
    def __init__(self, data, surfaces, native_reader, config, *, bank_pin):
        need(bank_pin == data.native.pin, 'LEGAL_SURFACE_BANK_PIN')
        need(surfaces.source is data.source and native_reader.data is data,
             'LEGAL_SURFACE_READER_SOURCE')
        need(surfaces.size == data.native.config['inputs'], 'LEGAL_SURFACE_INCOMPLETE_STORE')
        self.data, self.surfaces, self.reader, self.config = data, surfaces, native_reader, config
        self.specs, self.stamps = instruments(data.native)
        need(tuple(s.name for s in self.specs) == data.roster, 'LEGAL_SURFACE_ROSTER')
        self.binding = dict(data=data.binding, surface=surfaces.pin,
                            instruments={s.name:s.identity for s in self.specs},
                            use='decoder_inversion_true_same_atom_readings', scientific_admission=False)

    def fresh(self):
        self.data.fresh(); self.surfaces.fresh()
        need(all(stamp(p) == s for p,s in self.stamps.items()), 'LEGAL_SURFACE_INSTRUMENT_CHANGED')

    def wrap(self, occurrence, uid, raw, frame, *, side):
        need(int(self.data.mapping[occurrence]) == uid, 'LEGAL_SURFACE_UID_JOIN')
        need(isinstance(raw, bytes) and bool(raw)
             and hashlib.sha256(raw).digest() == self.data.source.arrays['hashes'][uid].tobytes(),
             'LEGAL_SURFACE_TEXT_JOIN')
        need(set(frame) == set(self.data.roster), 'LEGAL_SURFACE_NATIVE_ROSTER')
        row = digest(dict(source=self.data.source.pin, occurrence=occurrence, uid=uid))
        slots = {}
        for spec in self.specs:
            v = frame[spec.name]
            need(v.ndim == 2 and v.shape[1] == spec.dims, 'LEGAL_SURFACE_NATIVE_DIMENSIONS')
            value = NativeInput(spec.identity, (row,), v.unsqueeze(0),
                torch.zeros((1, len(v)), dtype=torch.bool, device=v.device),
                torch.zeros(1, dtype=torch.bool, device=v.device))
            value.check(spec, self.config); slots[spec.name] = value
        need(sum(v.values.numel() for v in slots.values()) <= self.config.max_native_elements,
             'LEGAL_SURFACE_NATIVE_MEMORY')
        return SurfaceExample(occurrence, uid, side, raw, slots)

    def fetch(self, occurrences, *, side, fit=False):
        # Eligibility must precede native access and byte access, including errors.
        indices = self.data.indices(occurrences, side, fit=fit)
        need(len(indices) <= 128, 'LEGAL_SURFACE_MEMORY_TILE')
        self.fresh(); uids = self.data.mapping[indices].tolist()
        raws = self.surfaces.read(uids); frames = self.reader.frames(uids)
        need(len(raws) == len(frames) == len(indices), 'LEGAL_SURFACE_JOIN_COUNT')
        result = tuple(self.wrap(i, u, raw, f, side=side)
                       for i,u,raw,f in zip(occurrences, uids, raws, frames))
        self.fresh(); return result

    def training_authority(self):
        raise ValueError('MNLM_LEGAL_SURFACE_SCIENTIFIC_ADMISSION_REQUIRED')
