"""Pinned surface characterization inputs; importing this module never masks CUDA."""
from dataclasses import replace
import json
from pathlib import Path

import torch

from model.contracts import Instrument, ModelConfig
from model.legal_surface_inputs import SurfaceExample
from model.native import NativeInput
from text.constellation_corpus import file_sha, need

ROOT = Path('/zfs/archive/mnlm/evidence/fsv/29')
REFS = dict(
    source=('legal-surface-decoder-proof-v1-20261023', '3400b9c5813d56859603a89d5f0fe3182370b94a68e014199b50a2ff605c8d45'),
    source_readback=('legal-surface-decoder-readback-v1-20261023', '61e974430b13b259135eee4123a26c13957d8545361d7a37a2255258f23d2c8f'),
    mixed=('legal-mixed-surface-cpu-v1-20261023', '792986624885936202b4a6f43e6dd307454dc8fa4e6f8bfb7b22a6a887f40754'),
    mixed_readback=('legal-mixed-surface-cpu-readback-v2-20261023', '30d066f0cadc5f42ee111c3666d083ccac9ba0dcdb56faf805dbd9b958a5c138'),
    gathered=('legal-gathered-surface-cpu-v2-20261023', '0f2aa3271891ef32a5a0a9f44b2b40653beceda6ba5b59cf744e68b5d49668c1'),
    gathered_readback=('legal-gathered-surface-cpu-readback-v2-20261023', '771869905ce319de748d7329587da0de53985b51fef833d9d0749b1ff966ed1b'))
NEW_CODE = ('audit/surface_device_authority.py', 'model/surface_device_runtime.py',
            'audit/surface_device_proof.py', 'audit/surface_device_readback.py', 'audit/surface_device_queue.py')
DEPENDENCIES = (
    'canon/common.py', 'canon/segment.py', 'canon/training_scope.py', 'census/ledger.py',
    'claims/capability_population.py', 'claims/codata_capture.py', 'claims/codata_source.py',
    'claims/episodic_memory.py', 'claims/process_conditions.py', 'claims/qasc_grounding.py',
    'kernel/legal_native_associations.py', 'kernel/native_window_index.py',
    'lens/lenscommon.py', 'lens/slot_layout.py', 'lens/slotio.py', 'lens/text_windows.py',
    'model/__init__.py', 'model/attention.py', 'model/contracts.py', 'model/identity_loop.py',
    'model/loop_checkpoint.py', 'model/loop_learning.py', 'model/loop_training_checkpoint.py',
    'model/native.py', 'model/native_population_training.py', 'model/native_rq_data.py',
    'model/network.py', 'model/stream_attention.py', 'model/stream_surface.py',
    'production/gate.py', 'production/publication.py', 'senses/__init__.py', 'senses/contract.py',
    'senses/scope.py', 'senses/trajectory_population.py', 'text/atomise_pilot.py',
    'text/constellation_corpus.py', 'text/constellation_input.py', 'text/evaluation_multiscale.py',
    'text/legal_continuation_input.py', 'text/legal_continuations.py', 'text/legal_corpus.py',
    'text/legal_native_batch.py', 'text/legal_native_store.py', 'text/legal_reuse.py',
    'text/legal_reuse_input.py', 'text/legal_stock.py', 'text/multiscale_spans.py')


def stamp(path):
    s = path.stat(); return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def pinned(path, pin):
    need(file_sha(path/'report.json') == pin, 'SURFACE_DEVICE_REPORT_PIN')
    record = json.loads((path/'report.json').read_bytes())
    need(record['complete'], 'SURFACE_DEVICE_INCOMPLETE')
    for n, h in record.get('files', {}).items():
        need(file_sha(path/n) == h, 'SURFACE_DEVICE_PHYSICAL_FILE')
    return record


def load(root=ROOT):
    records = {}; stamps = {}
    for key, (name, pin) in REFS.items():
        path = root/name; records[key] = pinned(path, pin)
        for p in [path/'report.json', *(path/n for n in records[key].get('files', {}))]:
            stamps[p] = stamp(p)
    for name in ['source', 'mixed', 'gathered']:
        need(records[name+'_readback']['input_pin'] == REFS[name][1], 'SURFACE_DEVICE_READBACK_BINDING')
    need(records['gathered']['same_cohort_exact'] and records['mixed_readback']['all_cold_arm_states_exact'],
         'SURFACE_DEVICE_PRIOR_EXACTNESS')
    base = Path(__file__).resolve().parents[1]
    code = dict(records['gathered']['code'])
    for n in (*NEW_CODE, *DEPENDENCIES): code[n] = file_sha(base/n)
    for record in records.values():
        for n, h in record.get('code', {}).items():
            need(code.get(n) == h, 'SURFACE_DEVICE_PRIOR_CODE')
    def fresh():
        for n, h in code.items(): need(file_sha(base/n) == h, 'SURFACE_DEVICE_CODE_CHANGED')
        for p, s in stamps.items(): need(stamp(p) == s, 'SURFACE_DEVICE_SOURCE_CHANGED')
    fresh()
    p = root/REFS['source'][0]; r = records['source']
    config = ModelConfig(**r['config']); specs = tuple(Instrument(**s).validate() for s in r['instruments'])
    examples = []; metadata = []
    for d in torch.load(p/'observations.pt', map_location='cpu', weights_only=True):
        for v in d['observations']:
            native = {s.name: NativeInput(s.identity, (v['row'],), v['native'][s.name][None],
                torch.zeros((1, len(v['native'][s.name])), dtype=torch.bool), torch.zeros(1, dtype=torch.bool)) for s in specs}
            examples.append(SurfaceExample(v['occurrence'], v['uid'], d['side'], bytes(v['raw'].tolist()), native))
            metadata.append(dict(document=d['document'], side=d['side'], train_present=v['train_present']))
    saved = json.loads((root/REFS['mixed'][0]/'results.json').read_bytes())
    need(saved['metadata'] == metadata, 'SURFACE_DEVICE_METADATA')
    donors = saved['donors']
    need(len(donors) == len(examples) == 215, 'SURFACE_DEVICE_POPULATION')
    for i, j in enumerate(donors):
        need(examples[j].side == 'train' and metadata[j]['document'] != metadata[i]['document']
             and examples[j].uid != examples[i].uid, 'SURFACE_DEVICE_DONOR')
    shuffled = [replace(e, native=examples[j].native) for e, j in zip(examples, donors)]
    initial = torch.load(p/'initial.pt', map_location='cpu', weights_only=True)
    fresh()
    return config, specs, examples, metadata, initial, shuffled, saved, code, fresh
