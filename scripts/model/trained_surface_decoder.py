"""Restore a completed, read-back inversion checkpoint for explicit-device inference."""
from copy import deepcopy
import json
import os
from pathlib import Path

import torch

from model.contracts import Instrument, ModelConfig, validate_bank
from model.native_population_training import fingerprint
from model.stream_surface import StreamSurfaceDecoder
from text.constellation_corpus import file_sha, need
from text.constellation_input import stamp


def check_state(state, report, reference):
    need(state['schema'] == 'surface-inversion-state-v1' and state['binding'] == report['binding']
         and all(state[k] == reference[k] for k in ('cursor','step','ledger'))
         and fingerprint(state) == reference['state'], 'TRAINED_DECODER_CHECKPOINT_STATE')


class TrainedSurfaceDecoder:
    def __init__(self, full, full_pin, plan, plan_pin, cold, cold_pin, final, final_pin,
                 *, device, cpu_diagnostic=False, retain_continuation=False):
        need(type(retain_continuation) is bool, 'TRAINED_DECODER_CONTINUATION_OPTION')
        self.model = None; self.closed = False; self.files = {}; self._continuation = None
        def read(path, pin):
            path = Path(path); before = stamp(path)
            need(file_sha(path) == pin, 'TRAINED_DECODER_PIN')
            value = json.loads(path.read_bytes()); need(stamp(path) == before, 'TRAINED_DECODER_CHANGED')
            self.files[path] = before; return value
        full, plan, cold, final = map(Path, (full,plan,cold,final))
        r = read(full/'report.json',full_pin); p = read(plan/'report.json',plan_pin)
        c = read(cold/'report.json',cold_pin); q = read(final/'report.json',final_pin)
        need(r['complete'] and r['full_training'] and not r['cpu_proof'] and r['resume'] is None
             and r['cursor'] == p['train']['atoms'] > 0 and r['steps'] == p['updates'] > 0
             and r['plan'] == r['binding']['plan'] == plan_pin and r['binding']['scope'] == 'full',
             'TRAINED_DECODER_TRAIN_SCOPE')
        need(p['complete'] and r['code'] == p['code'] == c['code']
             and c['complete'] and c['input_pin'] == full_pin and c['plan'] == plan_pin
             and c['every_evaluation_byte_exact'] and c['full_ledger_coverage']
             and c['comparison_tolerance'] == 0. and c['device'] == r['binding']['device'] == 'cuda:0',
             'TRAINED_DECODER_READBACK')
        need(q['complete'] and q['input_pin'] == full_pin and q['cold_pin'] == cold_pin
             and q['every_evaluation_byte_exact'] and q['source_strata_and_denominators_recounted']
             and q['checkpoint_count'] == len(r['checkpoints']) and q['optimizer_updates'] == r['steps']
             and q['train_atoms'] == r['cursor'], 'TRAINED_DECODER_FINAL_READBACK')
        self.config = ModelConfig(**p['config']); self.config.validate()
        self.specs = tuple(Instrument(**s).validate() for s in p['instruments']); validate_bank(self.specs)
        ref = r['checkpoints'][-1]; path = Path(ref['path'])
        need(path.parent.resolve() == full.resolve() and not path.is_symlink()
             and r['files'][path.name] == ref['pin'] and ref['cursor'] == r['cursor']
             and ref['step'] == r['steps'] and ref['ledger'] == r['ledger'], 'TRAINED_DECODER_FINAL_STATE')
        before = stamp(path); available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()
                                             if s.startswith('MemAvailable:'))
        need(path.stat().st_size*16 < available//4, 'TRAINED_DECODER_PHYSICAL_MEMORY')
        need(file_sha(path) == ref['pin'], 'TRAINED_DECODER_CHECKPOINT_PIN')
        state = torch.load(path,map_location='cpu',weights_only=True)
        need(stamp(path) == before, 'TRAINED_DECODER_CHANGED'); self.files[path] = before
        check_state(state,r,ref)
        self.state_pin = fingerprint(state['model'])
        # Reuse the producer's actual model implementation; do not load its data adapters.
        base = Path(__file__).resolve().parents[1]
        for name in ('model/contracts.py','model/network.py','model/native.py','model/attention.py',
                     'model/stream_attention.py','model/stream_surface.py','model/identity_loop.py'):
            path = base/name; before = stamp(path)
            need(file_sha(path) == r['code'][name], 'TRAINED_DECODER_IMPLEMENTATION')
            self.files[path] = before
        device = torch.device(device)
        need(device.type in ('cpu','cuda') and type(cpu_diagnostic) is bool, 'TRAINED_DECODER_DEVICE')
        if device.type == 'cpu': need(cpu_diagnostic, 'TRAINED_DECODER_EXPLICIT_CPU_DIAGNOSTIC')
        else:
            lease = Path.home()/'mnlm/.lease/gpu'
            need(lease.is_file() and json.loads(lease.read_bytes()).get('pid') == os.getpid(),
                 'TRAINED_DECODER_GPU_LEASE')
            need(torch.cuda.is_available(), 'TRAINED_DECODER_CUDA_UNAVAILABLE')
        self.model = StreamSurfaceDecoder(self.config,self.specs)
        self.model.load_state_dict(state['model'],strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.configuration = repr(self.config), tuple(self.specs)
        self.receipt = dict(full=full_pin,plan=plan_pin,cold=cold_pin,final=final_pin,
            checkpoint=ref['pin'],checkpoint_state=ref['state'],model=self.state_pin,
            optimizer=fingerprint(state['optimizer']),train_atoms=r['cursor'],optimizer_updates=r['steps'],
            original_device=r['binding']['device'],inference_device=str(device),cpu_diagnostic=cpu_diagnostic,
            numerical_runtime_equivalence_claimed=False,optimizer_allocated=False,serving=False,
            semantic_validity_established=False,full_update_replay_established=q['full_update_cold_replay'])
        if retain_continuation: self._continuation = state
        del state
        self.fresh()

    def fresh(self):
        need(not self.closed and self.model is not None, 'TRAINED_DECODER_CLOSED')
        need(all(stamp(p) == s for p,s in self.files.items()), 'TRAINED_DECODER_CHANGED')
        need(not self.model.training and all(not p.requires_grad for p in self.model.parameters())
             and fingerprint(self.model.state_dict()) == self.state_pin
             and (repr(self.model.config),tuple(self.model.specs)) == self.configuration,
             'TRAINED_DECODER_MODEL_CHANGED')
        if self._continuation is not None:
            need(fingerprint(self._continuation) == self.receipt['checkpoint_state'],
                 'TRAINED_DECODER_CONTINUATION_CHANGED')

    def export_continuation(self):
        """Fork the authenticated weights AND moments without another checkpoint read."""
        self.fresh()
        need(self._continuation is not None, 'TRAINED_DECODER_CONTINUATION_NOT_RETAINED')
        state = self._continuation
        need(state['step'] == self.receipt['optimizer_updates'], 'TRAINED_DECODER_CONTINUATION_STEP')
        return deepcopy(dict(after=state['model'], optimizer=state['optimizer'])), state['step']

    def close(self):
        self.model = None; self._continuation = None; self.closed = True
