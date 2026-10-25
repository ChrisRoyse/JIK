"""Read-only complete-answer checkpoint handoff and public-only generation."""
from copy import deepcopy
import json
from pathlib import Path
import time

import torch
from claims.process_conditions import write_committed
from model.complete_answer_comparison import MODES, grid
from model.complete_answer_trace import collect
from model.native_open_window_answer import answer_model
from model.native_placement_checked import optimizer_for, validate_optimizer
from model.native_population_training import fingerprint
from model.native_request_input import NativeRequest
from text.capability_answer_source import stamp
from text.constellation_corpus import file_sha, need

ARCHITECTURES = ('shared', 'own_slot', 'whole_proposals')
METADATA = ('schema', 'binding', 'cursor', 'step', 'noise_cursor', 'chain', 'counts')


class EvaluationCheckpoint:
    def __init__(self, path, reference, expected, runtime, target, target_pin, *, device):
        self.path = Path(path); self.reference = deepcopy(reference); self.expected = deepcopy(expected)
        need(file_sha(self.path) == reference['pin'], 'ANSWER_EVAL_CHECKPOINT_PIN')
        self.stamp = stamp(self.path)
        saved = torch.load(self.path, map_location='cpu', weights_only=True)
        need(fingerprint(saved) == reference['state'], 'ANSWER_EVAL_CHECKPOINT_STATE')
        need(set(saved) == {*METADATA, 'model', 'optimizer'} and set(expected) == set(METADATA), 'ANSWER_EVAL_SCHEMA')
        need(saved['schema'] == expected['schema'] == 'legal-complete-answer-epoch-v1'
             and saved['binding'] == expected['binding'], 'ANSWER_EVAL_BINDING')
        need(all(saved[k] == expected[k] for k in METADATA if k not in ('schema','binding')), 'ANSWER_EVAL_CURSOR')
        b = saved['binding']; self.architecture = b['architecture']; self.device = torch.device(device)
        need(self.architecture in ARCHITECTURES and str(self.device) == b['device'] == b['policy']['device']
             and b['torch'] == str(torch.__version__) and next(target.parameters()).device == self.device,
             'ANSWER_EVAL_DEVICE')
        need(type(saved['cursor']) is int and 0 <= saved['cursor'] <= b['documents']
             and saved['step'] == b['initial_step']+saved['cursor']
             and saved['noise_cursor'] == b['policy']['noise_seed']+saved['step']
             and saved['counts']['documents'] == saved['counts']['ends'] == saved['cursor'], 'ANSWER_EVAL_CURSOR')
        need(b['target'] == target_pin and b['target_state'] == fingerprint(target.state_dict())
             and b['policy'] == dict(runtime['policy'],device=str(self.device)), 'ANSWER_EVAL_TARGET_POLICY')
        self.model = answer_model(target,runtime,self.architecture,target_pin=target_pin).to(self.device)
        self.model.load_state_dict(saved['model'],strict=True)
        optimizer = optimizer_for(self.model,b['policy']); optimizer.load_state_dict(saved['optimizer'])
        validate_optimizer(self.model,optimizer,b['policy'],saved['step'])
        need(fingerprint((self.model.state_dict(),optimizer.state_dict())) == fingerprint((saved['model'],saved['optimizer'])),
             'ANSWER_EVAL_RESTORED_STATE')
        self.identity = dict(checkpoint=reference['pin'], state=reference['state'], model=fingerprint(saved['model']),
            optimizer=fingerprint(saved['optimizer']), architecture=self.architecture, step=saved['step'],
            cursor=saved['cursor'], chain=saved['chain'], counts=saved['counts'], target=target_pin,
            parameters=sum(p.numel() for p in self.model.parameters()), trained_device=b['device'])
        del optimizer, saved
        self.model.eval().requires_grad_(False); self.fresh()

    def fresh(self, *, physical=False):
        need(stamp(self.path) == self.stamp and (not physical or file_sha(self.path) == self.reference['pin']),
             'ANSWER_EVAL_CHECKPOINT_CHANGED')
        need(fingerprint(self.model.state_dict()) == self.identity['model'] and not self.model.training
             and all(not p.requires_grad and p.grad is None for p in self.model.parameters()), 'ANSWER_EVAL_MODEL_CHANGED')


def public_inputs(value, roster):
    need(set(value) == set(MODES), 'ANSWER_EVAL_PUBLIC_MODES')
    result = {}; pins = {}
    for mode in MODES:
        row = value[mode]
        need(set(row) == {'frames','descriptor'}, 'ANSWER_EVAL_PUBLIC_FIELDS')
        frames = row['frames']; request = NativeRequest(row['descriptor'])
        need(isinstance(frames,(tuple,list)) and frames and request.observed == len(frames)
             and all(set(frame) == set(roster) for frame in frames), 'ANSWER_EVAL_PUBLIC_PREFIX')
        for frame in frames:
            need(all(isinstance(v,torch.Tensor) and v.dtype == torch.float32 and v.ndim == 2 and v.shape[1] == 768
                     and len(v) > 0 and bool(torch.isfinite(v).all()) and v.device.type == 'cpu' for v in frame.values()),
                 'ANSWER_EVAL_PUBLIC_NATIVE')
        result[mode] = frames,request
        pins[mode] = dict(prefix=fingerprint(frames),request=request.pin,descriptor=json.loads(request.descriptor))
    need(all(pins[m]['request'] == pins['correct']['request'] for m in MODES), 'ANSWER_EVAL_PUBLIC_REQUEST_MATCH')
    need(fingerprint(result['reversed'][0]) == fingerprint(result['correct'][0][::-1]), 'ANSWER_EVAL_REVERSED_PREFIX')
    return result,pins


def generate(loaded, public, *, target, seed, context, output, authority):
    """Only observed readings enter this call. No private answer or length argument."""
    began = time.perf_counter(); output = Path(output); output.mkdir(exist_ok=False)
    need(set(loaded) == set(ARCHITECTURES), 'ANSWER_EVAL_ARCHITECTURES')
    roster = tuple(target.roster); contexts,pins = public_inputs(public,roster)
    expected = {a:list(loaded[a].model.proposals) for a in ARCHITECTURES}
    need(expected == dict(shared=['assembled'],own_slot=['assembled'],whole_proposals=list(roster)), 'ANSWER_EVAL_PROPOSALS')
    first = loaded['shared']
    need(all(v.identity['step'] == first.identity['step'] and v.identity['counts'] == first.identity['counts']
             and v.expected['binding']['source'] == first.expected['binding']['source']
             and v.expected['binding']['schedule'] == first.expected['binding']['schedule'] for v in loaded.values()),
         'ANSWER_EVAL_UNMATCHED_CHECKPOINTS')
    frozen = fingerprint(target.state_dict()); traces = []; phases = dict(prepare=time.perf_counter()-began)
    try:
        for architecture in ARCHITECTURES:
            tick = time.perf_counter(); checkpoint = loaded[architecture]; checkpoint.fresh()
            for mode in MODES:
                frames,request = contexts[mode]
                for ordinal,proposal in enumerate(expected[architecture]):
                    path = output/(architecture+'-'+mode+'-'+str(ordinal))
                    pin = collect(checkpoint.model,frames,request=request,target=target,proposal=proposal,
                        context=context,seed=seed,output=path,authority=authority)
                    traces.append(dict(architecture=architecture,proposal=proposal,mode=mode,path=str(path),pin=pin))
            checkpoint.fresh(physical=True); phases[architecture] = time.perf_counter()-tick
        tick = time.perf_counter()
        grid(traces,expected=expected,public=pins,roster=roster,authority=authority)
        need(fingerprint(target.state_dict()) == frozen, 'ANSWER_EVAL_FROZEN_CHANGED')
        report = dict(complete=True,schema='complete-answer-generation-v1',authority=authority,
            checkpoints={a:v.identity for a,v in loaded.items()},public=pins,expected=expected,roster=list(roster),
            traces=[dict(r,path=str(Path(r['path']).relative_to(output))) for r in traces],
            private_reference_received=False,reference_length_received=False,optimizer_updates=0,
            semantic_validity_established=False,architecture_winner=False)
        phases['seal'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
        need(.9 <= sum(phases.values())/seconds <= 1.001, 'ANSWER_EVAL_PHASES')
        report.update(phases=phases,seconds=seconds)
        return write_committed(output/'report.json',report)
    except BaseException as exc:
        write_committed(output/'failure.json',dict(error=type(exc).__name__,detail=str(exc))); raise
