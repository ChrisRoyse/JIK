"""Re-score sealed real answer traces; no inference, training or encoder reloads."""
import argparse
from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path
import resource
import shutil
import struct
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from claims.process_conditions import write_committed
from model.complete_answer_comparison import compare, grid, summarize, trace_score
from model.native_population_training import fingerprint
from model.native_request_input import NativeRequest
from text.constellation_corpus import file_sha, need

ROOT = Path('/zfs/archive/mnlm/evidence/fsv/29')
BASE = Path(__file__).resolve().parents[1]
REFS = dict(producer=('complete-answer-trace-proof-v2-20261024', '16efc3d7afe39494e82c0656a44c21b778006878e2d21637f8baa96a8c3d3188'),
    cold=('complete-answer-trace-cold-v2-20261024', '6510fd0988cf35c505587745acc069223eb07e1e9f5e31bb83d03f7034ab7cc5'),
    reference=('complete-answer-trace-readback-v1-20261024', '20cb9a4508f09697893fdf1285da3b4d8c4e696f3f4a8de6b5be0494b326fc08'))
NEW = ('model/complete_answer_comparison.py', 'audit/complete_answer_comparison.py')


def read(path, pin):
    need(file_sha(path) == pin, 'COMPARISON_PROOF_PIN'); return json.loads(path.read_bytes())


def reject(fn, code):
    try: fn()
    except ValueError as exc:
        need(code in str(exc), 'COMPARISON_PROOF_WRONG_REFUSAL:'+str(exc)); return str(exc)
    raise ValueError('COMPARISON_PROOF_CONTROL_ACCEPTED')


def prepare(root):
    reports = {k:read(root/d/'report.json', pin) for k,(d,pin) in REFS.items()}
    a,b,r = (reports[n] for n in ('producer','cold','reference'))
    need(a['complete'] and b['complete'] and r['complete'] and a['binding'] == b['binding']
         and b['previous_pin'] == REFS['producer'][1] and b['cold_reproduced']
         and r['direct_source_readback'] and r['metrics_independently_recomputed'], 'COMPARISON_PROOF_SOURCE')
    ref = root/REFS['reference'][0]/'reference.pt'; reference_pin = r['members']['readback/reference.pt']['sha256']
    need(file_sha(ref) == reference_pin, 'COMPARISON_PROOF_REFERENCE_PIN')
    stored = torch.load(ref, map_location='cpu', weights_only=True); item = stored['item']
    need(item['side'] != 'train' and item['document'] == a['binding']['document'] == r['document']
         and item['native_pin'] == fingerprint((item['context'],item['answers'])), 'COMPARISON_PROOF_REFERENCE')
    roster = tuple(stored['prefix'][0]); request = NativeRequest(stored['request']); donor = NativeRequest(stored['donor_request'])
    need(donor.pin == request.pin and fingerprint(item['context']) == fingerprint(stored['prefix']), 'COMPARISON_PROOF_REQUEST')
    public = {mode:dict(prefix=fingerprint(frames), request=req.pin, descriptor=json.loads(req.descriptor))
        for mode,frames,req in (('correct',stored['prefix'],request), ('reversed',stored['prefix'][::-1],request),
                               ('train_donor',stored['donor'],donor))}
    expected = dict(shared=['assembled'], own_slot=['assembled'], whole_proposals=list(roster))
    identity = dict(source=fingerprint(a['binding']['source']), side=item['side'], document=item['document'],
        component=item['provenance']['component'], exposure=a['binding']['exposure'], reference=reference_pin)
    options = dict(expected=expected, public=public, roster=roster, authority=a['binding'])
    return reports, stored, identity, options


def traces(root, label, report):
    return [dict(architecture=a, proposal=r['proposal'], mode=r['mode'], path=str(root/REFS[label][0]/r['path']), pin=r['report'])
            for a,rows in report['results'].items() for r in rows]


def normalized(case):
    value = deepcopy(case)
    for arm in value['arms']: arm.pop('pin')
    return value


def direct(path, future, roster):
    """Separate framed-payload reader and NumPy calculation, no production scorer."""
    values = {n:[-1.] * len(future) for n in roster}; generated = 0; ended = False
    with (path/'events.bin').open('rb') as stream:
        while header := stream.read(8):
            need(len(header) == 8, 'COMPARISON_DIRECT_FRAME')
            size = struct.unpack('>Q', header)[0]; need(0 < size <= 64<<20, 'COMPARISON_DIRECT_MEMORY')
            raw = stream.read(size); need(len(raw) == size, 'COMPARISON_DIRECT_BYTES')
            event = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
            need(not ended, 'COMPARISON_DIRECT_AFTER_END')
            if event['kind'] == 'end': ended = True; continue
            if generated < len(future):
                for name in roster:
                    x = event['slots'][name].numpy().astype('float64'); y = future[generated][name].numpy().astype('float64')
                    matrix = np.clip((x/np.linalg.norm(x,axis=1,keepdims=True)) @
                                     (y/np.linalg.norm(y,axis=1,keepdims=True)).T, -1, 1)
                    values[name][generated] = float(min(matrix.max(1).mean(),matrix.max(0).mean()))
            generated += 1
    extra = max(0,generated-len(future))
    return dict(per_slot=values, generated=generated, ended=ended, extra=extra,
        missing=max(0,len(future)-generated), means={n:(sum(v)-extra)/(len(future)+extra) for n,v in values.items()})


def run(a):
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    a.output.mkdir(exist_ok=False); began = tick = time.perf_counter(); phases = {}; controls = {}
    reports, stored, identity, options = prepare(a.root)
    rows = traces(a.root, a.label, reports[a.label]); future = stored['item']['answers']; roster = options['roster']
    code = {n:file_sha(BASE/n) for n in NEW}
    binding = dict(references=REFS, identity=identity, protocol=options, code=code,
        scope='previously_exposed_memory_pressure_mechanics_not_capability', optimizer_updates=0, frozen_encoder_calls=0)
    binding = json.loads(json.dumps(binding)); phases['authentication'] = time.perf_counter()-tick
    tick = time.perf_counter()
    case = compare(rows, **options, identity=identity, future=future, copy_frame=stored['prefix'][-1])
    summary = summarize([case]); case_pin = write_committed(a.output/'case.json', case)
    summary_pin = write_committed(a.output/'summary.json', summary)
    phases['paired_scoring'] = time.perf_counter()-tick; tick = time.perf_counter()
    old = {(arch,r['proposal'],r['mode']):r['result']['metrics'] for arch,entries in reports[a.label]['results'].items() for r in entries}
    for arm in case['arms']:
        need(dict(arm['metrics'], copy_current=case['copy_current']['per_slot']) == old[arm['architecture'],arm['proposal'],arm['mode']],
             'COMPARISON_PROOF_EXISTING_METRIC_CHANGED')
    controls['missing_arm'] = reject(lambda:grid(rows[:-1], **options), 'ANSWER_COMPARISON_GRID')
    controls['duplicate_arm'] = reject(lambda:grid(rows+[rows[0]], **options), 'ANSWER_COMPARISON_GRID')
    controls['duplicate_case'] = reject(lambda:summarize([case,case]), 'ANSWER_COMPARISON_DUPLICATE_CASE')
    partial = deepcopy(case); partial['arms'].pop()
    controls['partial_cohort'] = reject(lambda:summarize([partial]), 'ANSWER_COMPARISON_COHORT_GRID')
    partial = deepcopy(case); partial['paired'].pop()
    controls['partial_pairs'] = reject(lambda:summarize([partial]), 'ANSWER_COMPARISON_COHORT_PAIRS')
    changed_model = deepcopy(case); changed_model['identity']['document'] = -1
    changed_model['models']['shared']['model'] = '0'*64
    controls['cohort_checkpoint'] = reject(lambda:summarize([case,changed_model]), 'ANSWER_COMPARISON_COHORT_MODELS')
    wrong = deepcopy(rows); wrong[-1]['pin'] = '0'*64
    controls['wrong_pin'] = reject(lambda:grid(wrong, **options), 'ANSWER_COMPARISON_REPORT_PIN')
    for name in ('prefix', 'model', 'seed', 'cache_bytes'):
        directory = a.output/('changed-'+name); shutil.copytree(rows[-1]['path'], directory)
        manifest = json.loads((directory/'request.json').read_bytes())
        manifest[name] = '0'*64 if name in ('prefix','model') else manifest[name]+1
        (directory/'request.json').unlink(); request_pin = write_committed(directory/'request.json', manifest)
        report = json.loads((directory/'report.json').read_bytes()); report['files']['request.json'] = request_pin
        (directory/'report.json').unlink(); pin = write_committed(directory/'report.json',report)
        changed = deepcopy(rows); changed[-1].update(path=str(directory), pin=pin)
        error = 'ANSWER_COMPARISON_PUBLIC_CONTEXT' if name == 'prefix' else 'ANSWER_COMPARISON_UNMATCHED_ARMS'
        controls[name] = reject(lambda:grid(changed, **options), error)
    terminal = {}
    for architecture in options['expected']:
        for name in ('forced-end-control','physical-control'):
            path = a.root/REFS[a.label][0]/architecture/name; relative = str(path.relative_to(a.root/REFS[a.label][0]))+'/report.json'
            pin = reports[a.label]['files'][relative]
            result = trace_score(path,pin,future=future,roster=roster)
            need(result['generated_atoms'] == 0 and result['missing'] == len(future)
                 and not result['ended_at_reference_boundary'] and all(v == -1. for v in result['full_output_mean'].values())
                 and result['ended'] == (name == 'forced-end-control'), 'COMPARISON_PROOF_TERMINAL_CONTROL')
            terminal[architecture+'/'+name] = result
    controls['terminal'] = terminal
    phases['controls_and_parity'] = time.perf_counter()-tick; tick = time.perf_counter()
    previous = read(a.previous/'report.json',a.previous_pin) if a.previous else None
    if previous:
        need(previous['complete'] and previous['binding'] == binding and previous['label'] != a.label, 'COMPARISON_PROOF_PREVIOUS')
        prior_case = read(a.previous/'case.json',previous['files']['case.json'])
        prior_summary = read(a.previous/'summary.json',previous['files']['summary.json'])
        need(normalized(prior_case) == normalized(case) and prior_summary == summary and previous['controls'] == controls,
             'COMPARISON_PROOF_COLD_DIFFERENCE')
    if a.readback:
        need(previous is not None, 'COMPARISON_PROOF_READBACK_AUTHORITY')
        for row in rows:
            path = Path(row['path']); manifest = json.loads((path/'report.json').read_bytes())
            for n,h in manifest['files'].items(): need(file_sha(path/n) == h, 'COMPARISON_DIRECT_PIN')
            result = direct(path,future,roster)
            arm = next(x for x in case['arms'] if (x['architecture'],x['proposal'],x['mode']) ==
                       (row['architecture'],row['proposal'],row['mode']))['metrics']
            need(result['per_slot'] == arm['per_slot'] and result['means'] == arm['full_output_mean']
                 and result['generated'] == arm['generated_atoms'] and result['ended'] == arm['ended']
                 and result['missing'] == arm['missing'] and result['extra'] == arm['extra'], 'COMPARISON_DIRECT_METRICS')
        for pair in case['paired']:
            correct = next(r['metrics'] for r in case['arms'] if r['architecture'] == pair['architecture'] and
                           r['proposal'] == pair['proposal'] and r['mode'] == 'correct')
            other = next(r['metrics'] for r in case['arms'] if r['architecture'] == pair['architecture'] and
                         r['proposal'] == pair['proposal'] and r['mode'] == pair['control'])
            for n in roster:
                d = [v-w for v,w in zip(correct['per_slot'][n],other['per_slot'][n],strict=True)]
                need(pair['per_slot'][n]['reference_position_delta'] == d and
                     pair['per_slot'][n]['reference_mean_delta'] == math.fsum(d)/len(d) and
                     pair['per_slot'][n]['full_output_mean_delta'] == correct['full_output_mean'][n]-other['full_output_mean'][n],
                     'COMPARISON_DIRECT_PAIRED')
        for group in summary['groups']:
            metric = next(r['metrics'] for r in case['arms'] if all(r[k] == group[k] for k in ('architecture','proposal','mode')))
            need(group['cases'] == 1 and all(group['counts'][k] == int(metric[k]) for k in group['counts'])
                 and all(group['per_slot'][n]['case_mean'] == group['per_slot'][n]['atom_mean'] == metric['full_output_mean'][n]
                         for n in roster), 'COMPARISON_DIRECT_SUMMARY')
    phases['cold_and_direct_readback'] = time.perf_counter()-tick; tick = time.perf_counter()
    for n,h in code.items():
        need(file_sha(BASE/n) == h, 'COMPARISON_PROOF_CODE_CHANGED')
        path = a.output/'source'/n; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes((BASE/n).read_bytes())
    need(not torch.cuda.is_initialized(), 'COMPARISON_PROOF_DEVICE')
    files = {str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['seal'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
    need(.9 <= sum(phases.values())/seconds <= 1.001, 'COMPARISON_PROOF_PHASES')
    report = dict(complete=True,binding=binding,label=a.label,files=files,controls=controls,phases=phases,seconds=seconds,
        case_pin=case_pin,summary_pin=summary_pin,cold_exact=previous is not None,previous_pin=a.previous_pin,
        direct_independent_readback=a.readback,trajectories=len(rows),generated_atoms=sum(r['metrics']['generated_atoms'] for r in case['arms']),
        cases=1,components=1,optimizer_updates=0,frozen_encoder_calls=0,gpu_used=False,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        acceptance='author-checked',capability_demonstrated=False,architecture_winner=False)
    print(json.dumps(dict(complete=True,pin=write_committed(a.output/'report.json',report),seconds=seconds)),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--label',choices=('producer','cold'),required=True)
    p.add_argument('--previous',type=Path); p.add_argument('--previous-pin'); p.add_argument('--readback',action='store_true')
    a = p.parse_args()
    try: run(a)
    except BaseException as exc:
        if a.output.exists(): write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise
