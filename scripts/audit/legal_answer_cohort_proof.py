"""Real-source resident/fresh/cold whole-answer comparison and interruption proof."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit.complete_answer_comparison import direct, normalized
from audit.complete_answer_evaluation import read, save, BASE, ROOT, CACHE
from audit.complete_answer_trace import independent
from audit.legal_complete_answer_consumer import prepare, independent as source_readback
from claims.process_conditions import write_committed
from model import legal_answer_evaluation_plan as schedule_module
from model.complete_answer_cohort import case_authority
from model.complete_answer_comparison import compare, summarize
from model.complete_answer_evaluation import public_inputs
from model.native_population_training import fingerprint
from model.native_request_input import NativeRequest, legal
from text.constellation_corpus import file_sha, need
from text.legal_native_store import SHARD_INPUTS

EVALUATOR = ('complete-answer-evaluation-v3-20261024', 'dc3bccf168e96f87ebc8c0665040ed66ebd866b9065a433cdd558b6bce975b3c')
PLAN = ('legal-answer-evaluation-plan-v2-20261024', 'f0ffd23c01f9d9db5d388783fd414c9840bf93dff9f8ce872564146565f9a1a7')
COLD = ('legal-answer-evaluation-plan-cold-v2-20261024', '75362d95e072a272d6626f5a87ba2e705cf49e20cf7fba86ed7d4f5a77869f9e')
DOCUMENTS = (19295, 20107, 22924, 30520)
NEW = ('model/complete_answer_cohort.py', 'audit/complete_answer_cohort.py', 'audit/legal_answer_cohort_proof.py',
       'model/legal_answer_evaluation_plan.py')


def native_prefix(corpus, ids, frames):
    for uid, frame in zip(ids, frames, strict=True):
        start = uid//SHARD_INPUTS*SHARD_INPUTS
        for name in corpus.roster:
            d = corpus.data.native.descriptor(name, start); corpus.data.native.descriptor_fresh(d)
            values = np.load(d['root']/'vectors.npy', mmap_mode='r', allow_pickle=False)
            try:
                lo, hi = map(int, d['offsets'][uid-start:uid-start+2]); actual = frame[name].numpy()
                need(values[lo:hi].shape == actual.shape and values[lo:hi].tobytes() == actual.tobytes(), 'COHORT_PUBLIC_NATIVE')
            finally:values._mmap.close()
            corpus.data.native.descriptor_fresh(d)


def invoke(script, request, pin, output, log, *, receipt=None, fault=None, expected=0):
    command = [sys.executable, '-I', str(BASE/'audit'/script), '--cpu-proof', '--request', str(request),
               '--request-pin', pin, '--output', str(output)]
    if receipt is None:command += ['--worker']
    else:command += ['--receipt', receipt]
    if fault:command += ['--fault', fault]
    tick = time.perf_counter()
    with log.open('xb') as stream:
        child = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    need(child.returncode == expected, 'COHORT_CHILD_EXIT:'+str(child.returncode)+':'+str(log))
    return dict(exit_code=child.returncode, wall_seconds=time.perf_counter()-tick, log=file_sha(log))


def generation(root, result):
    path = root/'run'/result['path']; case = read(path/'case.json', result['pin'])
    need(case['status'] == 'generated', 'COHORT_PROOF_UNAVAILABLE_NATIVE_CONTROL')
    p = path/case['generation']['path']
    return p, read(p/'report.json', case['generation']['pin'])


def equal_events(left, a, right, b):
    need(len(a['traces']) == len(b['traces']) == 24, 'COHORT_PROOF_GRID')
    total = 0
    for x,y in zip(a['traces'], b['traces'], strict=True):
        need(all(x[k] == y[k] for k in ('architecture', 'proposal', 'mode')), 'COHORT_PROOF_TRACE_ORDER')
        xx, yy = independent(left/x['path']), independent(right/y['path'])
        need(fingerprint(xx) == fingerprint(yy), 'COHORT_PROOF_NATIVE_EVENT_DIFFERENCE'); total += len(xx)
    return total


def run(a):
    began = tick = time.perf_counter(); a.output.mkdir(exist_ok=False); phases = {}
    prior_eval = read(a.root/EVALUATOR[0]/'report.json', EVALUATOR[1])
    old = read(a.root/EVALUATOR[0]/'request.json', prior_eval['files']['request.json'])
    corpus, labels, continuation, bundle, code = prepare(a.root)
    need(corpus.binding == old['authority']['source'], 'COHORT_PROOF_SOURCE')
    del bundle
    pr, pc = [read(a.root/d/'report.json', pin) for d,pin in (PLAN, COLD)]
    need(pc['previous_pin'] == PLAN[1] and pr['summary'] == pc['summary'] and pr['controls'] == pc['controls']
         and pr['binding'] == pc['binding'] == corpus.source.binding, 'COHORT_PROOF_SCHEDULE_AUTHORITY')
    path = a.root/PLAN[0]/'evaluation-plan.npy'; need(file_sha(path) == pr['plan_sha256'], 'COHORT_PROOF_PLAN_BYTES')
    rows = np.load(path, mmap_mode='r', allow_pickle=False)
    public = schedule_module.prefixes(corpus.source); index = schedule_module.buckets(corpus.source, public)
    schedule = schedule_module.Schedule(corpus.source, rows, public, index)
    code.update(old['code']); code.update({n:file_sha(BASE/n) for n in NEW})
    for n,h in code.items():need(file_sha(BASE/n) == h, 'COHORT_PROOF_CODE')
    common = dict(evaluation_plan=PLAN[1], cold_plan=COLD[1], training=old['authority']['training'],
        cold_training=old['authority']['cold_training'], source=fingerprint(corpus.binding),
        code=fingerprint(code), scope='CPU_resident_mechanics_exposed_cases')
    cases = []; publics = {}; identities = {}
    for document in DOCUMENTS:
        ordinal = int(np.flatnonzero(rows['document'] == document)[0]); row = rows[ordinal]
        request = schedule.request(ordinal); donor_request = schedule.donor_request(ordinal)
        ids = list(request['observed_uids']); donor_ids = list(donor_request['observed_uids'])
        frames = corpus.reader.frames(ids); donor = corpus.reader.frames(donor_ids)
        native_prefix(corpus, ids, frames); native_prefix(corpus, donor_ids, donor)
        descriptor = legal(len(ids)); NativeRequest(descriptor)
        value = {m:dict(frames=f, descriptor=descriptor) for m,f in
                 (('correct', frames), ('reversed', frames[::-1]), ('train_donor', donor))}
        _, pins = public_inputs(value, corpus.roster)
        need(pins['correct']['prefix'] != pins['train_donor']['prefix'], 'COHORT_PROOF_IDENTICAL_NATIVE')
        for j in range(len(ids)):
            need([len(frames[j][n]) for n in corpus.roster] == row['layout'][j].tolist()
                 == [len(donor[j][n]) for n in corpus.roster], 'COHORT_PROOF_NATIVE_LAYOUT')
        ref = save(a.output/f'public-{document}.pt', value)
        cases.append(dict(ordinal=ordinal, document=document, public=ref, seed=20261024+document,
                          context='legal-cohort-'+str(document), reversal_distinct=bool(row['reversal_distinct'])))
        publics[document] = pins
        identities[document] = dict(source=fingerprint(corpus.binding), side=('train','development','query')[int(row['side'])],
            document=document, component=int(row['component']), exposure=corpus.source.binding['exposure'])
    cases.sort(key=lambda r:r['ordinal'])
    m = dict(schema='public-only-answer-cohort-v1', runtime=old['runtime'], checkpoints=old['checkpoints'],
             code=code, authority=common, cases=cases, intervention=old['intervention'])
    request_pin = write_committed(a.output/'request.json', m)
    phases['source_and_public_export'] = time.perf_counter()-tick; tick = time.perf_counter()
    launches = {}; worker = 'complete_answer_cohort.py'; root = a.output/'resident'
    launches['resident'] = invoke(worker, a.output/'request.json', request_pin, root, a.output/'resident.log', receipt='initial')
    resident = json.loads((root/'initial.json').read_bytes())
    need(resident['actions'] == dict(generated=4, reused=0, recovered=0, gap=0)
         and resident['checkpoint_loads'] == 3 and resident['frozen_target_loads'] == 1, 'COHORT_PROOF_RESIDENCY')
    launches['reopen'] = invoke(worker, a.output/'request.json', request_pin, root, a.output/'reopen.log', receipt='reopen')
    reopened = json.loads((root/'reopen.json').read_bytes())
    need(reopened['actions'] == dict(generated=0, reused=4, recovered=0, gap=0)
         and [(v['path'],v['pin']) for v in reopened['results']] == [(v['path'],v['pin']) for v in resident['results']],
         'COHORT_PROOF_REOPEN')
    phases['resident_and_reopen'] = time.perf_counter()-tick; tick = time.perf_counter()
    matched_events = 0; fresh = []
    previous = read(a.previous/'report.json', a.previous_pin) if a.previous else None
    if previous:
        need(previous['complete'] and previous['authority'] == common and previous['code'] == code, 'COHORT_PROOF_COLD_AUTHORITY')
        prior = read(a.previous/'resident/initial.json', previous['files']['resident/initial.json'])
        for current, other in zip(resident['results'], prior['results'], strict=True):
            x, xx = generation(root, current); y, yy = generation(a.previous/'resident', other)
            matched_events += equal_events(x, xx, y, yy)
    else:
        for row, result in zip(cases, resident['results'], strict=True):
            one = dict(schema='public-only-answer-evaluation-v1', runtime=m['runtime'], public=row['public'],
                checkpoints=m['checkpoints'], code=code, authority=case_authority(common,row),
                seed=row['seed'], context=row['context'], intervention=m['intervention'])
            path = a.output/f"single-{row['document']}.json"; pin = write_committed(path, one)
            dest = a.output/f"fresh-{row['document']}"
            launches[str(row['document'])] = invoke('complete_answer_evaluation.py', path, pin, dest,
                                                   a.output/f"fresh-{row['document']}.log")
            rr = json.loads((dest/'report.json').read_bytes()); fresh.append(rr)
            x, xx = generation(root, result)
            yy = read(dest/'generation/report.json', rr['generation'])
            matched_events += equal_events(x, xx, dest/'generation', yy)
    phases['fresh_or_cold_comparison'] = time.perf_counter()-tick; tick = time.perf_counter()
    scored = []; reference_rows = []
    for row, result in zip(cases, resident['results'], strict=True):
        item, req, _ = corpus.example(row['document'], fitting=False)
        reference_rows.append(source_readback(corpus,item,req))
        reference = save(a.output/f"private-{row['document']}.pt", item)
        identity = dict(identities[row['document']], reference=reference['pin'])
        path, g = generation(root, result)
        traces = [dict(v, path=str(path/v['path'])) for v in g['traces']]
        case = compare(traces, expected=g['expected'], public=publics[row['document']], authority=case_authority(common,row),
            identity=identity, future=item['answers'], roster=corpus.roster, copy_frame=item['context'][-1])
        for trace, arm in zip(sorted(traces,key=lambda x:(x['architecture'],x['proposal'],
            ('correct','reversed','train_donor').index(x['mode']))),case['arms'],strict=True):
            actual = direct(Path(trace['path']),item['answers'],corpus.roster); metric = arm['metrics']
            need(actual['per_slot'] == metric['per_slot'] and actual['means'] == metric['full_output_mean']
                 and actual['generated'] == metric['generated_atoms'] and actual['ended'] == metric['ended'], 'COHORT_PROOF_SCORES')
        write_committed(a.output/f"case-{row['document']}.json",case); scored.append(case)
        if previous:
            old_case = read(a.previous/f"case-{row['document']}.json", previous['files'][f"case-{row['document']}.json"])
            need(normalized(old_case) == normalized(case), 'COHORT_PROOF_COLD_SCORES')
    summary = summarize(scored); write_committed(a.output/'summary.json',summary)
    corpus.close(); corpus.reader.close(); corpus.cache.close(); labels.fresh(); continuation.fresh(); rows._mmap.close()
    phases['private_scoring_and_source_readback'] = time.perf_counter()-tick; tick = time.perf_counter()
    controls = {}; faults = {}
    if not previous:
        subset = dict(m, cases=cases[:1]); path = a.output/'fault-request.json'; pin = write_committed(path,subset)
        for fault, status in (('trace',71),('generation',72)):
            dest = a.output/('fault-'+fault)
            launches['fault-'+fault] = invoke(worker,path,pin,dest,a.output/('fault-'+fault+'.log'),receipt='fault',fault=fault,expected=status)
            case_path = dest/'run'/f"case-{cases[0]['ordinal']:06d}"; attempt = case_path/'attempt-000000'
            before = {str(p.relative_to(attempt)):file_sha(p) for p in attempt.rglob('*') if p.is_file()}
            need(before and not (case_path/'case.json').exists(), 'COHORT_PROOF_FAULT_BOUNDARY')
            launches['recover-'+fault] = invoke(worker,path,pin,dest,a.output/('recover-'+fault+'.log'),receipt='recovered')
            rr = json.loads((dest/'recovered.json').read_bytes())
            need(rr['actions'] == dict(generated=int(fault=='trace'), recovered=int(fault=='generation'), reused=0, gap=0),
                 'COHORT_PROOF_RECOVERY_ACTION')
            need(all(file_sha(attempt/n) == h for n,h in before.items()), 'COHORT_PROOF_ATTEMPT_CHANGED')
            x, xx = generation(dest,rr['results'][0]); y, yy = generation(root,resident['results'][0])
            events = equal_events(x,xx,y,yy)
            faults[fault] = dict(exit_code=status, before_files=before, actions=rr['actions'], exact_events=events)
        for label in ('order','binding','private'):
            changed = deepcopy(m)
            if label == 'order':changed['cases'].reverse(); error = 'ANSWER_COHORT_ORDER'
            elif label == 'binding':changed['cases'][0]['seed'] += 1; error = 'ANSWER_COHORT_BINDING_CHANGED'
            else:changed['cases'][0]['answers'] = []; error = 'ANSWER_COHORT_PUBLIC_FIELDS'
            p = a.output/(label+'-control.json'); h = write_committed(p,changed)
            launches[label] = invoke(worker,p,h,root,a.output/(label+'-control.log'),receipt=label,expected=1)
            failure = json.loads((root/(label+'-failure.json')).read_bytes())
            need(error in failure['detail'], 'COHORT_PROOF_WRONG_REFUSAL'); controls[label] = failure['detail']
        bad = a.output/'corrupt'; (bad/'run').mkdir(parents=True)
        shutil.copyfile(root/'run/binding.json',bad/'run/binding.json')
        sub = resident['results'][0]['path']; (bad/'run'/sub).mkdir()
        changed = json.loads((root/'run'/sub/'case.json').read_bytes()); changed['binding']['case']['document'] += 1
        write_committed(bad/'run'/sub/'case.json',changed)
        launches['corrupt'] = invoke(worker,a.output/'request.json',request_pin,bad,a.output/'corrupt.log',receipt='control',expected=1)
        failure = json.loads((bad/'control-failure.json').read_bytes())
        need('ANSWER_COHORT_CASE_CHANGED' in failure['detail'], 'COHORT_PROOF_COMMIT_REFUSAL'); controls['commit'] = failure['detail']
    phases['crash_and_negative_controls'] = time.perf_counter()-tick; tick = time.perf_counter()
    for n,h in code.items():
        need(file_sha(BASE/n) == h,'COHORT_PROOF_FINAL_CODE');dest = a.output/'source'/n
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((BASE/n).read_bytes())
    files = {str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['seal'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
    need(.9 <= sum(phases.values())/seconds <= 1.001 and not torch.cuda.is_initialized(), 'COHORT_PROOF_FINAL')
    report = dict(complete=True,authority=common,code=code,files=files,cases=len(cases),trajectories=len(cases)*24,
        actions=resident['actions'],reopen_actions=reopened['actions'],matched_native_events=matched_events,
        fresh_worker_checkpoint_loads=sum(v['checkpoint_loads'] for v in fresh),resident_checkpoint_loads=3,
        resident=resident, fresh_workers=[{k:v[k] for k in ('seconds','timings','checkpoint_loads','frozen_target_loads')} for v in fresh],
        reference_rows=reference_rows,controls=controls,faults=faults,launches=launches,phases_seconds=phases,
        seconds=seconds,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        previous_pin=a.previous_pin,cold_exact=previous is not None,
        controls_from_previous=a.previous_pin if previous else None,
        generated_atoms=sum(arm['metrics']['generated_atoms'] for c in scored for arm in c['arms']),
        ends=sum(arm['metrics']['ended'] for c in scored for arm in c['arms']),
        correct_length_ends=sum(arm['metrics']['ended_at_reference_boundary'] for c in scored for arm in c['arms']),
        acceptance='author-checked',gpu_used=False,optimizer_updates=0,architecture_winner=False,capability_demonstrated=False)
    pin = write_committed(a.output/'report.json',report)
    print(json.dumps(dict(pin=pin,cases=len(cases),trajectories=len(cases)*24,seconds=seconds,matched_events=matched_events)),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--previous',type=Path);p.add_argument('--previous-pin');a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise
