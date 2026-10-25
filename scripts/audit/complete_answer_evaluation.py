"""Separate public-only generation worker and later private-reference evaluation."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

if '--cpu-proof' in sys.argv: os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from claims.process_conditions import write_committed
from model.complete_answer_evaluation import EvaluationCheckpoint, generate, public_inputs, ARCHITECTURES
from model.native_population_training import fingerprint
from model.task_answer_training import extension
from text.constellation_corpus import file_sha, need

ROOT = Path('/zfs/archive/mnlm/evidence/fsv/29'); BASE = Path(__file__).resolve().parents[1]
NEW = ('model/complete_answer_evaluation.py','audit/complete_answer_evaluation.py')
TRAINED = ('legal-answer-epoch-proof-v3-20261024','50bed7838cb795e65c937550595789944aec3d087b81a1f8bd933450611637cf')
TRAINED_COLD = ('legal-answer-epoch-proof-cold-v3-20261024','cb6170cd7dc18cadc66bb6fb19f53395f062761ad4bc018fbf3d162eff48d3c6')
RUNTIME = ('dimensions','config','sizes','statistics','target','target_ref','policy','target_fingerprint')
CACHE = 64 << 10


def read(path,pin):
    need(file_sha(path) == pin,'ANSWER_EVAL_PROOF_PIN'); return json.loads(path.read_bytes())


def save(path,value):
    with path.open('xb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    need(fingerprint(torch.load(path,map_location='cpu',weights_only=True)) == fingerprint(value),'ANSWER_EVAL_PROOF_PERSIST')
    return dict(path=str(path),pin=file_sha(path),state=fingerprint(value))


def artifact(ref):
    path = Path(ref['path']); need(file_sha(path) == ref['pin'],'ANSWER_EVAL_PROOF_ARTIFACT_PIN')
    result = torch.load(path,map_location='cpu',weights_only=True)
    need(fingerprint(result) == ref['state'],'ANSWER_EVAL_PROOF_ARTIFACT_STATE'); return result


def worker(a):
    # The worker protocol contains no private reference path, answer frames or answer length.
    began = time.perf_counter(); a.output.mkdir(exist_ok=False)
    m = read(a.request,a.request_pin)
    need(set(m) == {'schema','runtime','public','checkpoints','code','authority','seed','context','intervention'}
         and m['schema'] == 'public-only-answer-evaluation-v1','ANSWER_EVAL_WORKER_SCHEMA')
    allowed = {a.request.resolve(),Path(m['runtime']['path']).resolve(),Path(m['public']['path']).resolve()}
    allowed.update(Path(r['reference']['path']).resolve() for r in m['checkpoints'].values())
    observed = set(); output_root = a.output.resolve()
    def access(event,args):
        if event != 'open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path = Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to('/zfs'):return
        need(path in allowed or path.is_relative_to(output_root),'ANSWER_EVAL_UNDECLARED_ARTIFACT_ACCESS')
        if path in allowed:observed.add(str(path))
    sys.addaudithook(access)
    for n,h in m['code'].items():need(file_sha(BASE/n) == h,'ANSWER_EVAL_WORKER_CODE')
    runtime = artifact(m['runtime']); public = artifact(m['public'])
    need(set(runtime) == set(RUNTIME),'ANSWER_EVAL_RUNTIME_FIELDS')
    target,target_pin = extension(runtime)
    need(fingerprint(target.state_dict()) == runtime['target_fingerprint'],'ANSWER_EVAL_RUNTIME_TARGET')
    public_inputs(public,target.roster)
    need(a.cpu_proof and m['intervention'] == dict(kind='CPU_memory_pressure_mechanics',cache_bytes=CACHE),
         'ANSWER_EVAL_PROOF_MODE')
    loaded = {}; timings = {}; start = time.perf_counter()
    need(set(m['checkpoints']) == set(ARCHITECTURES),'ANSWER_EVAL_WORKER_CHECKPOINTS')
    for architecture in ARCHITECTURES:
        row = m['checkpoints'][architecture]
        checkpoint = EvaluationCheckpoint(Path(row['reference']['path']),row['reference'],row['metadata'],
                                          runtime,target,target_pin,device='cpu')
        need(checkpoint.architecture == architecture,'ANSWER_EVAL_WORKER_ARCHITECTURE')
        checkpoint.model.forecaster.predictor.cache_bytes = CACHE
        loaded[architecture] = checkpoint
    timings['checkpoint_load'] = time.perf_counter()-start; start = time.perf_counter()
    pin = generate(loaded,public,target=target,seed=m['seed'],context=m['context'],output=a.output/'generation',authority=m['authority'])
    timings['generation'] = time.perf_counter()-start
    for checkpoint in loaded.values(): checkpoint.fresh(physical=True)
    for n,h in m['code'].items():need(file_sha(BASE/n) == h,'ANSWER_EVAL_WORKER_CODE_CHANGED')
    need(file_sha(a.request) == a.request_pin and not torch.cuda.is_initialized(),'ANSWER_EVAL_WORKER_FINAL')
    # A real but undeclared artifact must refuse before its bytes are read.
    outside = Path(m['checkpoints']['shared']['reference']['path']).parent.parent/'report.json'
    refused = reject(lambda:file_sha(outside),'ANSWER_EVAL_UNDECLARED_ARTIFACT_ACCESS')
    need(observed == {str(p) for p in allowed},'ANSWER_EVAL_DECLARED_INPUT_COVERAGE')
    result = dict(complete=True,request=a.request_pin,generation=pin,checkpoints={n:v.identity for n,v in loaded.items()},
        public_artifact=m['public'],runtime_artifact=m['runtime'],checkpoint_loads=len(loaded),frozen_target_loads=1,
        private_reference_received=False,optimizer_updates=0,gpu_used=False,timings=timings,seconds=time.perf_counter()-began,
        observed_artifact_inputs=sorted(observed),undeclared_artifact_refused=refused,
        access_guard='Python open audit hook over ZFS artifacts; not an OS sandbox',
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    print(json.dumps(dict(complete=True,pin=write_committed(a.output/'report.json',result))),flush=True)


def reject(fn,code):
    try:fn()
    except (ValueError,TypeError) as exc:
        need(code in str(exc),'ANSWER_EVAL_PROOF_WRONG_REFUSAL:'+str(exc));return str(exc)
    raise ValueError('ANSWER_EVAL_PROOF_NEGATIVE_ACCEPTED')


def run(a):
    from audit.complete_answer_comparison import prepare, direct, normalized
    from audit.legal_grouped_placement_training import source_bundle
    from model.complete_answer_comparison import compare,summarize
    need(a.cpu_proof,'ANSWER_EVAL_EXPLICIT_PROOF')
    began = tick = time.perf_counter(); a.output.mkdir(exist_ok=False); phases = {}; controls = {}
    trained = read(a.root/TRAINED[0]/'report.json',TRAINED[1]); cold = read(a.root/TRAINED_COLD[0]/'report.json',TRAINED_COLD[1])
    need(trained['complete'] and cold['complete'] and cold['previous_pin'] == TRAINED[1]
         and cold['cold_reproduced'] and trained['optimizer_updates'] == cold['optimizer_updates'] == 15
         and trained['code'] == cold['code'],'ANSWER_EVAL_TRAINED_AUTHORITY')
    old,stored,identity,options = prepare(a.root); original,bundle = source_bundle(a.root)
    runtime = {n:bundle[n] for n in RUNTIME}; del bundle
    source = trained['source']['binding']; need(source == old['producer']['binding']['source'],'ANSWER_EVAL_SOURCE_JOIN')
    code = dict(trained['code'])
    code.update({n:file_sha(BASE/n) for n in (*NEW,'model/complete_answer_comparison.py','model/complete_answer_trace.py',
        'model/complete_answer_scores.py','audit/complete_answer_comparison.py')})
    for n,h in code.items():need(file_sha(BASE/n) == h,'ANSWER_EVAL_CODE_AUTHORITY')
    checkpoints = {}
    for architecture in ARCHITECTURES:
        result = trained['results'][architecture]; c = cold['results'][architecture]
        need(result['updates'] == c['updates'] and result['counts'] == c['counts'] and result['binding'] == c['binding'],
             'ANSWER_EVAL_COLD_TRAINING')
        ref = result['checkpoints'][-1]; path = a.root/TRAINED[0]/architecture/'checkpoint-5.pt'
        need(Path(ref['path']) == path and file_sha(path) == ref['pin'] == trained['files'][architecture+'/checkpoint-5.pt'],
             'ANSWER_EVAL_TRAINED_BYTES')
        metadata = dict(schema='legal-complete-answer-epoch-v1',binding=result['binding'],cursor=5,step=10,
            noise_cursor=result['binding']['policy']['noise_seed']+10,chain=result['chain'],counts=result['counts'])
        checkpoints[architecture] = dict(reference=ref,metadata=metadata)
    public = {mode:dict(frames=frames,descriptor=descriptor) for mode,frames,descriptor in
        (('correct',stored['prefix'],stored['request']),('reversed',stored['prefix'][::-1],stored['request']),
         ('train_donor',stored['donor'],stored['donor_request']))}
    runtime_ref = save(a.output/'runtime.pt',runtime); public_ref = save(a.output/'public.pt',public)
    authority = dict(training=TRAINED[1],cold_training=TRAINED_COLD[1],document=identity['document'],
        source=source,code=code,scope='public_only_generation_memory_pressure_proof',
        checkpoint_pins={n:v['reference']['pin'] for n,v in checkpoints.items()})
    m = dict(schema='public-only-answer-evaluation-v1',runtime=runtime_ref,public=public_ref,
        checkpoints=checkpoints,code=code,authority=authority,seed=20261024+identity['document'],context='checkpoint-evaluation',
        intervention=dict(kind='CPU_memory_pressure_mechanics',cache_bytes=CACHE))
    request_pin = write_committed(a.output/'request.json',m)
    phases['authentication_and_public_export'] = time.perf_counter()-tick; tick = time.perf_counter()
    with (a.output/'worker.log').open('xb') as log:
        command=[sys.executable,'-I',str(Path(__file__).resolve()),'--worker','--cpu-proof','--request',str(a.output/'request.json'),
                 '--request-pin',request_pin,'--output',str(a.output/'worker')]
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    worker_report = json.loads((a.output/'worker/report.json').read_bytes()); worker_pin = file_sha(a.output/'worker/report.json')
    need(worker_report['complete'] and worker_report['request'] == request_pin and worker_report['checkpoint_loads'] == 3
         and worker_report['frozen_target_loads'] == 1 and worker_report['private_reference_received'] is False,'ANSWER_EVAL_WORKER_RECEIPT')
    generation = read(a.output/'worker/generation/report.json',worker_report['generation'])
    need(generation['complete'] and generation['private_reference_received'] is False
         and generation['reference_length_received'] is False,'ANSWER_EVAL_GENERATION_SEALED')
    phases['isolated_generation'] = time.perf_counter()-tick; tick = time.perf_counter()
    # Only the parent scorer has a private future; the worker above has exited.
    rows = [dict(r,path=str(a.output/'worker/generation'/r['path'])) for r in generation['traces']]
    options.update(authority=authority)
    case = compare(rows,**options,identity=identity,future=stored['item']['answers'],copy_frame=stored['prefix'][-1])
    summary = summarize([case]); write_committed(a.output/'case.json',case);write_committed(a.output/'summary.json',summary)
    for row in rows:
        actual = direct(Path(row['path']),stored['item']['answers'],options['roster'])
        metric = next(v['metrics'] for v in case['arms'] if all(v[k] == row[k] for k in ('architecture','proposal','mode')))
        need(actual['per_slot'] == metric['per_slot'] and actual['means'] == metric['full_output_mean']
             and actual['generated'] == metric['generated_atoms'] and actual['ended'] == metric['ended'], 'ANSWER_EVAL_DIRECT_SCORE')
    phases['private_scoring_and_direct_readback'] = time.perf_counter()-tick; tick = time.perf_counter()
    target,target_pin = extension(runtime); row = checkpoints['shared']; ref = row['reference']; metadata = row['metadata']
    loader=lambda path,r,expected:EvaluationCheckpoint(path,r,expected,runtime,target,target_pin,device='cpu')
    controls['wrong_pin'] = reject(lambda:loader(ref['path'],dict(ref,pin='0'*64),metadata),'ANSWER_EVAL_CHECKPOINT_PIN')
    wrong = deepcopy(metadata); wrong['binding']['architecture'] = 'own_slot'
    controls['binding'] = reject(lambda:loader(ref['path'],ref,wrong),'ANSWER_EVAL_BINDING')
    wrong = deepcopy(metadata); wrong['cursor'] += 1
    controls['cursor'] = reject(lambda:loader(ref['path'],ref,wrong),'ANSWER_EVAL_CURSOR')
    missing = torch.load(ref['path'],map_location='cpu',weights_only=True)
    missing['optimizer']['state'].pop(next(iter(missing['optimizer']['state'])))
    bad = save(a.output/'missing-moments.pt',missing)
    controls['moments'] = reject(lambda:loader(bad['path'],bad,metadata),'PLACEMENT_OPTIMIZER_INCOMPLETE_STATE_RESTORE_FULL_CHECKPOINT')
    leaked = deepcopy(public); leaked['correct']['answers'] = stored['item']['answers']
    controls['private_fields'] = reject(lambda:public_inputs(leaked,options['roster']),'ANSWER_EVAL_PUBLIC_FIELDS')
    leaked = deepcopy(public); leaked['correct']['target_length'] = 16
    controls['target_length'] = reject(lambda:public_inputs(leaked,options['roster']),'ANSWER_EVAL_PUBLIC_FIELDS')
    previous = read(a.previous/'report.json',a.previous_pin) if a.previous else None
    numerical = normalized(case)
    # request authorities are identical; output paths and elapsed timings are not numerical state.
    if previous:
        old_case = read(a.previous/'case.json',previous['files']['case.json'])
        old_summary = read(a.previous/'summary.json',previous['files']['summary.json'])
        need(previous['complete'] and previous['authority'] == authority and previous['controls'] == controls
             and normalized(old_case) == numerical and old_summary == summary,'ANSWER_EVAL_COLD_RESULTS')
        from audit.complete_answer_trace import independent
        prior_generation = read(a.previous/'worker/generation/report.json',previous['generation'])
        for left,right in zip(generation['traces'],prior_generation['traces'],strict=True):
            need(fingerprint(independent(a.output/'worker/generation'/left['path'])) ==
                 fingerprint(independent(a.previous/'worker/generation'/right['path'])),'ANSWER_EVAL_COLD_NATIVE_EVENTS')
    for n,h in code.items():
        need(file_sha(BASE/n) == h,'ANSWER_EVAL_FINAL_CODE');dest=a.output/'source'/n
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((BASE/n).read_bytes())
    for row in checkpoints.values():need(file_sha(row['reference']['path']) == row['reference']['pin'],'ANSWER_EVAL_FINAL_CHECKPOINT')
    phases['controls_cold_and_seal'] = time.perf_counter()-tick; tick=time.perf_counter()
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['hashes']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.001 and not torch.cuda.is_initialized(),'ANSWER_EVAL_PROOF_FINAL')
    report=dict(complete=True,authority=authority,files=files,controls=controls,phases=phases,seconds=seconds,
        worker=worker_pin,generation=worker_report['generation'],checkpoints=worker_report['checkpoints'],
        checkpoint_loads_per_worker=3,frozen_target_loads_per_worker=1,trajectories=len(rows),
        generated_atoms=sum(r['metrics']['generated_atoms'] for r in case['arms']),
        ends=sum(r['metrics']['ended'] for r in case['arms']),correct_length_ends=sum(r['metrics']['ended_at_reference_boundary'] for r in case['arms']),
        gpu_used=False,optimizer_updates=0,frozen_encoder_calls=0,cold_exact=previous is not None,previous_pin=a.previous_pin,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        capability_demonstrated=False,architecture_winner=False,acceptance='author-checked')
    print(json.dumps(dict(complete=True,pin=write_committed(a.output/'report.json',report),seconds=seconds)),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--cpu-proof',action='store_true')
    p.add_argument('--worker',action='store_true');p.add_argument('--request',type=Path);p.add_argument('--request-pin')
    p.add_argument('--previous',type=Path);p.add_argument('--previous-pin');a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    try:worker(a) if a.worker else run(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise
