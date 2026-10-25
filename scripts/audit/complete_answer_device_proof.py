"""Exposed checkpoint proof of declared-device evaluator mechanics; no GPU claim."""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit.complete_answer_evaluation import read,artifact,reject,BASE,ROOT
from audit.complete_answer_device_cohort import gpu_ready,validate,REQUIRED
from audit.legal_answer_cohort_proof import generation,equal_events,DOCUMENTS
from claims.process_conditions import write_committed
from model.complete_answer_cohort_scores import authority,sealed
from model.legal_answer_public_pack import PublicPack
from text.constellation_corpus import file_sha,need

PRIOR = ('legal-answer-cohort-scores-proof-v2-20261024/generation/worker',
         '0c94614b0672b89fc52096aa11e1120e9c1ac0482879051ab9c0234b0e214d67')
NEW = (*REQUIRED,'audit/complete_answer_device_proof.py')


def invoke(request,pin,output,log,flag,*,extra=(),error=None):
    command = [sys.executable,'-I',str(BASE/'audit/complete_answer_device_cohort.py'),flag,
               '--request',str(request),'--request-pin',pin,'--output',str(output),*extra]
    tick = time.perf_counter()
    with log.open('xb') as stream:
        child = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=False)
    need(child.returncode == (1 if error else 0), 'DEVICE_PROOF_CHILD_EXIT:'+str(log))
    if error:need(error in log.read_text(), 'DEVICE_PROOF_WRONG_REFUSAL:'+str(log))
    return dict(exit_code=child.returncode,seconds=time.perf_counter()-tick,log=file_sha(log),refusal=error)


def run(a):
    began = tick = time.perf_counter(); a.output.mkdir(exist_ok=False); launches = {}; controls = {}
    prior_root = a.root/PRIOR[0]; prior = read(prior_root/'report.json',PRIOR[1])
    m = read(prior_root.parent/'request.json',prior['request'])
    pack = PublicPack(m['pack']['path'],m['pack']['pin'])
    cases = [c for c in pack.cases if c['document'] in DOCUMENTS]
    need([c['document'] for c in cases] == list(DOCUMENTS), 'DEVICE_PROOF_DOCUMENTS')
    m['ordinals'] = [c['ordinal'] for c in cases]
    m['code'].update({n:file_sha(BASE/n) for n in NEW})
    request = a.output/'request.json'; request_pin = write_committed(request,m)
    phases = dict(prepare=time.perf_counter()-tick); tick = time.perf_counter()
    launches['inspect'] = invoke(request,request_pin,a.output/'inspect',a.output/'inspect.log','--inspect')
    inspection = read(a.output/'inspect/report.json',file_sha(a.output/'inspect/report.json'))
    need(inspection['checkpoint_loads'] == inspection['frozen_target_loads'] == 0
         and inspection['checkpoint_artifact_reads'] == 3 and not inspection['gpu_used']
         and inspection['optimizer_updates'] == 0 and not (a.output/'inspect/run').exists(), 'DEVICE_PROOF_INSPECT')
    phases['inspect'] = time.perf_counter()-tick; tick = time.perf_counter()
    out = a.output/'worker'
    launches['generate'] = invoke(request,request_pin,out,a.output/'worker.log','--cpu-proof')
    current_pin = file_sha(out/'report.json'); current = read(out/'report.json',current_pin)
    need(current['actions'] == dict(generated=4,reused=0,recovered=0,gap=0)
         and current['checkpoint_loads'] == 3 and current['frozen_target_loads'] == 1
         and not current['gpu_used'] and current['peak_cuda_bytes'] == 0, 'DEVICE_PROOF_RESIDENT')
    phases['generate'] = time.perf_counter()-tick; tick = time.perf_counter()
    old = {r['document']:r for r in prior['results']}; events = 0
    for row in current['results']:
        p,r = generation(out,row); q,s = generation(prior_root,old[row['document']])
        events += equal_events(p,r,q,s)
    previous = read(a.previous/'report.json',a.previous_pin) if a.previous else None
    if previous:
        before = read(a.previous/'worker/report.json',previous['worker_pin']); count = 0
        for x,y in zip(current['results'],before['results'],strict=True):
            p,r = generation(out,x); q,s = generation(a.previous/'worker',y); count += equal_events(p,r,q,s)
        need(count == events and previous['events'] == events, 'DEVICE_PROOF_COLD')
    r,b,actual = authority(out,current_pin,pack,'proof'); trajectories = 0
    for row,result in zip(actual,r['results'],strict=True):
        traces,_,_ = sealed(out,result,row,b,pack.load(row['public']),pack.roster); trajectories += len(traces)
    need(trajectories == 96, 'DEVICE_PROOF_SCORE_HANDOFF')
    controls['false_scored_full_scope'] = reject(lambda:authority(out,current_pin,pack,'all'),
                                                'COHORT_SCORE_INCOMPLETE_FULL_SCOPE')
    phases['independent_events_and_score_handoff'] = time.perf_counter()-tick; tick = time.perf_counter()
    launches['reopen'] = invoke(request,request_pin,out,a.output/'reopen.log','--cpu-proof')
    reopened = read(out/'reopen-000000.json',file_sha(out/'reopen-000000.json'))
    need(reopened['actions'] == dict(generated=0,reused=4,recovered=0,gap=0)
         and reopened['previous_report'] == current_pin and file_sha(out/'report.json') == current_pin,
         'DEVICE_PROOF_REOPEN')
    phases['reopen'] = time.perf_counter()-tick; tick = time.perf_counter()
    changed = deepcopy(m); changed['authority']['changed_for_negative'] = True
    changed_path = a.output/'changed-request.json'; changed_pin = write_committed(changed_path,changed)
    launches['changed_binding'] = invoke(changed_path,changed_pin,out,a.output/'changed.log','--cpu-proof',
                                        error='DEVICE_COHORT_BINDING_CHANGED')
    launches['wrong_device'] = invoke(request,request_pin,a.output/'wrong-device',a.output/'wrong-device.log',
        '--inspect',extra=('--inspect-device','cuda:0'),error='DEVICE_COHORT_DEVICE')
    runtime = artifact(m['runtime'])
    bad = deepcopy(m); bad['checkpoints'].pop('own_slot')
    controls['missing_architecture'] = reject(lambda:validate(bad,runtime,'cpu'),'DEVICE_COHORT_ARCHITECTURES')
    bad = deepcopy(m); bad['ordinals'] = None
    controls['false_generation_full_scope'] = reject(lambda:validate(bad,runtime,'cpu'),'DEVICE_COHORT_FULL_SCOPE')
    bad = deepcopy(m); bad['intervention'] = dict(kind='trained_runtime_policy')
    controls['cpu_as_production'] = reject(lambda:validate(bad,runtime,'cpu'),'DEVICE_COHORT_CPU_PROOF_ONLY')
    busy = a.output/'busy'; busy.mkdir()
    with (busy/'worker.lock').open('a+') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        launches['busy'] = invoke(request,request_pin,busy,a.output/'busy.log','--cpu-proof',
                                 error='DEVICE_COHORT_OWNER_BUSY')
    gpu = a.output/'occupied-gpu'; gpu.mkdir()
    controls['occupied_gpu'] = reject(lambda:gpu_ready(gpu),'DEVICE_COHORT_GPU_RESIDENT')
    controls['undeclared_zfs'] = current['undeclared_artifact_refused']
    need(not torch.cuda.is_initialized() and file_sha(out/'report.json') == current_pin, 'DEVICE_PROOF_FINAL')
    pack.close()
    for n,h in m['code'].items():need(file_sha(BASE/n) == h, 'DEVICE_PROOF_CODE_CHANGED')
    phases['controls'] = time.perf_counter()-tick; tick = time.perf_counter()
    files = {str(p.relative_to(a.output)):file_sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()}
    phases['seal'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
    need(.9 <= sum(phases.values())/seconds <= 1.001, 'DEVICE_PROOF_PHASES')
    result = dict(complete=True,mode='cold' if previous else 'publish',previous_pin=a.previous_pin,
        prior=PRIOR[1],worker_pin=current_pin,code=m['code'],files=files,documents=list(DOCUMENTS),
        cases=4,trajectories=trajectories,events=events,all_events_byte_exact=True,
        cold_reproduced=previous is not None,inspect_models=0,resident_models=3,resident_targets=1,reused_cases=4,
        controls=controls,launches=launches,phases_seconds=phases,seconds=seconds,
        cuda_qualified=False,full_runtime_policy_executed=False,admitted_senses=0,architecture_winner=False,
        capability_demonstrated=False)
    pin = write_committed(a.output/'report.json',result)
    print(json.dumps(dict(complete=True,pin=pin,events=events,seconds=seconds)),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--previous',type=Path); p.add_argument('--previous-pin')
    a = p.parse_args(); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise
