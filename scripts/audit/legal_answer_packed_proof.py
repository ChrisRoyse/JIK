"""Real checkpoint generations from packed public inputs versus retained traces."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
from audit.complete_answer_evaluation import read,BASE,ROOT
from audit.legal_answer_cohort_proof import generation,equal_events
from audit.legal_answer_public_pack import PRIOR
from claims.process_conditions import write_committed
from text.constellation_corpus import file_sha,need

NEW = ('model/legal_answer_public_pack.py','audit/legal_answer_public_pack.py',
       'audit/complete_answer_packed_cohort.py','audit/legal_answer_packed_proof.py')


def run(a):
    began = tick = time.perf_counter(); a.output.mkdir(exist_ok=False)
    producer = read(a.producer/'report.json',a.producer_pin)
    cold = read(a.cold/'report.json',a.cold_pin)
    need(producer['complete'] and cold['complete'] and cold['previous_pin'] == a.producer_pin
         and cold['pack'] == producer['pack'] and cold['consumed']['chain'] == producer['consumed']['chain']
         and cold['exact'] == producer['exact'], 'PACKED_PROOF_AUTHORITY')
    prior = read(a.root/PRIOR[0]/'report.json',PRIOR[1])
    m = read(a.root/PRIOR[0]/'request.json',prior['files']['request.json'])
    old_cases = m.pop('cases'); m['ordinals'] = [c['ordinal'] for c in old_cases]
    m['schema'] = 'public-pack-answer-cohort-v1'; m['pack'] = producer['pack']
    m['authority'] = dict(m['authority'],public_pack=m['pack']['pin'])
    m['code'].update({n:file_sha(BASE/n) for n in NEW})
    request = a.output/'request.json'; request_pin = write_committed(request,m)
    phases = dict(prepare=time.perf_counter()-tick); tick = time.perf_counter()
    with (a.output/'worker.log').open('xb') as stream:
        child = subprocess.run([sys.executable,'-I',str(BASE/'audit/complete_answer_packed_cohort.py'),
            '--cpu-proof','--request',str(request),'--request-pin',request_pin,
            '--output',str(a.output/'worker')],stdout=stream,stderr=subprocess.STDOUT,check=False)
    need(child.returncode == 0,'PACKED_PROOF_WORKER_EXIT:'+str(child.returncode))
    current_pin = file_sha(a.output/'worker/report.json'); current = read(a.output/'worker/report.json',current_pin)
    need(current['actions'] == dict(generated=4,reused=0,recovered=0,gap=0)
         and current['checkpoint_loads'] == 3 and current['frozen_target_loads'] == 1,
         'PACKED_PROOF_RESIDENT_LOADS')
    phases['worker'] = time.perf_counter()-tick; tick = time.perf_counter()
    old = read(a.root/PRIOR[0]/'resident/initial.json',prior['files']['resident/initial.json'])
    events = 0
    for x,y in zip(current['results'],old['results'],strict=True):
        need(x['ordinal'] == y['ordinal'] and x['document'] == y['document'],'PACKED_PROOF_CASE_ORDER')
        p,r = generation(a.output/'worker',x); q,s = generation(a.root/PRIOR[0]/'resident',y)
        events += equal_events(p,r,q,s)
    need(events == 1704,'PACKED_PROOF_EVENT_COVERAGE')
    previous = read(a.previous/'report.json',a.previous_pin) if a.previous else None
    if previous:
        before = read(a.previous/'worker/report.json',previous['worker_pin']); count = 0
        for x,y in zip(current['results'],before['results'],strict=True):
            p,r = generation(a.output/'worker',x); q,s = generation(a.previous/'worker',y)
            count += equal_events(p,r,q,s)
        need(count == events,'PACKED_PROOF_COLD_EVENTS')
    phases['independent_events'] = time.perf_counter()-tick; tick = time.perf_counter()
    for n,h in m['code'].items():need(file_sha(BASE/n) == h,'PACKED_PROOF_FINAL_CODE')
    files = {str(p.relative_to(a.output)):file_sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()}
    phases['seal'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
    need(.9 <= sum(phases.values())/seconds <= 1.001,'PACKED_PROOF_PHASES')
    pin = write_committed(a.output/'report.json',dict(complete=True,mode='cold' if previous else 'publish',
        previous_pin=a.previous_pin,producer=a.producer_pin,cold=a.cold_pin,pack=m['pack'],
        cases=4,trajectories=96,events=events,all_events_byte_exact=True,worker_pin=current_pin,
        files=files,code=m['code'],phases_seconds=phases,seconds=seconds,
        admitted_senses=0,architecture_winner=False,capability_demonstrated=False))
    print(json.dumps(dict(complete=True,pin=pin,events=events,seconds=seconds)),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT); p.add_argument('--producer',type=Path,required=True)
    p.add_argument('--producer-pin',required=True);p.add_argument('--cold',type=Path,required=True)
    p.add_argument('--cold-pin',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--previous',type=Path);p.add_argument('--previous-pin')
    a = p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise
