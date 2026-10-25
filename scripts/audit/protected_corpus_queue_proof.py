"""Real-source metadata and live-process boundary proof; never launches GPU work."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import protected_corpus_queue_contract as contract
from audit.surface_device_queue import identity,need,sha


def refused(fn,code):
    try:fn()
    except (ValueError,RuntimeError) as e:
        need(code in str(e),'PROTECTED_QUEUE_WRONG_REFUSAL:'+str(e));return str(e)
    raise ValueError('PROTECTED_QUEUE_NEGATIVE_ACCEPTED:'+code)


def run(a):
    began=time.perf_counter();a.output.mkdir(exist_ok=False);cfg,reports,fresh=contract.authenticate(a.authority,a.pin)
    owner=cfg['wait'];need(contract.process_current(owner)==owner['starttime'],'PROTECTED_QUEUE_PROOF_NEEDS_LIVE_OWNER')
    fd=os.pidfd_open(owner['pid'])
    try:
        need(contract.process_current(owner)==owner['starttime'] and not select.select([fd],[],[],0)[0],
             'PROTECTED_QUEUE_PROOF_PIDFD')
    finally:os.close(fd)
    launch=contract.read(contract.WAIT_ROOT/'launch.json',owner['launch_pin'])
    command=(Path('/proc')/str(owner['pid'])/'cmdline').read_bytes().decode().split('\0')
    need(command.count(str(contract.REPO/'scripts/audit/legal_answer_successor.py'))==1
         and '--output' in command and Path(command[command.index('--output')+1])==contract.WAIT_ROOT,
         'PROTECTED_QUEUE_PROOF_COMMAND')
    done=json.loads((contract.WAIT_ROOT/'completed-prior-readback.json').read_bytes())
    prior=contract.stage(done,'prior-readback',contract.WAIT_ROOT)
    controls={}
    controls['live_predecessor']=refused(lambda:contract.predecessor(cfg),'PROTECTED_QUEUE_LIVE_PREDECESSOR')
    bad=dict(owner,starttime=str(int(owner['starttime'])+1))
    controls['recycled_identity']=refused(lambda:contract.process_current(bad),'PROTECTED_QUEUE_PID_REUSED')
    bad=deepcopy(cfg);bad['execution']['automatic_retry']=True
    path=a.output/'negative-authority.json';pin=contract.persist(path,bad)
    controls['changed_authority']=refused(lambda:contract.authenticate(path,pin),'PROTECTED_QUEUE_AUTHORITY_BINDING')
    controls['wrong_authority_pin']=refused(lambda:contract.authenticate(a.authority,'0'*64),'PROTECTED_QUEUE_PIN')
    for label,change,code in [
        ('runtime',lambda r:r['cold']['runtime'].update(threads=2),'PROTECTED_QUEUE_PROOF_RUNTIME'),
        ('cohort',lambda r:r['selection']['selected']['query'].update(positions=114),'PROTECTED_QUEUE_DOCUMENT_COHORT'),
        ('cold',lambda r:r['cold'].update(midpoint_resume_exact=False),'PROTECTED_QUEUE_COLD_PROOF')]:
        bad=deepcopy(reports);change(bad);controls[label]=refused(lambda:contract.proof_scope(bad),code)
    bad=deepcopy(cfg);bad['code']['audit/protected_corpus_queue.py']='0'*64
    path=a.output/'negative-code.json';pin=contract.persist(path,bad)
    controls['changed_code']=refused(lambda:contract.authenticate(path,pin),'PROTECTED_QUEUE_AUTHORITY_BINDING')
    controls['partial_is_not_complete']=refused(lambda:contract.completion(prior,cfg),'PROTECTED_QUEUE_PREDECESSOR_INCOMPLETE')
    controls['cpu_not_gpu']=refused(lambda:contract.child_scope('gpu-proof',reports['cpu'],cfg,{}),'PROTECTED_QUEUE_CHILD_SCOPE')
    bad=deepcopy(done);bad['pin']='0'*64
    controls['changed_stage_pin']=refused(lambda:contract.stage(bad,'prior-readback',contract.WAIT_ROOT),'PROTECTED_QUEUE_PIN')
    controls['missing_stage']=refused(lambda:contract.stage({},'placement',contract.WAIT_ROOT),'PROTECTED_QUEUE_STAGE_PATH')
    root=a.output/'negative-stage';(root/'prior-readback').mkdir(parents=True);bad=deepcopy(prior);bad['complete']=False
    path=root/'prior-readback/report.json';pin=contract.persist(path,bad)
    controls['failed_stage']=refused(lambda:contract.stage(dict(done,path=str(path.parent),pin=pin),'prior-readback',root),
                                     'PROTECTED_QUEUE_STAGE_FAILED')
    with contract.lock():controls['second_owner']=refused(contract.lock,'PROTECTED_QUEUE_BUSY')
    child=None
    try:
        with contract.lock() as held:
            command=[sys.executable,'-I','-c',
                     'import os,sys; os.fstat(int(sys.argv[1])); print("ready",flush=True); sys.stdin.buffer.readline()',
                     str(held.fileno())]
            child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                   pass_fds=(held.fileno(),))
            need(child.stdout.readline()==b'ready\n','PROTECTED_QUEUE_LOCK_CHILD_READY')
            child_identity=identity(child.pid)
            need(child_identity is not None and os.readlink(f'/proc/{child.pid}/fd/{held.fileno()}')==str(contract.LOCK),
                 'PROTECTED_QUEUE_LOCK_CHILD_DESCRIPTOR')
        # Parent descriptor is now closed; the actual child alone must retain the lock.
        controls['inherited_owner']=refused(contract.lock,'PROTECTED_QUEUE_BUSY')
        lock_child=dict(pid=child.pid,starttime=child_identity,inherited_descriptor_checked=True,parent_closed=True)
    finally:
        if child is not None:
            child.stdin.write(b'release\n');child.stdin.flush();child.stdin.close()
            status=child.wait();child.stdout.close();child.stderr.close()
            need(status==0,'PROTECTED_QUEUE_LOCK_CHILD_EXIT')
    with contract.lock():controls['released_after_child_exit']=True
    outputs={label:dict(path=str(a.output/'planned'/label),pin='0'*64) for label in contract.STAGES}
    schedule={label:dict(script=script,args=args) for label in contract.STAGES
              for script,args in [contract.arguments(label,cfg,outputs)]}
    for label,entry in schedule.items():
        need('--cpu-proof' not in entry['args'] and entry['script'] in ('protected_corpus_program.py','protected_corpus_readback.py'),
             'PROTECTED_QUEUE_GPU_ARGUMENTS')
    need(schedule['full']['args'][-4:]==['--qualification',outputs['gpu-cold']['path'],'--qualification-pin','0'*64],
         'PROTECTED_QUEUE_FULL_QUALIFICATION_ARGUMENT')
    # Placeholder hashes above inspect argument wiring only. No command is run or accepted as evidence.
    predecessor_command=(Path('/proc')/str(owner['pid'])/'cmdline').read_bytes().decode().split('\0')
    observation=dict(wait=owner,launch=launch,command=predecessor_command,completed_prior=done,
                     proofs=cfg['proofs'],plan_train=reports['plan']['train'],schedule=schedule)
    # Canonical schedule destinations must be independent of this proof publication.
    observation['schedule']={k:dict(script=v['script'],args=[x.replace(str(a.output/'planned'),'QUEUE_OUTPUT')
                              for x in v['args']]) for k,v in schedule.items()}
    cold=False
    if a.previous:
        previous=contract.read(a.previous/'report.json',a.previous_pin)
        need(previous['complete'] and previous['authority']==a.pin and previous['code']==cfg['code']
             and previous['observation']==observation and previous['controls']==controls,'PROTECTED_QUEUE_PROOF_COLD')
        cold=True
    fresh();need('torch' not in sys.modules,'PROTECTED_QUEUE_PROOF_MODEL_IMPORT')
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True)
    contract.persist(a.output/'report.json',dict(complete=True,authority=a.pin,code=cfg['code'],observation=observation,
        controls=controls,previous=a.previous_pin,cold_readback_exact=cold,gpu_children_started=0,model_imported=False,
        live_predecessor_refused=True,three_boundaries_checked=True,actual_predecessor_complete=False,
        lock_child=lock_child,gpu_residents=gpu,seconds=time.perf_counter()-began,acceptance='author-checked'))
    print(json.dumps(dict(pin=sha(a.output/'report.json'),controls=len(controls),cold_readback_exact=cold)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('authority','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--pin',required=True);p.add_argument('--previous',type=Path);p.add_argument('--previous-pin')
    a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.is_dir():contract.persist(a.output/'failure.json',dict(error=type(e).__name__,message=str(e)))
        raise
