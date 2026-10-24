"""Run matched old/new reader proofs before the full placement comparison."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time


def need(value, code):
    if not value: raise ValueError(code)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def read(path, pin):
    need(sha(path)==pin,'GROUPED_QUEUE_PIN')
    return json.loads(Path(path).read_bytes())


def identity(pid):
    try: s=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError: return None
    return None if s[0]=='Z' else s[19]


def stamp(path):
    s=Path(path).stat()
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value,f,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())


def authenticate(path, pin):
    repo=Path(__file__).resolve().parents[2]; contract=read(path,pin)
    need(contract['complete'] and not contract['scientific_admission'],'GROUPED_QUEUE_AUTHORITY')
    records={}; authenticated={Path(path):pin}
    for name,ref in contract['proofs'].items():
        p=Path(ref['path']); records[name]=read(p,ref['pin']); authenticated[p]=ref['pin']
        for n,h in records[name].get('files',{}).items(): authenticated[p.parent/n]=h
    q,c,r,s=(records[n] for n in ['cpu','cold','readback','cold_readback'])
    n,v,m,plan=(records[k] for k in ['grouped_cpu','grouped_readback','cpu_match','source_plan'])
    need(all(x['complete'] for x in records.values()),'GROUPED_QUEUE_INCOMPLETE_PROOF')
    need(q['binding']==c['binding'] and c['previous_pin']==contract['proofs']['cpu']['pin']
         and q['results_pin']==c['results_pin'],'GROUPED_QUEUE_BASELINE_COLD')
    need(q['binding']['scope']=='small' and q['binding']['device']=='cpu'
         and n['binding']['scope']=='small' and n['binding']['device']=='cpu','GROUPED_QUEUE_CPU_SCOPE')
    for record,source in [(r,'cpu'),(s,'cold'),(v,'grouped_cpu')]:
        need(record['input_pin']==contract['proofs'][source]['pin'] and record['rows']==3200
             and record['checkpoints']==18,'GROUPED_QUEUE_CPU_READBACK')
    need(r['totals']==s['totals']==v['totals'],'GROUPED_QUEUE_CPU_RECOUNT')
    need(m['baseline_pin']==contract['proofs']['cpu']['pin']
         and m['candidate_pin']==contract['proofs']['grouped_cpu']['pin']
         and m['candidate_code']==n['binding']['code'] and m['device']=='cpu' and m['scope']=='small'
         and m['all_updates_exact'] and m['all_checkpoint_tensors_exact'] and m['all_evaluation_payloads_exact']
         and (m['updates'],m['training_rows'],m['checkpoints'])==(15,198,18),'GROUPED_QUEUE_CPU_MATCH')
    need(plan['code']==q['binding']['code'] and plan['results']['full']['train_documents']==91540
         and plan['results']['full']['expected_training_rows']==6455636
         and plan['results']['full']['eval_documents']==192
         and plan['results']['qualify']['actual_training_rows']==4104,'GROUPED_QUEUE_SOURCE_PLAN')
    need(all(n['binding']['code'].get(k)==p for k,p in q['binding']['code'].items()),'GROUPED_QUEUE_PRIOR_CODE')
    # Reader authority is checked again by the actual non-small consumer.
    sys.path.insert(0,str(repo/'scripts'))
    from audit.legal_grouped_io_authority import PROOFS, validate
    io={key:records['io_'+key] for key in PROOFS}; validate(io)
    need(all(contract['proofs']['io_'+k]['pin']==h for k,(_,h) in PROOFS.items()),'GROUPED_QUEUE_IO_PINS')
    code=dict(n['binding']['code']); code['audit/legal_grouped_placement_queue.py']=sha(__file__)
    need(code==contract['code'],'GROUPED_QUEUE_CODE_MANIFEST')
    for p,h in authenticated.items(): need(sha(p)==h,'GROUPED_QUEUE_PHYSICAL_PROOF')
    stamps={p:stamp(p) for p in authenticated}
    def fresh():
        for name,h in code.items(): need(sha(repo/'scripts'/name)==h,'GROUPED_QUEUE_CODE_CHANGED')
        for p,saved in stamps.items(): need(stamp(p)==saved,'GROUPED_QUEUE_PROOF_CHANGED')
    fresh()
    return contract, fresh


def run(a):
    repo=Path(__file__).resolve().parents[2]
    a.output.mkdir(parents=True,exist_ok=False)
    results=[]; lease=None
    def event(stage, **values):
        # A closed caller stdout must never kill a durable queue.
        with (a.output/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage=stage,timestamp=time.time(),**values),sort_keys=True)+'\n')
            f.flush(); os.fsync(f.fileno())
    try:
        contract,fresh=authenticate(a.authority,a.authority_pin)
        owner=contract['wait']; root=Path(owner['path']); pid=owner['pid']; start=owner['starttime']
        launch=json.loads((root/'launch.json').read_bytes())
        need(launch['pid']==pid and str(launch['starttime'])==start,'GROUPED_QUEUE_WAIT_OWNER')
        (a.output/'driver.py').write_bytes(Path(__file__).read_bytes())
        if a.mode=='probe':
            write(a.output/'report.json',dict(complete=True,authority=a.authority_pin,code=contract['code'],
                wait_identity=identity(pid),gpu_children_started=0,acceptance='author-checked',scientific_admission=False))
            event('probe_terminal',complete=True); return
        need(a.handoff and a.handoff_pin,'GROUPED_QUEUE_HANDOFF_REQUIRED')
        h=read(a.handoff,a.handoff_pin)
        need(h['complete'] and h['status']=='superseded_waiting_owner' and h['authority']==a.authority_pin
             and identity(h['old_pid'])!=h['old_starttime'] and h['children_before']==[]
             and h['original_queue_complete'] is False,'GROUPED_QUEUE_HANDOFF')
        lease=(repo/'.lease/legal-placement-comparison-queue').open('a+')
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(a.output/'launch.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),authority=a.authority_pin,
            handoff=a.handoff_pin,driver_pin=sha(__file__),wait_pid=pid,wait_start=start,wait_root=str(root)))
        if identity(pid)==start:
            fd=os.pidfd_open(pid)
            try:
                need(identity(pid) in (start,None),'GROUPED_QUEUE_PID_REUSED')
                event('waiting_for_runtime_owner',pid=pid,starttime=start,wait='kernel_pidfd')
                select.select([fd],[],[])
            finally: os.close(fd)
        prior=json.loads((root/'report.json').read_bytes())
        need(prior['complete'],'GROUPED_QUEUE_RUNTIME_FAILED')
        runtime={x['label']:x for x in prior['outputs']}
        need({'gpu','gpu-cold'}.issubset(runtime),'GROUPED_QUEUE_RUNTIME_MISSING')
        fresh(); event('runtime_owner_terminal',report_pin=sha(root/'report.json'))
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        need(env.get('CUDA_VISIBLE_DEVICES')!='','GROUPED_QUEUE_GPU_HIDDEN')
        def execute(label, script, arguments):
            fresh(); command=[sys.executable,'-I',str(repo/'scripts/audit'/script),*arguments,'--output',str(a.output/label)]
            with (a.output/(label+'.log')).open('xb') as log:
                child=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                event('running',label=label,pid=child.pid,starttime=identity(child.pid),command=command)
                status=child.wait(); log.flush(); os.fsync(log.fileno())
            event('terminal',label=label,exit_code=status); need(status==0,'GROUPED_QUEUE_CHILD_FAILED')
            path=a.output/label; report=json.loads((path/'report.json').read_bytes())
            need(report['complete'],'GROUPED_QUEUE_CHILD_INCOMPLETE')
            result=dict(label=label,path=str(path),pin=sha(path/'report.json')); results.append(result)
            fresh(); return result
        common=[]
        for argument,label in [('runtime','gpu'),('runtime-cold','gpu-cold')]:
            common+=['--'+argument,runtime[label]['path'],'--'+argument+'-pin',runtime[label]['pin']]
        def train(label, scope, grouped, extra=()):
            script='legal_grouped_placement_training.py' if grouped else 'legal_placement_comparison_training.py'
            return execute(label,script,['--scope',scope,*common,*extra])
        def recount(label, ref, grouped):
            script='legal_grouped_placement_readback.py' if grouped else 'legal_placement_comparison_readback.py'
            return execute(label,script,['--input',ref['path'],'--input-pin',ref['pin']])
        original=train('baseline','qualify',False); recount('baseline-readback',original,False)
        first=train('qualification','qualify',True)
        second=train('qualification-cold','qualify',True,['--previous',first['path'],'--previous-pin',first['pin']])
        recount('qualification-readback',first,True); recount('qualification-cold-readback',second,True)
        match=execute('reader-match','legal_grouped_placement_match.py',[
            '--baseline',original['path'],'--baseline-pin',original['pin'],
            '--candidate',first['path'],'--candidate-pin',first['pin']])
        full=train('full','full',True,['--qualification',first['path'],'--qualification-pin',first['pin'],
            '--qualification-cold',second['path'],'--qualification-cold-pin',second['pin'],
            '--reader-match',match['path'],'--reader-match-pin',match['pin']])
        recount('full-readback',full,True)
        write(a.output/'report.json',dict(complete=True,outputs=results,driver_pin=sha(__file__),
            authority=a.authority_pin,acceptance='author-checked',scientific_admission=False,capability_demonstrated=False))
        event('queue_terminal',complete=True,pin=sha(a.output/'report.json'))
    except BaseException as error:
        write(a.output/'failure.json',dict(error=repr(error),outputs=results))
        event('queue_terminal',complete=False); raise
    finally:
        if lease: lease.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['probe','full'],required=True)
    for n in ['authority','output']: p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--authority-pin',required=True)
    p.add_argument('--handoff',type=Path); p.add_argument('--handoff-pin')
    run(p.parse_args())
