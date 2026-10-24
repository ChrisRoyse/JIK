"""Sequential qualification or production continuation, with physical child identities."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
from audit.legal_graph_continuation import code,ident
from claims.process_conditions import write_committed
from kernel.certified_index_graph import pinned
from kernel.native_graph_continuation import read_publication
from text.constellation_corpus import file_sha,need

ORDER=['finbert-mean','pubmedbert-embeddings','codebert-mean']


def run(a):
    repo=Path(__file__).resolve().parents[2]
    lock=repo/'.lease'/('legal-continuation-'+a.mode+'-queue');lock.parent.mkdir(exist_ok=True)
    lease=lock.open('a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    a.output.mkdir(parents=True,exist_ok=False)
    pins=code();self_pin=file_sha(Path(__file__));owner=dict(pid=os.getpid(),starttime=ident(os.getpid()))
    (a.output/'driver.py').write_bytes(Path(__file__).read_bytes())
    write_committed(a.output/'launch.json',dict(**owner,code=pins,driver_pin=self_pin,mode=a.mode,instruments=ORDER))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    def check_code():
        need(code()==pins and file_sha(Path(__file__))==self_pin,'GRAPH_CONTINUATION_QUEUE_CODE_CHANGED')
    def emit(stage,**values):
        row=dict(stage=stage,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**values)
        with (a.output/'events.jsonl').open('a') as f:
            f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(row),flush=True)
    def launch(label,name,mode,target,extra=None,expect=0):
        check_code()
        command=[sys.executable,'-I',str(repo/'scripts/audit/legal_graph_continuation.py'),
                 '--instrument',name,'--mode',mode,'--output',str(target),*(extra or [])]
        with (a.output/(label+'.log')).open('xb') as log:
            child=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
            emit('running',label=label,pid=child.pid,starttime=ident(child.pid),command=command)
            status=child.wait();log.flush();os.fsync(log.fileno())
        emit('terminal',label=label,exit_code=status)
        need(status==expect,'GRAPH_CONTINUATION_CHILD_EXIT');check_code()
        return read_publication(target) if expect==0 else None
    emit('queue_started',**owner)
    if a.mode=='full':
        qualified=pinned(a.qualification/'report.json',a.qualification_pin)
        need(qualified['complete'] and qualified['mode']=='qualify' and qualified['code']==pins
             and qualified['driver_pin']==self_pin and list(qualified['results'])==sorted(ORDER),
             'GRAPH_CONTINUATION_QUEUE_QUALIFICATION')
    else:
        # Hold the actual qualification lease; the real child must refuse before gallery loading.
        gate=(repo/'.lease/legal-graph-continuation-qualification').open('a+')
        fcntl.flock(gate,fcntl.LOCK_EX|fcntl.LOCK_NB)
        blocked=a.output/'negative-busy-lease'
        launch('negative-busy-lease',ORDER[0],'qualify',blocked,expect=1)
        failures=list(blocked.glob('failure-*.json'));need(len(failures)==1,'GRAPH_CONTINUATION_LEASE_RECEIPT')
        failure=json.loads(failures[0].read_bytes())
        need(failure['error']=='ValueError:MNLM_TEXT_CORPUS_GRAPH_CONTINUATION_LEASE_BUSY',
             'GRAPH_CONTINUATION_LEASE_REFUSAL')
        fcntl.flock(gate,fcntl.LOCK_UN);gate.close()
    results={}
    for name in ORDER:
        try:
            if a.mode=='qualify':
                root=a.output/(name+'-qualification')
                q,qp=launch(name+'-qualification',name,'qualify',root)
                old_files={str(f.relative_to(root)):file_sha(f) for f in root.rglob('*') if f.is_file()}
                _,original_pin=launch(name+'-resume',name,'qualify',root,['--resume'])
                need(original_pin==qp,'GRAPH_CONTINUATION_RESUME_RESEALED')
                need(all(file_sha(root/n)==pin for n,pin in old_files.items()),'GRAPH_CONTINUATION_RESUME_CHANGED_BYTES')
                resume_path=root/'runs/000001/report.json';resume=json.loads(resume_path.read_bytes())
                need(resume['complete'] and resume['result_pin']==q['result_pin'] and resume['created_shards']==0
                     and resume['reused_shards']==len(q['new_shards']),'GRAPH_CONTINUATION_RESUME_RECOMPUTED')
                newly=set(str(f.relative_to(root)) for f in root.rglob('*') if f.is_file())-set(old_files)
                need(newly=={'runs/000001/report.json','runs/000001/events.jsonl'},'GRAPH_CONTINUATION_RESUME_EXTRA_WRITES')
                cold=a.output/(name+'-cold')
                c,cp=launch(name+'-cold',name,'cold',cold,['--previous',str(root),'--previous-pin',qp])
                need(c['previous_pin']==qp and c['result_pin']==q['result_pin'] and c['created_shards']==0,
                     'GRAPH_CONTINUATION_COLD_DIFFERENT')
                results[name]=dict(complete=True,qualification=dict(path=str(root),pin=qp),cold=dict(path=str(cold),pin=cp),
                    resume=dict(path=str(resume_path),pin=file_sha(resume_path),original_files_unchanged=True,
                                new_files=sorted(newly)),coverage=q['coverage'],result_pin=q['result_pin'])
            else:
                qr=qualified['results'][name];root=a.graphs/f'legal-native-{name}-continuation-v1-20261023'
                options=['--wait','--qualification',qr['qualification']['path'],'--qualification-pin',qr['qualification']['pin'],
                         '--qualification-cold',qr['cold']['path'],'--qualification-cold-pin',qr['cold']['pin']]
                q,qp=launch(name+'-full',name,'full',root,options)
                cold=a.output/(name+'-cold')
                c,cp=launch(name+'-full-cold',name,'cold',cold,options+['--previous',str(root),'--previous-pin',qp])
                need(q['full_graph_complete'] and c['full_graph_complete'] and c['previous_pin']==qp
                     and c['result_pin']==q['result_pin'] and q['coverage']['queries']==6360378,
                     'GRAPH_CONTINUATION_QUEUE_INCOMPLETE_FULL')
                results[name]=dict(complete=True,full=dict(path=str(root),pin=qp),cold=dict(path=str(cold),pin=cp),
                                   coverage=q['coverage'],result_pin=q['result_pin'])
        except Exception as error:
            results[name]=dict(complete=False,error=type(error).__name__+':'+str(error))
        emit('lane_terminal',instrument=name,result=results[name])
    report=dict(complete=all(r['complete'] for r in results.values()),mode=a.mode,code=pins,driver_pin=self_pin,
        results=results,owner=owner,scientific_admission=False,model_updates=0,acceptance='author-checked',number_status='measured')
    pin=write_committed(a.output/'report.json',report);emit('queue_terminal',pin=pin,complete=report['complete'])
    return 0 if report['complete'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['qualify','full'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--graphs',type=Path,default=Path('/zfs/hot/mnlm/associations'))
    p.add_argument('--qualification',type=Path);p.add_argument('--qualification-pin')
    raise SystemExit(run(p.parse_args()))
