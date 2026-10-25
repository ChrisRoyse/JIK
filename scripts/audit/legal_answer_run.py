"""Actual source-fed epoch publication, crash recovery, and physical cold continuation."""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit.legal_complete_answer_consumer import prepare
from claims.process_conditions import write_committed
from model.legal_answer_epoch import AnswerEpoch, persist
from model.legal_answer_run import AnswerRun, read, encoded
from model.native_population_training import fingerprint
from model.task_answer_training import extension
from text.constellation_corpus import file_sha, need

ROOT=Path('/zfs/archive/mnlm/evidence/fsv/29')
BASE=Path(__file__).resolve().parents[1]
PARENT=('open-window-answer-cpu-v2-20261024','a62c0402a911f7e7c7cafaf3f88e760c1fd6c332159931e10923028a3d3ae339')
REFERENCE=('legal-answer-epoch-proof-v3-20261024','50bed7838cb795e65c937550595789944aec3d087b81a1f8bd933450611637cf')
DOCUMENTS=(15639,27195,32791,105748,106288)
STAGES=('rows_durable','checkpoint_durable','receipt_durable','commit_linked','directory_synced')
ARCHITECTURES=('shared','own_slot','whole_proposals')
NEW=('model/legal_answer_run.py','audit/legal_answer_run.py')


def pinned(path,pin):
    need(file_sha(path)==pin,'ANSWER_RUN_PROOF_PIN');return json.loads(path.read_bytes())


def numeric(value):
    return dict(records=value['record']['records'],**{n:value[n] for n in ('gradients','early','model','optimizer')})


def run(a):
    began=tick=time.perf_counter();a.output.mkdir(exist_ok=False);phases={};corpus=None
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    need(not torch.cuda.is_initialized(),'ANSWER_RUN_CPU_PROOF')
    try:
        parent=pinned(a.root/PARENT[0]/'report.json',PARENT[1]);ref=pinned(a.root/REFERENCE[0]/'report.json',REFERENCE[1])
        physical={a.root/PARENT[0]/'report.json':PARENT[1],a.root/REFERENCE[0]/'report.json':REFERENCE[1]}
        need(ref['complete'] and ref['optimizer_updates']==15 and not ref['gpu_used'],'ANSWER_RUN_REFERENCE_SCOPE')
        physical.update({a.root/REFERENCE[0]/n:h for n,h in ref['files'].items()})
        initial={};expected={}
        for arch in ARCHITECTURES:
            name=arch+'/update-5.pt';path=a.root/PARENT[0]/name;physical[path]=parent['files'][name]
            need(file_sha(path)==physical[path],'ANSWER_RUN_INITIAL_BYTES')
            initial[arch]=torch.load(path,map_location='cpu',weights_only=True)
            expected[arch]={i:torch.load(a.root/REFERENCE[0]/arch/f'update-{i+6}.pt',map_location='cpu',weights_only=True) for i in range(5)}
        for path,h in physical.items():need(file_sha(path)==h,'ANSWER_RUN_REFERENCE_BYTES')
        corpus,labels,continuation,bundle,source_code=prepare(a.root)
        code=dict(ref['code'],**{n:file_sha(BASE/n) for n in NEW})
        need(all(code.get(n)==h for n,h in source_code.items())
             and all(file_sha(BASE/n)==h for n,h in code.items()),'ANSWER_RUN_PROOF_CODE')
        target,target_pin=extension(bundle);policy=dict(bundle['policy'],device='cpu')
        def epoch(arch):return AnswerEpoch(corpus,bundle,target,target_pin,arch,initial[arch],5,policy,code,documents=DOCUMENTS)
        def compare(arch,index,value):
            need(fingerprint(numeric(value))==fingerprint(numeric(expected[arch][index])),'ANSWER_RUN_NUMERICAL_REFERENCE')
            need(value['record']['storage']['remaining_saved_storage_bytes']==0,'ANSWER_RUN_RETAINED_GRAPH')
        source=dict(parent=PARENT,reference=REFERENCE,code=code,binding=corpus.binding,documents=DOCUMENTS,policy=policy)
        write_committed(a.output/'source.json',source)
        if a.crash_stage:
            def crash(at):
                if at==a.crash_stage:os._exit(9)
            with AnswerRun(a.output/'run',epoch('shared'),checkpoint_documents=2,fault=crash) as journal:
                while journal.epoch.cursor<5:journal.step()
            raise ValueError('ANSWER_RUN_CRASH_NOT_REACHED')
        previous=pinned(a.previous/'report.json',a.previous_pin) if a.previous else None
        if previous:need(previous['complete'] and previous['code']==code,'ANSWER_RUN_COLD_CODE')
        phases['authentication']=time.perf_counter()-tick;results={};numerical={};accepted=0;decisions=0;crashes={}
        for arch in ARCHITECTURES:
            tick=time.perf_counter();out=a.output/arch;out.mkdir();book=out/'run';values=[];boundaries=[]
            with AnswerRun(book,epoch(arch),checkpoint_documents=2,fault=boundaries.append) as journal:
                for i in range(2):
                    value,row=journal.step();compare(arch,i,value);values.append(fingerprint(numeric(value)))
                    persist(out/f'update-{i+6}.pt',value);accepted+=1;decisions+=value['record']['decisions']
                checkpoint=fingerprint(journal.epoch.snapshot())
                try:duplicate=AnswerRun(book,epoch(arch),checkpoint_documents=2)
                except BlockingIOError:duplicate_refused=True
                else:duplicate.close();raise ValueError('ANSWER_RUN_DUPLICATE_WRITER_ACCEPTED')
            with AnswerRun(book,epoch(arch),checkpoint_documents=2,fault=boundaries.append) as journal:
                need(journal.epoch.cursor==2 and fingerprint(journal.epoch.snapshot())==checkpoint,'ANSWER_RUN_RESUMED_MIDPOINT')
                for i in range(2,5):
                    value,row=journal.step();compare(arch,i,value);values.append(fingerprint(numeric(value)))
                    persist(out/f'update-{i+6}.pt',value);accepted+=1;decisions+=value['record']['decisions']
                result=journal.finish();state=fingerprint(journal.epoch.snapshot())
                persist(out/'final-state.pt',journal.epoch.snapshot())
            with AnswerRun(book,epoch(arch),checkpoint_documents=2) as journal:
                need(fingerprint(journal.epoch.snapshot())==state and journal.finish()['counts']['decisions']==66,
                     'ANSWER_RUN_COMPLETED_REOPEN')
            need(set(boundaries)==set(STAGES),'ANSWER_RUN_BOUNDARY_COVERAGE')
            results[arch]=dict(result=result,state=state,updates=values,duplicate_writer_refused=duplicate_refused,boundaries=boundaries)
            numerical[arch]=dict(state=state,updates=values,chain=result['chain'],counts=result['counts'])
            phases[arch]=time.perf_counter()-tick
            print(json.dumps(dict(architecture=arch,completed=5,decisions=66,seconds=phases[arch])),flush=True)
            del value,journal;gc.collect()
        tick=time.perf_counter()
        # Fresh interpreters avoid inheriting PyTorch's background threads across fork.
        # Reauthentication is paid once per forced-crash process, never per source atom.
        for stage in STAGES:
            out=a.output/('crash-'+stage);out.mkdir();book=out/'child/run'
            need(not torch.cuda.is_initialized(),'ANSWER_RUN_CRASH_WITH_CUDA')
            cmd=[sys.executable,'-I',str(Path(__file__).resolve()),'--root',str(a.root),
                 '--output',str(out/'child'),'--crash-stage',stage]
            with (out/'child.log').open('xb') as log:
                child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                write_committed(out/'launch.json',dict(pid=child.pid,mode='fresh_interpreter',stage=stage,command=cmd))
                status=child.wait();log.flush();os.fsync(log.fileno())
            need(status==9,'ANSWER_RUN_CRASH_NOT_REACHED')
            before={str(p.relative_to(book)):file_sha(p) for p in book.rglob('*') if p.is_file()}
            child_rows=[json.loads(raw) for n in before if n.endswith('/rows.jsonl') for raw in (book/n).read_bytes().splitlines()]
            with AnswerRun(book,epoch('shared'),checkpoint_documents=2) as journal:
                recovered=journal.epoch.cursor
                need(recovered==(2 if stage in ('commit_linked','directory_synced') else 0),'ANSWER_RUN_CRASH_CURSOR')
                if recovered:
                    need(fingerprint(journal.epoch.model.state_dict())==fingerprint(expected['shared'][recovered-1]['model'])
                         and fingerprint(journal.epoch.optimizer.state_dict())==fingerprint(expected['shared'][recovered-1]['optimizer']),
                         'ANSWER_RUN_CRASH_COMMITTED_STATE')
                replayed=0;replayed_decisions=0
                for i in range(recovered,5):
                    value,row=journal.step();compare('shared',i,value);accepted+=1;decisions+=value['record']['decisions']
                    replayed+=1;replayed_decisions+=value['record']['decisions']
                result=journal.finish();state=fingerprint(journal.epoch.snapshot())
                need(state==numerical['shared']['state'],'ANSWER_RUN_CRASH_FINAL_STATE')
                persist(out/'final-state.pt',journal.epoch.snapshot())
                retained=journal.uncommitted
            need(all((book/n).is_file() and file_sha(book/n)==h for n,h in before.items()),'ANSWER_RUN_INTERRUPTED_BYTES_CHANGED')
            crashes[stage]=dict(exit_code=9,recovered_cursor=recovered,replayed_updates=replayed,
                replayed_decisions=replayed_decisions,state=state,retained_attempts=retained,before=before)
            crashes[stage].update(child_updates=len(child_rows),child_decisions=sum(r['entry']['update']['decisions'] for r in child_rows))
            write_committed(out/'recovery.json',crashes[stage]);del journal,value;gc.collect()
        phases['crash_recovery']=time.perf_counter()-tick;tick=time.perf_counter();controls={}
        def refuse(name,fn,error):
            try:obj=fn()
            except (ValueError,RuntimeError) as exc:
                need(error in str(exc),'ANSWER_RUN_WRONG_REFUSAL:'+str(exc));controls[name]=str(exc)
            else:obj.close();raise ValueError('ANSWER_RUN_NEGATIVE_ACCEPTED:'+name)
        original=a.output/'shared/run'
        refuse('binding',lambda:AnswerRun(original,epoch('shared'),checkpoint_documents=3),'ANSWER_RUN_BINDING')
        for name in ('checkpoint','rows','gap','moments','chain'):
            book=a.output/('wrong-'+name);shutil.copytree(original,book)
            final=book/'commit-000003.json';manifest=read(final)
            if name in ('checkpoint','rows'):
                key='checkpoint' if name=='checkpoint' else 'rows';path=book/manifest[key]['path']
                with path.open('r+b') as f:raw=f.read(1);f.seek(0);f.write(bytes([raw[0]^1]));f.flush();os.fsync(f.fileno())
                error='ANSWER_EPOCH_CHECKPOINT_PIN' if name=='checkpoint' else 'ANSWER_RUN_ROWS_PIN'
            elif name=='gap':
                (book/'commit-000002.json').rename(book/'retained-missing-commit.json');error='ANSWER_RUN_COMMIT_GAP'
            elif name=='moments':
                path=book/manifest['checkpoint']['path'];bad=torch.load(path,map_location='cpu',weights_only=True)
                bad['optimizer']['state'].pop(next(iter(bad['optimizer']['state'])))
                with path.open('wb') as f:torch.save(bad,f);f.flush();os.fsync(f.fileno())
                manifest['checkpoint'].update(pin=file_sha(path),state=fingerprint(bad),bytes=path.stat().st_size)
                final.write_bytes(encoded(manifest));error='PLACEMENT_OPTIMIZER_INCOMPLETE_STATE_RESTORE_FULL_CHECKPOINT'
            else:manifest['chain']='0'*64;final.write_bytes(encoded(manifest));error='ANSWER_RUN_BLOCK_CHAIN'
            refuse(name,lambda:AnswerRun(book,epoch('shared'),checkpoint_documents=2),error)
        phases['corruption_controls']=time.perf_counter()-tick;tick=time.perf_counter()
        if previous:
            need(numerical==previous['numerical'] and controls==previous['controls'],'ANSWER_RUN_COLD_NUMERICAL')
            for arch in ARCHITECTURES:
                path=a.previous/arch/'final-state.pt';need(file_sha(path)==previous['files'][str(path.relative_to(a.previous))],'ANSWER_RUN_COLD_BYTES')
                need(fingerprint(torch.load(path,map_location='cpu',weights_only=True))==numerical[arch]['state'],'ANSWER_RUN_COLD_STATE')
            need(all(crashes[s]['state']==previous['crashes'][s]['state'] and crashes[s]['recovered_cursor']==previous['crashes'][s]['recovered_cursor']
                     for s in STAGES),'ANSWER_RUN_COLD_CRASH')
        corpus.close();corpus.cache.close();corpus.reader.close();labels.fresh();continuation.fresh()
        need(all(file_sha(BASE/n)==h for n,h in code.items()),'ANSWER_RUN_FINAL_CODE')
        need(fingerprint(target.state_dict())==bundle['target_fingerprint'] and not torch.cuda.is_initialized(),'ANSWER_RUN_FROZEN_OR_DEVICE')
        for path,h in physical.items():need(file_sha(path)==h,'ANSWER_RUN_FINAL_AUTHORITY')
        phases['final_readback']=time.perf_counter()-tick;seconds=time.perf_counter()-began
        need(.9<=sum(phases.values())/seconds<=1.001,'ANSWER_RUN_PHASE_ATTRIBUTION')
        files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
        report=dict(complete=True,code=code,source=source,results=results,numerical=numerical,crashes=crashes,controls=controls,
            files=files,accepted_updates_compared=accepted,compared_decisions=decisions,
            crash_child_updates=sum(r['child_updates'] for r in crashes.values()),
            crash_child_decisions=sum(r['child_decisions'] for r in crashes.values()),
            crash_child_updates_published=sum(r['recovered_cursor'] for r in crashes.values()),
            crash_child_updates_uncommitted=sum(r['child_updates']-r['recovered_cursor'] for r in crashes.values()),
            previous_pin=a.previous_pin,cold_reproduced=previous is not None,phases=phases,seconds=seconds,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,gpu_used=False,
            full_population_trained=False,promoted=False,capability_demonstrated=False,acceptance='author-checked')
        print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),updates=accepted,decisions=decisions,seconds=seconds)),flush=True)
    finally:
        if corpus is not None and not corpus.closed:corpus.close();corpus.cache.close();corpus.reader.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path);p.add_argument('--previous-pin')
    p.add_argument('--crash-stage',choices=STAGES)
    a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise
