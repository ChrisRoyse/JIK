"""Real seven-case scoring, cold/reopen/readback and durable interruption proof."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit.complete_answer_evaluation import read,BASE,ROOT
from audit.complete_answer_comparison import direct,reject
from audit.legal_answer_public_pack import authenticate
from audit.legal_answer_cohort_scores import References,CODE
from claims.process_conditions import write_committed
from model import complete_answer_cohort_scores as scoring
from model.complete_answer_comparison import summarize
from model.complete_answer_summary import Summary
from model.legal_answer_public_pack import PublicPack
from text.constellation_corpus import file_sha,need

PRIOR=('legal-answer-packed-proof-v1-20261024','e8acd43b5e2a01d035aabcc5424e266a580a97b4819b32f5446e2e49aed3b535')
DOCUMENTS=(7,4830,19295,20107,22924,30520,84190)


def launch(script,args,log,expected=0):
    tick=time.perf_counter()
    with log.open('xb') as stream:
        child=subprocess.run([sys.executable,'-I',str(BASE/'audit'/script),*args],stdout=stream,stderr=subprocess.STDOUT,check=False)
    need(child.returncode==expected,'COHORT_SCORE_PROOF_CHILD:'+str(child.returncode)+':'+str(log))
    return dict(exit_code=child.returncode,seconds=time.perf_counter()-tick)


def payloads(root,result):
    for row in result['cases']:
        c=read(root/'scores'/row['path'],row['pin']);yield read(root/'scores'/c['payload']['path'],c['payload']['pin'])


def worker(cohort,pin,output,receipt,log,*,fault=False):
    args=['--cohort',str(cohort),'--cohort-pin',pin,'--scope','proof','--output',str(output),'--receipt',receipt]
    if fault:args+=['--fault']
    return launch('legal_answer_cohort_scores.py',args,log,73 if fault else 0)


def run(a):
    began=tick=time.perf_counter();a.output.mkdir(exist_ok=False);phases={};launches={}
    previous=read(a.previous/'report.json',a.previous_pin) if a.previous else None
    prior=read(a.root/PRIOR[0]/'report.json',PRIOR[1])
    m=read(a.root/PRIOR[0]/'request.json',prior['files']['request.json'])
    pack=PublicPack(m['pack']['path'],m['pack']['pin'])
    ordinals=[int(np.flatnonzero(pack.rows['document']==n)[0]) for n in DOCUMENTS]
    if previous:
        need(previous['complete'] and previous['documents']==list(DOCUMENTS),'COHORT_SCORE_PROOF_PREVIOUS')
        generation=previous['generation'];cohort=Path(generation['path']);cohort_pin=generation['pin']
    else:
        m['ordinals']=ordinals;m['authority']=dict(m['authority'],scope='CPU_score_mechanics_exposed_cases')
        m['code'].update({n:file_sha(BASE/n) for n in CODE})
        dest=a.output/'generation';dest.mkdir();request_pin=write_committed(dest/'request.json',m)
        phases['prepare']=time.perf_counter()-tick;tick=time.perf_counter()
        launches['generation']=launch('complete_answer_packed_cohort.py',['--cpu-proof','--request',str(dest/'request.json'),
            '--request-pin',request_pin,'--output',str(dest/'worker')],a.output/'generation.log')
        cohort=dest/'worker';cohort_pin=file_sha(cohort/'report.json')
        generation=dict(path=str(cohort),pin=cohort_pin)
    phases['generation_or_prior']=time.perf_counter()-tick;tick=time.perf_counter()
    out=a.output/'scorer';launches['score']=worker(cohort,cohort_pin,out,'initial',a.output/'score.log')
    first=read(out/'initial.json',file_sha(out/'initial.json'))
    need(first['actions']==dict(scored=7,reused=0,gap=0) and len(first['reference_calls'])==7
         and not first['full_population'] and first['scored_cases']==7,'COHORT_SCORE_PROOF_INITIAL')
    launches['reopen']=worker(cohort,cohort_pin,out,'reopen',a.output/'reopen.log')
    reopened=read(out/'reopen.json',file_sha(out/'reopen.json'))
    need(reopened['actions']==dict(scored=0,reused=7,gap=0) and not reopened['reference_calls']
         and reopened['summary']==first['summary'] and reopened['cases']==first['cases'],'COHORT_SCORE_PROOF_REOPEN')
    if previous:
        old=read(a.previous/'scorer/initial.json',previous['files']['scorer/initial.json'])
        need(first['summary']==old['summary'] and first['by_collection']==old['by_collection']
             and first['cases']==old['cases'],'COHORT_SCORE_PROOF_COLD')
    phases['score_and_reopen']=time.perf_counter()-tick;tick=time.perf_counter()
    source,rows,data,continuation,reader=authenticate(a.root)
    refs=References(source,data,reader,pack,a.output);r,b,cases=scoring.authority(cohort,cohort_pin,pack,'proof')
    saved=list(payloads(out,first));baseline=summarize(saved)
    need(baseline==first['summary'],'COHORT_SCORE_PROOF_SUMMARY_EXACT')
    for collection,summary in first['by_collection'].items():
        need(summarize(v for v in saved if v['identity']['collection']==collection)==summary,'COHORT_SCORE_PROOF_COLLECTION_EXACT')
    metrics=0;atoms=0;window_counts={n:0 for n in pack.roster};ends=exact_ends=refusals=0
    for row,result,score in zip(cases,r['results'],saved,strict=True):
        public=pack.load(row['public']);traces,_,_=scoring.sealed(cohort,result,row,b,public,pack.roster)
        private=refs(row['ordinal'],public);future=private['future'];atoms+=len(future)
        need(score['identity']==private['identity'],'COHORT_SCORE_PROOF_REFERENCE_IDENTITY')
        for n in pack.roster:window_counts[n]+=sum(len(f[n]) for f in future)
        lookup={(v['architecture'],v['proposal'],v['mode']):v for v in traces}
        for arm in score['arms']:
            t=lookup[arm['architecture'],arm['proposal'],arm['mode']];d=direct(Path(t['path']),future,pack.roster);v=arm['metrics']
            need(d['per_slot']==v['per_slot'] and d['means']==v['full_output_mean'] and d['generated']==v['generated_atoms']
                 and d['missing']==v['missing'] and d['extra']==v['extra'] and d['ended']==v['ended'], 'COHORT_SCORE_PROOF_DIRECT_METRICS')
            metrics+=1;ends+=v['ended'];exact_ends+=v['ended_at_reference_boundary'];refusals+=bool(v['refusal'])
        for n in pack.roster:
            x=public['correct']['frames'][-1][n].numpy().astype('float64');x=x/np.linalg.norm(x,axis=1,keepdims=True);values=[]
            for f in future:
                y=f[n].numpy().astype('float64');z=np.clip(x@(y/np.linalg.norm(y,axis=1,keepdims=True)).T,-1,1)
                values.append(float(min(z.max(1).mean(),z.max(0).mean())))
            need(values==score['copy_current']['per_slot'][n],'COHORT_SCORE_PROOF_COPY_BASELINE')
    need(atoms==2917 and metrics==168 and len(first['by_collection'])==3,'COHORT_SCORE_PROOF_SCOPE')
    phases['independent_native_and_metrics']=time.perf_counter()-tick;tick=time.perf_counter()
    controls={};controls['false_full_scope']=reject(lambda:scoring.authority(cohort,cohort_pin,pack,'all'),'COHORT_SCORE_INCOMPLETE_FULL_SCOPE')
    s=Summary();s.add(saved[0]);controls['duplicate_case']=reject(lambda:s.add(saved[0]),'ANSWER_COMPARISON_DUPLICATE_CASE')
    broken=deepcopy(saved[0]);broken['arms'].pop();controls['missing_arm']=reject(lambda:Summary().add(broken),'ANSWER_COMPARISON_COHORT_GRID')
    for kind in ('missing','duplicate','reordered'):
        dest=a.output/('bad-'+kind);(dest/'worker/run').mkdir(parents=True)
        shutil.copyfile(cohort.parent/'request.json',dest/'request.json');shutil.copyfile(cohort/'run/binding.json',dest/'worker/run/binding.json')
        bad=deepcopy(r)
        if kind=='missing':bad['results'].pop()
        elif kind=='duplicate':bad['results'].append(bad['results'][0])
        else:bad['results'].reverse()
        pin=write_committed(dest/'worker/report.json',bad)
        controls[kind]=reject(lambda:scoring.authority(dest/'worker',pin,pack,'proof'),'COHORT_SCORE_COVERAGE')
    row,result=cases[0],r['results'][0];public=pack.load(row['public']);calls=reader.stats['calls'];refs.maximum_allowance=1
    controls['reference_memory_before_read']=reject(lambda:refs(row['ordinal'],public),'COHORT_SCORE_REFERENCE_MEMORY')
    need(reader.stats['calls']==calls,'COHORT_SCORE_PROOF_MEMORY_READ');refs.maximum_allowance=None
    dest=a.output/'bad-trace';casepath=dest/'run'/result['path'];casepath.mkdir(parents=True)
    c=read(cohort/'run'/result['path']/'case.json',result['pin']);gpath=cohort/'run'/result['path']/c['generation']['path']
    g=read(gpath/'report.json',c['generation']['pin']);badtrace=a.output/'corrupt-trace';original=Path(g['traces'][0]['path'])
    if not original.is_absolute():original=gpath/original
    shutil.copytree(original,badtrace)
    with (badtrace/'events.bin').open('r+b') as stream:stream.seek(-1,os.SEEK_END);v=stream.read(1);stream.seek(-1,os.SEEK_END);stream.write(bytes([v[0]^1]))
    # Other traces still belong to the original generation. Resolve their
    # relative paths before relocating this deliberately corrupted manifest.
    for trace in g['traces']:
        p=Path(trace['path'])
        if not p.is_absolute():trace['path']=str(gpath/p)
    g['traces'][0]['path']=str(badtrace);(casepath/c['generation']['path']).mkdir()
    c['generation']['pin']=write_committed(casepath/c['generation']['path']/'report.json',g)
    badresult=dict(result,pin=write_committed(casepath/'case.json',c))
    controls['corrupt_trace_before_reference']=reject(lambda:scoring.sealed(dest,badresult,row,b,public,pack.roster),'TRACE')
    broken=a.output/'bad-saved-score';shutil.copytree(out/'scores',broken)
    record=read(broken/first['cases'][0]['path'],first['cases'][0]['pin']);target=broken/record['payload']['path']
    target.write_bytes(target.read_bytes()+b' ')
    before=len(refs.calls)
    controls['changed_saved_score']=reject(lambda:scoring.run(cohort,cohort_pin,pack,refs,output=broken,
        code=first['code'],source=source.binding,scope='proof'),'ANSWER_COHORT_RECEIPT_PIN')
    need(len(refs.calls)==before,'COHORT_SCORE_PROOF_CORRUPT_REFERENCE_READ')
    reader.close();source.close();rows._mmap.close();continuation.fresh();pack.close()
    phases['controls_and_source_close']=time.perf_counter()-tick;tick=time.perf_counter()
    interrupted=a.output/'interrupted';launches['exit']=worker(cohort,cohort_pin,interrupted,'exit',a.output/'exit.log',fault=True)
    committed=list((interrupted/'scores').glob('case-*.json'));need(len(committed)==1,'COHORT_SCORE_PROOF_EXIT_COMMIT')
    committed_pin=file_sha(committed[0]);launches['resume']=worker(cohort,cohort_pin,interrupted,'resume',a.output/'resume.log')
    resumed=read(interrupted/'resume.json',file_sha(interrupted/'resume.json'))
    need(resumed['actions']==dict(scored=6,reused=1,gap=0) and len(resumed['reference_calls'])==6
         and resumed['summary']==first['summary'] and resumed['cases']==first['cases']
         and file_sha(committed[0])==committed_pin,'COHORT_SCORE_PROOF_RESUME')
    phases['exit_and_resume']=time.perf_counter()-tick;tick=time.perf_counter()
    files={str(p.relative_to(a.output)):file_sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()}
    phases['seal']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.001,'COHORT_SCORE_PROOF_PHASES')
    result=dict(complete=True,mode='cold' if previous else 'publish',previous_pin=a.previous_pin,generation=generation,
        documents=list(DOCUMENTS),cases=7,trajectories=metrics,reference_atoms=atoms,native_windows=window_counts,
        ends=ends,correct_length_ends=exact_ends,refusals=refusals,all_metrics_exact=True,summary_exact=True,
        controls=controls,launches=launches,files=files,code={n:file_sha(BASE/n) for n in CODE},
        phases_seconds=phases,seconds=seconds,admitted_senses=0,architecture_winner=False)
    pin=write_committed(a.output/'report.json',result);print(json.dumps(dict(complete=True,pin=pin,seconds=seconds,metrics=metrics)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path);p.add_argument('--previous-pin')
    a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise
