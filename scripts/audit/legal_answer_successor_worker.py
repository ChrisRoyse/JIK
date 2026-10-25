"""Bound scientific stages used only by the versioned successor coordinator."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
from audit import legal_answer_successor_authority as authority
from audit.legal_answer_successor import Children,paired_args
from audit.surface_device_queue import identity,need,sha,write


def placement(a,cfg,children):
    retirement=authority.handoff(cfg,a.pin,a.handoff,a.handoff_pin);refs={}
    common=['--root',str(authority.ROOT)]
    for key,ref in cfg['runtime'].items():common+=paired_args(key,ref)
    def run(label,script,args,report='report.json'):
        ref=children.run(label,script,args,a.output/label,report);refs[label]=ref;return ref
    run('original','legal_grouped_placement_training.py',['--scope','small',*common])
    run('boundary','legal_placement_resume_training.py',['--scope','small','--boundary-step','2',*common],'boundary.json')
    resume=paired_args('resume',refs['boundary'])
    run('resumed','legal_placement_resume_training.py',['--scope','small',*common,*resume])
    run('cold','legal_placement_resume_training.py',['--scope','small',*common,*resume,*paired_args('previous',refs['resumed'])])
    args=[]
    for key,label in (('original','original'),('candidate','resumed'),('cold','cold'),('boundary','boundary')):
        args+=paired_args(key,refs[label])
    run('match','legal_placement_resume_match.py',args)
    run('source-readback','legal_grouped_placement_readback.py',paired_args('input',refs['resumed']))
    args=['--scope','full',*common,*paired_args('resume',retirement['boundary'])]
    for key,label in (('qualification','resumed'),('qualification-cold','cold'),('match','match')):
        args+=paired_args(key,refs[label])
    run('full','legal_placement_resume_training.py',args)
    run('full-readback','legal_grouped_placement_readback.py',paired_args('input',refs['full']))
    need(tuple(refs)==authority.PLACEMENT,'SUCCESSOR_PLACEMENT_STAGES')
    return dict(outputs=refs,full_placement_compared=True,full_population_trained=False)


def prior(old):
    from audit import legal_answer_recovery_authority as recovery
    refs={}
    for name,ref in old['completed_original'].items():
        value=authority.read(Path(ref['path'])/'report.json',ref['pin'])
        need(value['complete'],'SUCCESSOR_PRIOR_INCOMPLETE');refs[name]=ref
    records={n:authority.read(authority.ROOT/d/'report.json',h) for n,(d,h) in recovery.RECEIPTS.items()}
    recovery.correction_scope(records)
    path=Path(old['owners']['driver']['output'])/'inversion-cold-failure.json'
    need(json.loads(path.read_bytes())==old['retained_failure'],'SUCCESSOR_FAILURE_CHANGED')
    return dict(completed_original=refs,correction=old['correction'],retained_failure=dict(path=str(path),pin=sha(path)),
                metadata_reconciled=True,numerical_payloads_reauthenticated=False,original_complete=False)


def capacity(cfg,inputs):
    from audit.legal_answer_recovery_queue import final_capacity
    from audit.legal_answer_source_training import qualification
    source=inputs['source'];epoch=inputs['epoch'];r=final_capacity(Path(source['path']))
    need(sha(Path(source['path'])/'report.json')==source['pin'] and r['predecessor_pin']==epoch['pin'],
         'SUCCESSOR_CAPACITY_CHAIN')
    qualification(Path(epoch['path']),epoch['pin'],cfg['code'])
    pair=authority.evaluation.training.stages(r['outputs'],('proof','proof-cold'))
    first=authority.read(Path(pair['proof']['path'])/'report.json',pair['proof']['pin'])
    inspected=authority.read(Path(cfg['inspection']['path'])/'report.json',cfg['inspection']['pin'])
    need(first['source']==inspected['source'],'SUCCESSOR_CAPACITY_SOURCE')
    args=['--mode','full','--root',str(authority.ROOT)]
    for name,ref in (('inspection',cfg['inspection']),('qualification',epoch),('proof',pair['proof']),('cold',pair['proof-cold'])):
        args+=paired_args(name,ref)
    return dict(arguments=args,source=first['source'],binding=first['binding'],capacity_qualified=True,
                full_population_trained=False)


def trained(cfg,inputs):
    cap=inputs['capacity'];value=authority.read(Path(cap['path'])/'report.json',cap['pin'])
    ref=inputs['training'];path=Path(ref['path']);report=authority.read(path/'report.json',ref['pin'])
    actual=authority.evaluation.training.completed(path,cfg,value)
    need(actual['pin']==ref['pin'] and report['code']==value['source']['code'],'SUCCESSOR_TRAINING_CODE')
    for name,h in report['files'].items():need(sha(path/name)==h,'SUCCESSOR_TRAINING_FILES')
    finals={}
    for arch in authority.evaluation.training.ARCHES:
        row=report['results'][arch]['full']['result']
        finals[arch]={k:row[k] for k in ('model','optimizer','chain','counts')}
    return dict(training=actual,finals=finals,full_population_trained=True,optimizer_updates=3*91540,
                source=value['source'],binding=value['binding'])


def run(a):
    began=time.perf_counter();cfg,old,_,fresh=authority.authenticate(a.authority,a.pin)
    inputs=authority.read(a.inputs,a.inputs_pin)
    expected=authority.STAGES[:authority.STAGES.index(a.stage)]
    need(tuple(inputs)==expected or set(inputs)==set(expected),'SUCCESSOR_INPUT_STAGES')
    a.output.mkdir(exist_ok=False)
    write(a.output/'launch.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),authority=a.pin,
                                     driver_pin=sha(__file__),stage=a.stage))
    children=Children(a.output,fresh,a.owner_fd)
    if a.stage=='prior-readback':value=prior(old)
    elif a.stage=='placement':value=placement(a,cfg,children)
    elif a.stage=='capacity':value=capacity(cfg,inputs)
    else:value=trained(cfg,inputs)
    fresh();write(a.output/'report.json',dict(complete=True,stage=a.stage,authority=a.pin,driver_pin=sha(__file__),
        inputs=a.inputs_pin,**value,seconds=time.perf_counter()-began,acceptance='author-checked'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('prior-readback','placement','capacity','training-readback'),required=True)
    for n in ('authority','inputs','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('pin','inputs-pin'):p.add_argument('--'+n,required=True)
    p.add_argument('--handoff',type=Path);p.add_argument('--handoff-pin');p.add_argument('--owner-fd',type=int)
    a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write(a.output/'failure.json',dict(error=repr(error)))
        raise
