"""Qualify a mixed continuation on CUDA, then run the unchanged complete comparison."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import answer_refresh_contract as contract
from audit import legal_answer_successor_authority as original
from audit.legal_answer_successor import Children,paired_args
from audit.surface_device_queue import identity,need,sha,write


def choose(first,cold):
    need(first['gpu_qualified'] and cold['gpu_qualified'] and first['all_updates_exact']
         and cold['all_updates_exact'] and first['result_pin']==cold['result_pin'] and cold['cold_readback_exact'],
         'MIXED_PROGRAM_VALIDATION')
    per={}
    for name in ('shared','own_slot','whole_proposals'):
        rows=first['timing'][name]+cold['timing'][name]
        per[name]={key:statistics.median(row[key] for row in rows) for key in ('original','candidate')}
    # The GPU timings are a bounded probe, not a claim of full-corpus throughput.
    selected='batched' if all(v['candidate']<v['original'] for v in per.values()) else 'original'
    return dict(implementation=selected,per_architecture_median_seconds=per,scope='same_5_documents_66_targets')


def run(a):
    began=time.perf_counter();cfg,old,_,fresh=contract.authenticate(a.authority,a.pin)
    contract.terminal(cfg)
    # A retirement receipt is required before this entry can launch even a small GPU proof.
    from audit.answer_refresh_retirement import handoff
    retired=handoff(cfg,a.pin,a.handoff,a.handoff_pin)
    a.output.mkdir(exist_ok=False)
    write(a.output/'launch.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),authority=a.pin,driver_pin=sha(__file__)))
    children=Children(a.output,fresh,a.owner_fd);refs={}
    def child(label,script,args,report='report.json'):
        r=children.run(label,script,args,a.output/label,report);refs[label]=r;return r
    common=['--root',str(contract.ROOT)]
    first=child('validation','placement_validation_proof.py',[*common,*paired_args('qualification',cfg['validation'])])
    cold=child('validation-cold','placement_validation_proof.py',[
        *common,*paired_args('qualification',cfg['validation']),*paired_args('previous',first)])
    previous_report=contract.read(cold,'report.json')
    need(previous_report['previous']==first['pin'],'MIXED_PROGRAM_VALIDATION_COLD')
    selected=choose(contract.read(first,'report.json'),previous_report)
    write(a.output/'selection.json',dict(**selected,validation=first,validation_cold=cold))
    oldref=cfg['cuda_reference'];boundary=cfg['cuda_boundary']
    made=child('mixed-boundary','mixed_placement_boundary.py',[
        *common,*paired_args('reference',oldref),*paired_args('boundary',boundary)])
    resume=contract.read(made,'report.json')['boundary'];implementation=selected['implementation']
    arguments=[*common,'--implementation',implementation,*paired_args('cpu-qualification',cfg['cpu'][implementation])]
    for n,ref in old['runtime'].items():arguments+=paired_args(n,ref)
    first=child('resumed','mixed_placement_training.py',['--scope','small',*arguments,*paired_args('resume',resume)])
    cold=child('cold','mixed_placement_training.py',['--scope','small',*arguments,*paired_args('resume',resume),*paired_args('previous',first)])
    match=child('match','mixed_placement_match.py',[*paired_args('original',oldref),*paired_args('candidate',first),
        *paired_args('cold',cold),*paired_args('boundary',resume)])
    child('source-readback','legal_grouped_placement_readback.py',paired_args('input',first))
    full=child('full','mixed_placement_training.py',['--scope','full',*arguments,*paired_args('resume',retired['boundary']),
        *paired_args('qualification',first),*paired_args('qualification-cold',cold),*paired_args('match',match)])
    child('full-readback','legal_grouped_placement_readback.py',paired_args('input',full))
    fresh();write(a.output/'report.json',dict(complete=True,driver_pin=sha(__file__),authority=a.pin,
        outputs=refs,selection=selected,full_placement_compared=True,full_population_trained=False,
        scientific_admission=False,architecture_winner=False,seconds=time.perf_counter()-began,acceptance='author-checked'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('authority','handoff','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('pin','handoff-pin'):p.add_argument('--'+n,required=True)
    p.add_argument('--owner-fd',type=int);a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.exists():write(a.output/'failure.json',dict(error=repr(e)))
        raise
