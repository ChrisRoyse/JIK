"""Qualify energy batching on CUDA or explicitly resume the qualified reference."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import energy_pipeline_contract as contract
from audit.legal_answer_successor import Children,paired_args
from audit.surface_device_queue import identity,need,sha,write


def choose(first,cold,first_pin):
    need(first['complete'] and cold['complete'] and first['device']==cold['device']=='cuda'
         and first['cuda_producer_exact'] and cold['gpu_qualified'] and first['all_updates_exact']
         and cold['all_updates_exact'] and cold['reference_replay_exact'] and cold['restarts_exact']
         and cold['source_pin']==first_pin and first['code']==cold['code']
         and first['source_state']==cold['source_state'],'ENERGY_PROGRAM_CUDA_PROOF')
    per={n:dict(reference=v['reference_median'],candidate=v['candidate_median'])
         for n,v in cold['speed'].items()}
    need(set(per)=={'shared','own_slot','whole_proposals'}
         and all(v[k]>0 for v in per.values() for k in ('reference','candidate')),
         'ENERGY_PROGRAM_TIMING')
    selected='energy' if all(v['candidate']<v['reference'] for v in per.values()) else 'batched'
    return dict(implementation=selected,per_architecture_median_seconds=per,
                scope='same_5_documents_66_targets',cuda_equivalence=True)


def run(a):
    began=time.perf_counter();cfg,old,_,fresh=contract.authenticate(a.authority,a.pin)
    contract.terminal(cfg)
    from audit.energy_pipeline_retirement import handoff
    retired=handoff(cfg,a.pin,a.handoff,a.handoff_pin)
    a.output.mkdir(exist_ok=False)
    write(a.output/'launch.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),authority=a.pin,
                                    driver_pin=sha(__file__),mode=a.mode))
    children=Children(a.output,fresh,a.owner_fd);refs={}
    def child(label,script,args,report='report.json'):
        r=children.run(label,script,args,a.output/label,report);refs[label]=r;return r
    common=['--root',str(contract.ROOT)]
    selected=dict(implementation='batched',reason='explicit_reference_mode',cuda_equivalence=None)
    first=cold=match=None; energy_args=None
    if a.mode=='qualify':
        runtime=child('energy-runtime','energy_placement_qualification.py',[
            *common,'--cpu-source',cfg['cpu']['update']['path'],'--cpu-pin',cfg['cpu']['update']['pin'],
            *paired_args('cpu-cold',cfg['cpu']['cold'])])
        runtime_cold=child('energy-runtime-cold','energy_placement_readback.py',[
            *common,'--source',runtime['path'],'--pin',runtime['pin']])
        selected=choose(contract.read(runtime,'report.json'),contract.read(runtime_cold,'report.json'),runtime['pin'])
        if selected['implementation']=='energy':
            energy_args=[*common,'--implementation','energy',*paired_args('runtime',runtime),
                *paired_args('runtime-cold',runtime_cold),*paired_args('cpu-qualification',cfg['cpu']['continuation'])]
            resume=cfg['cuda_boundary']
            first=child('resumed','energy_placement_training.py',[
                '--scope','small',*energy_args,*paired_args('resume',resume)])
            cold=child('cold','energy_placement_training.py',[
                '--scope','small',*energy_args,*paired_args('resume',resume),*paired_args('previous',first)])
            match=child('match','energy_continuation_readback.py',[
                *common,*paired_args('original',cfg['cuda_reference']),*paired_args('candidate',first),
                *paired_args('cold',cold),*paired_args('boundary',resume)])
            child('source-readback','legal_grouped_placement_readback.py',paired_args('input',first))
    write(a.output/'selection.json',dict(**selected,mode=a.mode,qualification_outputs=refs))
    if selected['implementation']=='energy':
        args=[*energy_args,*paired_args('qualification',first),*paired_args('qualification-cold',cold),
              *paired_args('match',match)]
        script='energy_placement_training.py'
    else:
        # This is the exact retained, CUDA-qualified implementation, explicitly reported.
        parent=contract.read(contract.PARENT)
        args=[*common,'--implementation','batched',*paired_args('cpu-qualification',parent['cpu']['batched'])]
        for name,ref in old['runtime'].items():args+=paired_args(name,ref)
        for name,ref in cfg['reference'].items():args+=paired_args(name,ref)
        script='mixed_placement_training.py'
    full=child('full',script,['--scope','full',*args,*paired_args('resume',retired['boundary'])])
    child('full-readback','legal_grouped_placement_readback.py',paired_args('input',full))
    fresh();write(a.output/'report.json',dict(complete=True,driver_pin=sha(__file__),authority=a.pin,
        outputs=refs,selection=selected,mode=a.mode,full_placement_compared=True,full_population_trained=False,
        scientific_admission=False,architecture_winner=False,seconds=time.perf_counter()-began,acceptance='author-checked'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('authority','handoff','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('pin','handoff-pin'):p.add_argument('--'+n,required=True)
    p.add_argument('--mode',choices=('qualify','reference'),default='qualify')
    p.add_argument('--owner-fd',type=int);a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.exists():write(a.output/'failure.json',dict(error=repr(e),mode=a.mode))
        raise
