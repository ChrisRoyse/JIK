"""Bind successor checkpoints into the unchanged whole-answer evaluation pipeline.

Only the authority provider changes. The original module is restored in finally;
the numerical export, generation, cold comparison and scoring functions are reused.
"""
import argparse
import json
import os
from pathlib import Path
import sys

if '--cpu-proof' in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
from audit import legal_answer_successor_authority as authority
from audit import legal_answer_evaluation_pipeline as pipeline
from audit.surface_device_queue import need,sha,write


class Provider:
    ROOT=authority.ROOT;REPO=authority.REPO;training=authority.evaluation.training
    exported=staticmethod(authority.evaluation.exported)

    def __init__(self,a):self.a=a;self.result=None

    @staticmethod
    def planned(output):
        value=authority.evaluation.planned(output)
        value['pipeline']=str(authority.REPO/'scripts/audit/legal_answer_successor_evaluation.py')
        return value

    def authenticate(self,path,pin):
        cfg,_,template,fresh=authority.authenticate(path,pin)
        if self.a.cuda:
            inputs=authority.read(self.a.inputs,self.a.inputs_pin)
            need(set(inputs)==set(authority.STAGES[:-1]),'SUCCESSOR_EVALUATION_INPUTS')
            ref=inputs['training-readback'];r=authority.read(Path(ref['path'])/'report.json',ref['pin'])
            need(r['complete'] and r['authority']==pin and r['stage']=='training-readback'
                 and r['driver_pin']==cfg['code']['audit/legal_answer_successor_worker.py']
                 and r['full_population_trained'] and r['optimizer_updates']==3*91540,
                 'SUCCESSOR_EVALUATION_TRAINING_READBACK')
            actual=r['training'];raw=inputs['training']
            need(actual['path']==raw['path'] and actual['pin']==raw['pin'],'SUCCESSOR_EVALUATION_TRAINING_PATH')
            t=authority.read(Path(actual['path'])/'report.json',actual['pin'])
            inspected=authority.read(Path(cfg['inspection']['path'])/'report.json',cfg['inspection']['pin'])
            need(t['complete'] and t['mode']=='full' and t['gpu_used'] and t['full_population_trained']
                 and t['source']==r['source']==inspected['source'] and t['binding']==r['binding']
                 and t['optimizer_updates']==3*91540,'SUCCESSOR_EVALUATION_TRAINING_SCOPE')
            for n,h in t['files'].items():need(sha(Path(actual['path'])/n)==h,'SUCCESSOR_EVALUATION_TRAINING_BYTES')
            self.result=dict(queue=ref['pin'],training=actual,finals=r['finals'])
        else:
            need(self.a.cpu_proof,'SUCCESSOR_EVALUATION_MODE')
        return cfg,None,template,fresh

    def terminal(self,cfg,parent):
        authority.terminal_owners(cfg)
        need(self.result is not None,'SUCCESSOR_EVALUATION_NO_TRAINING');return self.result


def run(a):
    old=pipeline.handoff
    try:
        pipeline.handoff=Provider(a)
        pipeline.run(a)
    finally:pipeline.handoff=old
    need(pipeline.handoff is old,'SUCCESSOR_EVALUATION_PROVIDER_RESTORE')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--cuda',action='store_true');g.add_argument('--cpu-proof',action='store_true')
    for n in ('authority','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--pin',required=True);p.add_argument('--inputs',type=Path);p.add_argument('--inputs-pin')
    p.add_argument('--owner-fd',type=int);a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write(a.output/'failure.json',dict(error=repr(error)))
        raise
