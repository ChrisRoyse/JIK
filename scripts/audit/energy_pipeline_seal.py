"""Capture completed, partial and fresh branches only under complete owner quiescence."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import energy_pipeline_contract as contract
from audit.answer_refresh_seal import branch
from audit import legal_grouped_placement_training as original
from audit import legal_placement_resume_training as driver
from audit.legal_placement_successor_seal import observation
from claims.process_conditions import write_committed
from model import completed_placement_branch as completed
from model import placement_resume_publication as publication
from model import placement_resume_tail as tail
from model.legal_placement_resume import Census,recount
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha,need


def run(a):
    began=time.perf_counter();cfg,_,_,fresh=contract.authenticate(a.authority,a.pin)
    before,q=observation(cfg,a.transition,a.transition_pin)
    need(q is not None,'ENERGY_PIPELINE_SEAL_QUIESCENCE_REQUIRED')
    a.output.mkdir(exist_ok=False);torch.set_num_threads(1);job=contract.SOURCE
    chosen={}
    for n in original.runtime.original.ARCHITECTURES:
        if not (job/n).exists():continue
        cps=sorted(int(p.stem.split('-')[-1]) for p in (job/n).glob('checkpoint-*.pt'))
        need(cps and cps[-1]>0,'ENERGY_PIPELINE_SEAL_NO_CHECKPOINT');chosen[n]=cps[-1]
    need(set(chosen)=={'shared','own_slot'} and chosen['shared']==91540 and chosen['own_slot']<91540,
         'ENERGY_PIPELINE_SEAL_MIXED_SCOPE')
    v=torch.load(job/'shared/checkpoint-91540.pt',map_location='cpu',weights_only=True)
    binding={k:v for k,v in v['binding'].items() if k!='architecture'}
    source,bundle=original.source_bundle(a.root)
    need(binding['code']==original.code(source) and binding['scope']=='full' and binding['device']=='cuda',
         'ENERGY_PIPELINE_SEAL_NUMERICAL_BINDING')
    data=original.load_data(argparse.Namespace(scope='full',root=a.root),bundle)
    census=driver.load_census(argparse.Namespace(scope='full',root=a.root),data)
    need(Census.load(torch.load(job/'census.pt',map_location='cpu',weights_only=True)).pin==census.pin
         and binding['source']==data['binding'],'ENERGY_PIPELINE_SEAL_CENSUS')
    donor=job/'donors.json';metadata=original.measure.metadata(data['items'])
    need(file_sha(donor)==binding['donors'] and json.loads(donor.read_bytes())==json.loads(json.dumps(
        dict(metadata=metadata,donors=original.measure.donor_plan(metadata)))),'ENERGY_PIPELINE_SEAL_DONORS')
    publication.copy_file(donor,a.output/'donors.json',binding['donors'])
    original.measure.persist(a.output/'census.pt',census.value())
    execution=publication.code();source_pins={str(donor):binding['donors']}
    for n,h in {**binding['code'],**execution,**{n:cfg['code'][n] for n in contract.NEW}}.items():
        publication.copy_file(contract.REPO/'scripts'/n,a.output/'source'/n,h)
    branches={};cuts={}
    for n in original.runtime.original.ARCHITECTURES:
        if n not in chosen:branches[n]=dict(fresh=True,step=0);continue
        branches[n],pins,cuts[n]=branch(job,a.output,n,chosen[n],binding,census,data['items'],data['labels'].roster)
        source_pins.update(pins)
    data['close']();after,_=observation(cfg,a.transition,a.transition_pin)
    need(set(chosen)=={n for n in original.runtime.original.ARCHITECTURES if (job/n).exists()},'ENERGY_PIPELINE_SEAL_BRANCH_CHANGED')
    for n,cut in cuts.items():need(contract.stamp(job/n/'steps.jsonl')==cut['stat'],'ENERGY_PIPELINE_SEAL_JOURNAL_CHANGED')
    need(all(file_sha(p)==h for p,h in source_pins.items()),'ENERGY_PIPELINE_SEAL_SOURCE_CHANGED')
    fresh();need(not torch.cuda.is_initialized(),'ENERGY_PIPELINE_SEAL_GPU_USED')
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    m=dict(complete=True,kind='placement-resume-boundary-v1',training_complete=False,binding=binding,
        execution=execution,census=census.pin,branches=branches,owners=list(cfg['owners'].values()),source_job=str(job),
        owner_authority=dict(path=str(a.authority),pin=a.pin),files=files,source_pins=source_pins,journal_cuts=cuts,
        retirement_ready=True,quiescence=q,before=before,after=after,gpu_used=False,optimizer_updates=0,
        seconds=time.perf_counter()-began,acceptance='author-checked',number_status='measured')
    pin=write_committed(a.output/'boundary.json',m);publication.boundary(a.output,pin,execution=execution)
    print(json.dumps(dict(pin=pin,steps=chosen,seconds=m['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=contract.ROOT)
    for n in ('authority','transition','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('pin','transition-pin'):p.add_argument('--'+n,required=True)
    a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(e)))
        raise
