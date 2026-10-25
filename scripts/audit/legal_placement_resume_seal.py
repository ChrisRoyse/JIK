"""Seal fixed completed prefixes of a live comparison without editing or stopping it."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import legal_grouped_placement_training as original
from audit import legal_placement_resume_training as driver
from audit.surface_device_queue import identity
from claims.process_conditions import write_committed
from model import placement_resume_publication as publication
from model.legal_placement_resume import recount
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha,need


def run(a):
    began=tick=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    execution=publication.code()
    for name,pin in {**execution,'audit/legal_placement_resume_seal.py':file_sha(__file__)}.items():
        publication.copy_file(Path(__file__).resolve().parents[1]/name,a.output/'source'/name,pin)
    need(file_sha(a.owners)==a.owners_pin,'PLACEMENT_SEAL_OWNERS_PIN')
    authority=json.loads(a.owners.read_bytes());need(authority['complete'],'PLACEMENT_SEAL_OWNERS_INCOMPLETE')
    owners=[dict(pid=int(pid),starttime=row['starttime']) for pid,row in authority['processes'].items()]
    need(len(owners)==5 and all(row['exists'] for row in authority['processes'].values()),'PLACEMENT_SEAL_OWNER_SCOPE')
    for row in owners:need(identity(row['pid'])==row['starttime'],'PLACEMENT_SEAL_OWNER_CHANGED')
    workers=[]
    for row in owners:
        raw=(Path('/proc')/str(row['pid'])/'cmdline').read_bytes();argv=raw.decode().split('\0')
        if any(Path(arg).name=='legal_grouped_placement_training.py' for arg in argv) and '--output' in argv:
            need(Path(argv[argv.index('--output')+1]).resolve()==a.job.resolve(),'PLACEMENT_SEAL_OWNER_JOB')
            workers.append(row)
    need(len(workers)==1,'PLACEMENT_SEAL_WORKER_IDENTITY')
    chosen={}
    for value in a.branch:
        name,cursor=value.split(':',1);need(name in original.runtime.original.ARCHITECTURES and name not in chosen,
                                         'PLACEMENT_SEAL_BRANCH')
        chosen[name]=int(cursor)
    need(chosen,'PLACEMENT_SEAL_NO_PREFIX')
    for name in original.runtime.original.ARCHITECTURES:
        if (a.job/name).exists():need(name in chosen,'PLACEMENT_SEAL_UNDECLARED_BRANCH')
    reference=torch.load(a.job/next(iter(chosen))/f'checkpoint-{next(iter(chosen.values()))}.pt',map_location='cpu',weights_only=True)
    binding={k:v for k,v in reference['binding'].items() if k!='architecture'}
    need(binding['scope']=='full' and binding['device']=='cuda','PLACEMENT_SEAL_SOURCE_SCOPE')
    source,bundle=original.source_bundle(a.root);need(binding['code']==original.code(source),'PLACEMENT_SEAL_NUMERICAL_CODE')
    for name,pin in binding['code'].items():need(file_sha(a.job/'source'/name)==pin,'PLACEMENT_SEAL_OLD_SOURCE_COPY')
    data=original.load_data(argparse.Namespace(scope='full',root=a.root),bundle)
    census=driver.load_census(argparse.Namespace(scope='full',root=a.root),data)
    need(binding['source']==data['binding'],'PLACEMENT_SEAL_DATA_BINDING')
    donor_path=a.job/'donors.json';need(file_sha(donor_path)==binding['donors'],'PLACEMENT_SEAL_DONORS')
    donors=json.loads(donor_path.read_bytes());metadata=original.measure.metadata(data['items'])
    need(donors==json.loads(json.dumps(dict(metadata=metadata,donors=original.measure.donor_plan(metadata)))),
         'PLACEMENT_SEAL_DONOR_PLAN')
    publication.copy_file(donor_path,a.output/'donors.json',binding['donors'])
    original.measure.persist(a.output/'census.pt',census.value())
    phases=dict(authentication=time.perf_counter()-tick);branches={};source_pins={donor_path:binding['donors']}
    for name in original.runtime.original.ARCHITECTURES:
        tick=time.perf_counter()
        if name not in chosen:branches[name]=dict(fresh=True,step=0);continue
        step=chosen[name];need(0<step<=len(census.documents),'PLACEMENT_SEAL_CURSOR')
        old=a.job/name;out=a.output/name;out.mkdir();cps={}
        paths=[p for p in old.glob('checkpoint-*.pt') if int(p.stem.split('-')[-1])<=step]
        need(any(p.name==f'checkpoint-{step}.pt' for p in paths),'PLACEMENT_SEAL_CHECKPOINT_MISSING')
        for p in sorted(paths,key=lambda p:int(p.stem.split('-')[-1])):
            pin=file_sha(p);value=torch.load(p,map_location='cpu',weights_only=True);cursor=int(p.stem.split('-')[-1])
            need(value['binding']==dict(binding,architecture=name) and value['step']==cursor,'PLACEMENT_SEAL_CHECKPOINT_BINDING')
            publication.copy_file(p,out/p.name,pin);source_pins[p]=pin;cps[str(cursor)]=dict(pin=pin,state=fingerprint(value))
        initial=torch.load(out/'checkpoint-0.pt',map_location='cpu',weights_only=True)
        saved=torch.load(out/f'checkpoint-{step}.pt',map_location='cpu',weights_only=True)
        count=recount(old/'steps.jsonl',saved,initial,census,copy_to=out/'steps.jsonl',allow_suffix=True)
        rows,names=publication.evaluation(old/'initial',data['items'],name,census.roster)
        for rel in names:
            path=old/'initial'/rel;pin=file_sha(path);source_pins[path]=pin
            publication.copy_file(path,out/'initial'/rel,pin)
        branch=dict(step=step,checkpoints=cps,counts=count,initial=original.measure.summary(rows,census.roster),
                    initial_rows=fingerprint(rows))
        _,_,reopened=publication.validate_checkpoints(out,branch,dict(binding,architecture=name),census)
        need(reopened==count,'PLACEMENT_SEAL_PREFIX_READBACK');branches[name]=branch
        phases[name+'_copy_and_readback']=time.perf_counter()-tick
    tick=time.perf_counter();data['close']()
    need(all(file_sha(p)==pin for p,pin in source_pins.items()),'PLACEMENT_SEAL_IMMUTABLE_SOURCE_CHANGED')
    for row in owners:need(identity(row['pid'])==row['starttime'],'PLACEMENT_SEAL_OWNER_CHANGED')
    need(not torch.cuda.is_initialized(),'PLACEMENT_SEAL_UNEXPECTED_GPU')
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['final_readback']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.000001,'PLACEMENT_SEAL_PHASES')
    result=dict(complete=True,kind='placement-resume-boundary-v1',training_complete=False,binding=binding,
        execution=execution,census=census.pin,branches=branches,owners=owners,source_job=str(a.job),files=files,
        worker_owner=workers[0],
        owner_authority=dict(path=str(a.owners),pin=a.owners_pin),census_authority=driver.CENSUS,
        source_pins={str(p):pin for p,pin in source_pins.items()},phases=phases,seconds=seconds,
        gpu_used=False,optimizer_updates=0,live_processes_changed=False,acceptance='author-checked',number_status='measured')
    pin=write_committed(a.output/'boundary.json',result)
    publication.boundary(a.output,pin,execution=execution)
    # A sealed prefix is evidence, not permission to run alongside its live owners.
    try:driver.terminal_owners(owners)
    except ValueError as error:
        need('LIVE_PREDECESSOR' in str(error),'PLACEMENT_SEAL_REFUSAL_CODE')
        control=dict(complete=True,boundary_pin=pin,live_owner_refused=str(error),gpu_used=False)
    else:raise ValueError('PLACEMENT_SEAL_LIVE_CONTROL_ACCEPTED')
    control_pin=write_committed(a.output/'readback.json',control)
    print(json.dumps(dict(boundary_pin=pin,readback_pin=control_pin,branches={n:b['step'] for n,b in branches.items()})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    for n in ('job','owners','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--owners-pin',required=True);p.add_argument('--branch',action='append',required=True)
    a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error)))
        raise
