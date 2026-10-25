"""Seal all nine owners' placement prefix and suffix; reuse immutable initial readings."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import legal_answer_successor_authority as authority
from audit import legal_grouped_placement_training as original
from audit import legal_placement_resume_training as driver
from claims import process_handoff as processes
from claims import process_retirement as retirement
from claims.process_conditions import write_committed
from model import placement_resume_publication as publication
from model import placement_resume_tail as tail
from model.legal_placement_resume import recount
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha,need


def observation(cfg,transition=None,pin=None):
    order=[cfg['owners'][n]['pid'] for n in cfg['stop_order']]
    expected=list(cfg['owners'].values());rows=processes.capture(expected,order)
    by_pid={r['pid']:r for r in expected}
    need(all(processes.binding(r)==processes.binding(by_pid[r['pid']]) for r in rows),'SUCCESSOR_SEAL_CLOSURE')
    if transition is None:
        need(all(r['state'] in ('R','S','D') for r in rows),'SUCCESSOR_SEAL_LIVE_STATE')
        return rows,None
    path=transition/'contract.json';need(file_sha(path)==pin,'SUCCESSOR_SEAL_TRANSITION_PIN')
    value=json.loads(path.read_bytes())
    need(value['schema']=='process-retirement-v1' and value['sources']==retirement.sources()
         and value['boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
         and value['before_decision']=='resume' and value['after_decision']=='retire'
         and not value['may_launch_successor'] and not (transition/'decision.json').exists(),
         'SUCCESSOR_SEAL_TRANSITION_SCOPE')
    need([processes.binding(r) for r in rows]==[processes.binding(r) for r in value['owners']]
         and value['order']==order and all(r['state']=='T' for r in rows),'SUCCESSOR_SEAL_QUIESCENCE')
    coordinator=processes.process(value['coordinator']['pid'])
    need(coordinator is not None and coordinator['starttime']==value['coordinator']['starttime'],'SUCCESSOR_SEAL_COORDINATOR')
    return rows,dict(path=str(transition),pin=pin)


def run(a):
    began=tick=time.perf_counter();a.output.mkdir(exist_ok=False);torch.set_num_threads(1)
    cfg,_,_,fresh=authority.authenticate(a.authority,a.pin)
    before,transition=observation(cfg,a.transition,a.transition_pin)
    job=Path(cfg['owners']['producer']['output']);execution=publication.code()
    initial,initial_census=publication.boundary(a.initial,a.initial_pin)
    need(initial['source_job']==str(job) and initial['binding']['scope']=='full'
         and initial['binding']['device']=='cuda','SUCCESSOR_SEAL_INITIAL_SCOPE')
    chosen={}
    for value in a.branch:
        name,step=value.split(':');need(name not in chosen and name in original.runtime.original.ARCHITECTURES,'SUCCESSOR_SEAL_BRANCH')
        chosen[name]=int(step)
    need(chosen and set(chosen)=={n for n in original.runtime.original.ARCHITECTURES if (job/n).exists()},'SUCCESSOR_SEAL_BRANCH_SCOPE')
    for name in chosen:need(not initial['branches'][name].get('fresh'),'SUCCESSOR_SEAL_INITIAL_MISSING')
    first=next(iter(chosen));saved=torch.load(job/first/f'checkpoint-{chosen[first]}.pt',map_location='cpu',weights_only=True)
    binding={k:v for k,v in saved['binding'].items() if k!='architecture'}
    need(binding==initial['binding'],'SUCCESSOR_SEAL_BINDING')
    source,bundle=original.source_bundle(a.root);need(binding['code']==original.code(source),'SUCCESSOR_SEAL_NUMERICAL_SOURCE')
    data=original.load_data(argparse.Namespace(scope='full',root=a.root),bundle)
    census=driver.load_census(argparse.Namespace(scope='full',root=a.root),data)
    need(census.pin==initial_census.pin and data['binding']==binding['source'],'SUCCESSOR_SEAL_CENSUS')
    for name,h in binding['code'].items():need(file_sha(job/'source'/name)==h,'SUCCESSOR_SEAL_OLD_SOURCE')
    for name,h in {**execution,'audit/legal_placement_successor_seal.py':file_sha(__file__)}.items():
        publication.copy_file(Path(__file__).resolve().parents[1]/name,a.output/'source'/name,h)
    donor=job/'donors.json';need(file_sha(donor)==binding['donors'],'SUCCESSOR_SEAL_DONORS')
    metadata=original.measure.metadata(data['items'])
    need(json.loads(donor.read_bytes())==json.loads(json.dumps(dict(metadata=metadata,
        donors=original.measure.donor_plan(metadata)))),'SUCCESSOR_SEAL_DONOR_PLAN')
    publication.copy_file(donor,a.output/'donors.json',binding['donors'])
    original.measure.persist(a.output/'census.pt',census.value())
    data['close']();del data,bundle
    phases=dict(authentication=time.perf_counter()-tick);branches={};source_pins={str(donor):binding['donors']};cuts={}
    for name in original.runtime.original.ARCHITECTURES:
        tick=time.perf_counter()
        if name not in chosen:branches[name]=dict(fresh=True,step=0);continue
        old=job/name;out=a.output/name;out.mkdir();step=chosen[name]
        need(0<step<=len(census.documents),'SUCCESSOR_SEAL_CURSOR')
        cuts[name]=dict(bytes=(old/'steps.jsonl').stat().st_size,stat=authority.stamp(old/'steps.jsonl'))
        cps={};paths=[p for p in old.glob('checkpoint-*.pt') if int(p.stem.split('-')[-1])<=step]
        for p in sorted(paths,key=lambda p:int(p.stem.split('-')[-1])):
            h=file_sha(p);value=torch.load(p,map_location='cpu',weights_only=True);cursor=int(p.stem.split('-')[-1])
            need(value['step']==cursor and value['binding']==dict(binding,architecture=name),'SUCCESSOR_SEAL_CHECKPOINT')
            publication.copy_file(p,out/p.name,h);source_pins[str(p)]=h;cps[str(cursor)]=dict(pin=h,state=fingerprint(value))
        need(str(step) in cps and '0' in cps,'SUCCESSOR_SEAL_CHECKPOINT_SET')
        saved=torch.load(out/f'checkpoint-{step}.pt',map_location='cpu',weights_only=True)
        zero=torch.load(out/'checkpoint-0.pt',map_location='cpu',weights_only=True)
        counts=recount(old/'steps.jsonl',saved,zero,census,copy_to=out/'steps.jsonl',allow_suffix=True)
        suffix=tail.seal(old/'steps.jsonl',out,counts,census,dict(binding,architecture=name),cuts[name]['bytes'])
        # These initial tensors were already authenticated in a sealed publication.
        # Reuse their exact bytes and metrics, without loading every tensor again.
        previous=initial['branches'][name]
        need(cps['0']['state']==previous['checkpoints']['0']['state'],'SUCCESSOR_SEAL_INITIAL_MODEL')
        paths={r:h for r,h in initial['files'].items() if r.startswith(name+'/initial/')}
        need(paths and {Path(r).name for r in paths}=={p.name for p in (old/'initial').iterdir() if p.is_file()},
             'SUCCESSOR_SEAL_INITIAL_FILES')
        for rel,h in paths.items():
            src=job/rel;need(initial['source_pins'][str(src)]==h and file_sha(src)==h,'SUCCESSOR_SEAL_INITIAL_ORIGINAL')
            publication.copy_file(a.initial/rel,a.output/rel,h);source_pins[str(src)]=h
        branch=dict(step=step,checkpoints=cps,counts=counts,tail=suffix,
                    initial=previous['initial'],initial_rows=previous['initial_rows'])
        _,_,again=publication.validate_checkpoints(out,branch,dict(binding,architecture=name),census)
        need(again==counts,'SUCCESSOR_SEAL_CHECKPOINT_READBACK');branches[name]=branch
        phases[name+'_copy_and_readback']=time.perf_counter()-tick
    tick=time.perf_counter();after,_=observation(cfg,a.transition,a.transition_pin)
    need([processes.binding(r) for r in before]==[processes.binding(r) for r in after],'SUCCESSOR_SEAL_OWNER_CHANGED')
    if transition is not None:
        for name,cut in cuts.items():need(authority.stamp(job/name/'steps.jsonl')==cut['stat'],'SUCCESSOR_SEAL_FROZEN_SOURCE_CHANGED')
    need(set(chosen)=={n for n in original.runtime.original.ARCHITECTURES if (job/n).exists()},'SUCCESSOR_SEAL_BRANCH_CHANGED')
    need(all(file_sha(p)==h for p,h in source_pins.items()),'SUCCESSOR_SEAL_SOURCE_CHANGED');fresh()
    need(publication.code()==execution and not torch.cuda.is_initialized(),'SUCCESSOR_SEAL_EXECUTION')
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['final_readback']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.000001,'SUCCESSOR_SEAL_PHASES')
    manifest=dict(complete=True,kind='placement-resume-boundary-v1',training_complete=False,binding=binding,
        execution=execution,census=census.pin,branches=branches,owners=list(cfg['owners'].values()),source_job=str(job),files=files,
        owner_authority=dict(path=str(a.authority),pin=a.pin),census_authority=driver.CENSUS,
        initial_reuse=dict(path=str(a.initial),pin=a.initial_pin),source_pins=source_pins,journal_cuts=cuts,
        retirement_ready=transition is not None,quiescence=transition,phases=phases,seconds=seconds,
        gpu_used=False,optimizer_updates=0,live_processes_changed=False,acceptance='author-checked',number_status='measured')
    pin=write_committed(a.output/'boundary.json',manifest);publication.boundary(a.output,pin,execution=execution)
    print(json.dumps(dict(pin=pin,retirement_ready=manifest['retirement_ready'],seconds=seconds,
                         branches={n:dict(step=b['step'],tail=b.get('tail',{}).get('records',0)) for n,b in branches.items()})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    for n in ('authority','initial','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('pin','initial-pin'):p.add_argument('--'+n,required=True)
    p.add_argument('--branch',action='append',required=True);p.add_argument('--transition',type=Path);p.add_argument('--transition-pin')
    a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error)))
        raise
