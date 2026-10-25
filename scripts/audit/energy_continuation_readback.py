"""Independent complete-experiment readback and candidate continuation fault controls."""
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
from audit import captured_answer_readback as rd
from audit import legal_grouped_placement_training as original
from audit import energy_placement_training as driver
from audit import energy_placement_qualification as energy
from model.energy_placement_continuation import EnergyPlacementContinuation
from model.placement_resume_tail import Replay
from model.native_population_training import fingerprint
from senses.contract import canonical
from claims.process_conditions import write_committed
from model import placement_resume_publication as publication
from text.constellation_corpus import file_sha,need


def controls(a):
    """Fault controls run on the retained CPU checkpoint even for CUDA readback."""
    _,bundle,target,items,_=energy.sources(a.root,'cpu')
    old_root=a.root/'placement-driver-original-v1-20261024'
    old=original.pinned(old_root,'af334059da30b3a6cf1961b97c4c3b838fb00aeb8d2762a80f68c54a6202ce40')
    boundary_root=a.root/'mixed-placement-boundary-cpu-v1-20261025/boundary'
    boundary,census=publication.boundary(boundary_root,'bc4e6c80b45666bc600c6fd39f465b35d5f02dbf6586233cab5ef32c053408bd',execution=publication.code())
    name='own_slot';binding=dict(old['binding'],architecture=name);branch=boundary['branches'][name]
    need(binding==dict(boundary['binding'],architecture=name),'ENERGY_CONTROL_BINDING')
    initial=rd.load(old_root,name+'/checkpoint-0.pt');saved=rd.load(old_root,name+'/checkpoint-2.pt')
    path=old_root/name/'checkpoint-2.pt';receipt=old['results'][name]['checkpoints']['2']
    prefix=boundary_root/name/'steps.jsonl';out=a.output/'continuation-controls';out.mkdir()
    engine=EnergyPlacementContinuation(bundle,target,binding,census)
    result={}
    def reject(label,call,code):
        try:call()
        except (ValueError,RuntimeError) as error:
            need(code in str(error),'ENERGY_CONTROL_WRONG_REFUSAL:'+label);result[label]=str(error)
        else:raise ValueError('ENERGY_CONTROL_ACCEPTED:'+label)
    before=fingerprint(engine.snapshot())
    reject('checkpoint_pin',lambda:engine.restore(path,dict(receipt,pin='0'*64),prefix,initial,cursor=2,chain=saved['chain']), 'CHECKPOINT_PIN')
    reject('cursor',lambda:engine.restore(path,receipt,prefix,initial,cursor=3,chain=saved['chain']), 'CHECKPOINT_BINDING')
    need(fingerprint(engine.snapshot())==before,'ENERGY_CONTROL_BAD_RESTORE_MUTATED')
    engine.restore(path,receipt,prefix,initial,cursor=2,chain=saved['chain'])
    need(type(engine.model).__name__=='EnergyBatchedForecaster','ENERGY_CONTROL_RESTORE_CLASS')
    before=fingerprint(engine.snapshot())
    bad=deepcopy(saved);state=next(iter(bad['optimizer']['state'].values()));state.pop('exp_avg')
    bad_path=out/'missing-moment.pt';bad_ref=original.measure.persist(bad_path,bad)
    reject('missing_moment',lambda:engine.reopen(bad_path,bad_ref),'OPTIMIZER_CURSOR')
    bad=deepcopy(saved);bad['noise_cursor']+=1
    bad_path=out/'wrong-noise.pt';bad_ref=original.measure.persist(bad_path,bad)
    reject('noise_cursor',lambda:engine.reopen(bad_path,bad_ref),'CHECKPOINT_CURSOR')
    need(fingerprint(engine.snapshot())==before,'ENERGY_CONTROL_BAD_REOPEN_MUTATED')
    prior_model=engine.model
    engine.model,_=original.runtime.make(bundle,target,name,'cpu')
    try:reject('wrong_implementation',engine.fresh,'IMPLEMENTATION')
    finally:engine.model=prior_model
    reject('wrong_device',lambda:EnergyPlacementContinuation(bundle,target,dict(binding,device='cuda'),census),'DEVICE')
    bad_binding=deepcopy(binding);bad_binding['target']['pin']='0'*64
    reject('wrong_target',lambda:EnergyPlacementContinuation(bundle,target,bad_binding,census),'TARGET')
    altered=deepcopy(target);other=EnergyPlacementContinuation(bundle,altered,binding,census)
    with torch.no_grad():next(altered.parameters()).add_(0)
    reject('target_version',other.fresh,'TARGET_CHANGED')
    tail_path=out/'bad-tail.jsonl';rows=(boundary_root/name/'retained-tail.jsonl').read_bytes().splitlines(keepends=True)
    first=json.loads(rows[0]);first['update']['loss']+=1;rows[0]=canonical(first)+b'\n';tail_path.write_bytes(b''.join(rows))
    tail_ref=deepcopy(branch['tail']);tail_ref.update(bytes=tail_path.stat().st_size,sha256=file_sha(tail_path))
    ledger_path=out/'refusal-ledger.jsonl';ledger_path.write_bytes(prefix.read_bytes());ledger_pin=file_sha(ledger_path)
    with ledger_path.open('ab') as ledger,Replay(tail_path,tail_ref,2) as replay:
        reject('tail_mismatch',lambda:engine.commit(items[2],ledger,replay),'REPLAY_MISMATCH')
    need(engine.poisoned and file_sha(ledger_path)==ledger_pin,'ENERGY_CONTROL_BAD_TAIL_COMMITTED')
    reject('poisoned_continuation',lambda:engine.update(items[2]),'POISONED')
    return dict(scope='retained_real_cpu_checkpoint',refusals=result,bad_restore_and_reopen_leave_state_unchanged=True,
        tail_mismatch_commits_no_bytes=True,failed_execution_poisons_continuation=True)


def run(a):
    began=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    roots=dict(original=a.original,candidate=a.candidate,cold=a.cold)
    reports={n:original.pinned(p,getattr(a,n+'_pin')) for n,p in roots.items()}
    for n,p in roots.items():rd.manifest(p,reports[n])
    old,new,cold=(reports[n] for n in ('original','candidate','cold'));execution=driver.code()
    need(all(r['complete'] for r in reports.values()) and old['binding']==new['binding']==cold['binding']
         and new['execution']['code']==cold['execution']['code']==execution
         and new['execution']['resume'] and cold['execution']['resume']
         and cold['previous_pin']==a.candidate_pin and new['pid']!=cold['pid'], 'PLACEMENT_MATCH_EXPERIMENT')
    need(old['results_pin']==new['results_pin']==cold['results_pin'],'PLACEMENT_MATCH_RESULTS_PIN')
    for name,p in roots.items():need(file_sha(p/'results.json')==reports[name]['results_pin'],'PLACEMENT_MATCH_RESULTS_BYTES')
    totals=dict(tensors=0,bytes=0);updates=targets=evaluations=0;details={}
    for name in original.runtime.original.ARCHITECTURES:
        old_rows=(a.original/name/'steps.jsonl').read_bytes()
        need(old_rows==(a.candidate/name/'steps.jsonl').read_bytes()==(a.cold/name/'steps.jsonl').read_bytes(),
             'PLACEMENT_MATCH_LEDGER_BYTES')
        result=old['results'][name];need(len(old_rows.splitlines())==result['updates'],'PLACEMENT_MATCH_UPDATE_COUNT')
        paths=sorted(p for p in old['files'] if p.startswith(name+'/') and p.endswith('.pt'))
        need(paths==sorted(p for p in new['files'] if p.startswith(name+'/') and p.endswith('.pt'))
             ==sorted(p for p in cold['files'] if p.startswith(name+'/') and p.endswith('.pt')),
             'PLACEMENT_MATCH_PHYSICAL_SCOPE')
        for rel in paths:
            expected=rd.load(a.original,rel)
            for root in (a.candidate,a.cold):
                c=rd.exact(expected,rd.load(root,rel))
                for k in totals:totals[k]+=c[k]
            if '/initial/' in rel or '/final/' in rel:evaluations+=len(expected)
        updates+=result['updates'];targets+=result['rows']
        details[name]=dict(updates=result['updates'],rows=result['rows'],files=len(paths),ledger=old['files'][name+'/steps.jsonl'])
    boundary,_=publication.boundary(a.boundary,a.boundary_pin,execution=publication.code())
    roles=['completed' if 'completed' in b else 'fresh' if b.get('fresh') else 'partial'
           for n in original.runtime.original.ARCHITECTURES for b in [boundary['branches'][n]]]
    need(roles==['completed','partial','fresh'],'MIXED_PLACEMENT_BRANCH_ROLES')
    need(new['execution']['completed_branches_reused']==cold['execution']['completed_branches_reused']==['shared'],
         'MIXED_PLACEMENT_COMPLETED_REUSE')
    need(new['execution']['implementation']==cold['execution']['implementation']=='energy','MIXED_PLACEMENT_IMPLEMENTATION')
    need(new['execution']['resume']==cold['execution']['resume']==a.boundary_pin,'PLACEMENT_MATCH_RESUME_SOURCE')
    expected_constructions=[]
    for name in original.runtime.original.ARCHITECTURES:
        expected_constructions.append(dict(architecture=name,kind='initial',step=0,model='EnergyBatchedForecaster'))
        if name!='whole_proposals':
            expected_constructions.append(dict(architecture=name,kind='restore',step=5 if name=='shared' else 2,model='EnergyBatchedForecaster'))
        for step in (range(3,6) if name=='own_slot' else range(1,6) if name=='whole_proposals' else ()):
            expected_constructions.append(dict(architecture=name,kind='reopen',step=step,model='EnergyBatchedForecaster'))
    need(new['candidate_class_at_all_boundaries'] and cold['candidate_class_at_all_boundaries']
         and new['constructions']==cold['constructions']==expected_constructions,
         'ENERGY_READBACK_CONSTRUCTIONS')
    continuation_controls=controls(a)
    negatives={}
    def reject(name,fn,code):
        try:fn()
        except (ValueError,RuntimeError) as error:
            need(code in str(error),'PLACEMENT_MATCH_WRONG_REFUSAL');negatives[name]=str(error)
        else:raise ValueError('PLACEMENT_MATCH_CONTROL_ACCEPTED:'+name)
    reject('wrong_boundary_pin',lambda:publication.boundary(a.boundary,'0'*64),'BOUNDARY_PIN')
    reject('wrong_execution',lambda:publication.boundary(a.boundary,a.boundary_pin,execution={}), 'PUBLICATION_EXECUTION')
    owner=dict(pid=os.getpid(),starttime=driver.identity(os.getpid()))
    reject('live_predecessor',lambda:driver.mixed.terminal_owners([owner]),'LIVE_PREDECESSOR')
    reject('reused_pid',lambda:driver.mixed.terminal_owners([dict(owner,starttime='0')]),'PID_REUSED')
    reject('missing_owners',lambda:driver.mixed.terminal_owners([]),'OWNERS_REQUIRED')
    reject('missing_cuda_runtime',lambda:driver.gpu_gate(argparse.Namespace(cpu_qualification=None,cpu_qualification_pin=None),execution,boundary),
           'CPU_CONTINUATION_REQUIRED')
    # Deliberate changes are made to independent copies of actual persisted inputs.
    for name,rel in (('donor_bytes','donors.json'),('checkpoint_bytes','shared/checkpoint-2.pt'),
                     ('evaluation_bytes',next(p for p in boundary['files'] if p.startswith('shared/initial/')))):
        root=a.output/name;root.mkdir();target=root/rel;target.parent.mkdir(parents=True,exist_ok=True)
        raw=bytearray((a.boundary/rel).read_bytes());raw[-1]^=1;target.write_bytes(raw)
        bad=deepcopy(boundary);bad['files']={rel:boundary['files'][rel]}
        pin=write_committed(root/'boundary.json',bad)
        reject(name,lambda root=root,pin=pin:publication.boundary(root,pin,execution=publication.code()),'PUBLICATION_FILE_PIN')
    root=a.output/'incomplete';root.mkdir();bad=deepcopy(boundary);bad['complete']=False
    pin=write_committed(root/'boundary.json',bad)
    reject('incomplete_boundary',lambda:publication.boundary(root,pin),'BOUNDARY_SCOPE')
    need(not torch.cuda.is_initialized(),'PLACEMENT_MATCH_UNEXPECTED_GPU')
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    report=dict(complete=True,original_pin=a.original_pin,candidate_pin=a.candidate_pin,cold_pin=a.cold_pin,
        boundary_pin=a.boundary_pin,device=new['binding']['device'],execution=execution,mixed_branch_roles=roles,
        same_complete_comparison=True,results_pin=new['results_pin'],updates=updates,targets=targets,
        implementation='energy',candidate_class_at_all_boundaries=True,constructions=expected_constructions,
        continuation_controls=continuation_controls,
        evaluation_rows=evaluations,details=details,comparison=totals,controls=negatives,files=files,
        seconds=time.perf_counter()-began,gpu_used=False,acceptance='author-checked',number_status='measured')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),comparison=totals,evaluation_rows=evaluations)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('original','candidate','cold','boundary'):
        p.add_argument('--'+n,type=Path,required=True);p.add_argument('--'+n+'-pin',required=True)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error)))
        raise
