"""Cold context-only continuation readback: protected state, updates and matched outcomes."""
import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from torch.nn import functional as F
from audit.capability_conditioning import fingerprint
from audit.capability_conditioning_readback import refused, independent_score
from audit.capability_kernel_learning import parameter_hash
from audit.capability_order_diagnostic import snapshot
from audit.capability_recognition_learning import frozen
from audit.capability_recognition_readback import validate, native_match
from audit.capability_structure_learning import restore
from audit.capability_structure_readback import independent_context
from audit.fsv_identity_loop import configure
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import take_lease, run_cli
from model.capability_kernel_learning import pinned
from model.capability_structure import predict_structured, ARMS
from model.capability_structure_input import StructuredContexts


def replay(model,optimizer,contexts,previous,row):
    corpus=contexts.corpus;key=row['task'];need(key in corpus.train,'STRUCTURE_COLD_FIT_QUERY')
    optimizer.zero_grad(set_to_none=True);world=model['world'];kind=corpus.tasks[key]['kind']
    output=predict_structured(model,contexts.get(key,world.config,'cuda'),kind,mode=row['arm'])
    _,target=corpus.target(key,world.config,'cuda',fitting=True)
    label=torch.tensor([previous[key]['recognition']['label']],device='cuda')
    chosen=world.supervised_placement(output,label,allowed=torch.ones(len(corpus.regions),dtype=torch.bool,device='cuda'),
        event_ids=[digest(dict(key=key,operation=kind))])
    regions=[];placements=[]
    for spec in world.specs:
        r=F.cross_entropy(output['regions'][spec.name],label);p=1-native_match(chosen.slots[spec.name],target[spec.name])
        need(dict(region=float(r.detach()),placement=float(p.detach()))==row['objective']['lanes'][spec.name],'STRUCTURE_COLD_OBJECTIVE')
        regions.append(r);placements.append(p)
    loss=torch.stack(regions).mean()+torch.stack(placements).mean()
    need(float(loss.detach())==row['objective']['loss'] and int(label)==row['objective']['label']
         and chosen.regions[0]==row['objective']['placement_region'],'STRUCTURE_COLD_LOSS')
    loss.backward();need(all(p.grad is None for p in model.parameters() if not p.requires_grad),'STRUCTURE_COLD_PARENT_GRADIENT')
    for branch,evidence in row['gradients'].items():
        values={n:p.grad for n,p in model.named_parameters() if p.grad is not None
                and next((s.name for s in world.specs if '.'+s.name+'.' in n),'shared')==branch}
        parameters={n:p for n,p in model.named_parameters()
                    if next((s.name for s in world.specs if '.'+s.name+'.' in n),'shared')==branch}
        need(fingerprint(values)==evidence['sha256'] and fingerprint(parameters)==evidence['parameters'],'STRUCTURE_COLD_GRADIENT')
    optimizer.step()


def compare(all_rows):
    comparisons={}
    for family in ('coding','conversation','reasoning'):
        selected={n:[r for r in rows if r['split']=='query' and r['family']==family] for n,rows in all_rows.items()}
        comparisons[family]={}
        for arm in ('boundaries','mismatched'):
            left,right=selected['structured'],selected[arm]
            need([r['task'] for r in left]==[r['task'] for r in right],'STRUCTURE_COLD_MATCH')
            delta=[r['surface']['loss']-s['surface']['loss'] for r,s in zip(left,right)]
            groups={g:[d for d,r in zip(delta,left) if r['group']==g] for g in sorted({r['group'] for r in left})}
            comparisons[family]['structured_vs_'+arm]=dict(mean_delta=math.fsum(delta)/len(delta),better=sum(x<0 for x in delta),
                worse=sum(x>0 for x in delta),ties=sum(x==0 for x in delta),tasks=len(delta),
                groups={g:dict(tasks=len(ds),mean_delta=math.fsum(ds)/len(ds)) for g,ds in groups.items()})
    return comparisons


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiments',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();need(not a.output.exists(),'STRUCTURE_COLD_EXISTS')
    refs=json.loads(a.experiments.read_bytes());need(set(refs)==set(ARMS),'STRUCTURE_COLD_ARMS')
    reports={n:pinned(refs[n]) for n in ARMS};source=reports['boundaries']['source']
    need(all(r['source']==source for r in reports.values()) and source['code_sha256']==sha(Path(__file__).with_name('capability_structure_learning.py').read_bytes()),
         'STRUCTURE_COLD_SOURCE')
    gpu=subprocess.check_output(['nvidia-smi'],text=True);take_lease('29','capability-structure-learning-readback');configure()
    cases=[];queries=[];controls=[];all_rows={};cost={};fixed={}
    for arm,report in reports.items():
        model,optimizer,corpus,previous,parent,saved=restore(source['structure'],source['readback'])
        baseline=frozen(model,optimizer);contexts=StructuredContexts(corpus)
        root=Path(refs[arm]['path']).parent;plan=json.loads((root/'plan.json').read_bytes())
        need(plan['source']==source and plan['arm']==arm and plan['train']==list(corpus.train) and plan['query']==list(corpus.query)
             and plan['catalogue']==contexts.identity and plan['census']==contexts.census()
             and plan['trainable']==[n for n,p in model.named_parameters() if p.requires_grad]
             and plan['protected']==[n for n,p in model.named_parameters() if not p.requires_grad]
             and report['initial_optimizer']==saved['optimizer_hash'] and report['new_updates']==len(corpus.train),'STRUCTURE_COLD_PLAN')
        fixed[arm]=[]
        for key in corpus.train:
            item=contexts.get(key,model['world'].config,'cuda')
            if torch.equal(item.field_metadata,item.field_metadata.roll(1,0)) and all(torch.equal(x,x.flip(0)) for values in item.windows.values() for x in values):
                fixed[arm].append(key)
        need(fixed[arm]==plan['control_fixed_tasks'],'STRUCTURE_COLD_CONTROL_FIXED')
        raw=(root/'steps.jsonl').read_bytes();need(sha(raw)==report['steps_sha256'],'STRUCTURE_COLD_STEPS')
        steps=[json.loads(x) for x in raw.splitlines()];need(len(steps)==len(corpus.train),'STRUCTURE_COLD_ROSTER')
        initial=torch.load(root/'initial.pt',map_location='cpu',weights_only=True)
        need(fingerprint(initial['model'])==fingerprint(model.state_dict()) and initial['optimizer_hash']==fingerprint(optimizer.state_dict())
             and fingerprint(initial['optimizer'])==initial['optimizer_hash'] and initial['updates']==saved['updates'], 'STRUCTURE_COLD_INITIAL')
        chain=sha((root/'initial.pt').read_bytes());current=parameter_hash(model)
        for index,row in enumerate(steps):
            need(row['index']==index and row['task']==corpus.train[index] and row['arm']==arm and row['previous']==chain
                 and row['before']==current and digest({k:v for k,v in row.items() if k!='sha256'})==row['sha256'],'STRUCTURE_COLD_CHAIN')
            chain,current=row['sha256'],row['after']
        need(current==report['final_parameters'] and sha((root/'final.pt').read_bytes())==report['final_checkpoint'],'STRUCTURE_COLD_FINAL')
        expected_boundaries=sorted({0,len(corpus.train)-1,*[next(i for i,k in enumerate(corpus.train) if corpus.tasks[k]['family']==f)
            for f in ('coding','conversation','reasoning')]})
        need(plan['boundaries']==expected_boundaries,'STRUCTURE_COLD_BOUNDARIES')
        for index in expected_boundaries:
            before=torch.load(root/('initial.pt' if index==0 else f'before-{index}.pt'),map_location='cpu',weights_only=True)
            after=torch.load(root/f'after-{index}.pt',map_location='cpu',weights_only=True)
            need(before['source']==after['source']==source and before['updates']==report['parent_updates']+index
                 and after['updates']==before['updates']+1 and after['previous']==steps[index]['sha256'],'STRUCTURE_COLD_CHECKPOINT')
            model.load_state_dict(before['model'],strict=True);optimizer.load_state_dict(before['optimizer'])
            need(parameter_hash(model)==steps[index]['before']==before['parameter_hash']
                 and fingerprint(optimizer.state_dict())==before['optimizer_hash'] and frozen(model,optimizer)==baseline,'STRUCTURE_COLD_BEFORE')
            replay(model,optimizer,contexts,previous,steps[index])
            need(fingerprint(model.state_dict())==fingerprint(after['model']) and fingerprint(optimizer.state_dict())==after['optimizer_hash']
                 and fingerprint(after['optimizer'])==after['optimizer_hash'] and parameter_hash(model)==steps[index]['after']
                 and frozen(model,optimizer)==baseline,'STRUCTURE_COLD_AFTER')
            cases.append(dict(arm=arm,index=index,objective_gradients_parameters_moments_exact=True))
        final=torch.load(root/'final.pt',map_location='cpu',weights_only=True)
        need(fingerprint(model.state_dict())==fingerprint(final['model']) and fingerprint(optimizer.state_dict())==final['optimizer_hash']
             and frozen(model,optimizer)==report['frozen_identity'],'STRUCTURE_COLD_FINAL_STATE')
        raw=(root/'evaluation.jsonl').read_bytes();need(sha(raw)==report['evaluation_sha256'],'STRUCTURE_COLD_EVALUATION')
        records=[json.loads(x) for x in raw.splitlines()];validate(corpus,records,report,previous);all_rows[arm]=records
        mode='boundaries' if arm=='boundaries' else 'structured'
        with torch.no_grad():
            for family in ('coding','conversation','reasoning'):
                key=min(r['task'] for r in records if r['split']=='query' and r['family']==family)
                row=next(r for r in records if r['task']==key);item=contexts.get(key,model['world'].config,'cuda')
                actual=predict_structured(model,item,corpus.tasks[key]['kind'],mode=mode)
                output,residual=independent_context(model,item,corpus.tasks[key]['kind'],mode)
                need(torch.equal(residual,actual['structural_context']) and fingerprint(snapshot(output))==fingerprint(snapshot(actual)),'STRUCTURE_COLD_ATTENTION')
                need(int(next(iter(output['world_owner'].values()))[0])==row['selected'],'STRUCTURE_COLD_REGION')
                raw,target=corpus.target(key,model['world'].config,'cuda',fitting=False);pred=output['predicted_constellation'].slots
                for name,slots in (('surface',pred),('target_surface',target)):
                    need(independent_score(model['decoder'],raw,slots,skipped=False,size=row[name]['physical_batch'])==row[name]['loss'],'STRUCTURE_COLD_SCORE')
                for name,slot in pred.items():need(float(native_match(slot,target[name]))==row['lanes'][name]['cosine'],'STRUCTURE_COLD_GEOMETRY')
                queries.append(dict(arm=arm,task=key,family=family,context_region_surface_geometry_exact=True))
        need(sha(Path(corpus.config['journal']['path']).read_bytes())==corpus.config['journal']['pin'],'STRUCTURE_COLD_JOURNAL')
        cpu=math.fsum(r['cpu_seconds'] for r in steps);wall=math.fsum(r['seconds'] for r in steps)
        cost[arm]=dict(step_cpu_seconds=cpu,step_wall_seconds=wall,cpu_wall_ratio=cpu/wall,
            phases=report['phases'],seconds=report['seconds'],peak_cuda_bytes=report['peak_cuda_bytes'])
    for mode,code in (('missing','POPULATION'),('target','TARGET'),('score','SUMMARY')):
        damaged=copy.deepcopy(all_rows['structured'])
        if mode=='missing':damaged.pop()
        elif mode=='target':damaged[0]['label']=-1
        else:next(r for r in damaged if r['split']=='query')['surface']['loss']+=1
        controls.append(refused('RECOGNITION_READER_'+code,lambda:validate(corpus,damaged,reports['structured'],previous)))
    controls.append(refused('STRUCTURE_COLD_FIT_QUERY',lambda:replay(model,optimizer,contexts,previous,dict(task=corpus.query[0]))))
    damaged=copy.deepcopy(final['optimizer']);next(iter(damaged['state'].values()))['exp_avg'].reshape(-1)[0]+=1
    controls.append(refused('STRUCTURE_COLD_MOMENT',lambda:need(fingerprint(damaged)==final['optimizer_hash'],'STRUCTURE_COLD_MOMENT')))
    result=dict(schema='capability-structure-learning-readback-v1',sources=refs,cases=cases,queries=queries,controls=controls,
        reader_sha256=sha(Path(__file__).read_bytes()),all_steps=sum(r['new_updates'] for r in reports.values()),
        all_evaluation_rows=sum(len(r) for r in all_rows.values()),comparisons=compare(all_rows),control_fixed_tasks=fixed,cost=cost,
        protected_state_exact=True,parent_optimizer_carried=True,gpu_before=gpu,number_status='measured',acceptance='author-checked',
        scientific_admission=False,learned_capability=False,serving=False,exposure='development_exposed_group_holdout')
    print(json.dumps(dict(pin=write_committed(a.output,result),comparisons=result['comparisons'])),flush=True)


if __name__=='__main__':run_cli(main)
