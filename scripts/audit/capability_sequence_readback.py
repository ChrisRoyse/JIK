"""Cold matched prefix learning readback: source path, gradients, moments and outcomes."""
import argparse
import copy
from dataclasses import replace
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
from audit.capability_recognition_learning import prepare, frozen
from audit.capability_recognition_readback import validate, native_match
from audit.fsv_identity_loop import configure
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import take_lease, run_cli
from model.capability_kernel_learning import pinned, predict
from model.capability_sequence import source_prefix


def forward(model,corpus,key,arm,*,fitting):
    need(not fitting or key in corpus.train,'SEQUENCE_READER_FIT_QUERY')
    world=model['world'];task=corpus.tasks[key];native=corpus.batch(key,'context',world.config,'cuda')
    if arm=='baseline' or task['kind']!='continue_recorded_steps':
        return predict(world,model['request'],native,key=key,operation=task['kind'])
    history=source_prefix(corpus,key,world.config,'cuda')
    if fitting and arm=='reversed':history=replace(history,frames=history.frames[::-1],rows=history.rows[::-1])
    state=model['request'](world.start('capability:'+key),operation=task['kind'])
    return world(native,(state,),allowed=torch.ones(len(corpus.regions),dtype=torch.bool,device='cuda'),phase=0,
        event_ids=[digest(dict(key=key,operation=task['kind']))],evidence_origin=['source_observation'],advance_phase=False,history=history)


def replay(model,optimizer,corpus,previous,row):
    key=row['task'];optimizer.zero_grad(set_to_none=True);world=model['world']
    output=forward(model,corpus,key,row['arm'],fitting=True)
    _,target=corpus.target(key,world.config,'cuda',fitting=True)
    label=torch.tensor([previous[key]['recognition']['label']],device='cuda')
    chosen=world.supervised_placement(output,label,allowed=torch.ones(len(corpus.regions),dtype=torch.bool,device='cuda'),
        event_ids=[digest(dict(key=key,operation=corpus.tasks[key]['kind']))])
    regions=[];placements=[]
    for spec in world.specs:
        r=F.cross_entropy(output['regions'][spec.name],label);p=1-native_match(chosen.slots[spec.name],target[spec.name])
        need(dict(region=float(r.detach()),placement=float(p.detach()))==row['objective']['lanes'][spec.name],'SEQUENCE_READER_OBJECTIVE')
        regions.append(r);placements.append(p)
    loss=torch.stack(regions).mean()+torch.stack(placements).mean()
    need(float(loss.detach())==row['objective']['loss'] and int(label)==row['objective']['label']
         and chosen.regions[0]==row['objective']['placement_region'],'SEQUENCE_READER_LOSS')
    loss.backward()
    for branch,evidence in row['gradients'].items():
        values={n:p.grad for n,p in model.named_parameters() if p.grad is not None
                and next((s.name for s in world.specs if '.'+s.name+'.' in n),'shared')==branch}
        need(fingerprint(values)==evidence['sha256'],'SEQUENCE_READER_GRADIENT')
    optimizer.step()


def summarize(records):
    result={}
    for split in ('train','query'):
        selected=[r for r in records if r['split']==split and r['sequence']]
        value=dict(tasks=len(selected),hits=sum(r['selected']==r['label'] for r in selected),
            parent_hits=sum(r['parent_selected']==r['label'] for r in selected))
        if split=='query':
            value.update(mean_loss=math.fsum(r['surface']['loss'] for r in selected)/len(selected),
                parent_mean_loss=math.fsum(r['parent_loss'] for r in selected)/len(selected))
        result[split]=value
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiments',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();need(not a.output.exists(),'SEQUENCE_READER_EXISTS')
    refs=json.loads(a.experiments.read_bytes());need(set(refs)=={'baseline','ordered','reversed'},'SEQUENCE_READER_ARMS')
    reports={n:pinned(r) for n,r in refs.items()};source=reports['baseline']['source']
    need(all(r['source']==source for r in reports.values()),'SEQUENCE_READER_SOURCE')
    for name,pin in source['code'].items():need(sha((Path(__file__).resolve().parents[1]/name).read_bytes())==pin,'SEQUENCE_READER_CODE')
    gpu=subprocess.check_output(['nvidia-smi'],text=True);take_lease('29','capability-sequence-readback');configure()
    cases=[];queries=[];controls=[];all_rows={};summaries={}
    for arm,report in reports.items():
        model,optimizer,corpus,previous,parent,saved=prepare(source['diagnostic'],source['readback']);baseline=frozen(model,optimizer)
        root=Path(refs[arm]['path']).parent;plan=json.loads((root/'plan.json').read_bytes())
        train=[k for k in corpus.train if corpus.tasks[k]['kind']=='continue_recorded_steps']
        query=[k for k in corpus.query if corpus.tasks[k]['kind']=='continue_recorded_steps']
        need(plan['source']==source and plan['arm']==arm and plan['train']==train and plan['query']==query
             and plan['all_train']==list(corpus.train) and plan['all_query']==list(corpus.query)
             and report['initial_optimizer']==saved['optimizer_hash'] and report['new_updates']==len(train),'SEQUENCE_READER_PLAN')
        raw=(root/'steps.jsonl').read_bytes();need(sha(raw)==report['steps_sha256'],'SEQUENCE_READER_STEPS')
        steps=[json.loads(x) for x in raw.splitlines()];need(len(steps)==len(train),'SEQUENCE_READER_ROSTER')
        chain=sha((root/'initial.pt').read_bytes());current=saved['parameter_hash']
        for index,row in enumerate(steps):
            need(row['index']==index and row['task']==train[index] and row['arm']==arm and row['previous']==chain
                 and row['before']==current and digest({k:v for k,v in row.items() if k!='sha256'})==row['sha256'],'SEQUENCE_READER_CHAIN')
            chain,current=row['sha256'],row['after']
        need(current==report['final_parameters'] and sha((root/'final.pt').read_bytes())==report['final_checkpoint'],'SEQUENCE_READER_FINAL')
        for index in plan['boundaries']:
            before=torch.load(root/('initial.pt' if index==0 else f'before-{index}.pt'),map_location='cpu',weights_only=True)
            after=torch.load(root/f'after-{index}.pt',map_location='cpu',weights_only=True)
            need(before['source']==after['source']==source and before['updates']==report['parent_updates']+index
                 and after['updates']==before['updates']+1,'SEQUENCE_READER_CHECKPOINT')
            model.load_state_dict(before['model'],strict=True);optimizer.load_state_dict(before['optimizer'])
            need(fingerprint(optimizer.state_dict())==before['optimizer_hash'] and frozen(model,optimizer)==baseline,'SEQUENCE_READER_BEFORE')
            replay(model,optimizer,corpus,previous,steps[index])
            need(fingerprint(model.state_dict())==fingerprint(after['model']) and fingerprint(optimizer.state_dict())==after['optimizer_hash']
                 and frozen(model,optimizer)==baseline,'SEQUENCE_READER_AFTER')
            cases.append(dict(arm=arm,index=index,objective_gradients_parameters_moments_exact=True))
        final=torch.load(root/'final.pt',map_location='cpu',weights_only=True)
        need(fingerprint(model.state_dict())==fingerprint(final['model']) and frozen(model,optimizer)==report['frozen_identity'],'SEQUENCE_READER_FINAL_STATE')
        raw=(root/'evaluation.jsonl').read_bytes();need(sha(raw)==report['evaluation_sha256'],'SEQUENCE_READER_EVALUATION')
        records=[json.loads(x) for x in raw.splitlines()];validate(corpus,records,report,previous)
        need(all(r['sequence']==(corpus.tasks[r['task']]['kind']=='continue_recorded_steps') for r in records),'SEQUENCE_READER_SUBSET')
        all_rows[arm]=records;summaries[arm]=summarize(records)
        with torch.no_grad():
            chosen={min(r['task'] for r in records if r['split']=='query' and r['family']==family) for family in ('coding','conversation','reasoning')}
            chosen.add(min(query))
            for key in sorted(chosen):
                row=next(r for r in records if r['task']==key);output=forward(model,corpus,key,arm,fitting=False)
                need(int(next(iter(output['world_owner'].values()))[0])==row['selected'],'SEQUENCE_READER_REGION')
                raw,target=corpus.target(key,model['world'].config,'cuda',fitting=False);pred=output['predicted_constellation'].slots
                for name,slots in (('surface',pred),('target_surface',target)):
                    need(independent_score(model['decoder'],raw,slots,skipped=False,size=row[name]['physical_batch'])==row[name]['loss'],
                         'SEQUENCE_READER_SCORE')
                for name,slot in pred.items():need(float(native_match(slot,target[name]))==row['lanes'][name]['cosine'],'SEQUENCE_READER_GEOMETRY')
                queries.append(dict(arm=arm,task=key,family=row['family'],sequence=row['sequence'],surface_and_geometry_exact=True))
        need(sha(Path(corpus.config['journal']['path']).read_bytes())==corpus.config['journal']['pin'],'SEQUENCE_READER_JOURNAL')
    for mode,code in (('missing','POPULATION'),('target','TARGET'),('score','SUMMARY')):
        damaged=copy.deepcopy(all_rows['ordered'])
        if mode=='missing':damaged.pop()
        elif mode=='target':damaged[0]['label']=-1
        else:next(r for r in damaged if r['split']=='query')['surface']['loss']+=1
        controls.append(refused('RECOGNITION_READER_'+code,lambda:validate(corpus,damaged,reports['ordered'],previous)))
    controls.append(refused('SEQUENCE_READER_FIT_QUERY',lambda:forward(model,corpus,query[0],'ordered',fitting=True)))
    damaged=copy.deepcopy(final['optimizer']);next(iter(damaged['state'].values()))['exp_avg'].reshape(-1)[0]+=1
    controls.append(refused('SEQUENCE_READER_MOMENT',lambda:need(fingerprint(damaged)==final['optimizer_hash'],'SEQUENCE_READER_MOMENT')))
    comparisons={}
    for scope in ('prefix','coding','conversation','reasoning'):
        selected={n:[r for r in records if r['split']=='query' and (r['sequence'] if scope=='prefix' else r['family']==scope)]
                  for n,records in all_rows.items()}
        comparisons[scope]={}
        for arm in ('baseline','reversed'):
            delta=[r['surface']['loss']-s['surface']['loss'] for r,s in zip(selected['ordered'],selected[arm])]
            comparisons[scope]['ordered_vs_'+arm]=dict(mean_delta=math.fsum(delta)/len(delta),better=sum(x<0 for x in delta),
                worse=sum(x>0 for x in delta),ties=sum(x==0 for x in delta),tasks=len(delta))
    result=dict(schema='capability-sequence-readback-v1',sources=refs,cases=cases,queries=queries,controls=controls,
        all_steps=sum(r['new_updates'] for r in reports.values()),all_evaluation_rows=sum(len(r) for r in all_rows.values()),
        prefix=summaries,comparisons=comparisons,protected_state_exact=True,parent_optimizer_carried=True,gpu_before=gpu,
        number_status='measured',acceptance='author-checked',scientific_admission=False,learned_capability=False,serving=False)
    print(json.dumps(dict(pin=write_committed(a.output,result),prefix=summaries,comparisons=comparisons)),flush=True)


if __name__=='__main__':run_cli(main)
