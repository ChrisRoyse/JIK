"""Cold continuation replay, protected-state checks and matched recognition comparisons."""
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
from audit.capability_recognition_learning import prepare, frozen, ARMS
from audit.fsv_identity_loop import configure
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import take_lease, run_cli
from model.capability_kernel_learning import pinned, predict


def read(path):return json.loads(Path(path).read_bytes())


def native_match(left,right):
    a=F.normalize(left.values[0,~left.padding[0]],dim=-1)
    b=F.normalize(right.values[0,~right.padding[0]],dim=-1)
    scores=a@b.T
    return (scores.amax(0).mean()+scores.amax(1).mean())/2


def validate(corpus,rows,report,previous):
    need([(r['split'],r['task']) for r in rows]==[('train',k) for k in corpus.train]+[('query',k) for k in corpus.query],
         'RECOGNITION_READER_POPULATION')
    for row in rows:
        task=corpus.tasks[row['task']]; old=previous[row['task']]
        need(row['family']==task['family'] and row['group']==task['split_group']
             and row['label']==old['recognition']['label'] and row['parent_selected']==old['recognition']['selected']
             and 0<=row['selected']<len(corpus.regions),'RECOGNITION_READER_TARGET')
        if row['split']=='query':
            field=task['selections']['targets']['fields'][0]['id']; raw=corpus.sources[task['population']].population.fields[field]['text'].encode()
            need(row['source_sha256']==sha(raw) and row['source_bytes']==len(raw) and row['surface']['positions']==len(raw)+1
                 and row['target_surface']==old['scores']['target'] and row['parent_loss']==old['scores']['predicted']['loss'],
                 'RECOGNITION_READER_TARGET')
            need(set(row['lanes'])=={s.name for s in corpus.specs},'RECOGNITION_READER_ROSTER')
            for name,lane in row['lanes'].items():
                need(math.isfinite(lane['cosine']) and lane['parent_cosine']==old['lanes'][name]['conditions']['predicted']['coverage_cosine'],
                     'RECOGNITION_READER_LANES')
    for split in ('train','query'):
        for family in ('coding','conversation','reasoning'):
            selected=[r for r in rows if r['split']==split and r['family']==family]; c=report['families'][split][family]
            need(c['tasks']==len(selected) and c['groups']==len({r['group'] for r in selected})
                 and c['hits']==sum(r['label']==r['selected'] for r in selected)
                 and c['parent_hits']==sum(r['label']==r['parent_selected'] for r in selected),'RECOGNITION_READER_SUMMARY')
            if split=='query':
                delta=[r['surface']['loss']-r['parent_loss'] for r in selected]
                expected=dict(mean_loss=math.fsum(r['surface']['loss'] for r in selected)/len(selected),
                    mean_delta=math.fsum(delta)/len(delta),better=sum(x<0 for x in delta),worse=sum(x>0 for x in delta),ties=sum(x==0 for x in delta))
                need(all(c[k]==v for k,v in expected.items()),'RECOGNITION_READER_SUMMARY')
                for name,lane in c['lanes'].items():
                    delta=[r['lanes'][name]['cosine']-r['lanes'][name]['parent_cosine'] for r in selected]
                    expected=dict(mean_cosine=math.fsum(r['lanes'][name]['cosine'] for r in selected)/len(selected),
                        mean_delta=math.fsum(delta)/len(delta),better=sum(x>0 for x in delta),worse=sum(x<0 for x in delta),ties=sum(x==0 for x in delta))
                    need(expected==lane,'RECOGNITION_READER_SUMMARY')


def independent_update(model,optimizer,corpus,previous,row):
    optimizer.zero_grad(set_to_none=True); world=model['world']; key=row['task']; kind=corpus.tasks[key]['kind']
    native=corpus.batch(key,'context',world.config,'cuda')
    output=predict(world,model['request'],native,key=key,operation=kind)
    _,target=corpus.target(row['target'],world.config,'cuda',fitting=True)
    label=torch.tensor([previous[row['target']]['recognition']['label']],device='cuda')
    chosen=output['predicted_constellation']
    if row['mode']!='legacy':
        chosen=world.supervised_placement(output,label,allowed=torch.ones(len(corpus.regions),dtype=torch.bool,device='cuda'),
                                         event_ids=[digest(dict(key=key,operation=kind))])
    region=[]; placement=[]
    for spec in world.specs:
        r=F.cross_entropy(output['regions'][spec.name],label)
        q=1-native_match(chosen.slots[spec.name],target[spec.name])
        need(dict(region=float(r.detach()),placement=float(q.detach()))==row['objective']['lanes'][spec.name],
             'RECOGNITION_READER_OBJECTIVE')
        region.append(r); placement.append(q)
    loss=torch.stack(region).mean()+torch.stack(placement).mean()
    need(float(loss.detach())==row['objective']['loss'] and chosen.regions[0]==row['objective']['placement_region'],
         'RECOGNITION_READER_LOSS')
    loss.backward()
    for branch,evidence in row['gradients'].items():
        values={n:p.grad for n,p in model.named_parameters() if p.grad is not None
            and next((s.name for s in world.specs if '.'+s.name+'.' in n),'shared')==branch}
        need(fingerprint(values)==evidence['sha256'],'RECOGNITION_READER_GRADIENT')
    optimizer.step()


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--experiments',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args(); need(not a.output.exists(),'RECOGNITION_READER_EXISTS')
    refs=read(a.experiments); need(set(refs)==set(ARMS),'RECOGNITION_READER_ARMS')
    reports={n:pinned(r) for n,r in refs.items()}; source=reports['legacy']['source']
    need(all(r['source']==source for r in reports.values()) and source['code_sha256']==sha(Path(__file__).with_name('capability_recognition_learning.py').read_bytes()),
         'RECOGNITION_READER_SOURCE')
    gpu=subprocess.check_output(['nvidia-smi'],text=True); take_lease('29','capability-recognition-readback'); configure()
    cases=[]; query_cases=[]; controls=[]; all_rows={}
    for arm,report in reports.items():
        model,optimizer,corpus,previous,parent,saved=prepare(source['diagnostic'],source['readback'])
        baseline=frozen(model,optimizer); root=Path(refs[arm]['path']).parent; plan=read(root/'plan.json')
        need(plan['source']==source and plan['arm']==arm and plan['train']==list(corpus.train) and plan['query']==list(corpus.query)
             and report['initial_optimizer']==saved['optimizer_hash'],'RECOGNITION_READER_PLAN')
        raw=(root/'steps.jsonl').read_bytes(); need(sha(raw)==report['steps_sha256'],'RECOGNITION_READER_CHAIN')
        steps=[json.loads(x) for x in raw.splitlines()]; need(len(steps)==len(corpus.train),'RECOGNITION_READER_STEPS')
        chain=sha((root/'initial.pt').read_bytes()); current=saved['parameter_hash']
        for index,row in enumerate(steps):
            target=plan['mapping'][corpus.train[index]] if arm=='shuffled' else corpus.train[index]
            need(row['index']==index and row['task']==corpus.train[index] and row['target']==target and row['mode']==arm
                 and row['previous']==chain and row['before']==current
                 and digest({k:v for k,v in row.items() if k!='sha256'})==row['sha256'],'RECOGNITION_READER_CHAIN')
            chain,current=row['sha256'],row['after']
        need(current==report['final_parameters'] and sha((root/'final.pt').read_bytes())==report['final_checkpoint'],'RECOGNITION_READER_FINAL')
        for index in plan['boundaries']:
            before=torch.load(root/('initial.pt' if index==0 else f'before-{index}.pt'),map_location='cpu',weights_only=True)
            after=torch.load(root/f'after-{index}.pt',map_location='cpu',weights_only=True)
            need(before['source']==after['source']==source and before['updates']==report['parent_updates']+index
                 and after['updates']==before['updates']+1,'RECOGNITION_READER_CHECKPOINT')
            model.load_state_dict(before['model'],strict=True); optimizer.load_state_dict(before['optimizer'])
            need(frozen(model,optimizer)==baseline and fingerprint(optimizer.state_dict())==before['optimizer_hash'],'RECOGNITION_READER_BEFORE')
            independent_update(model,optimizer,corpus,previous,steps[index])
            need(fingerprint(model.state_dict())==fingerprint(after['model']) and fingerprint(optimizer.state_dict())==after['optimizer_hash']
                 and frozen(model,optimizer)==baseline,'RECOGNITION_READER_AFTER')
            cases.append(dict(arm=arm,index=index,family=steps[index]['family'],loss_gradients_parameters_moments_exact=True))
            if len(cases)==1:
                damaged=copy.deepcopy(before['optimizer']); next(iter(damaged['state'].values()))['exp_avg'].reshape(-1)[0]+=1
                controls.append(refused('RECOGNITION_READER_MOMENT',lambda:need(fingerprint(damaged)==before['optimizer_hash'],'RECOGNITION_READER_MOMENT')))
                controls.append(refused('LEARNING_TARGET_SPLIT',lambda:corpus.target(corpus.query[0],model['world'].config,'cuda',fitting=True)))
        final=torch.load(root/'final.pt',map_location='cpu',weights_only=True)
        need(fingerprint(model.state_dict())==fingerprint(final['model']) and frozen(model,optimizer)==report['frozen_identity'],
             'RECOGNITION_READER_FINAL_STATE')
        raw=(root/'evaluation.jsonl').read_bytes(); need(sha(raw)==report['evaluation_sha256'],'RECOGNITION_READER_EVALUATION')
        records=[json.loads(x) for x in raw.splitlines()]; validate(corpus,records,report,previous); all_rows[arm]=records
        with torch.no_grad():
            for family in ('coding','conversation','reasoning'):
                row=min((r for r in records if r['split']=='query' and r['family']==family),key=lambda r:r['task'])
                key=row['task']; world=model['world']; native=corpus.batch(key,'context',world.config,'cuda')
                output=predict(world,model['request'],native,key=key,operation=corpus.tasks[key]['kind'])
                need(int(next(iter(output['world_owner'].values()))[0])==row['selected'],'RECOGNITION_READER_REGION')
                raw,target=corpus.target(key,world.config,'cuda',fitting=False); predicted=output['predicted_constellation'].slots
                for name,slots in (('surface',predicted),('target_surface',target)):
                    measured=independent_score(model['decoder'],raw,slots,skipped=False,size=row[name]['physical_batch'])
                    need(measured==row[name]['loss'],'RECOGNITION_READER_SCORE')
                for name,slot in predicted.items():
                    need(float(native_match(slot,target[name]))==row['lanes'][name]['cosine'],'RECOGNITION_READER_GEOMETRY')
                query_cases.append(dict(arm=arm,family=family,task=key,region_and_both_surface_scores_exact=True))
    for mutation,code in (('missing','POPULATION'),('target','TARGET'),('score','SUMMARY')):
        damaged=copy.deepcopy(all_rows['supervised'])
        if mutation=='missing':damaged.pop()
        elif mutation=='target':damaged[0]['label']=-1
        else:next(r for r in damaged if r['split']=='query')['surface']['loss']+=1
        controls.append(refused('RECOGNITION_READER_'+code,lambda:validate(corpus,damaged,reports['supervised'],previous)))
    comparisons={}
    for family in ('coding','conversation','reasoning'):
        by_arm={n:[r for r in rows if r['split']=='query' and r['family']==family] for n,rows in all_rows.items()}
        comparisons[family]={}
        for arm in ('legacy','shuffled'):
            delta=[r['surface']['loss']-s['surface']['loss'] for r,s in zip(by_arm['supervised'],by_arm[arm])]
            comparisons[family]['supervised_vs_'+arm]=dict(mean_delta=math.fsum(delta)/len(delta),better=sum(x<0 for x in delta),
                worse=sum(x>0 for x in delta),ties=sum(x==0 for x in delta))
    result=dict(schema='capability-recognition-readback-v1',sources=refs,cases=cases,query_cases=query_cases,controls=controls,
        all_steps=sum(r['new_updates'] for r in reports.values()),all_evaluation_rows=sum(len(v) for v in all_rows.values()),
        comparisons=comparisons,protected_state_exact=True,parent_optimizer_carried=True,gpu_before=gpu,
        number_status='measured',acceptance='author-checked',scientific_admission=False,learned_capability=False,serving=False)
    print(json.dumps(dict(pin=write_committed(a.output,result),comparisons=comparisons)),flush=True)


if __name__=='__main__':run_cli(main)
