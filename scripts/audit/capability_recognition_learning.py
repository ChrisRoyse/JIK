"""Matched region/placement continuation with protected native-to-surface decoding."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from torch.nn import functional as F
from audit.capability_conditioning import fingerprint, score
from audit.capability_kernel_learning import shuffled, branch_evidence, parameter_hash, save_checkpoint
from audit.capability_region_diagnostic import load
from audit.fsv_identity_loop import configure
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import take_lease, run_cli
from model.capability_kernel_learning import pinned, predict
from model.capability_objective import native_similarity


ARMS=('legacy','supervised','shuffled')


def prepare(diagnostic_ref, readback_ref):
    diagnostic=pinned(diagnostic_ref); readback=pinned(readback_ref)
    need(readback['source']==diagnostic_ref and readback['schema']=='capability-region-readback-v3', 'RECOGNITION_READBACK')
    root=Path(diagnostic_ref['path']).parent; raw=(root/'tasks.jsonl').read_bytes()
    need(sha(raw)==diagnostic['tasks_sha256'],'RECOGNITION_LABEL_CACHE')
    rows={r['task']:r for r in map(json.loads,raw.splitlines())}
    model,corpus,parent,prior=load(diagnostic['plan']['source'])
    need(fingerprint(model.state_dict())==diagnostic['model_identity'] and set(rows)==set(corpus.train)|set(corpus.query),
         'RECOGNITION_PARENT')
    path=Path(diagnostic['plan']['source']['path']).parent/'final.pt'
    saved=torch.load(path,map_location='cpu',weights_only=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=corpus.config['learning_rate'],
                               weight_decay=corpus.config['weight_decay'],foreach=False)
    optimizer.load_state_dict(saved['optimizer'])
    need(fingerprint(optimizer.state_dict())==saved['optimizer_hash'],'RECOGNITION_MOMENTS')
    model['world'].requires_grad_(True); model['request'].requires_grad_(True)
    # Reciprocal inputs share the real world projectors. Freeze those aliases too;
    # this protects the entire prior reciprocal function, not just its output heads.
    model['reciprocal'].requires_grad_(False); model['decoder'].requires_grad_(False)
    return model,optimizer,corpus,rows,parent,saved


def frozen(model,optimizer):
    return fingerprint(dict(buffers=dict(model.named_buffers()),
        weights={n:p for n,p in model.named_parameters() if not p.requires_grad},
        moments={n:optimizer.state.get(p,{}) for n,p in model.named_parameters() if not p.requires_grad}))


def objective(model,corpus,rows,key,target_key,mode):
    need(key in corpus.train and target_key in corpus.train,'RECOGNITION_FIT_QUERY')
    need(corpus.tasks[key]['kind']==corpus.tasks[target_key]['kind'],'RECOGNITION_TARGET_OPERATION')
    need(mode in ARMS,'RECOGNITION_MODE'); world=model['world']; task=corpus.tasks[key]
    native=corpus.batch(key,'context',world.config,'cuda')
    output=predict(world,model['request'],native,key=key,operation=task['kind'])
    _,target=corpus.target(target_key,world.config,'cuda',fitting=True)
    need(rows[target_key]['split']=='train','RECOGNITION_LABEL_SPLIT')
    label=torch.tensor([rows[target_key]['recognition']['label']],device='cuda')
    prediction=output['predicted_constellation']
    if mode!='legacy':
        prediction=world.supervised_placement(output,label,
            allowed=torch.ones(len(corpus.regions),dtype=torch.bool,device='cuda'),
            event_ids=[digest(dict(key=key,operation=task['kind']))])
    region=[]; placement=[]; lanes={}
    for spec in world.specs:
        r=F.cross_entropy(output['regions'][spec.name],label)
        actual,guess=target[spec.name],prediction.slots[spec.name]
        p=1-native_similarity(guess.values[0,~guess.padding[0]],actual.values[0,~actual.padding[0]])
        region.append(r); placement.append(p)
        lanes[spec.name]=dict(region=float(r.detach()),placement=float(p.detach()))
    loss=torch.stack(region).mean()+torch.stack(placement).mean()
    need(bool(torch.isfinite(loss)),'RECOGNITION_LOSS')
    return loss,dict(loss=float(loss.detach()),label=int(label),selected=int(next(iter(output['world_owner'].values()))[0]),
        lanes=lanes,placement_region=prediction.regions[0])


def step(model,optimizer,corpus,rows,key,target_key,mode):
    before=parameter_hash(model); optimizer.zero_grad(set_to_none=True)
    loss,metrics=objective(model,corpus,rows,key,target_key,mode); loss.backward()
    need(all(p.grad is None for p in model.parameters() if not p.requires_grad),'RECOGNITION_FROZEN_GRADIENT')
    need(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'RECOGNITION_GRADIENT')
    evidence=branch_evidence(model); optimizer.step(); after=parameter_hash(model)
    need(before!=after and all(bool(torch.isfinite(p).all()) for p in model.parameters()),'RECOGNITION_UPDATE')
    return dict(task=key,target=target_key,family=corpus.tasks[key]['family'],mode=mode,before=before,after=after,
                objective=metrics,gradients=evidence)


def aggregate(records):
    result={}
    for split in ('train','query'):
        result[split]={}
        for family in ('coding','conversation','reasoning'):
            selected=[r for r in records if r['split']==split and r['family']==family]
            cell=dict(tasks=len(selected),groups=len({r['group'] for r in selected}),
                hits=sum(r['label']==r['selected'] for r in selected),parent_hits=sum(r['label']==r['parent_selected'] for r in selected))
            if split=='query':
                deltas=[r['surface']['loss']-r['parent_loss'] for r in selected]
                cell.update(mean_loss=math.fsum(r['surface']['loss'] for r in selected)/len(selected),
                    mean_delta=math.fsum(deltas)/len(deltas),better=sum(x<0 for x in deltas),worse=sum(x>0 for x in deltas),
                    ties=sum(x==0 for x in deltas),lanes={})
                for name in selected[0]['lanes']:
                    delta=[r['lanes'][name]['cosine']-r['lanes'][name]['parent_cosine'] for r in selected]
                    cell['lanes'][name]=dict(mean_cosine=math.fsum(r['lanes'][name]['cosine'] for r in selected)/len(selected),
                        mean_delta=math.fsum(delta)/len(delta),better=sum(x>0 for x in delta),worse=sum(x<0 for x in delta),ties=sum(x==0 for x in delta))
            result[split][family]=cell
    return result


@torch.no_grad()
def evaluate(model,corpus,parent_rows,root):
    world=model['world']; result=[]
    with (root/'evaluation.jsonl').open('x') as out:
        for split,keys in (('train',corpus.train),('query',corpus.query)):
            for key in keys:
                task=corpus.tasks[key]; old=parent_rows[key]
                native=corpus.batch(key,'context',world.config,'cuda')
                output=predict(world,model['request'],native,key=key,operation=task['kind'])
                row=dict(task=key,split=split,family=task['family'],group=task['split_group'],label=old['recognition']['label'],
                    selected=int(next(iter(output['world_owner'].values()))[0]),parent_selected=old['recognition']['selected'])
                if split=='query':
                    raw,target=corpus.target(key,world.config,'cuda',fitting=False)
                    pred=output['predicted_constellation'].slots
                    row.update(source_sha256=sha(raw),source_bytes=len(raw),surface=score(model['decoder'],raw,pred),
                        parent_loss=old['scores']['predicted']['loss'],target_surface=score(model['decoder'],raw,target),lanes={})
                    need(row['target_surface']==old['scores']['target'],'RECOGNITION_RENDERER_DRIFT')
                    for spec in world.specs:
                        a,b=pred[spec.name],target[spec.name]
                        row['lanes'][spec.name]=dict(cosine=float(native_similarity(a.values[0,~a.padding[0]],b.values[0,~b.padding[0]])),
                            parent_cosine=old['lanes'][spec.name]['conditions']['predicted']['coverage_cosine'])
                out.write(json.dumps(row,sort_keys=True)+'\n'); out.flush(); result.append(row)
        os.fsync(out.fileno())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('diagnostic','readback'):
        p.add_argument('--'+name,type=Path,required=True); p.add_argument('--'+name+'-pin',required=True)
    p.add_argument('--arm',choices=ARMS,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    need(not a.output.exists(),'RECOGNITION_EXISTS'); a.output.mkdir(); start=tick=time.perf_counter(); phases={}
    source={name:dict(path=str(getattr(a,name)),pin=getattr(a,name+'_pin')) for name in ('diagnostic','readback')}
    source['code_sha256']=sha(Path(__file__).read_bytes())
    try:
        gpu=subprocess.check_output(['nvidia-smi'],text=True); take_lease('29','capability-recognition-'+a.arm); configure()
        model,optimizer,corpus,rows,parent,saved=prepare(source['diagnostic'],source['readback'])
        mapping=shuffled(corpus); base=saved['updates']; old=frozen(model,optimizer)
        boundaries=sorted({0,len(corpus.train)-1,*[next(i for i,k in enumerate(corpus.train) if corpus.tasks[k]['family']==f)
                    for f in ('coding','conversation','reasoning')]})
        plan=dict(source=source,arm=a.arm,train=corpus.train,query=corpus.query,mapping=mapping,boundaries=boundaries,
            same_group_shuffle=sum(corpus.tasks[k]['split_group']==corpus.tasks[v]['split_group'] for k,v in mapping.items()),
            trainable=[n for n,p in model.named_parameters() if p.requires_grad],protected=[n for n,p in model.named_parameters() if not p.requires_grad])
        write_committed(a.output/'plan.json',plan)
        chain=save_checkpoint(a.output/'initial.pt',model,optimizer,base,source,parent['final_checkpoint'])
        phases['restore']=time.perf_counter()-tick; tick=time.perf_counter()
        with (a.output/'steps.jsonl').open('x') as out:
            for index,key in enumerate(corpus.train):
                if index in boundaries and index:save_checkpoint(a.output/f'before-{index}.pt',model,optimizer,base+index,source,chain)
                at=time.perf_counter(); cpu=time.process_time()
                row=step(model,optimizer,corpus,rows,key,mapping[key] if a.arm=='shuffled' else key,a.arm)
                row.update(index=index,previous=chain,seconds=time.perf_counter()-at,cpu_seconds=time.process_time()-cpu)
                chain=digest(row); out.write(json.dumps(dict(row,sha256=chain),sort_keys=True)+'\n'); out.flush(); os.fsync(out.fileno())
                if index in boundaries:save_checkpoint(a.output/f'after-{index}.pt',model,optimizer,base+index+1,source,chain)
                print(json.dumps(dict(arm=a.arm,index=index,loss=row['objective']['loss'])),flush=True)
        phases['learn']=time.perf_counter()-tick; tick=time.perf_counter()
        need(frozen(model,optimizer)==old,'RECOGNITION_FROZEN_DRIFT'); model['world'].check_reference()
        final=save_checkpoint(a.output/'final.pt',model,optimizer,base+len(corpus.train),source,chain)
        records=evaluate(model,corpus,rows,a.output)
        need(frozen(model,optimizer)==old,'RECOGNITION_EVALUATION_DRIFT')
        phases['evaluate']=time.perf_counter()-tick; tick=time.perf_counter()
        need(sha(Path(corpus.config['journal']['path']).read_bytes())==corpus.config['journal']['pin'],'RECOGNITION_JOURNAL')
        result=dict(schema='capability-recognition-learning-v1',source=source,arm=a.arm,parent_updates=base,new_updates=len(corpus.train),
            initial_optimizer=saved['optimizer_hash'],final_checkpoint=final,final_parameters=parameter_hash(model),frozen_identity=old,
            steps_sha256=sha((a.output/'steps.jsonl').read_bytes()),evaluation_sha256=sha((a.output/'evaluation.jsonl').read_bytes()),
            families=aggregate(records),gpu_before=gpu,scientific_admission=False,learned_capability=False,serving=False,
            generated_surface_evaluated=False,exposure='development_exposed_group_holdout',number_status='measured',acceptance='author-checked')
        phases['finalize']=time.perf_counter()-tick; wall=time.perf_counter()-start
        need(sum(phases.values())/wall>=.9,'RECOGNITION_PHASES'); result.update(phases=phases,seconds=wall)
        print(json.dumps(dict(pin=write_committed(a.output/'report.json',result))),flush=True)
    except BaseException as exc:
        write_committed(a.output/'failure.json',dict(error=str(exc),trace=traceback.format_exc())); raise


if __name__=='__main__':run_cli(main)
