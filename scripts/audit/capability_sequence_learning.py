"""Matched continuation on complete attested reasoning prefixes, with an order control."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from torch.nn import functional as F
from audit.capability_conditioning import score
from audit.capability_kernel_learning import branch_evidence, parameter_hash, save_checkpoint
from audit.capability_recognition_learning import prepare, frozen, aggregate
from audit.fsv_identity_loop import configure
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import take_lease, run_cli
from model.capability_kernel_learning import predict, pinned
from model.capability_objective import native_similarity
from model.capability_sequence import source_prefix

ARMS=('baseline','ordered','reversed')


class Inputs:
    def __init__(self,corpus,world):
        self.corpus=corpus; self.native={}; self.history={}; self.world=world
        self.train=tuple(k for k in corpus.train if corpus.tasks[k]['kind']=='continue_recorded_steps')
        self.query=tuple(k for k in corpus.query if corpus.tasks[k]['kind']=='continue_recorded_steps')
        for key in (*corpus.train,*corpus.query):
            self.native[key]=corpus.batch(key,'context',world.config,'cuda')
            if corpus.tasks[key]['kind']=='continue_recorded_steps':
                self.history[key]=source_prefix(corpus,key,world.config,'cuda')

    def forward(self,model,key,arm,*,fitting):
        need(arm in ARMS and (not fitting or key in self.train),'SEQUENCE_FIT_QUERY')
        world=model['world']; task=self.corpus.tasks[key]; native=self.native[key]
        if arm=='baseline' or key not in self.history:
            return predict(world,model['request'],native,key=key,operation=task['kind'])
        history=self.history[key]
        if arm=='reversed' and fitting:history=replace(history,frames=history.frames[::-1],rows=history.rows[::-1])
        state=model['request'](world.start('capability:'+key),operation=task['kind'])
        return world(native,(state,),allowed=torch.ones(len(self.corpus.regions),dtype=torch.bool,device='cuda'),
            phase=0,event_ids=[digest(dict(key=key,operation=task['kind']))],evidence_origin=['source_observation'],
            advance_phase=False,history=history)


def step(model,optimizer,inputs,rows,key,arm):
    corpus=inputs.corpus;world=model['world'];optimizer.zero_grad(set_to_none=True);before=parameter_hash(model)
    output=inputs.forward(model,key,arm,fitting=True)
    _,target=corpus.target(key,world.config,'cuda',fitting=True)
    need(rows[key]['split']=='train','SEQUENCE_LABEL_SPLIT')
    label=torch.tensor([rows[key]['recognition']['label']],device='cuda')
    chosen=world.supervised_placement(output,label,allowed=torch.ones(len(corpus.regions),dtype=torch.bool,device='cuda'),
        event_ids=[digest(dict(key=key,operation=corpus.tasks[key]['kind']))])
    region=[];placement=[];lanes={}
    for spec in world.specs:
        r=F.cross_entropy(output['regions'][spec.name],label);x,y=chosen.slots[spec.name],target[spec.name]
        p=1-native_similarity(x.values[0,~x.padding[0]],y.values[0,~y.padding[0]])
        region.append(r);placement.append(p);lanes[spec.name]=dict(region=float(r.detach()),placement=float(p.detach()))
    loss=torch.stack(region).mean()+torch.stack(placement).mean();need(bool(torch.isfinite(loss)),'SEQUENCE_LOSS')
    loss.backward()
    need(all(p.grad is None for p in model.parameters() if not p.requires_grad),'SEQUENCE_FROZEN_GRADIENT')
    need(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'SEQUENCE_GRADIENT')
    gradients=branch_evidence(model);optimizer.step();after=parameter_hash(model)
    need(after!=before and all(bool(torch.isfinite(p).all()) for p in model.parameters()),'SEQUENCE_UPDATE')
    return dict(task=key,arm=arm,before=before,after=after,gradients=gradients,
        objective=dict(loss=float(loss.detach()),label=int(label),lanes=lanes,placement_region=chosen.regions[0]))


@torch.no_grad()
def evaluate(model,inputs,previous,arm,root):
    corpus=inputs.corpus;world=model['world'];records=[]
    with (root/'evaluation.jsonl').open('x') as out:
        for split,keys in (('train',corpus.train),('query',corpus.query)):
            for key in keys:
                task=corpus.tasks[key];old=previous[key];output=inputs.forward(model,key,arm,fitting=False)
                row=dict(task=key,split=split,family=task['family'],group=task['split_group'],label=old['recognition']['label'],
                    selected=int(next(iter(output['world_owner'].values()))[0]),parent_selected=old['recognition']['selected'],
                    sequence=key in inputs.history)
                if split=='query':
                    raw,target=corpus.target(key,world.config,'cuda',fitting=False);pred=output['predicted_constellation'].slots
                    row.update(source_sha256=sha(raw),source_bytes=len(raw),surface=score(model['decoder'],raw,pred),
                        parent_loss=old['scores']['predicted']['loss'],target_surface=score(model['decoder'],raw,target),lanes={})
                    need(row['target_surface']==old['scores']['target'],'SEQUENCE_RENDERER_DRIFT')
                    for spec in world.specs:
                        x,y=pred[spec.name],target[spec.name]
                        row['lanes'][spec.name]=dict(cosine=float(native_similarity(x.values[0,~x.padding[0]],y.values[0,~y.padding[0]])),
                            parent_cosine=old['lanes'][spec.name]['conditions']['predicted']['coverage_cosine'])
                out.write(json.dumps(row,sort_keys=True)+'\n');out.flush();records.append(row)
        os.fsync(out.fileno())
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('diagnostic','readback','order_readback'):
        p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-pin',required=True)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    need(not a.output.exists(),'SEQUENCE_EXISTS');a.output.mkdir();start=tick=time.perf_counter();phases={}
    source={n:dict(path=str(getattr(a,n)),pin=getattr(a,n+'_pin')) for n in ('diagnostic','readback','order_readback')}
    source['code']={n:sha((Path(__file__).resolve().parents[1]/n).read_bytes()) for n in
        ('audit/capability_sequence_learning.py','model/capability_sequence.py','audit/capability_recognition_learning.py')}
    try:
        gpu=subprocess.check_output(['nvidia-smi'],text=True);take_lease('29','capability-sequence-'+a.arm);configure()
        order=pinned(source['order_readback']);need(order['schema']=='capability-order-readback-v1','SEQUENCE_ORDER_READBACK')
        model,optimizer,corpus,rows,parent,saved=prepare(source['diagnostic'],source['readback']);base=saved['updates'];old=frozen(model,optimizer)
        phases['restore']=time.perf_counter()-tick;tick=time.perf_counter();inputs=Inputs(corpus,model['world'])
        boundaries=sorted({0,len(inputs.train)//2,len(inputs.train)-1})
        plan=dict(source=source,arm=a.arm,train=inputs.train,query=inputs.query,all_train=corpus.train,all_query=corpus.query,
            boundaries=boundaries,evaluation_history='attested_for_both_history_arms',
            trainable=[n for n,p in model.named_parameters() if p.requires_grad],protected=[n for n,p in model.named_parameters() if not p.requires_grad])
        write_committed(a.output/'plan.json',plan)
        chain=save_checkpoint(a.output/'initial.pt',model,optimizer,base,source,parent['final_checkpoint'])
        phases['build_inputs']=time.perf_counter()-tick;tick=time.perf_counter()
        with (a.output/'steps.jsonl').open('x') as out:
            for index,key in enumerate(inputs.train):
                if index in boundaries and index:save_checkpoint(a.output/f'before-{index}.pt',model,optimizer,base+index,source,chain)
                at=time.perf_counter();cpu=time.process_time();row=step(model,optimizer,inputs,rows,key,a.arm)
                row.update(index=index,previous=chain,seconds=time.perf_counter()-at,cpu_seconds=time.process_time()-cpu)
                chain=digest(row);out.write(json.dumps(dict(row,sha256=chain),sort_keys=True)+'\n');out.flush();os.fsync(out.fileno())
                if index in boundaries:save_checkpoint(a.output/f'after-{index}.pt',model,optimizer,base+index+1,source,chain)
        phases['learn']=time.perf_counter()-tick;tick=time.perf_counter();need(frozen(model,optimizer)==old,'SEQUENCE_FROZEN_DRIFT')
        final=save_checkpoint(a.output/'final.pt',model,optimizer,base+len(inputs.train),source,chain)
        records=evaluate(model,inputs,rows,a.arm,a.output)
        need(frozen(model,optimizer)==old,'SEQUENCE_EVALUATION_DRIFT');model['world'].check_reference()
        phases['evaluate']=time.perf_counter()-tick;tick=time.perf_counter()
        need(sha(Path(corpus.config['journal']['path']).read_bytes())==corpus.config['journal']['pin'],'SEQUENCE_JOURNAL')
        report=dict(schema='capability-sequence-learning-v1',source=source,arm=a.arm,parent_updates=base,new_updates=len(inputs.train),
            initial_optimizer=saved['optimizer_hash'],final_checkpoint=final,final_parameters=parameter_hash(model),frozen_identity=old,
            steps_sha256=sha((a.output/'steps.jsonl').read_bytes()),evaluation_sha256=sha((a.output/'evaluation.jsonl').read_bytes()),
            families=aggregate(records),gpu_before=gpu,scientific_admission=False,learned_capability=False,serving=False,
            generated_surface_evaluated=False,exposure='development_exposed_group_holdout',number_status='measured',acceptance='author-checked')
        phases['finalize']=time.perf_counter()-tick;wall=time.perf_counter()-start
        need(sum(phases.values())/wall>=.9,'SEQUENCE_PHASES');report.update(phases=phases,seconds=wall)
        print(json.dumps(dict(pin=write_committed(a.output/'report.json',report))),flush=True)
    except BaseException as exc:
        write_committed(a.output/'failure.json',dict(error=str(exc),trace=traceback.format_exc()));raise


if __name__=='__main__':run_cli(main)
