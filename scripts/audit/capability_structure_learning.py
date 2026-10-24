"""Matched context-only continuation, preserving every inherited parameter and Adam moment."""
import argparse
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
from audit.capability_conditioning import fingerprint, score
from audit.capability_kernel_learning import branch_evidence, parameter_hash, save_checkpoint
from audit.capability_recognition_learning import prepare, frozen, aggregate
from audit.fsv_identity_loop import configure
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import take_lease, run_cli
from model.capability_kernel_learning import pinned
from model.capability_objective import native_similarity
from model.capability_structure_input import StructuredContexts
from model.capability_structure import attach_context, predict_structured, ARMS


def restore(structure_ref,reader_ref):
    report=pinned(structure_ref);reader=pinned(reader_ref);source=report['plan']['source']
    need(reader['source']==structure_ref and reader['schema']=='capability-structure-readback-v1','STRUCTURE_LEARN_READBACK')
    for name,pin in source['code'].items():need(sha((Path(__file__).resolve().parents[1]/name).read_bytes())==pin,'STRUCTURE_LEARN_CODE')
    model,optimizer,corpus,rows,parent,saved=prepare(source['diagnostic'],source['readback'])
    extension=attach_context(model,optimizer,corpus.config['seed'])
    initial=Path(structure_ref['path']).parent/'adapter-initial.pt'
    need(sha(initial.read_bytes())==report['files']['adapter-initial.pt']
         and fingerprint(extension.state_dict())==fingerprint(torch.load(initial,weights_only=True)),'STRUCTURE_LEARN_INITIAL')
    return model,optimizer,corpus,rows,parent,saved


def step(model,optimizer,contexts,previous,key,arm):
    corpus=contexts.corpus;need(key in corpus.train,'STRUCTURE_LEARN_FIT_QUERY')
    world=model['world'];task=corpus.tasks[key];item=contexts.get(key,world.config,'cuda')
    optimizer.zero_grad(set_to_none=True);before=parameter_hash(model)
    output=predict_structured(model,item,task['kind'],mode=arm)
    _,target=corpus.target(key,world.config,'cuda',fitting=True)
    need(previous[key]['split']=='train','STRUCTURE_LEARN_LABEL_SPLIT')
    label=torch.tensor([previous[key]['recognition']['label']],device='cuda')
    chosen=world.supervised_placement(output,label,allowed=torch.ones(len(corpus.regions),dtype=torch.bool,device='cuda'),
        event_ids=[digest(dict(key=key,operation=task['kind']))])
    regions=[];placements=[];lanes={}
    for spec in world.specs:
        r=F.cross_entropy(output['regions'][spec.name],label);x,y=chosen.slots[spec.name],target[spec.name]
        p=1-native_similarity(x.values[0,~x.padding[0]],y.values[0,~y.padding[0]])
        regions.append(r);placements.append(p);lanes[spec.name]=dict(region=float(r.detach()),placement=float(p.detach()))
    loss=torch.stack(regions).mean()+torch.stack(placements).mean();need(bool(torch.isfinite(loss)),'STRUCTURE_LEARN_LOSS')
    loss.backward();need(all(p.grad is None for p in model.parameters() if not p.requires_grad),'STRUCTURE_LEARN_PARENT_GRADIENT')
    need(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'STRUCTURE_LEARN_GRADIENT')
    gradients=branch_evidence(model);optimizer.step();after=parameter_hash(model)
    need(after!=before and all(bool(torch.isfinite(p).all()) for p in model.parameters()),'STRUCTURE_LEARN_UPDATE')
    return dict(task=key,arm=arm,family=task['family'],before=before,after=after,gradients=gradients,
        objective=dict(loss=float(loss.detach()),label=int(label),lanes=lanes,placement_region=chosen.regions[0]))


@torch.no_grad()
def evaluate(model,contexts,previous,arm,root):
    corpus=contexts.corpus;world=model['world'];records=[];mode='boundaries' if arm=='boundaries' else 'structured'
    with (root/'evaluation.jsonl').open('x') as out:
        for split,keys in (('train',corpus.train),('query',corpus.query)):
            for key in keys:
                task=corpus.tasks[key];old=previous[key];output=predict_structured(model,contexts.get(key,world.config,'cuda'),task['kind'],mode=mode)
                row=dict(task=key,split=split,family=task['family'],group=task['split_group'],label=old['recognition']['label'],
                    selected=int(next(iter(output['world_owner'].values()))[0]),parent_selected=old['recognition']['selected'])
                if split=='query':
                    raw,target=corpus.target(key,world.config,'cuda',fitting=False);pred=output['predicted_constellation'].slots
                    row.update(source_sha256=sha(raw),source_bytes=len(raw),surface=score(model['decoder'],raw,pred),
                        parent_loss=old['scores']['predicted']['loss'],target_surface=score(model['decoder'],raw,target),lanes={})
                    need(row['target_surface']==old['scores']['target'],'STRUCTURE_LEARN_RENDERER_DRIFT')
                    for spec in world.specs:
                        x,y=pred[spec.name],target[spec.name]
                        row['lanes'][spec.name]=dict(cosine=float(native_similarity(x.values[0,~x.padding[0]],y.values[0,~y.padding[0]])),
                            parent_cosine=old['lanes'][spec.name]['conditions']['predicted']['coverage_cosine'])
                out.write(json.dumps(row,sort_keys=True)+'\n');out.flush();records.append(row)
        os.fsync(out.fileno())
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('structure','readback'):
        p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-pin',required=True)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    need(not a.output.exists(),'STRUCTURE_LEARN_EXISTS');a.output.mkdir();start=tick=time.perf_counter();phases={}
    source={n:dict(path=str(getattr(a,n)),pin=getattr(a,n+'_pin')) for n in ('structure','readback')}
    source['code_sha256']=sha(Path(__file__).read_bytes())
    try:
        gpu=subprocess.check_output(['nvidia-smi'],text=True);take_lease('29','capability-structure-learn-'+a.arm);configure()
        torch.cuda.reset_peak_memory_stats();model,optimizer,corpus,previous,parent,saved=restore(source['structure'],source['readback'])
        base=saved['updates'];protected=frozen(model,optimizer);phases['restore']=time.perf_counter()-tick;tick=time.perf_counter()
        contexts=StructuredContexts(corpus);census=contexts.census();free,total=torch.cuda.mem_get_info()
        need(census['native_bytes']+model['world'].config.max_native_elements*4<free,'STRUCTURE_LEARN_MEMORY')
        for key in (*corpus.train,*corpus.query):contexts.get(key,model['world'].config,'cuda')
        fixed=[]
        for key in corpus.train:
            item=contexts.get(key,model['world'].config,'cuda')
            if torch.equal(item.field_metadata,item.field_metadata.roll(1,0)) and all(torch.equal(x,x.flip(0)) for values in item.windows.values() for x in values):fixed.append(key)
        boundaries=sorted({0,len(corpus.train)-1,*[next(i for i,k in enumerate(corpus.train) if corpus.tasks[k]['family']==f)
                                               for f in ('coding','conversation','reasoning')]})
        plan=dict(source=source,arm=a.arm,train=corpus.train,query=corpus.query,boundaries=boundaries,catalogue=contexts.identity,
            census=census,control_fixed_tasks=fixed,evaluation_metadata='attested_for_structured_and_mismatched',
            trainable=[n for n,p in model.named_parameters() if p.requires_grad],protected=[n for n,p in model.named_parameters() if not p.requires_grad])
        write_committed(a.output/'plan.json',plan);chain=save_checkpoint(a.output/'initial.pt',model,optimizer,base,source,parent['final_checkpoint'])
        phases['compile_materialize']=time.perf_counter()-tick;tick=time.perf_counter()
        with (a.output/'steps.jsonl').open('x') as out:
            for index,key in enumerate(corpus.train):
                if index in boundaries and index:save_checkpoint(a.output/f'before-{index}.pt',model,optimizer,base+index,source,chain)
                at=time.perf_counter();cpu=time.process_time();row=step(model,optimizer,contexts,previous,key,a.arm)
                row.update(index=index,previous=chain,seconds=time.perf_counter()-at,cpu_seconds=time.process_time()-cpu)
                chain=digest(row);out.write(json.dumps(dict(row,sha256=chain),sort_keys=True)+'\n');out.flush();os.fsync(out.fileno())
                if index in boundaries:save_checkpoint(a.output/f'after-{index}.pt',model,optimizer,base+index+1,source,chain)
        phases['learn']=time.perf_counter()-tick;tick=time.perf_counter();need(frozen(model,optimizer)==protected,'STRUCTURE_LEARN_PARENT_DRIFT')
        final=save_checkpoint(a.output/'final.pt',model,optimizer,base+len(corpus.train),source,chain)
        records=evaluate(model,contexts,previous,a.arm,a.output);need(frozen(model,optimizer)==protected,'STRUCTURE_LEARN_EVALUATION_DRIFT')
        model['world'].check_reference();phases['evaluate']=time.perf_counter()-tick;tick=time.perf_counter()
        need(sha(Path(corpus.config['journal']['path']).read_bytes())==corpus.config['journal']['pin'],'STRUCTURE_LEARN_JOURNAL')
        report=dict(schema='capability-structure-learning-v1',source=source,arm=a.arm,parent_updates=base,new_updates=len(corpus.train),
            initial_optimizer=saved['optimizer_hash'],final_checkpoint=final,final_parameters=parameter_hash(model),frozen_identity=protected,
            steps_sha256=sha((a.output/'steps.jsonl').read_bytes()),evaluation_sha256=sha((a.output/'evaluation.jsonl').read_bytes()),
            families=aggregate(records),gpu_before=gpu,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            scientific_admission=False,learned_capability=False,serving=False,generated_surface_evaluated=False,
            exposure='development_exposed_group_holdout',number_status='measured',acceptance='author-checked')
        phases['finalize']=time.perf_counter()-tick;wall=time.perf_counter()-start
        need(sum(phases.values())/wall>=.9,'STRUCTURE_LEARN_PHASES');report.update(phases=phases,seconds=wall)
        print(json.dumps(dict(pin=write_committed(a.output/'report.json',report))),flush=True)
    except BaseException as exc:
        write_committed(a.output/'failure.json',dict(error=str(exc),trace=traceback.format_exc()));raise


if __name__=='__main__':run_cli(main)
