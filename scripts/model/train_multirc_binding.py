"""Sixteen fixed-budget, source-scoped MultiRC binding fits; no serving path."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.multirc_context import load,check,LANES,POPULATION,NATIVE,QUALITY
from model.multirc_binding import CONFIG,ARMS,Store,Binding,schedule,donors,infer,controls
from model.multirc_metrics import plan,evaluate,comparisons,intervention_change
from model.renderer_conditioning import gpu_lease
from model.scifact_readout import array_sha
from audit.multirc_binding_readback import files,preparation
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha
from lens.lenscommon import run_cli

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('population','native','quality','raw','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();check(not a.output.exists(),'OUTPUT_EXISTS');start=time.perf_counter();phases={};torch.set_num_threads(32)
    ctx=load(a.population,a.native,a.quality,a.raw);a.output.mkdir(parents=True);root=a.output/'preparation';root.mkdir()
    for name,value in [('rows',ctx['rows']),('lanes',ctx['lanes']),('before',ctx['before']),('cold',ctx['cold'])]:publish(root/(name+'.json'),value)
    np.save(root/'labels.npy',ctx['labels']);np.save(root/'train.npy',ctx['train'])
    for k,name in enumerate(LANES):np.savez(root/(name+'.npz'),**ctx['maps'][k],mean=ctx['means'][k])
    publish(root/'report.json',dict(schema='multirc-binding-preparation-v1',population_sha256=POPULATION,native_sha256=NATIVE,quality_payloads=QUALITY,
                files=files(root),world_truth_assessed=False,scientific_admission=False,served=False))
    publish(a.output/'preparation-readback.json',preparation(root,ctx,a.native,a.raw))
    phases['source_native_scope_publication_and_independent_readback']=time.perf_counter()-start;tick=time.perf_counter()
    publish(a.output/'config.json',CONFIG);residents=gpu_lease(None,'multirc-source-binding-sixteen-matched-fits');(a.output/'gpu-before.txt').write_text(residents,encoding='utf-8')
    store=Store(ctx);labels=torch.as_tensor(ctx['labels'],device='cuda');pp=plan(ctx)
    ids=np.flatnonzero(store.available);heldout=np.array([i for i in ids if ctx['rows'][i]['partition']!='train'],np.int64)
    donor=donors(ctx['rows']);np.savez(a.output/'evaluation-scope.npz',ids=ids,heldout=heldout,donors=donor)
    prior=float(ctx['labels'][ctx['train']].mean());prior_logits=np.broadcast_to(np.log([1-prior,prior]),(len(ids),2)).copy()
    prior_metrics,prior_aggregate,_=evaluate(prior_logits,ids,pp);np.save(a.output/'prior-logits.npy',prior_logits);publish(a.output/'prior-metrics.json',prior_metrics)
    source_code={n:file_sha(Path(sys.modules[n].__file__)) for n in ('model.multirc_context','model.multirc_binding','model.multirc_metrics','audit.multirc_binding_readback')}
    source_code['model.train_multirc_binding']=file_sha(Path(__file__))
    publish(a.output/'runtime.json',dict(torch=torch.__version__,cuda=torch.version.cuda,platform=sys.platform,device=torch.cuda.get_device_name(0),source_code=source_code))
    phases['gpu_lease_upload_and_baselines']=time.perf_counter()-tick;results={};errors=[];aggregates={}
    for seed in CONFIG['seeds']:
        tick=time.perf_counter();seedroot=a.output/f'seed-{seed}';seedroot.mkdir();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        model=Binding();initial={k:v.detach().clone() for k,v in model.state_dict().items()};parameters=sum(p.numel() for p in model.parameters());del model
        torch.save(initial,seedroot/'initial.pt');steps=schedule(ctx['train'],seed);np.save(seedroot/'training-rows.npy',steps)
        phases[f'seed-{seed}/shared_initialization']=time.perf_counter()-tick
        for arm in (*ARMS,*(['shuffled'] if seed==25 else [])):
            out=seedroot/arm;out.mkdir();fit=time.perf_counter();local={};model=opt=None;key=f'seed-{seed}/{arm}';scope='joint' if arm=='shuffled' else arm
            try:
                target=ctx['labels'].copy()
                if arm=='shuffled':target[ctx['train']]=np.random.default_rng(seed).permutation(target[ctx['train']])
                np.save(out/'training-labels.npy',target[ctx['train']]);tg=torch.as_tensor(target,device='cuda')
                model=Binding().cuda();model.load_state_dict(initial);opt=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
                torch.cuda.reset_peak_memory_stats();losses=[];local['initialization']=time.perf_counter()-fit;at=time.perf_counter()
                for step,ii in enumerate(steps):
                    opt.zero_grad(set_to_none=True);logits=model(store.batch(ii),scope);loss=torch.nn.functional.cross_entropy(logits,tg[ii])
                    check(bool(torch.isfinite(loss)),'NONFINITE_LOSS');loss.backward()
                    if step==0:
                        gradients={k:float(v.grad.norm()) if v.grad is not None else None for k,v in model.named_parameters()}
                        check(any(v is not None and v>0 for v in gradients.values()) and all(v is None or np.isfinite(v) for v in gradients.values()),'GRADIENTS')
                        np.savez(out/'objective.npz',rows=ii,labels=target[ii],logits=logits.detach().cpu().numpy())
                        publish(out/'objective.json',dict(loss=float(loss.detach()),gradients=gradients))
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),CONFIG['max_gradient_norm']);check(bool(torch.isfinite(norm)),'NONFINITE_GRADIENT')
                    opt.step();losses.append(float(loss.detach()))
                    if (step+1)%500==0:print(json.dumps(dict(fit=key,updates=step+1,seconds=time.perf_counter()-at)),flush=True)
                torch.cuda.synchronize();local['training']=time.perf_counter()-at;at=time.perf_counter()
                np.save(out/'losses.npy',np.array(losses));torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'model.pt')
                publish(out/'training-time.json',dict(seconds=local['training'],updates=2000,case_updates=128000,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
                model.requires_grad_(False).eval();logits=infer(model,store,scope,ids);np.save(out/'logits.npy',logits)
                metrics,aggregate,_=evaluate(logits,ids,pp);publish(out/'metrics.json',metrics);aggregates[key]=aggregate
                if arm=='joint':
                    intervention={};summaries={};base=logits[np.searchsorted(ids,heldout)]
                    for kind in ('remove-context','question-swap','passage-swap'):
                        value=infer(model,store,scope,heldout,kind,donor);intervention[kind]=value;summaries[kind]=intervention_change(base,value)
                    np.savez(out/'interventions.npz',**intervention);publish(out/'interventions.json',summaries)
                publish(out/'controls.json',controls(model,store,ids));local['checkpoint_inference_metrics_and_controls']=time.perf_counter()-at
                wall=time.perf_counter()-fit
                publish(out/'report.json',dict(arm=arm,scope=scope,seed=seed,parameters=parameters,initial_sha256=file_sha(seedroot/'initial.pt'),
                        files=files(out),phases_seconds=local,wall_seconds=wall,phase_coverage=sum(local.values())/wall,
                        world_truth_assessed=False,scientific_admission=False,served=False,epistemic_understanding_proven=False))
                results[key]=file_sha(out/'report.json');print(json.dumps(dict(fit=key,evaluation=metrics['evaluation'],seconds=wall)),flush=True)
            except (ValueError,RuntimeError) as exc:
                error=dict(fit=key,error=str(exc));publish(out/'failure.json',error);errors.append(error);print(json.dumps(error),flush=True)
            finally:del model,opt;torch.cuda.empty_cache()
            phases[key]=time.perf_counter()-fit
    tick=time.perf_counter();comparison={}
    for seed in CONFIG['seeds']:
        key=f'seed-{seed}/joint'
        if key not in aggregates:continue
        for arm in ARMS[1:]:
            other=f'seed-{seed}/{arm}'
            if other in aggregates:comparison[key+'/'+arm]=comparisons(aggregates[key],aggregates[other])
        comparison[key+'/train-prior']=comparisons(aggregates[key],prior_aggregate)
    publish(a.output/'comparisons.json',comparison)
    check(all(array_sha(v.cpu().numpy())==array_sha(x) for v,x in zip(store.native,ctx['native'])),'NATIVE_DRIFT')
    after={p:file_sha(Path(p)) for p in ctx['before']};check(after==ctx['before'],'PARENT_DRIFT');publish(a.output/'after.json',after)
    metadata=files(a.output);phases['comparisons_immutable_native_and_parent_hashes']=time.perf_counter()-tick;wall=time.perf_counter()-start
    publish(a.output/'report.json',dict(schema='multirc-typed-binding-v1',files=metadata,results=results,errors=errors,complete=not errors,
            phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,scientific_admission=False,served=False,
            epistemic_understanding_proven=False,world_truth_assessed=False,production_assertions=0,sealed_evaluation=False,
            assertion_verdict='REFUSE_UNADMITTED_PANEL',anchor_scope='source_relative_answer_correctness'))
    print(json.dumps(dict(complete=not errors,report_sha256=file_sha(a.output/'report.json'),seconds=wall)),flush=True)
    return 2 if errors else 0

if __name__=='__main__':sys.exit(run_cli(main))
