"""Fourteen matched, provisional typed product-region fits on original supplied facts."""
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
from model.qasc_generation_data import load,TARGETS,TARGET_NATIVE
from model.qasc_context import LANES,POLICIES,POPULATION,NATIVE,check
from model.typed_regions import CONFIG,Store,Predictor,build_books,objective,schedule,infer,controls
from model.qasc_region_metrics import summarize,scores
from model.qasc_combination import donors
from model.renderer_conditioning import gpu_lease
from model.scifact_readout import array_sha
from text.evaluation_multiscale import file_sha
from senses.trajectory_population import publish
from lens.lenscommon import run_cli

def files(root):
    return {p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in root.rglob('*') if p.is_file()}

def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('population','native','targets','target-native','raw','proof','output'):p.add_argument('--'+n,type=Path,required=True)
    return p.parse_args()

def main():
    a=arguments();check(not a.output.exists(),'REGION_OUTPUT_EXISTS');start=time.perf_counter();phases={};torch.set_num_threads(32)
    ctx=load(a.population,a.native,a.targets,a.target_native,a.raw,a.proof);a.output.mkdir(parents=True)
    for n,v in [('config',CONFIG),('before',ctx['before']),('cold',ctx['cold']),('lanes',ctx['lanes']),('policies',ctx['policies'])]:publish(a.output/(n+'.json'),v)
    np.savez(a.output/'gather.npz',maps=ctx['maps'],units=ctx['units'],original=ctx['original'],target_index=ctx['target_index'])
    phases['source_native_readback_and_scope']=time.perf_counter()-start;tick=time.perf_counter()
    residents=gpu_lease(None,'typed-qasc-regions-matched-fourteen-fits');(a.output/'gpu-before.txt').write_text(residents,encoding='utf-8')
    store=Store(ctx);source_code={Path(sys.modules[n].__file__).name:file_sha(Path(sys.modules[n].__file__))
                               for n in ('model.qasc_generation_data','model.typed_regions','model.qasc_region_metrics','model.qasc_combination')}
    source_code[Path(__file__).name]=file_sha(Path(__file__))
    publish(a.output/'runtime.json',dict(torch=torch.__version__,cuda=torch.version.cuda,platform=sys.platform,
            device=torch.cuda.get_device_name(0),source_code=source_code,precision='fp32',batch=32))
    phases['gpu_resident_lease_and_upload']=time.perf_counter()-tick;results={};errors=[]
    for policy in POLICIES:
        tick=time.perf_counter();root=a.output/policy;root.mkdir();rr=ctx['policies'][policy]
        train=np.array([i for i,r in enumerate(rr) if r['partition']=='train'],np.int64);unique=store.prepare(train)
        books,codes,histories,initial=build_books(store,unique);truth=[v.cpu().numpy() for v in store.y];code_array=codes.cpu().numpy()
        payload=dict(train_rows=train,unique_training_targets=unique.cpu().numpy(),initial_positions=initial,codes=code_array)
        for k,name in enumerate(LANES):payload.update({name+'_mean':store.means[k].cpu().numpy(),name+'_centers':books[k].cpu().numpy(),name+'_counts':histories[k]})
        np.savez(root/'codebook.npz',**payload);codebook_pin=file_sha(root/'codebook.npz')
        baseline={};baseline_cos={}
        for name,j in [('copy-fact1',1),('copy-fact2',2)]:
            values=[x[store.maps[:,j]].cpu().numpy() for x in store.x];cos,_=scores(values,truth,ctx['target_index'])
            baseline[name]=summarize(cos,None,rr);baseline_cos[name]=cos
        modes=[torch.bincount(codes[unique,k],minlength=64).argmax() for k in range(6)]
        values=[b[m].expand(8134,-1).cpu().numpy() for b,m in zip(books,modes)];cos,_=scores(values,truth,ctx['target_index'])
        baseline['training-modal-region']=summarize(cos,None,rr);baseline_cos['training-modal-region']=cos
        np.savez(root/'baselines.npz',**baseline_cos);publish(root/'baselines.json',baseline)
        donor=donors(rr);np.save(root/'donors.npy',donor);phases[policy+'/codebook_and_baselines']=time.perf_counter()-tick
        for seed in CONFIG['seeds']:
            tick=time.perf_counter();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            seedroot=root/f'seed-{seed}';seedroot.mkdir();initial_model=Predictor(books)
            initial_state={k:v.detach().cpu().clone() for k,v in initial_model.state_dict().items()};parameters=sum(p.numel() for p in initial_model.parameters());del initial_model
            torch.save(initial_state,seedroot/'initial.pt');steps=schedule(rr,seed);np.save(seedroot/'training-rows.npy',steps)
            for arm in ['region','direct',*(['shuffled'] if seed==25 else [])]:
                out=seedroot/arm;out.mkdir();fit=time.perf_counter();local={};model=opt=None
                try:
                    target=ctx['target_index'].copy()
                    if arm=='shuffled':target[train]=np.random.default_rng(seed).permutation(target[train])
                    np.save(out/'training-targets.npy',target[train]);tg=torch.as_tensor(target,device='cuda')
                    model=Predictor(books).cuda();model.load_state_dict(initial_state);placement='region' if arm=='shuffled' else arm
                    opt=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
                    torch.cuda.reset_peak_memory_stats();losses=[];local['initialization']=time.perf_counter()-fit;at=time.perf_counter()
                    for step,ids in enumerate(steps):
                        opt.zero_grad(set_to_none=True);values,logits=model(store.batch(ids),placement);targets=tg[ids]
                        y=[v[targets] for v in store.y];labels=codes[targets];loss,terms=objective(values,logits,y,labels,targets)
                        check(bool(torch.isfinite(loss)),'REGION_NONFINITE_LOSS');loss.backward()
                        if step==0:
                            gradients={k:float(v.grad.norm()) if v.grad is not None else None for k,v in model.named_parameters()}
                            check(all(v is not None and np.isfinite(v) for v in gradients.values()),'REGION_GRADIENTS')
                            snap=dict(rows=ids,targets=targets.cpu().numpy(),codes=labels.cpu().numpy())
                            for k in range(6):snap.update({f'slots_{k}':values[k].detach().cpu().numpy(),f'logits_{k}':logits[k].detach().cpu().numpy()})
                            np.savez(out/'objective.npz',**snap);publish(out/'objective.json',dict(loss=float(loss.detach()),terms=terms.detach().cpu().tolist(),gradients=gradients))
                        opt.step();losses.append(terms.detach().cpu().tolist())
                    torch.cuda.synchronize();local['training']=time.perf_counter()-at;at=time.perf_counter()
                    np.save(out/'losses.npy',np.array(losses));torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'model.pt')
                    publish(out/'training-time.json',dict(seconds=local['training'],updates=600,case_updates=19200,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
                    model.requires_grad_(False).eval();values,logits=infer(model,store,placement)
                    np.savez(out/'outputs.npz',**{**{f'slots_{k}':x for k,x in enumerate(values)},**{f'logits_{k}':x for k,x in enumerate(logits)}})
                    cos,correct=scores(values,truth,ctx['target_index'],logits,code_array)
                    np.savez(out/'scores.npz',cosine=cos,correct=correct);metrics={'primary':summarize(cos,correct,rr)}
                    if arm=='region':
                        interventions={}
                        for scope in ('fact1-only','fact2-only','question-only','facts-only','other-group-facts'):
                            v,l=infer(model,store,placement,'both-facts' if scope=='other-group-facts' else scope,donor if scope=='other-group-facts' else None)
                            c,ok=scores(v,truth,ctx['target_index'],l,code_array);interventions[scope+'_cosine']=c;interventions[scope+'_correct']=ok
                            metrics[scope]=summarize(c,ok,rr)
                        np.savez(out/'interventions.npz',**interventions)
                    publish(out/'metrics.json',metrics);publish(out/'controls.json',controls(model,store,placement))
                    check(all(torch.equal(getattr(model,'book_'+str(k)),books[k]) for k in range(6)),'REGION_BOOK_DRIFT')
                    local['checkpoint_inference_controls_and_metrics']=time.perf_counter()-at;wall=time.perf_counter()-fit
                    publish(out/'report.json',dict(policy=policy,seed=seed,arm=arm,placement=placement,parameters=parameters,
                            initial_sha256=file_sha(seedroot/'initial.pt'),codebook_sha256=codebook_pin,files=files(out),
                            phases_seconds=local,wall_seconds=wall,phase_coverage=sum(local.values())/wall,
                            scientific_admission=False,served=False,free_text_composition_proven=False))
                    results[f'{policy}/seed-{seed}/{arm}']=file_sha(out/'report.json')
                    print(json.dumps(dict(policy=policy,seed=seed,arm=arm,evaluation=metrics['primary']['evaluation'],seconds=wall)),flush=True)
                except (ValueError,RuntimeError) as exc:
                    error=dict(policy=policy,seed=seed,arm=arm,error=str(exc));publish(out/'failure.json',error);errors.append(error);print(json.dumps(error),flush=True)
                finally:del model,opt;torch.cuda.empty_cache()
            phases[policy+f'/seed-{seed}']=time.perf_counter()-tick
    tick=time.perf_counter()
    check(all(array_sha(v.cpu().numpy())==array_sha(x) for v,x in zip(store.native+store.truth,ctx['inputs']+ctx['truth'])),'REGION_NATIVE_DRIFT')
    after={p:file_sha(Path(p)) for p in ctx['before']};check(after==ctx['before'],'REGION_PARENT_DRIFT');publish(a.output/'after.json',after)
    metadata=files(a.output);phases['native_parent_immutability_and_final_hashes']=time.perf_counter()-tick;wall=time.perf_counter()-start
    publish(a.output/'report.json',dict(schema='qasc-typed-regions-v1',population_sha256=POPULATION,native_sha256=NATIVE,
            targets_sha256=TARGETS,target_native_sha256=TARGET_NATIVE,files=metadata,results=results,errors=errors,complete=not errors,
            phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,
            scientific_admission=False,served=False,sealed_evaluation=False,free_text_composition_proven=False,
            assertion_verdict='REFUSE_UNADMITTED_PANEL',production_assertions=0,teacher_forced_regions=False))
    print(json.dumps(dict(complete=not errors,report_sha256=file_sha(a.output/'report.json'),wall_seconds=wall)),flush=True)
    return 2 if errors else 0

if __name__=='__main__':sys.exit(run_cli(main))
