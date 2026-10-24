"""Predeclared two-policy, 32-fit supplied-fact diagnostic; provisional only."""
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
from model.qasc_context import load,check,POLICIES,POPULATION,NATIVE
from model.qasc_combination import CONFIG,ARMS,Store,Combination,schedule,donors,evaluate,controls
from model.qasc_metrics import measure,comparisons
from model.scifact_readout import array_sha
from model.renderer_conditioning import gpu_lease
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha
from lens.lenscommon import run_cli

def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('population','native','raw','native-proof','output'):p.add_argument('--'+n,type=Path,required=True)
    return p.parse_args()

def main():
    a=arguments();check(not a.output.exists(),'OUTPUT_EXISTS');start=time.perf_counter();phases={};torch.set_num_threads(32)
    proof=json.loads((a.native_proof/'replay-linux-v1/report.json').read_bytes())
    check(proof['publication_sha256']==NATIVE and proof['source_and_instruments_unchanged'] is True
          and all(v['all_replayed_vectors_bitwise_equal'] for v in proof['lanes'].values()),'NATIVE_REPLAY')
    ctx=load(a.population,a.native,a.raw);a.output.mkdir(parents=True)
    for n,v in (('before',ctx['before']),('config',CONFIG),('population-readback',ctx['source']),('native-readback',ctx['readings']),('lanes',ctx['lanes'])):publish(a.output/(n+'.json'),v)
    np.savez(a.output/'gather.npz',maps=ctx['maps'],units=ctx['units'],original=ctx['original']);np.save(a.output/'labels.npy',ctx['labels'])
    publish(a.output/'policies.json',ctx['policies']);phases['source_native_and_gather_readback']=time.perf_counter()-start;tick=time.perf_counter()
    gpu=gpu_lease(None,'qasc-two-policy-32-matched-fits');(a.output/'gpu-before.txt').write_text(gpu,encoding='utf-8')
    store=Store(ctx,'cuda');code={p.name:file_sha(p) for p in [Path(__file__),*[Path(__file__).with_name(n) for n in ('qasc_context.py','qasc_combination.py','qasc_metrics.py')]]}
    publish(a.output/'runtime.json',dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),platform=sys.platform,code=code))
    phases['lease_native_upload_and_runtime']=time.perf_counter()-tick;results={};errors=[]
    for policy in POLICIES:
        rows=ctx['policies'][policy];train=np.array([i for i,r in enumerate(rows) if r['partition']=='train'],np.int64)
        donor=donors(rows);folder=a.output/policy;folder.mkdir();np.save(folder/'donors.npy',donor)
        for seed in CONFIG['seeds']:
            tick=time.perf_counter();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);initial=Combination()
            state={k:v.detach().clone() for k,v in initial.state_dict().items()};parameters=sum(v.numel() for v in initial.parameters());del initial
            seedroot=folder/f'seed-{seed}';seedroot.mkdir();torch.save(state,seedroot/'initial.pt')
            steps=schedule(rows,seed);np.save(seedroot/'training-rows.npy',steps);out={}
            for name in [*ARMS,*(['shuffled'] if seed==25 else [])]:
                root=seedroot/name;root.mkdir();at=time.perf_counter();local={};model=opt=None;scope='both-facts' if name=='shuffled' else name
                try:
                    target=ctx['labels'].copy()
                    if name=='shuffled':target[train]=np.random.default_rng(seed).permutation(target[train])
                    np.save(root/'training-labels.npy',target[train]);target_gpu=torch.from_numpy(target).cuda()
                    model=Combination().cuda();model.load_state_dict(state);opt=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
                    torch.cuda.reset_peak_memory_stats();losses=[];local['initialization']=time.perf_counter()-at;fit=time.perf_counter()
                    for step,ids in enumerate(steps):
                        opt.zero_grad(set_to_none=True);logits=model(store.batch(ids),scope);loss=torch.nn.functional.cross_entropy(logits,target_gpu[ids])
                        check(bool(torch.isfinite(loss)),'NONFINITE_LOSS');loss.backward()
                        if step==0:
                            gradients={k:float(p.grad.norm()) if p.grad is not None else None for k,p in model.named_parameters()}
                            check(all(v is None or np.isfinite(v) for v in gradients.values()) and any(v and v>0 for v in gradients.values()),'GRADIENTS')
                            np.savez(root/'objective.npz',rows=ids,logits=logits.detach().cpu().numpy(),labels=target[ids]);publish(root/'objective.json',dict(loss=float(loss.detach()),gradients=gradients))
                        opt.step();losses.append(float(loss.detach()))
                    torch.cuda.synchronize();local['training']=time.perf_counter()-fit;fit=time.perf_counter()
                    publish(root/'training-time.json',dict(seconds=local['training'],updates=400,case_updates=12800,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
                    np.save(root/'losses.npy',np.array(losses));torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},root/'model.pt')
                    model.eval();o={'probabilities':evaluate(model,store,len(rows),scope)}
                    if name=='both-facts':
                        for intervention in ('fact1-only','fact2-only','question-only'):o[intervention]=evaluate(model,store,len(rows),intervention)
                        o['other-group-facts']=evaluate(model,store,len(rows),'both-facts',donor)
                    np.savez(root/'outputs.npz',**o);publish(root/'controls.json',controls(model,store));score=measure(o['probabilities'],rows,ctx['labels']);publish(root/'metrics.json',score)
                    local['checkpoint_inference_controls_and_metrics']=time.perf_counter()-fit;wall=time.perf_counter()-at
                    rep=dict(seed=seed,arm=name,scope=scope,policy=policy,parameters=parameters,model_sha256=file_sha(root/'model.pt'),
                             initial_sha256=file_sha(seedroot/'initial.pt'),output_sha256={k:array_sha(v) for k,v in o.items()},phases_seconds=local,wall_seconds=wall,phase_coverage=sum(local.values())/wall)
                    publish(root/'report.json',rep);results[f'{policy}/seed-{seed}/{name}']=file_sha(root/'report.json');out[name]=o
                    print(json.dumps(dict(policy=policy,seed=seed,arm=name,evaluation=score['evaluation']['accuracy'],seconds=wall)),flush=True)
                except (ValueError,RuntimeError) as exc:
                    error=dict(policy=policy,seed=seed,arm=name,error=str(exc));publish(root/'failure.json',error);errors.append(error);print(json.dumps(error),flush=True)
                finally:del model,opt;torch.cuda.empty_cache()
            if set(out)==set(ARMS)|({'shuffled'} if seed==25 else set()):
                plan,samples,summary=comparisons(out,rows,ctx['labels'],seed);np.savez(seedroot/'bootstrap-plan.npz',**plan);np.savez(seedroot/'bootstrap-samples.npz',**samples);publish(seedroot/'comparisons.json',summary)
            phases[policy+f'/seed-{seed}']=time.perf_counter()-tick
    tick=time.perf_counter();check(all(array_sha(x.cpu().numpy())==array_sha(y) for x,y in zip(store.native,ctx['native'])),'NATIVE_DRIFT')
    after={p:file_sha(Path(p)) for p in ctx['before']};check(after==ctx['before'],'SOURCE_DRIFT');publish(a.output/'after.json',after)
    files={p.relative_to(a.output).as_posix():dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in a.output.rglob('*') if p.is_file()}
    phases['unchanged_parent_and_result_hashes']=time.perf_counter()-tick;wall=time.perf_counter()-start
    report=dict(schema='qasc-combination-results-v1',population_sha256=POPULATION,native_sha256=NATIVE,files=files,arms=results,errors=errors,complete=not errors,
                scientific_admission=False,served=False,sealed_evaluation=False,assertion_verdict='REFUSE_UNADMITTED_PANEL',production_assertions=0,
                source_supplied_evidence=True,free_text_composition_proven=False,phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(a.output/'report.json',report);print(json.dumps(dict(complete=not errors,report_sha256=file_sha(a.output/'report.json'),wall_seconds=wall)),flush=True)
    return 2 if errors else 0

if __name__=='__main__':sys.exit(run_cli(main))
