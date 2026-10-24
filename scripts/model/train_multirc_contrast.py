"""Fixed matched-sampling versus paired-loss experiment on pinned source labels."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.multirc_context import load,check
from model.multirc_binding import Store,Binding,donors,infer,controls
from model.multirc_contrast import BASELINE,CONFIG,ARMS,scope,groups,schedule,objective,objective_controls
from model.multirc_metrics import plan,evaluate,comparisons,intervention_change
from model.multirc_contrast_metrics import pair_sums,compare,advancement
from model.renderer_conditioning import gpu_lease
from model.scifact_readout import array_sha
from audit.multirc_binding_readback import files,manifest,preparation,read_json
from audit.multirc_contrast_scope import validate
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha
from lens.lenscommon import run_cli

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('population','native','quality','raw','baseline','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();check(not a.output.exists(),'CONTRAST_OUTPUT_EXISTS');start=time.perf_counter();phases={};torch.set_num_threads(32)
    ctx=load(a.population,a.native,a.quality,a.raw);check(file_sha(a.baseline/'report.json')==BASELINE,'CONTRAST_BASELINE_PIN')
    parent=read_json(a.baseline/'report.json');manifest(a.baseline,parent);prep=preparation(a.baseline/'preparation',ctx,a.native,a.raw)
    before={**ctx['before'],str(a.baseline/'report.json'):BASELINE}
    for name,v in parent['files'].items():before[str(a.baseline/name)]=v['sha256']
    a.output.mkdir(parents=True);shutil.copytree(a.baseline/'preparation',a.output/'preparation');publish(a.output/'preparation-readback.json',prep)
    publish(a.output/'source-before.json',before);publish(a.output/'config.json',CONFIG);grouped=groups(ctx);publish(a.output/'training-groups.json',grouped)
    selections={}
    for seed in CONFIG['seeds']:
        seedroot=a.output/f'seed-{seed}';seedroot.mkdir();selected=schedule(ctx['train'],grouped,seed)
        np.savez(seedroot/'schedule.npz',**selected)
        with np.load(seedroot/'schedule.npz',allow_pickle=False) as z:cold={n:z[n] for n in z.files}
        publish(seedroot/'schedule-readback.json',validate(grouped,cold,ctx));selections[seed]=cold
        shutil.copyfile(a.baseline/f'seed-{seed}/initial.pt',seedroot/'initial.pt')
    phases['cold_sources_reused_preparation_and_all_pair_schedules']=time.perf_counter()-start;tick=time.perf_counter()
    residents=gpu_lease(None,'multirc-matched-contrast-sixteen-fits');(a.output/'gpu-before.txt').write_text(residents,encoding='utf-8')
    store=Store(ctx);pp=plan(ctx);ids=np.flatnonzero(store.available);heldout=np.array([i for i in ids if ctx['rows'][i]['partition']!='train'],np.int64)
    donor=donors(ctx['rows']);np.savez(a.output/'evaluation-scope.npz',ids=ids,heldout=heldout,donors=donor)
    prior=float(ctx['labels'][ctx['train']].mean());prior_logits=np.broadcast_to(np.log([1-prior,prior]),(len(ids),2)).copy()
    prior_metrics,prior_aggregate,_=evaluate(prior_logits,ids,pp);np.save(a.output/'prior-logits.npy',prior_logits);publish(a.output/'prior-metrics.json',prior_metrics)
    aggregates={'prior':prior_aggregate};pair_aggregates={'prior':pair_sums(prior_logits,ids,pp)}
    for seed in CONFIG['seeds']:
        logits=np.load(a.baseline/f'seed-{seed}/joint/logits.npy',allow_pickle=False)
        _,aggregates[f'seed-{seed}/uniform-ce'],_=evaluate(logits,ids,pp);pair_aggregates[f'seed-{seed}/uniform-ce']=pair_sums(logits,ids,pp)
    modules=('model.multirc_context','model.multirc_binding','model.multirc_contrast','model.multirc_metrics','model.multirc_contrast_metrics','audit.multirc_binding_readback','audit.multirc_contrast_scope')
    code={n:file_sha(Path(sys.modules[n].__file__)) for n in modules};code['model.train_multirc_contrast']=file_sha(Path(__file__))
    publish(a.output/'runtime.json',dict(torch=torch.__version__,cuda=torch.version.cuda,platform=sys.platform,device=torch.cuda.get_device_name(0),source_code=code))
    phases['lease_upload_and_fixed_baselines']=time.perf_counter()-tick;results={};errors=[]
    for seed in CONFIG['seeds']:
        seedroot=a.output/f'seed-{seed}';initial=torch.load(seedroot/'initial.pt',map_location='cpu',weights_only=True)
        selected=selections[seed];steps=selected['rows'];pair_positions=torch.as_tensor(selected['pairs'],device='cuda')
        for arm in (*ARMS,*(['shuffled-rank'] if seed==25 else [])):
            out=seedroot/arm;out.mkdir();fit=time.perf_counter();local={};model=opt=None;key=f'seed-{seed}/{arm}';model_scope=scope(arm);ranked=arm!='joint-ce'
            try:
                target=ctx['labels'].copy()
                if arm=='shuffled-rank':target[ctx['train']]=np.random.default_rng(seed).permutation(target[ctx['train']])
                np.save(out/'training-labels.npy',target[ctx['train']]);tg=torch.as_tensor(target,device='cuda')
                model=Binding().cuda();model.load_state_dict(initial);parameters=sum(p.numel() for p in model.parameters());check(parameters==1391330,'CONTRAST_MODEL_SIZE')
                opt=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
                torch.cuda.reset_peak_memory_stats();losses=[];local['initialization']=time.perf_counter()-fit;at=time.perf_counter()
                for step,ii in enumerate(steps):
                    opt.zero_grad(set_to_none=True);logits=model(store.batch(ii),model_scope)
                    loss,terms,counts=objective(logits,tg[ii],pair_positions[step],ranked);check(bool(torch.isfinite(loss)),'CONTRAST_NONFINITE_LOSS');loss.backward()
                    if step==0:
                        gradients={k:float(v.grad.norm()) if v.grad is not None else None for k,v in model.named_parameters()}
                        check(any(v is not None and v>0 for v in gradients.values()) and all(v is None or np.isfinite(v) for v in gradients.values()),'CONTRAST_GRADIENTS')
                        np.savez(out/'objective.npz',rows=ii,labels=target[ii],pairs=selected['pairs'][step],logits=logits.detach().cpu().numpy())
                        publish(out/'objective.json',dict(loss=float(loss.detach()),terms=terms.detach().cpu().tolist(),counts=counts.cpu().tolist(),gradients=gradients,ranked=ranked))
                        publish(out/'objective-controls.json',objective_controls(logits.detach(),tg[ii],pair_positions[step],ranked))
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);check(bool(torch.isfinite(norm)),'CONTRAST_NONFINITE_GRADIENT');opt.step()
                    losses.append(torch.cat((loss.detach()[None],terms.detach(),counts)).cpu().numpy())
                    if (step+1)%500==0:print(json.dumps(dict(fit=key,updates=step+1,seconds=time.perf_counter()-at)),flush=True)
                torch.cuda.synchronize();local['training']=time.perf_counter()-at;at=time.perf_counter()
                np.save(out/'losses.npy',np.asarray(losses));torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'model.pt')
                publish(out/'training-time.json',dict(seconds=local['training'],updates=2000,case_updates=128000,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
                model.requires_grad_(False).eval();logits=infer(model,store,model_scope,ids);np.save(out/'logits.npy',logits)
                metrics,aggregates[key],_=evaluate(logits,ids,pp);pair_aggregates[key]=pair_sums(logits,ids,pp);publish(out/'metrics.json',metrics)
                if arm=='joint-rank':
                    intervention={};summary={};base=logits[np.searchsorted(ids,heldout)]
                    for kind in ('remove-context','question-swap','passage-swap'):
                        value=infer(model,store,model_scope,heldout,kind,donor);intervention[kind]=value;summary[kind]=intervention_change(base,value)
                    np.savez(out/'interventions.npz',**intervention);publish(out/'interventions.json',summary)
                publish(out/'controls.json',controls(model,store,ids));local['checkpoint_all_outputs_metrics_controls']=time.perf_counter()-at;wall=time.perf_counter()-fit
                publish(out/'report.json',dict(arm=arm,scope=model_scope,ranked=ranked,seed=seed,parameters=parameters,initial_sha256=file_sha(seedroot/'initial.pt'),
                    files=files(out),phases_seconds=local,wall_seconds=wall,phase_coverage=sum(local.values())/wall,
                    world_truth_assessed=False,scientific_admission=False,served=False,epistemic_understanding_proven=False))
                results[key]=file_sha(out/'report.json');print(json.dumps(dict(fit=key,evaluation=metrics['evaluation'],seconds=wall)),flush=True)
            except (ValueError,RuntimeError) as exc:
                failure=dict(fit=key,error=str(exc));publish(out/'failure.json',failure);errors.append(failure);print(json.dumps(failure),flush=True)
            finally:del model,opt;torch.cuda.empty_cache()
            phases[key]=time.perf_counter()-fit
    tick=time.perf_counter();comparison={};paired={}
    for seed in CONFIG['seeds']:
        left=f'seed-{seed}/joint-rank'
        if left not in aggregates:continue
        for other in ('joint-ce','no-context-rank','no-question-rank','no-answer-rank','uniform-ce',*(['shuffled-rank'] if seed==25 else [])):
            right=f'seed-{seed}/{other}'
            if right in aggregates:comparison[left+'/'+other]=comparisons(aggregates[left],aggregates[right]);paired[left+'/'+other]=compare(pair_aggregates[left],pair_aggregates[right])
        comparison[left+'/train-prior']=comparisons(aggregates[left],aggregates['prior']);paired[left+'/train-prior']=compare(pair_aggregates[left],pair_aggregates['prior'])
    publish(a.output/'comparisons.json',comparison);publish(a.output/'pair-comparisons.json',paired)
    if not errors:publish(a.output/'research-advancement.json',advancement(paired))
    check(all(array_sha(v.cpu().numpy())==array_sha(x) for v,x in zip(store.native,ctx['native'])),'CONTRAST_NATIVE_DRIFT')
    after={p:file_sha(Path(p)) for p in before};check(after==before,'CONTRAST_PARENT_DRIFT');publish(a.output/'after.json',after)
    metadata=files(a.output);phases['paired_comparisons_and_immutable_parent_hashes']=time.perf_counter()-tick;wall=time.perf_counter()-start
    publish(a.output/'report.json',dict(schema='multirc-matched-contrast-v1',baseline_report_sha256=BASELINE,files=metadata,results=results,errors=errors,complete=not errors,
        phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,scientific_admission=False,served=False,world_truth_assessed=False,
        epistemic_understanding_proven=False,production_assertions=0,sealed_evaluation=False,assertion_verdict='REFUSE_UNADMITTED_PANEL',anchor_scope='source_relative_answer_correctness'))
    print(json.dumps(dict(complete=not errors,report_sha256=file_sha(a.output/'report.json'),seconds=wall)),flush=True)
    return 2 if errors else 0

if __name__=='__main__':sys.exit(run_cli(main))
