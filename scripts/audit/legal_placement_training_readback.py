"""Real-source qualification of connected native distribution learning and resume."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch

from audit import legal_stochastic_placement_readback as prior
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model import native_placement_training as train
from model.native_population_training import cpu,fingerprint
from model.native_stochastic_placement import energy_loss
from text.constellation_corpus import file_sha,need

QUAL=('legal-stochastic-placement-qualification-v3-20261023','313a64d75cea56d898f5f604e9598779301cb8ecbc3b127b9b1bfc60232f9706')
COLD=('legal-stochastic-placement-cold-v3-20261023','d49c8143616d80f36e25e5ccda2a23d558993e99d53e89a24031303d9fbdd2cb')
POLICY=dict(device='cpu',threads=1,seed=20261025,placement_seed=20261029,noise_seed=20261030,
            tile=8,samples=4,noise_width=8,learning_rate=1e-4,weight_decay=.01,
            working_bytes=512<<20,cache_bytes=16<<20,output_bytes=16<<20,score_error=1e-10)


def code():
    root=Path(__file__).resolve().parents[1]
    return dict(prior.code(),**{n:file_sha(root/n) for n in
        ('model/native_placement_training.py','audit/legal_placement_training_readback.py','model/loop_training_checkpoint.py')})


def make(target,ref,name):
    return train.NativePlacementForecaster(target,target_pin=ref['pin'],architecture=name,seed=POLICY['seed'],
        placement_seed=POLICY['placement_seed'],cache_bytes=POLICY['cache_bytes'],output_bytes=POLICY['output_bytes'])


def donor_selection(items):
    # Only attested identity/layout metadata participates in selection, never a score.
    shapes={i['document']:tuple(tuple(f[n].shape for n in f) for f in i['frames']) for i in items}
    identities={i['document']:tuple(fingerprint(f) for f in i['frames']) for i in items}
    result={}
    for item in items:
        rows=[];d=item['document']
        for offset in range(len(item['codes'])):
            candidates=[other['document'] for other in items if other['side']==item['side']
                and other['component']!=item['component'] and len(other['frames'])>offset
                and shapes[other['document']][:offset+1]==shapes[d][:offset+1]
                and identities[d][offset+1] not in identities[other['document']][:offset+1]]
            chosen=min(candidates,key=lambda other:hashlib.sha256(f'placement-donor:{d}:{offset}:{other}'.encode()).digest()) if candidates else -1
            rows.append(chosen)
        result[str(d)]=rows
    return result


@torch.no_grad()
def contexts(model,items):
    result={}
    for item in items:
        stream=model.predictor.stream();state=None;parts=[]
        for lo in range(0,len(item['frames']),POLICY['tile']):
            h,state=stream.step(item['frames'][lo:lo+POLICY['tile']],context=str(item['document']),offset=lo,
                                state=state,detach_history=True)
            parts.append(h)
        result[item['document']]=torch.cat(parts)
    return result


@torch.no_grad()
def evaluate(model,target,labels,items,donors,output):
    output.mkdir();cache=contexts(model,items);rows=[]
    for item in items:
        d=item['document'];hidden=cache[d][:-1];target_frames=item['frames'][1:];codes=item['codes'];counts=item['counts']
        noise=torch.randn(len(codes),4,8,generator=torch.Generator().manual_seed(20261031+d))
        donor=donors[str(d)];eligible=[i for i,v in enumerate(donor) if v>=0]
        wrong_hidden=torch.stack([cache[donor[i]][i] for i in eligible]) if eligible else None
        values={}
        for name,(head,h) in model.contexts(hidden).items():
            placement=model.placement[name];forecast=head.forecast(h);discrete=head.loss(h,codes,counts)
            generated=placement(h,codes,counts,noise,target=target);score=energy_loss(generated,target_frames,placement.variances())
            np_scores=[prior.numpy_energy(r,t,placement.variances()) for r,t in zip(generated,target_frames)]
            err=abs(float(score['loss'])-sum(r['loss'] for r in np_scores)/len(np_scores))
            need(err<=POLICY['score_error'],'PLACEMENT_FIT_NUMPY_ENERGY')
            for level,x in enumerate(discrete['logits']):
                ce=prior.branch.proof.recount(x,codes[:,level])['ce'].mean()
                need(abs(float(ce)-float(discrete['region'][level]))<=5e-7,'PLACEMENT_FIT_NUMPY_CE')
            base=target.reconstruct(target.values(codes),[{n:int(counts[n][i]) for n in model.roster} for i in range(len(codes))])
            baseline=energy_loss(tuple({n:x[None].expand(4,*x.shape) for n,x in f.items()} for f in base),target_frames,placement.variances())
            measured=prior.reencode(generated,labels);same=measured==codes.numpy()[:,None,:]
            wrong=None;matched=None
            if eligible:
                wrong_context=model.contexts(wrong_hidden)[name][1]
                wrong=placement(wrong_context,codes[eligible],{n:x[eligible] for n,x in counts.items()},noise[eligible],target=target)
                ws=energy_loss(wrong,tuple(target_frames[i] for i in eligible),placement.variances())
                cs=energy_loss(tuple(generated[i] for i in eligible),tuple(target_frames[i] for i in eligible),placement.variances())
                matched=dict(rows=len(eligible),joint_correct=float(cs['joint'].mean()),joint_wrong=float(ws['joint'].mean()),
                    loss_correct=float(cs['loss']),loss_wrong=float(ws['loss']),
                    lane_correct={n:float(x.mean()) for n,x in cs['lanes'].items()},lane_wrong={n:float(x.mean()) for n,x in ws['lanes'].items()})
            autonomous=[];auto_prefix=np.zeros(3,dtype=np.int64);shape={n:0 for n in model.roster};all_shapes=0
            for i in range(len(codes)):
                fc=forecast['codes'][i:i+1];fw={n:x[i:i+1] for n,x in forecast['counts'].items()}
                # One explicit draw per autonomous forecast, no best-of-K selection.
                sample=placement(h[i:i+1],fc,fw,noise[i:i+1,:1],target=target)
                mc=prior.reencode(sample,labels);auto_prefix+=np.logical_and.accumulate(mc==fc.numpy()[:,None,:],axis=2).sum((0,1))
                matches={n:int(fw[n][0])==len(target_frames[i][n]) for n in model.roster}
                for n,ok in matches.items():shape[n]+=int(ok)
                all_shapes+=int(all(matches.values()))
                autonomous.append(cpu(dict(sample=sample,measured_codes=torch.from_numpy(mc.copy()),forecast_codes=fc,forecast_counts=fw)))
            exact=(forecast['codes']==codes).numpy()
            record=dict(document=d,side=item['side'],proposal=name,targets=len(codes),matched=matched,
                eligible_donors=len(eligible),missing_donors=len(codes)-len(eligible),
                native_loss=float(score['loss']),native_joint=float(score['joint'].mean()),
                native_lanes={n:float(v.mean()) for n,v in score['lanes'].items()},baseline_loss=float(baseline['loss']),
                baseline_joint=float(baseline['joint'].mean()),baseline_lanes={n:float(v.mean()) for n,v in baseline['lanes'].items()},
                region_ce=[float(x) for x in discrete['region']],count_ce={n:float(x) for n,x in discrete['counts'].items()},
                code_hits=exact.sum(0).tolist(),prefix_hits=np.logical_and.accumulate(exact,axis=1).sum(0).tolist(),
                count_hits={n:int((forecast['counts'][n]==counts[n]).sum()) for n in model.roster},
                conditional_roundtrip=np.logical_and.accumulate(same,axis=2).sum((0,1)).tolist(),
                autonomous_roundtrip=auto_prefix.tolist(),autonomous_shape_matches=shape,autonomous_all_shapes_match=all_shapes,
                score_recount_error=err)
            values[name]=cpu(dict(noise=noise,forecast=forecast,discrete=discrete,generated=generated,score=score,baseline=baseline,
                       measured_codes=torch.from_numpy(measured.copy()),wrong=wrong,autonomous=autonomous,record=record))
            rows.append(record)
        prior.branch.persist(output/f'document-{d}.pt',values)
    pin=write_committed(output/'metrics.json',rows)
    return pin,rows


def gradient_route(model,item,target):
    model.zero_grad(set_to_none=True);frames=item['frames'][:2];stream=model.predictor.stream()
    hidden,_=stream.step(frames[:1],context='energy-only',offset=0,detach_history=True)
    noises=torch.randn(1,4,8,generator=torch.Generator().manual_seed(20261101))
    out=model.losses(hidden,item['codes'][:1],{n:x[:1] for n,x in item['counts'].items()},frames[1:],noises,target=target)
    loss=torch.stack([r['native']['loss'] for r in out['proposals'].values()]).mean();loss.backward()
    grads=cpu({n:p.grad for n,p in model.named_parameters()})
    needed=['predictor.projectors.'+n+'.' for n in model.roster]
    needed+=['placement.'+p+'.residual.'+n+'.' for p in model.proposals for n in model.roster]
    for prefix in needed:
        need(any(g is not None and bool(g.abs().max()>0) for n,g in grads.items() if n.startswith(prefix)),
             'PLACEMENT_FIT_ENERGY_ROUTE')
    model.zero_grad(set_to_none=True)
    return dict(native_loss=float(loss.detach()),branches={p:fingerprint({n:g for n,g in grads.items() if n.startswith(p)}) for p in needed}),grads


def negative_controls(model,optimizer,item,held,target,step):
    before=fingerprint((model.state_dict(),optimizer.state_dict()))
    changed=deepcopy(item);changed['frames'][0][model.roster[0]][0,0]+=1
    label=deepcopy(item);label['codes'][0,0]=(label['codes'][0,0]+1)%model.sizes[0]
    run=lambda row=item,opt=optimizer,number=step,policy=POLICY,tt=target:train.update(model,opt,row,number,policy,target=tt)
    fresh=train.optimizer_for(model,POLICY)
    result=dict(held_out_fit=rejected(lambda:run(row=held)),changed_native=rejected(lambda:run(row=changed)),
        changed_labels=rejected(lambda:run(row=label)),stale_cursor=rejected(lambda:run(number=step+1)),
        weights_only_restart=rejected(lambda:run(opt=fresh)),wrong_target=rejected(lambda:run(tt=deepcopy(target))),
        oversized_microtile=rejected(lambda:run(policy=dict(POLICY,working_bytes=1))))
    parameter=next(model.parameters());hook=parameter.register_hook(lambda g:torch.full_like(g,float('nan')))
    try:result['nonfinite_gradient']=rejected(run)
    finally:hook.remove();optimizer.zero_grad(set_to_none=True)
    need(before==fingerprint((model.state_dict(),optimizer.state_dict())),'PLACEMENT_FIT_NEGATIVE_COMMIT')
    return result


def run(a):
    began=tick=time.perf_counter();process=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    for n,pin in code().items():
        dest=a.output/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((Path(__file__).resolve().parents[1]/n).read_bytes());need(file_sha(dest)==pin,'PLACEMENT_FIT_CODE_COPY')
    q,c=prior.branch.proof.pinned(a.root/QUAL[0],QUAL[1]),prior.branch.proof.pinned(a.root/COLD[0],COLD[1])
    need(q['complete'] and c['complete'] and c['previous_pin']==QUAL[1] and q['binding']==c['binding']
         and q['results_pin']==c['results_pin'] and q['binding']['code']==prior.code(),'PLACEMENT_FIT_PREREQUISITE')
    target,ref,labels,items,source_binding=prior.source(a.root);frozen=fingerprint(target.state_dict())
    need(source_binding==q['binding']['source'] and ref==q['binding']['target'],'PLACEMENT_FIT_SOURCE')
    for item in items:item['label_pin']=fingerprint((item['codes'],item['counts']))
    donors=donor_selection(items);donor_pin=write_committed(a.output/'donors.json',donors)
    fitting=[i for i in items if i['side']=='train'];held=next(i for i in items if i['side']=='query')
    binding=dict(code=code(),policy=POLICY,source=source_binding,target=ref,placement_proof=QUAL[1],placement_cold=COLD[1],donors=donor_pin)
    previous=prior.branch.proof.pinned(a.previous,a.previous_pin) if a.previous else None
    if previous:
        need(previous['complete'] and previous['binding']==binding,'PLACEMENT_FIT_COLD_BINDING')
        for n,pin in previous['files'].items():need(file_sha(a.previous/n)==pin,'PLACEMENT_FIT_COLD_BYTES')
    phases=dict(authentication=time.perf_counter()-tick);results={}
    for name in train.ARCHITECTURES:
        start=time.perf_counter();out=a.output/name;out.mkdir();model=make(target,ref,name);optimizer=train.optimizer_for(model,POLICY)
        before=prior.branch.persist(out/'initial.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=0,noise_cursor=POLICY['noise_seed']))
        route,grads=gradient_route(model,fitting[0],target);route_pin=prior.branch.persist(out/'energy-only-gradients.pt',grads)
        initial,initial_metrics=evaluate(model,target,labels,items,donors,out/'initial-evaluation');updates=[]
        for step,item in enumerate(fitting):
            record,gradients,noise=train.update(model,optimizer,item,step,POLICY,target=target)
            state=prior.branch.persist(out/f'checkpoint-{step+1}.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                gradients=gradients,noise=noise,step=step+1,noise_cursor=POLICY['noise_seed']+step+1,record=record,binding=binding))
            updates.append(dict(record=record,state=state))
        restored=make(target,ref,name);opt=train.optimizer_for(restored,POLICY)
        cp=torch.load(out/'checkpoint-2.pt',map_location='cpu',weights_only=True)
        need(cp['binding']==binding and cp['noise_cursor']==POLICY['noise_seed']+2,'PLACEMENT_FIT_RESTORE_BINDING')
        restored.load_state_dict(cp['model'],strict=True);opt.load_state_dict(cp['optimizer'])
        record,grads,noise=train.update(restored,opt,fitting[2],2,POLICY,target=target)
        resumed=prior.branch.persist(out/'resumed-3.pt',dict(model=restored.state_dict(),optimizer=opt.state_dict(),
            gradients=grads,noise=noise,step=3,noise_cursor=POLICY['noise_seed']+3,record=record,binding=binding))
        need(resumed==updates[2]['state'],'PLACEMENT_FIT_RESUME_EXACT')
        final,final_metrics=evaluate(model,target,labels,items,donors,out/'final-evaluation')
        controls=negative_controls(model,optimizer,fitting[-1],held,target,len(fitting))
        need(fingerprint(target.state_dict())==frozen,'PLACEMENT_FIT_FROZEN_TARGET_CHANGED')
        results[name]=dict(parameters=sum(p.numel() for p in model.parameters()),initial_state=before,initial=initial,final=final,
            gradient_route=route,gradient_route_payload=route_pin,updates=updates,resumed=resumed,controls=controls,
            initial_metrics=initial_metrics,final_metrics=final_metrics,final_state=fingerprint((model.state_dict(),optimizer.state_dict())))
        phases[name]=time.perf_counter()-start;print(json.dumps(dict(architecture=name,parameters=results[name]['parameters'],seconds=phases[name])),flush=True)
    tick=time.perf_counter();need(not torch.cuda.is_initialized(),'PLACEMENT_FIT_UNEXPECTED_CUDA')
    result_pin=write_committed(a.output/'results.json',results)
    if previous:need(result_pin==previous['results_pin'],'PLACEMENT_FIT_COLD_EXACT')
    files={str(f.relative_to(a.output)):file_sha(f) for f in a.output.rglob('*') if f.is_file()}
    phases['readback']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(sum(phases.values())/seconds>=.9,'PLACEMENT_FIT_PHASE_ACCOUNTING')
    report=dict(complete=True,binding=binding,files=files,results_pin=result_pin,previous_pin=a.previous_pin,documents=15,
        train_targets=66,development_targets=64,query_targets=70,updates_per_architecture=5,
        phase_seconds=phases,seconds=seconds,cpu_seconds=time.process_time()-process,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,number_status='measured',acceptance='author-checked',
        capacity_matched=False,gpu_qualified=False,scientific_admission=False,capability_demonstrated=False)
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),seconds=seconds)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--previous',type=Path);parser.add_argument('--previous-pin')
    args=parser.parse_args();need(bool(args.previous)==bool(args.previous_pin),'PLACEMENT_FIT_PREVIOUS_ARGUMENTS');run(args)
