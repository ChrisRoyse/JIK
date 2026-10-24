"""Real-native forward/backward/score qualification of conditional sample placement."""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch

from audit import legal_branch_predictor_readback as branch
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model.native_population_training import cpu,fingerprint
from model.native_region_predictor import NativeRegionPredictor
from model.native_stochastic_placement import NativeStochasticPlacement,energy_loss
from model.native_target_cache import FrozenLabels
from text.constellation_corpus import file_sha,need

POLICY=dict(seed=20261029,samples=4,noise_width=8,output_bytes=16<<20,device='cpu',threads=1,
            coordinate_error=1e-5,score_error=1e-10,gradient_error=5e-10,optimizer_updates=0)


def code():
    root=Path(__file__).resolve().parents[1]
    return dict(branch.code(),**{n:file_sha(root/n) for n in
        ('model/native_stochastic_placement.py','audit/legal_stochastic_placement_readback.py')})


def source(root):
    # Reuse the persisted causal contexts and native proof, once per document.
    p=branch.proof;prior=p.prior;q=p.pinned(root/branch.QUAL[0],branch.QUAL[1])
    cold=p.pinned(root/branch.COLD[0],branch.COLD[1])
    need(q['complete'] and cold['complete'] and q['binding']==cold['binding']
         and cold['previous_pin']==branch.QUAL[1] and q['results_pin']==cold['results_pin'],'PLACEMENT_PROOF_SOURCE')
    plan=prior.training.pinned(root/branch.QUAL[0]/'selection.json',q['binding']['selection']);authority=plan['source']
    src,split=root/'legal-input-reuse-v1-20261022',root/'legal-case-split-v1-20261022'
    sr,sp=p.pinned(src,authority['source']),p.pinned(split,authority['split'])
    mapping=p.array(src,sr,'mapping.npy');docs=p.array(src,sr,'documents.npy')
    sides=p.array(split,sp,'document-side.npy');components=p.array(split,sp,'component.npy')
    stats_root=root/prior.initialization.STATS[0];stats_report=p.pinned(stats_root,prior.initialization.STATS[1])
    need(file_sha(stats_root/'statistics.pt')==stats_report['files']['statistics.pt'],'PLACEMENT_PROOF_STATS')
    stats=torch.load(stats_root/'statistics.pt',map_location='cpu',weights_only=True)
    target,fit,ref=prior.load_target(root,stats,{n:len(v) for n,v in stats['centers'].items()})
    need(fit['binding']['source']==authority,'PLACEMENT_PROOF_FIT_SOURCE');labels=FrozenLabels(target)
    cache,cc=p.pinned(root/branch.CACHE[0],branch.CACHE[1]),p.pinned(root/branch.CACHE_COLD[0],branch.CACHE_COLD[1])
    need(cache['complete'] and cc['complete'] and cc['previous_pin']==branch.CACHE[1]
         and cache['binding']==cc['binding'] and cache['binding']['source']==authority
         and cc['result']['all_rows_canonical_singleton_exact'],'PLACEMENT_PROOF_CACHE')
    maps=[]
    for n in ('codes','counts'):
        f=root/branch.CACHE[0]/'export'/(n+'.npy')
        need(file_sha(f)==cache['files']['export/'+n+'.npy'],'PLACEMENT_PROOF_CACHE_BYTES')
        maps.append(np.load(f,mmap_mode='r',allow_pickle=False))
    items=[]
    for row in plan['documents']:
        f=root/branch.QUAL[0]/f"document-{row['document']}/outputs.pt"
        need(file_sha(f)==q['files'][str(f.relative_to(root/branch.QUAL[0]))],'PLACEMENT_PROOF_NATIVE_BYTES')
        saved=torch.load(f,map_location='cpu',weights_only=True);start,stop=map(int,docs[row['document']])
        need((start,stop)==(row['start'],row['stop']) and row['side']==('train','development','query')[int(sides[row['document']])]
             and len(saved['frames'])==len(saved['hidden'])==stop-start
             and fingerprint(saved['frames'])==saved['native_pin'],'PLACEMENT_PROOF_LINEAGE')
        uids=np.asarray(mapping[start+1:stop]);codes=np.array(maps[0][uids],dtype=np.int64);counts=np.array(maps[1][uids],dtype=np.int64)
        direct,dc=labels.encode(saved['frames'][1:],reference=True,canonical=True)
        need(np.array_equal(direct,codes) and np.array_equal(dc,counts),'PLACEMENT_PROOF_LABEL_RECOUNT')
        items.append(dict(**row,component=int(components[row['document']]),uids=uids.tolist(),
             frames=saved['frames'],hidden=saved['hidden'][:-1],codes=torch.from_numpy(codes),
             counts={n:torch.from_numpy(counts[:,i].copy()) for i,n in enumerate(target.roster)},native_pin=saved['native_pin']))
    for a in [mapping,docs,sides,components]+maps:a._mmap.close()
    need(len(items)==15 and sum(len(i['codes']) for i in items)==200,'PLACEMENT_PROOF_SCOPE')
    return target,ref,labels,items,dict(native=branch.QUAL[1],cold=branch.COLD[1],cache=branch.CACHE[1],
                                     cache_cold=branch.CACHE_COLD[1],source=authority,selection=q['binding']['selection'])


def numpy_energy(row,target,variances,empirical=False,gradient=False):
    names=tuple(variances);k=len(names);s=len(row[names[0]]);lanes={};pair={};td={};arrays={};scales={}
    for n in names:
        x=np.asarray(row[n].detach().numpy(),dtype=np.float64);y=np.asarray(target[n].numpy(),dtype=np.float64)
        arrays[n]=(x,y);scales[n]=y.size*float(variances[n]);td[n]=np.sqrt(np.square(x-y).sum((1,2))/scales[n])
        pair[n]=np.array([[np.sqrt(np.square(a-b).sum()/scales[n]) for b in x] for a in x])
        denominator=s*s if empirical else s*(s-1)
        lanes[n]=float(td[n].mean()-.5*pair[n].sum()/denominator)
    jt=np.sqrt(sum(v*v for v in td.values())/k);jp=np.sqrt(sum(v*v for v in pair.values())/k)
    joint=float(jt.mean()-.5*jp.sum()/denominator);out=dict(lanes=lanes,joint=joint,loss=sum(lanes.values())/k+joint)
    if gradient:
        grads={}
        for n in names:
            x,y=arrays[n];g=np.zeros_like(x)
            for a in range(s):
                if td[n][a]>0:g[a]+=(x[a]-y)/(k*s*scales[n]*td[n][a])
                if jt[a]>0:g[a]+=(x[a]-y)/(k*s*scales[n]*jt[a])
                for b in range(s):
                    if a==b:continue
                    if pair[n][a,b]>0:g[a]-=(x[a]-x[b])/(k*denominator*scales[n]*pair[n][a,b])
                    if jp[a,b]>0:g[a]-=(x[a]-x[b])/(k*denominator*scales[n]*jp[a,b])
            grads[n]=g
        out['gradients']=grads
    return out


def numpy_head(model,hidden,codes,counts,noise,target):
    def arr(t):return t.detach().double().numpy()
    def linear(x,m):return x@arr(m.weight).T+arr(m.bias)
    with torch.no_grad():z=target.values(codes);base=target.reconstruct(z,[{n:int(counts[n][0]) for n in model.roster}])
    result={}
    for lane,n in enumerate(model.roster):
        h=linear(arr(hidden[:,lane]),model.context[n])+linear(arr(z),model.region[n])
        e=linear(arr(noise),model.noise[n])[0];w=int(counts[n][0])
        phase=np.arange(w)[:,None]*arr(target.encoder.frequency)[None,:]
        position=np.stack((np.sin(phase),np.cos(phase)),axis=-1).reshape(w,-1)
        x=h[0][None,None,:]+e[:,None,:]+linear(position,model.position[n])[None,:,:]
        norm,up,_,down=model.residual[n]
        x=(x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+norm.eps)
        x=x*arr(norm.weight)+arr(norm.bias);x=linear(x,up)
        x=.5*x*(1+np.vectorize(math.erf)(x/math.sqrt(2)))
        result[n]=arr(base[0][n])[None]+math.sqrt(float(model.variances()[n]))*linear(x,down)
    return result


@torch.no_grad()
def reencode(samples,labels):
    frames=tuple({n:x[s].detach() for n,x in row.items()} for row in samples for s in range(len(next(iter(row.values())))))
    codes=[]
    for lo in range(0,len(frames),128):codes.append(labels.encode(frames[lo:lo+128],reference=True,canonical=True)[0])
    return np.concatenate(codes).reshape(len(samples),len(next(iter(samples[0].values()))),-1)


def score_controls(items,variances):
    eligible=[f for item in items if item['side']=='train' for f in item['frames']]
    first=eligible[0];second=next(f for f in eligible[1:] if all(f[n].shape==first[n].shape for n in first)
                                and all(not torch.equal(f[n],first[n]) for n in first))
    roster=tuple(first);middle={n:(first[n]+second[n])/2 for n in roster}
    correct={n:torch.stack((first[n],second[n])) for n in roster}
    collapsed={n:torch.stack((middle[n],middle[n])) for n in roster}
    wrong={n:(correct[n] if i<len(roster)//2 else correct[n].flip(0)) for i,n in enumerate(roster)}
    result={}
    for name,sample in [('correct',correct),('collapsed',collapsed),('wrong_coupling',wrong)]:
        scores=[energy_loss((sample,),(y,),variances,empirical=True) for y in (first,second)]
        result[name]=dict(joint=sum(float(v['joint'][0]) for v in scores)/2,
                         loss=sum(float(v['loss']) for v in scores)/2,
                         lanes={n:sum(float(v['lanes'][n][0]) for v in scores)/2 for n in roster})
    need(result['correct']['loss']<result['collapsed']['loss'],'PLACEMENT_PROOF_COLLAPSE_CONTROL')
    need(max(abs(result['correct']['lanes'][n]-result['wrong_coupling']['lanes'][n]) for n in roster)<1e-12
         and result['correct']['joint']<result['wrong_coupling']['joint'],'PLACEMENT_PROOF_COUPLING_CONTROL')
    zero={n:torch.stack((v,v)).double().requires_grad_(True) for n,v in first.items()}
    loss=energy_loss((zero,),(first,),variances)['loss'];loss.backward()
    need(float(loss)==0 and all(torch.equal(x.grad,torch.zeros_like(x)) for x in zero.values()),'PLACEMENT_PROOF_ZERO_DISTANCE_GRADIENT')
    return result,dict(first=first,second=second,correct=correct,collapsed=collapsed,wrong_coupling=wrong)


def negatives(model,item,target):
    h=item['hidden'][:1];c=item['codes'][:1];w={n:x[:1] for n,x in item['counts'].items()}
    z=torch.randn(1,4,POLICY['noise_width'],generator=torch.Generator().manual_seed(POLICY['seed']))
    run=lambda hh=h,cc=c,ww=w,zz=z,tt=target:model(hh,cc,ww,zz,target=tt)
    before=fingerprint(model.state_dict());nan=h.clone();nan[0,0,0]=float('nan');nz=z.clone();nz[0,0,0]=float('nan')
    bad=c.clone();bad[0,0]=model.sizes[0];zero={n:x.clone() for n,x in w.items()};zero[model.roster[0]].zero_()
    excessive={n:x.clone() for n,x in w.items()};excessive[model.roster[0]].fill_(model.max_windows[model.roster[0]]+1)
    result=dict(missing_context_slot=rejected(lambda:run(hh=h[:,:-1])),nonfinite_context=rejected(lambda:run(hh=nan)),
        nonfinite_noise=rejected(lambda:run(zz=nz)),missing_region=rejected(lambda:run(cc=None)),
        wrong_region=rejected(lambda:run(cc=bad)),missing_count_slot=rejected(lambda:run(ww={})),
        zero_count=rejected(lambda:run(ww=zero)),count_range=rejected(lambda:run(ww=excessive)),
        changed_frozen_target=rejected(lambda:run(tt=deepcopy(target))))
    bound=model.output_bytes;model.output_bytes=1;result['output_memory']=rejected(run);model.output_bytes=bound
    sample=run();y=(item['frames'][1],);missing={n:x for n,x in sample[0].items() if n!=model.roster[0]}
    bad_target=cpu(y);bad_target[0][model.roster[0]]=bad_target[0][model.roster[0]][:,:-1]
    result['missing_sample_slot']=rejected(lambda:energy_loss((missing,),y,model.variances()))
    result['target_layout']=rejected(lambda:energy_loss(sample,bad_target,model.variances()))
    result['one_score_sample']=rejected(lambda:energy_loss(tuple({n:x[:1] for n,x in row.items()} for row in sample),y,model.variances()))
    result['score_memory']=rejected(lambda:energy_loss(sample,y,model.variances(),working_bytes=1))
    bad_variance=dict(model.variances());bad_variance[model.roster[0]]=0.
    result['invalid_variance']=rejected(lambda:energy_loss(sample,y,bad_variance))
    disposable=deepcopy(target)
    probe=NativeStochasticPlacement(disposable,target_pin=model.target_pin,seed=POLICY['seed'])
    with torch.no_grad():disposable.books[0].add_(1.)
    result['mutated_target_weights']=rejected(lambda:probe(h,c,w,z,target=disposable))
    try:model(h,c,w,z,target=target,target_readings=y)
    except TypeError:result['forbidden_target_input']='TypeError'
    else:raise AssertionError('PLACEMENT_PROOF_TARGET_ARGUMENT')
    need(before==fingerprint(model.state_dict()),'PLACEMENT_PROOF_NEGATIVE_MUTATION')
    return result


def run(a):
    began=tick=time.perf_counter();process=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    for n,pin in code().items():
        dest=a.output/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((Path(__file__).resolve().parents[1]/n).read_bytes());need(file_sha(dest)==pin,'PLACEMENT_PROOF_CODE_COPY')
    target,ref,labels,items,source_binding=source(a.root)
    model=NativeStochasticPlacement(target,target_pin=ref['pin'],seed=POLICY['seed'],output_bytes=POLICY['output_bytes'])
    region=NativeRegionPredictor(target,target_pin=ref['pin'],seed=branch.POLICY['seed'])
    before=fingerprint(model.state_dict());frozen=fingerprint(target.state_dict())
    binding=dict(code=code(),policy=POLICY,source=source_binding,target=ref,initial_state=before)
    previous=branch.proof.pinned(a.previous,a.previous_pin) if a.previous else None
    if previous:
        need(previous['complete'] and previous['binding']==binding,'PLACEMENT_PROOF_COLD_BINDING')
        for n,pin in previous['files'].items():need(file_sha(a.previous/n)==pin,'PLACEMENT_PROOF_COLD_BYTES')
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter();results=[];auto=[]
    for item in items:
        h,c,w=item['hidden'],item['codes'],item['counts'];y=item['frames'][1:]
        noise=torch.randn(len(h),POLICY['samples'],POLICY['noise_width'],generator=torch.Generator().manual_seed(POLICY['seed']+item['document']))
        model.zero_grad(set_to_none=True);samples=model(h,c,w,noise,target=target);score=energy_loss(samples,y,model.variances())
        references=[numpy_energy(row,t,model.variances()) for row,t in zip(samples,y)]
        err=max([abs(float(score['loss'].detach())-sum(r['loss'] for r in references)/len(references))]
                +[abs(float(score['joint'][i].detach())-r['joint']) for i,r in enumerate(references)]
                +[abs(float(score['lanes'][n][i].detach())-r['lanes'][n]) for i,r in enumerate(references) for n in model.roster])
        need(err<=POLICY['score_error'],'PLACEMENT_PROOF_SCORE_RECOUNT')
        gradients=None
        if item['side']=='train':
            score['loss'].backward();gradients=cpu({n:p.grad for n,p in model.named_parameters()})
            need(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients.values()),'PLACEMENT_PROOF_GRADIENTS')
            for n in model.roster:
                for module in ('context','region','noise','position','residual'):
                    need(any(bool(g.abs().max()>0) for k,g in gradients.items() if k.startswith(module+'.'+n+'.')),'PLACEMENT_PROOF_BRANCH_GRADIENT')
        sampled_codes=reencode(samples,labels);equal=sampled_codes==c.numpy()[:,None,:]
        out=a.output/f"document-{item['document']}";out.mkdir()
        payload=cpu(dict(hidden=h,codes=c,counts=w,noise=noise,samples=samples,targets=y,score=score,
                         sampled_codes=torch.from_numpy(sampled_codes.copy()),gradients=gradients))
        pin=branch.persist(out/'conditional.pt',payload)
        results.append(dict(document=item['document'],side=item['side'],targets=len(h),candidates=len(h)*POLICY['samples'],
            score_recount_error=err,mean_loss=float(score['loss'].detach()),payload=pin,
            mean_joint=float(score['joint'].mean().detach()),per_lane={n:float(v.mean().detach()) for n,v in score['lanes'].items()},
            region_prefix_hits=np.logical_and.accumulate(equal,axis=2).sum((0,1)).tolist(),
            distinct_sample_delta=max(float((x[0]-x[1]).detach().abs().max()) for row in samples for x in row.values()),
            train_gradient_pin=fingerprint(gradients) if gradients else None))
        with torch.no_grad():
            prediction=region.forecast(h);records=[];prefix=np.zeros(len(model.sizes),dtype=np.int64)
            shape={n:0 for n in model.roster};all_shapes=0
            # One context per autonomous tile: forecast counts can be much larger
            # than the true counts. Samples remain batched within every native slot.
            for i in range(len(h)):
                pc=prediction['codes'][i:i+1];pw={n:v[i:i+1] for n,v in prediction['counts'].items()}
                generated=model(h[i:i+1],pc,pw,noise[i:i+1],target=target)
                measured=reencode(generated,labels);e=measured==pc.numpy()[:,None,:]
                prefix+=np.logical_and.accumulate(e,axis=2).sum((0,1))
                matching={n:int(pw[n][0])==len(y[i][n]) for n in model.roster}
                for n,good in matching.items():shape[n]+=int(good)
                all_shapes+=int(all(matching.values()))
                records.append(cpu(dict(codes=pc,counts=pw,noise=noise[i:i+1],samples=generated,
                                        measured_codes=torch.from_numpy(measured.copy()),target_shape_matches=matching)))
            auto_pin=branch.persist(out/'autonomous.pt',records)
            auto.append(dict(document=item['document'],side=item['side'],targets=len(h),candidates=len(h)*POLICY['samples'],
                region_prefix_hits=prefix.tolist(),shape_matches=shape,all_shapes_match=all_shapes,payload=auto_pin,
                wrong_shape_score='not_defined_no_padding_or_target_count_substitution'))
        print(json.dumps(dict(document=item['document'],targets=len(h),conditional_region_hits=results[-1]['region_prefix_hits'],autonomous_region_hits=auto[-1]['region_prefix_hits'])),flush=True)
    phases['generation_score_gradients_region_readback']=time.perf_counter()-tick;tick=time.perf_counter()
    item=next(i for i in items if i['side']=='train');h=item['hidden'][:1];c=item['codes'][:1];w={n:x[:1] for n,x in item['counts'].items()}
    noise=torch.randn(1,4,POLICY['noise_width'],generator=torch.Generator().manual_seed(POLICY['seed']))
    sample=model(h,c,w,noise,target=target);np_head=numpy_head(model,h,c,w,noise,target)
    ce=max(float(np.abs(sample[0][n].detach().numpy()-v).max()) for n,v in np_head.items())
    need(ce<=POLICY['coordinate_error'],'PLACEMENT_PROOF_NUMPY_HEAD')
    leaves={n:x.detach().double().requires_grad_(True) for n,x in sample[0].items()};y=(item['frames'][1],)
    loss=energy_loss((leaves,),y,model.variances());loss['loss'].backward()
    reference=numpy_energy(leaves,y[0],model.variances(),gradient=True)
    ge=max(float(np.abs(leaves[n].grad.numpy()-g).max()) for n,g in reference['gradients'].items())
    need(ge<=POLICY['gradient_error'],'PLACEMENT_PROOF_NUMPY_GRADIENT')
    controls,control_vectors=score_controls(items,model.variances());negative=negatives(model,item,target)
    integration={}
    with torch.no_grad():
        # Same producer works for the two alternative typed context structures.
        for name,constructor in branch.ARMS.items():
            predictor=constructor(target,target_pin=ref['pin'],seed=branch.POLICY['seed'])
            hidden=branch.encode(predictor,item['frames'][:-1]);integration[name]={}
            for proposal,(head,context) in branch.outputs(predictor,hidden[:1]).items():
                forecast=head.forecast(context);out=model(context,forecast['codes'],forecast['counts'],noise,target=target)
                integration[name][proposal]=fingerprint(out)
        zero=model(h,c,w,torch.zeros_like(noise),target=target)
        need(all(torch.equal(x[0],x[j]) for row in zero for x in row.values() for j in range(1,len(x))), 'PLACEMENT_PROOF_NOISE_CONTROL')
        replay=model(h,c,w,noise,target=target)
        need(fingerprint(sample)==fingerprint(replay),'PLACEMENT_PROOF_EXPLICIT_NOISE_REPLAY')
    proof_payload=dict(sample=cpu(sample),noise=noise,score_gradients={n:torch.from_numpy(g) for n,g in reference['gradients'].items()},
                       control_vectors=control_vectors,initial=cpu(model.state_dict()))
    proof_pin=branch.persist(a.output/'controls.pt',proof_payload)
    need(fingerprint(model.state_dict())==before and fingerprint(target.state_dict())==frozen,'PLACEMENT_PROOF_STATE_CHANGED')
    need(not torch.cuda.is_initialized(),'PLACEMENT_PROOF_UNEXPECTED_CUDA')
    rp=write_committed(a.output/'results.json',dict(conditional=results,autonomous=auto,controls=controls,negative=negative,
        integration=integration,numpy_head_error=ce,numpy_gradient_error=ge,proof_payload=proof_pin))
    if previous:need(rp==previous['results_pin'],'PLACEMENT_PROOF_COLD_EXACT')
    phases['controls_and_integration']=time.perf_counter()-tick;tick=time.perf_counter()
    files={str(f.relative_to(a.output)):file_sha(f) for f in a.output.rglob('*') if f.is_file()}
    phases['publication_hashes']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(sum(phases.values())/seconds>=.9,'PLACEMENT_PROOF_PHASE_ACCOUNTING')
    report=dict(complete=True,binding=binding,results_pin=rp,files=files,previous_pin=a.previous_pin,
        parameters=sum(p.numel() for p in model.parameters()),targets=200,conditional_candidates=800,autonomous_candidates=800,
        optimizer_updates=0,scientific_admission=False,capability_demonstrated=False,gpu_qualified=False,
        phases_seconds=phases,seconds=seconds,cpu_seconds=time.process_time()-process,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,acceptance='author-checked',number_status='measured')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),seconds=seconds)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--previous',type=Path);parser.add_argument('--previous-pin')
    args=parser.parse_args();need(bool(args.previous)==bool(args.previous_pin),'PLACEMENT_PROOF_PREVIOUS_ARGUMENTS');run(args)
