"""Matched TRAIN-only interventions, with exact per-arm cold continuation."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit.legal_prefix_stream_readback import native_reader
from audit.legal_residual_codebook_readback import CONFIG,POLICY as PRIOR_POLICY,fit_scope,measure,pinned,reconstruction_baselines
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model.legal_learning_source import LegalLearningSource
from model.loop_training_checkpoint import tensor_fingerprint
from model.native_rq_training import NativeRQTrainingModel,RECIPES
from senses.contract import digest
from text.constellation_corpus import file_sha,need
from text.legal_continuation_input import Continuations
from text.legal_reuse_input import ReuseInputs

POLICY=dict(PRIOR_POLICY,updates=512,warmup_steps=256,lloyd_iterations=16,recipes=list(RECIPES),
            diagnostic_steps=[0,32,128,256,512],selection='TRAIN_only_no_promotion',continuation_step=511)


def pins():
    base=Path(__file__).resolve().parents[1]
    return {n:file_sha(base/n) for n in ['audit/legal_rq_training_diagnostics.py','model/native_rq_training.py',
        'model/native_residual_codebook.py','model/native_prefix_transformer.py','model/attention.py',
        'audit/legal_residual_codebook_readback.py']}


def make(dimensions,recipe):
    torch.manual_seed(POLICY['seed']); model=NativeRQTrainingModel(dimensions,CONFIG,POLICY['sizes'],recipe)
    opt=torch.optim.AdamW(model.parameters(),lr=POLICY['learning_rate'],weight_decay=POLICY['weight_decay'],foreach=False,fused=False)
    return model,opt


def checkpoint(path,pin,recipe,native_pin):
    need(file_sha(path)==pin,'RQ_DIAGNOSTIC_CHECKPOINT_PIN')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    need(saved['recipe']==recipe and saved['native_pin']==native_pin and saved['step']==511,'RQ_DIAGNOSTIC_CHECKPOINT_SCOPE')
    return saved


def step(model,opt,frames,i):
    batch=tuple(frames[(i*8+j)%len(frames)] for j in range(8))
    before=digest(tensor_fingerprint(model.state_dict()));opt.zero_grad(set_to_none=True)
    out=model.training_loss(batch,i); out['loss'].backward()
    gradients={n:p.grad for n,p in model.named_parameters()}
    expected_none={n for n in gradients if n.startswith('books.')} if out['continuous'] else set()
    need({n for n,g in gradients.items() if g is None}==expected_none,'RQ_DIAGNOSTIC_DISCONNECTED')
    need(all(g is None or torch.isfinite(g).all() for g in gradients.values()),'RQ_DIAGNOSTIC_NONFINITE_GRADIENT')
    norms={}
    for prefix in ['encoder.','books.','to_latent.']+['projectors.'+n+'.' for n in model.roster]+['decoders.'+n+'.' for n in model.roster]:
        norms[prefix]=sum(float(g.double().square().sum()) for n,g in gradients.items() if n.startswith(prefix) and g is not None)**.5
    pin=digest(tensor_fingerprint(gradients));opt.step()
    need(all(torch.isfinite(p).all() for p in model.parameters()),'RQ_DIAGNOSTIC_NONFINITE_PARAMETER')
    return dict(step=i+1,before=before,after=digest(tensor_fingerprint(model.state_dict())),gradient=pin,gradient_norms=norms,
                optimizer=digest(tensor_fingerprint(opt.state_dict())),loss=float(out['loss'].detach()),
                reconstruction=float(out['reconstruction'].detach()),quantizer=float(out['quantizer'].detach()),continuous=out['continuous'])


def loss_reference(model,frames,i):
    out=model.training_loss(frames[:8],i)
    per_lane={n:float(np.mean([np.mean((d[n].detach().numpy().astype(np.float64)-f[n].numpy().astype(np.float64))**2)
                                    for d,f in zip(out['decoded'],frames[:8])])) for n in model.roster}
    expected=float(np.mean([v*(model.dimensions[n] if model.recipe=='squared_norm_losses' else 1) for n,v in per_lane.items()]))
    error=abs(expected-float(out['reconstruction'].detach()));need(error<=5e-6,'RQ_DIAGNOSTIC_LOSS_REFERENCE')
    rg=torch.autograd.grad(out['reconstruction'],out['latent'],retain_graph=True)[0]
    qg=torch.autograd.grad(out['quantizer'],out['latent'],retain_graph=True)[0]
    return dict(reconstruction_reference_error=error,reconstruction_latent_gradient_norm=float(rg.norm()),
                quantizer_latent_gradient_norm=float(qg.norm()))


@torch.no_grad()
def diagnostic(model,frames,donors,step_number):
    stats,codes=measure(model,frames);latents=model.latent(frames);_,values,_=model.quantize(latents)
    counts=[{n:len(f[n]) for n in model.roster} for f in frames]
    correct=model.reconstruct(values,counts);shuffled=model.reconstruct(values[donors],counts)
    gaps={n:float(torch.stack([(w[n]-f[n]).square().mean()-(d[n]-f[n]).square().mean()
                   for f,d,w in zip(frames,correct,shuffled)]).mean()) for n in model.roster}
    contingency={}
    for frame,code in zip(frames,codes):
        key=','.join(str(len(frame[n])) for n in model.roster)
        if key not in contingency:contingency[key]=[0]*model.sizes[0]
        contingency[key][int(code[0])]+=1
    usage=[]
    for row in stats['usage']:
        p=np.asarray(row,dtype=float)/sum(row);nz=p[p>0]
        usage.append(dict(active=int((p>0).sum()),entropy_bits=float(-(nz*np.log2(nz)).sum()),perplexity=float(np.exp(-(nz*np.log(nz)).sum()))))
    normed=torch.nn.functional.normalize(latents,dim=-1);N=len(latents)
    cosine=float((normed.sum(0).square().sum()-N)/(N*(N-1)))
    return dict(step=step_number,**stats,usage_summary=usage,shuffled_minus_correct_mse=gaps,
                first_code_by_window_counts=contingency,latent_mean_coordinate_sd=float(latents.std(0).mean()),
                mean_pairwise_latent_cosine=cosine),dict(codes=codes,latent=latents,decoded=correct,shuffled=shuffled)


@torch.no_grad()
def input_scales(model,frames):
    result={}
    for n in model.roster:
        x=torch.stack([model.projectors[n](f[n]).mean(0) for f in frames])
        if model.recipe=='input_dimension_scale':
            x=torch.stack([model.projectors[n](f[n]*model.dimensions[n]**.5).mean(0) for f in frames])
        pos=torch.stack([model.encoder.position(torch.arange(len(f[n]))).mean(0) for f in frames])
        result[n]=dict(projected_between_atom_variance=float(x.var(0).sum()),
                       position_between_atom_variance=float(pos.var(0).sum()),projected_rms=float(x.square().mean().sqrt()),
                       position_rms=float(pos.square().mean().sqrt()))
    return result


def run(a):
    start=tick=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    prior=pinned(a.prior/'report.json',a.prior_pin);baseline=pinned(a.baseline/'report.json',a.baseline_pin)
    source=ReuseInputs(baseline['source']['path'],baseline['source']['pin']);continuation=Continuations(Path(baseline['continuations']['path']),baseline['continuations']['pin'])
    split,cold=continuation.report['split'],continuation.report['cold'];bank,proof=baseline['bank'],baseline['proof']
    data=LegalLearningSource(source,split['path'],split['pin'],cold['path'],cold['pin'],bank['path'],bank['pin'])
    need(prior['binding']==data.binding,'RQ_DIAGNOSTIC_SOURCE_BINDING')
    read,_=native_reader(data,Path(proof['path']),proof['pin']);uids=prior['result']['groups']['train'];fit_scope(uids,continuation.presence)
    frames=read(uids);native_pin=digest(tensor_fingerprint(frames));dimensions={n:frames[0][n].shape[1] for n in data.roster}
    need(sum(x.numel()*x.element_size() for f in frames for x in f.values())<=POLICY['buffer_bytes'],'RQ_DIAGNOSTIC_BUFFER')
    # Pick one real TRAIN occurrence per unique input, then require every shuffled donor to be in another case component.
    offsets,occurrences=source.arrays['offsets'],source.arrays['occurrences'];train_occ=[]
    for uid in uids:
        loc=occurrences[int(offsets[uid]):int(offsets[uid+1])];valid=loc[data.sides[loc]==0]
        need(len(valid)>0,'RQ_DIAGNOSTIC_TRAIN_OCCURRENCE');train_occ.append(int(valid[0]))
    ix=np.asarray(train_occ,dtype='<i8');donors=None
    for shift in range(1,len(ix)):
        candidate=np.roll(np.arange(len(ix)),shift)
        try:data.validate_donors(ix,ix[candidate])
        except ValueError:continue
        donors=candidate;break
    need(donors is not None,'RQ_DIAGNOSTIC_CASE_DISJOINT_DONORS')
    phases=dict(authenticate_and_native=time.perf_counter()-tick);tick=time.perf_counter();results={}
    if a.previous is not None:
        previous=pinned(a.previous/'report.json',a.previous_pin)
        need(previous['code']==pins() and previous['native_pin']==native_pin and previous['policy']==POLICY,'RQ_DIAGNOSTIC_CONTINUATION_BINDING')
        for recipe in RECIPES:
            root=a.previous/recipe;ref=previous['results'][recipe]
            saved=checkpoint(root/'resume.pt',ref['resume_pin'],recipe,native_pin)
            model,opt=make(dimensions,recipe);model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
            actual=step(model,opt,frames,511)
            need(actual==ref['ledger'][-1],'RQ_DIAGNOSTIC_COLD_UPDATE')
            final,raw=diagnostic(model,frames,donors,512)
            need(final==ref['checkpoints'][-1],'RQ_DIAGNOSTIC_COLD_DIAGNOSTIC')
            need(file_sha(root/'arithmetic.pt')==ref['arithmetic_pin'],'RQ_DIAGNOSTIC_ARITHMETIC_PIN')
            expected=torch.load(root/'arithmetic.pt',map_location='cpu',weights_only=True)
            need(tensor_fingerprint(raw)==tensor_fingerprint(expected),'RQ_DIAGNOSTIC_COLD_OUTPUT')
            results[recipe]=dict(cold_step_exact=True,cold_codes_and_outputs_exact=True,final_weight_pin=actual['after'],final_optimizer_pin=actual['optimizer'])
    else:
        for recipe in RECIPES:
            began=time.perf_counter();root=a.output/recipe;root.mkdir();model,opt=make(dimensions,recipe);model.prepare(frames)
            scales=input_scales(model,frames);initial,_=diagnostic(model,frames,donors,0);checkpoints=[initial];ledger=[]
            loss_checks=[dict(step=0,**loss_reference(model,frames,0))]
            init=getattr(model,'initialization',[])
            for row in init:need(all(b<=a+1e-6 for a,b in zip(row['objective'],row['objective'][1:])),'RQ_DIAGNOSTIC_LLOYD_INCREASE')
            for i in range(512):
                if i==256 and recipe=='continuous_warmup':model.finish_warmup(frames,opt)
                if i==511:torch.save(dict(recipe=recipe,step=i,native_pin=native_pin,model=model.state_dict(),optimizer=opt.state_dict()),root/'resume.pt')
                ledger.append(step(model,opt,frames,i))
                if i+1 in POLICY['diagnostic_steps']:
                    stats,raw=diagnostic(model,frames,donors,i+1);checkpoints.append(stats)
                    loss_checks.append(dict(step=i+1,**loss_reference(model,frames,i)))
            if recipe=='baseline':
                # Additional registered zero buffers do not alter phase36's actual numerical fit.
                need(all(ledger[i][k]==prior['result']['ledger'][i][k] for i in range(32) for k in ('gradient','loss','optimizer')),
                     'RQ_DIAGNOSTIC_BASELINE_NOT_REPRODUCED')
            torch.save(raw,root/'arithmetic.pt');torch.save(model.state_dict(),root/'final.pt')
            results[recipe]=dict(checkpoints=checkpoints,ledger=ledger,loss_checks=loss_checks,initialization=init,input_scales=scales,
                seconds=time.perf_counter()-began,resume_pin=file_sha(root/'resume.pt'),arithmetic_pin=file_sha(root/'arithmetic.pt'),final_pin=file_sha(root/'final.pt'))
            print(json.dumps(dict(recipe=recipe,seconds=results[recipe]['seconds'],train_codes=checkpoints[-1]['distinct_codes'],per_lane_mse=checkpoints[-1]['per_lane_mse'])),flush=True)
        need(native_pin==digest(tensor_fingerprint(frames)),'RQ_DIAGNOSTIC_NATIVE_CHANGED')
        controls=[dict(name='non_train',code=rejected(lambda:fit_scope(prior['result']['groups']['development'],continuation.presence))),
                  dict(name='invalid_recipe',code=rejected(lambda:make(dimensions,'unknown')))]
        altered=dict(frames[0]);altered.pop(data.roster[0]);controls.append(dict(name='missing_slot',code=rejected(lambda:model.latent((altered,)))))
        target=a.output/'baseline'/'resume.pt';pin=results['baseline']['resume_pin']
        for label,expected_pin,recipe,input_pin in [('wrong_checkpoint_pin','0'*64,'baseline',native_pin),
                ('wrong_recipe',pin,'mean_residual',native_pin),('changed_native',pin,'baseline','0'*64)]:
            controls.append(dict(name=label,code=rejected(lambda:checkpoint(target,expected_pin,recipe,input_pin))))
        results['controls']=controls
    phases['matched_experiments_or_cold_readback']=time.perf_counter()-tick;tick=time.perf_counter()
    data.fresh();continuation.fresh();phases['freshness']=time.perf_counter()-tick;seconds=time.perf_counter()-start
    need(sum(phases.values())/seconds>=.9,'RQ_DIAGNOSTIC_PHASE_ATTRIBUTION')
    report=dict(schema='legal-rq-training-diagnostics-v1',complete=True,policy=POLICY,code=pins(),binding=data.binding,native_pin=native_pin,
                train_uids=uids,train_occurrences=train_occ,donor_indices=donors.tolist(),results=results,
                baselines=reconstruction_baselines({'train':frames},data.roster)['train'],phases_seconds=phases,seconds=seconds,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,number_status='measured',acceptance='author-checked',
                model_promotion=False,development_query_read=False)
    pin=write_committed(a.output/'report.json',report);print(json.dumps(dict(pin=pin,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('baseline','prior'):p.add_argument('--'+n,type=Path,required=True);p.add_argument('--'+n+'-pin',required=True)
    p.add_argument('--previous',type=Path);p.add_argument('--previous-pin');p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
