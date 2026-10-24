"""Matched-source qualification of shared, independent-slot, and whole-proposal heads.

Small explicit CPU proof; not a GPU fallback or a capability/admission campaign.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch

from audit import legal_reciprocal_region_readback as proof
from audit.legal_predictor_training import QUAL, COLD
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model.native_branch_predictor import IndependentNativeRegionPredictor, PerInstrumentConstellationPredictor
from model.native_population_training import cpu, fingerprint
from model.native_region_predictor import NativeRegionPredictor
from model.native_target_cache import FrozenLabels
from text.constellation_corpus import file_sha, need

POLICY = dict(seed=20261025, device='cpu', threads=1, tile=8, cache_bytes=16<<20,
              output_error=5e-6, gradient_error=2e-5, loss_error=5e-7,
              lr=1e-4, weight_decay=.01, scope='exposed_architecture_qualification')
CACHE = ('legal-target-cache-full-full-v2-20261023', '77319fe3eaac3ea9db2ede167519a418151133b5c4641ef19ce6c901d2d5fc72')
CACHE_COLD = ('legal-target-cache-full-cold-v2-20261023', 'd4498a44902db8e642ab368e1e1ead435e7e87caed44cae274245215b2a9a77f')
ARMS = dict(shared=NativeRegionPredictor, own_slot=IndependentNativeRegionPredictor,
            whole_proposals=PerInstrumentConstellationPredictor)


def code():
    root = Path(__file__).resolve().parents[1]
    names = ['model/native_branch_predictor.py', 'audit/legal_branch_predictor_readback.py',
             'audit/legal_reciprocal_region_readback.py', 'audit/legal_predictor_training.py',
             'model/native_target_cache.py']
    return dict(proof.prior.code(), **{n:file_sha(root/n) for n in names})


def source(root):
    q, cold = proof.pinned(root/QUAL[0], QUAL[1]), proof.pinned(root/COLD[0], COLD[1])
    need(q['complete'] and cold['complete'] and cold['previous_pin']==QUAL[1]
         and q['binding']==cold['binding'] and q['results_pin']==cold['results_pin'], 'BRANCH_PROOF_SOURCE')
    selection = proof.prior.training.pinned(root/QUAL[0]/'selection.json',q['binding']['selection'])
    authority = selection['source']
    src, split = root/'legal-input-reuse-v1-20261022', root/'legal-case-split-v1-20261022'
    sr, sp = proof.pinned(src,authority['source']), proof.pinned(split,authority['split'])
    mapping = proof.array(src,sr,'mapping.npy'); documents = proof.array(src,sr,'documents.npy')
    components = proof.array(split,sp,'component.npy'); sides = proof.array(split,sp,'document-side.npy')
    stats_root = root/proof.prior.initialization.STATS[0]
    stats_report = proof.pinned(stats_root,proof.prior.initialization.STATS[1])
    need(file_sha(stats_root/'statistics.pt')==stats_report['files']['statistics.pt'],'BRANCH_PROOF_STATS')
    stats = torch.load(stats_root/'statistics.pt',map_location='cpu',weights_only=True)
    target,fit,ref = proof.prior.load_target(root,stats,{n:len(v) for n,v in stats['centers'].items()})
    need(fit['binding']['source']==authority,'BRANCH_PROOF_FIT_SOURCE')
    labels = FrozenLabels(target)
    cache, cc = proof.pinned(root/CACHE[0],CACHE[1]), proof.pinned(root/CACHE_COLD[0],CACHE_COLD[1])
    need(cache['complete'] and cc['complete'] and cc['previous_pin']==CACHE[1]
         and cache['binding']==cc['binding'] and cache['binding']['source']==authority
         and cc['result']['all_rows_canonical_singleton_exact'], 'BRANCH_PROOF_CACHE_AUTHORITY')
    exports=[]
    for name in ('codes','counts'):
        path=root/CACHE[0]/'export'/(name+'.npy')
        need(file_sha(path)==cache['files']['export/'+name+'.npy'],'BRANCH_PROOF_CACHE_BYTES')
        exports.append(np.load(path,mmap_mode='r',allow_pickle=False))
    baseline=proof.prior.training.pinned(root/CACHE[0]/'baselines.json',cache['files']['baselines.json'])
    need(baseline['rows']==6455636 and not baseline['fit_uses_held_out'],'BRANCH_PROOF_BASELINE_SUPPORT')
    modes=dict(codes=np.array(baseline['majority_codes']),counts={n:int(np.argmax(row))
               for n,row in zip(target.roster,baseline['count_histogram'])})
    items=[]
    for row in selection['documents']:
        path=root/QUAL[0]/f"document-{row['document']}/outputs.pt"
        need(file_sha(path)==q['files'][str(path.relative_to(root/QUAL[0]))],'BRANCH_PROOF_NATIVE_BYTES')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        start,stop=map(int,documents[row['document']])
        need((start,stop)==(row['start'],row['stop']) and len(saved['frames'])==stop-start
             and row['side']==('train','development','query')[int(sides[row['document']])]
             and fingerprint(saved['frames'])==saved['native_pin'], 'BRANCH_PROOF_DOCUMENT_JOIN')
        uids=np.asarray(mapping[start:stop]); codes=np.array(exports[0][uids],dtype=np.int64)
        counts=np.array(exports[1][uids],dtype=np.int64)
        direct,dc=labels.encode(saved['frames'],reference=True,canonical=True)
        need(np.array_equal(direct,codes) and np.array_equal(dc,counts),'BRANCH_PROOF_LABEL_RECOUNT')
        items.append(dict(**row,component=int(components[row['document']]),uids=uids.tolist(),
            frames=saved['frames'],native_pin=saved['native_pin'], all_codes=torch.from_numpy(codes),
            codes=torch.from_numpy(codes[1:].copy()),counts={n:torch.from_numpy(counts[1:,i].copy())
                for i,n in enumerate(target.roster)},
            old_label_changes=int(np.any(codes[1:]!=saved['codes'].numpy(),axis=1).sum())))
    for value in [mapping,documents,components,sides]+exports:value._mmap.close()
    need(len(items)==15 and sum(len(i['codes']) for i in items)==200,'BRANCH_PROOF_POPULATION')
    return target,ref,items,modes,dict(source=authority,native=QUAL[1],cold=COLD[1],cache=CACHE[1],
                               cache_cold=CACHE_COLD[1],selection=q['binding']['selection'],
                               baselines=cache['files']['baselines.json'])


def encode(model, frames, *, tile=8):
    stream=model.stream(); state=None; parts=[]
    for lo in range(0,len(frames),tile):
        hidden,state=stream.step(frames[lo:lo+tile],context='proof-document',offset=lo,
                                state=state,detach_history=False)
        parts.append(hidden)
    return torch.cat(parts)


def outputs(model, hidden):
    if isinstance(model,PerInstrumentConstellationPredictor):
        return {n:(model.readouts[n],model.proposal_context(hidden,n)) for n in model.roster}
    return {'assembled':(model,hidden)}


@torch.no_grad()
def evaluate(model, items, target, modes):
    rows=[]; payload={}
    for item in items:
        hidden=encode(model,item['frames'][:-1])
        for name,(head,context) in outputs(model,hidden).items():
            forecast=head.forecast(context); loss=head.loss(context,item['codes'],item['counts'])
            region=[proof.recount(x,item['codes'][:,i]) for i,x in enumerate(loss['logits'])]
            count={n:proof.recount(forecast['count_logits'][n],item['counts'][n]-1) for n in model.roster}
            error=max([abs(float(r['ce'].mean())-float(v)) for r,v in zip(region,loss['region'])]
                      +[abs(float(count[n]['ce'].mean())-float(loss['counts'][n])) for n in model.roster])
            need(error<=POLICY['loss_error'],'BRANCH_PROOF_NUMPY_LOSS')
            eq=(forecast['codes']==item['codes']).numpy()
            decoded=head.render(forecast,target)
            # Independent singleton frozen reconstruction from forecast-only codes/counts.
            reference=tuple(target.reconstruct(target.values(forecast['codes'][i:i+1]),
                [{n:int(forecast['counts'][n][i]) for n in model.roster}])[0] for i in range(len(eq)))
            de=max(float((a[n]-b[n]).abs().max()) for a,b in zip(decoded,reference) for n in model.roster)
            need(de<=POLICY['output_error'],'BRANCH_PROOF_RENDER_RECOUNT')
            rows.append(dict(document=item['document'],side=item['side'],proposal=name,targets=len(eq),
                code_hits=eq.sum(0).tolist(),prefix_hits=np.logical_and.accumulate(eq,axis=1).sum(0).tolist(),
                region_ce=[float(r['ce'].sum()) for r in region],region_brier=[float(r['brier'].sum()) for r in region],
                count_hits={n:int((forecast['counts'][n]==item['counts'][n]).sum()) for n in model.roster},
                count_ce={n:float(count[n]['ce'].sum()) for n in model.roster},
                mode_code_hits=(item['codes'].numpy()==modes['codes']).sum(0).tolist(),
                copy_code_hits=(item['all_codes'][:-1]==item['codes']).sum(0).tolist(),
                mode_count_hits={n:int((item['counts'][n]==modes['counts'][n]).sum()) for n in model.roster},
                copy_count_hits={n:sum(len(f[n])==int(c) for f,c in zip(item['frames'][:-1],item['counts'][n])) for n in model.roster},
                recount_error=error,render_error=de))
            payload[f"{item['document']}:{name}"]=cpu(dict(forecast=forecast,decoded=decoded,teacher_logits=loss['logits']))
    return rows,payload


def check_item(item, training_documents):
    need(item['side']=='train' and item['document'] in training_documents,'BRANCH_PROOF_HELD_OUT_FIT')
    need(fingerprint(item['frames'])==item['native_pin'],'BRANCH_PROOF_CHANGED_NATIVE')


def update(model,optimizer,item,training_documents):
    check_item(item,training_documents)
    before=fingerprint(model.state_dict()); optimizer.zero_grad(set_to_none=True)
    loss=model.loss(encode(model,item['frames'][:-1]),item['codes'],item['counts'])['loss']
    loss.backward(); grads=cpu({n:p.grad for n,p in model.named_parameters()})
    need(all(g is not None and bool(torch.isfinite(g).all()) for g in grads.values()),'BRANCH_PROOF_GRADIENT_FINITE')
    branches={}
    for n in model.roster:
        values={k:v for k,v in grads.items() if k.startswith('projectors.'+n+'.')}
        need(values and any(bool(v.abs().max()>0) for v in values.values()),'BRANCH_PROOF_BRANCH_GRADIENT')
        branches[n]=fingerprint(values)
    optimizer.step()
    need(all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()),'BRANCH_PROOF_PARAMETER_FINITE')
    return dict(loss=float(loss.detach()),before=before,after=fingerprint(model.state_dict()),
                gradient=fingerprint(grads),branches=branches,optimizer=fingerprint(optimizer.state_dict())),grads


def controls(model, item, target, held, training_documents):
    frames=item['frames'][:-1]; before=fingerprint(model.state_dict())
    stream=model.stream()
    _,state=stream.step(frames[:1],context='one',offset=0,detach_history=False)
    bad=tuple({n:x for n,x in f.items() if n!=model.roster[0]} for f in frames[:1])
    nonfinite=cpu(frames[:1]);nonfinite[0][model.roster[0]][0,0]=float('nan')
    wrong_shape=cpu(frames[:1]);wrong_shape[0][model.roster[0]]=wrong_shape[0][model.roster[0]][:,:-1]
    other=model.stream()
    result=dict(missing_lane=rejected(lambda:encode(model,bad)),nonfinite=rejected(lambda:encode(model,nonfinite)),
        wrong_shape=rejected(lambda:encode(model,wrong_shape)),
        wrong_cache_owner=rejected(lambda:other.step(frames[:1],context='one',offset=1,state=state,detach_history=False)),
        wrong_context=rejected(lambda:stream.step(frames[:1],context='two',offset=1,state=state,detach_history=False)),
        wrong_offset=rejected(lambda:stream.step(frames[:1],context='one',offset=2,state=state,detach_history=False)),
        held_out_fit=rejected(lambda:check_item(held,training_documents)))
    with torch.no_grad():
        h=encode(model,frames)
        try:model.forecast(h,target_codes=item['codes'])
        except TypeError:result['future_target_argument']='TypeError'
        else:raise AssertionError('BRANCH_PROOF_TARGET_LEAK')
        head,context=next(iter(outputs(model,h).values()))
        result['changed_target']=rejected(lambda:head.render(head.forecast(context),deepcopy(target)))
    # A changed encoder parameter must invalidate the existing history cache.
    parameter=next(model.encoder.parameters());saved=parameter.detach().clone()
    with torch.no_grad():parameter.add_(1.)
    result['stale_weights']=rejected(lambda:stream.step(frames[:1],context='one',offset=1,state=state,detach_history=False))
    with torch.no_grad():parameter.copy_(saved)
    old=model.cache_bytes;model.cache_bytes=1
    result['memory_refusal']=rejected(lambda:encode(model,frames[:1]));model.cache_bytes=old
    need(fingerprint(model.state_dict())==before,'BRANCH_PROOF_NEGATIVE_MUTATION')
    return result


def boundaries(model, item):
    frames=item['frames']; model.zero_grad(set_to_none=True)
    with torch.no_grad():
        streamed=encode(model,frames);dense=model.encoder.encode(frames,model.projectors)
        error=float((streamed-dense).abs().max());need(error<=POLICY['output_error'],'BRANCH_PROOF_DENSE_STREAM')
        short=encode(model,frames[:-1]);future=float((streamed[:-1]-short).abs().max())
        need(future<=POLICY['output_error'],'BRANCH_PROOF_FUTURE_EXCLUSION')
        order=float((encode(model,tuple(reversed(frames[:-1]))+(frames[-1],))[-1]-streamed[-1]).abs().max())
        need(order>0,'BRANCH_PROOF_ORDER_INVISIBLE')
        isolated={}
        if hasattr(model.encoder,'lanes'):
            for i,name in enumerate(model.roster):
                changed=tuple({n:(frames[1][n] if j==0 and n==name else x) for n,x in f.items()}
                              for j,f in enumerate(frames))
                new=encode(model,changed)
                unaffected=[j for j in range(len(model.roster)) if j!=i]
                need(torch.equal(new[:,unaffected],streamed[:,unaffected]),'BRANCH_PROOF_CROSS_SOURCE_LEAK')
                delta=float((new[:,i]-streamed[:,i]).abs().max());need(delta>0,'BRANCH_PROOF_IGNORED_SOURCE')
                if isinstance(model,PerInstrumentConstellationPredictor):
                    a,b=model.forecast(streamed),model.forecast(new)
                    need(all(fingerprint(a[n])==fingerprint(b[n]) for n in model.roster if n!=name),
                         'BRANCH_PROOF_PROPOSAL_LEAK')
                isolated[name]=delta
    # Recount dense versus incremental gradients over the complete real document.
    model.loss(model.encoder.encode(frames[:-1],model.projectors),item['codes'],item['counts'])['loss'].backward()
    first=cpu({n:p.grad for n,p in model.named_parameters()});model.zero_grad(set_to_none=True)
    model.loss(encode(model,frames[:-1]),item['codes'],item['counts'])['loss'].backward()
    second=cpu({n:p.grad for n,p in model.named_parameters()})
    ge=max(float((first[n]-second[n]).abs().max()) for n in first)
    need(ge<=POLICY['gradient_error'],'BRANCH_PROOF_DENSE_STREAM_GRADIENT')
    proposal_gradient={}
    if isinstance(model,PerInstrumentConstellationPredictor):
        model.zero_grad(set_to_none=True);h=encode(model,frames[:-1]);name=model.roster[0]
        head,context=outputs(model,h)[name];head.loss(context,item['codes'],item['counts'])['loss'].backward()
        for n in model.roster:
            values=[p.grad for k,p in model.named_parameters() if k.startswith('encoder.lanes.'+n+'.')]
            nonzero=any(g is not None and bool(g.abs().max()>0) for g in values)
            need(nonzero==(n==name),'BRANCH_PROOF_CROSS_SOURCE_GRADIENT');proposal_gradient[n]=nonzero
    model.zero_grad(set_to_none=True)
    return dict(dense_stream_error=error,future_exclusion_error=future,order_delta=order,
                source_isolation=isolated,gradient_error=ge,proposal_gradient=proposal_gradient)


def persist(path,value):
    torch.save(cpu(value),path)
    loaded=torch.load(path,map_location='cpu',weights_only=True)
    need(fingerprint(value)==fingerprint(loaded),'BRANCH_PROOF_PAYLOAD_READBACK')
    return fingerprint(loaded)


def run(a):
    began=tick=time.perf_counter();process=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    for name,pin in code().items():
        dest=a.output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes())
        need(file_sha(dest)==pin,'BRANCH_PROOF_CODE_COPY')
    target,ref,items,modes,source_binding=source(a.root);target_before=fingerprint(target.state_dict())
    train=[i for i in items if i['side']=='train'];allowed={i['document'] for i in train}
    binding=dict(code=code(),policy=POLICY,source=source_binding,target=ref)
    previous=proof.pinned(a.previous,a.previous_pin) if a.previous else None
    if previous:
        need(previous['complete'] and previous['binding']==binding,'BRANCH_PROOF_COLD_BINDING')
        for name,pin in previous['files'].items():need(file_sha(a.previous/name)==pin,'BRANCH_PROOF_COLD_BYTES')
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter();results={}
    for name,constructor in ARMS.items():
        out=a.output/name;out.mkdir();start=time.perf_counter()
        model=constructor(target,target_pin=ref['pin'],seed=POLICY['seed'],cache_bytes=POLICY['cache_bytes'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=POLICY['lr'],weight_decay=POLICY['weight_decay'],foreach=False)
        initial_state=persist(out/'initial.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict()))
        checks=boundaries(model,train[0]);negative=controls(model,train[0],target,next(i for i in items if i['side']=='query'),allowed)
        initial,payload=evaluate(model,items,target,modes);initial_predictions=persist(out/'initial-predictions.pt',payload)
        updates=[]
        for step,item in enumerate(train,1):
            update_row,gradient=update(model,optimizer,item,allowed)
            update_row['state']=persist(out/f'step-{step}.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                                             gradients=gradient,document=item['document'],step=step))
            updates.append(dict(document=item['document'],**update_row))
        # Restore actual AdamW moments/update counts, then independently reproduce
        # the next committed update. Restarting from weights alone is not continuation.
        restored=constructor(target,target_pin=ref['pin'],seed=POLICY['seed'],cache_bytes=POLICY['cache_bytes'])
        resumed=torch.optim.AdamW(restored.parameters(),lr=POLICY['lr'],weight_decay=POLICY['weight_decay'],foreach=False)
        checkpoint=torch.load(out/'step-2.pt',map_location='cpu',weights_only=True)
        restored.load_state_dict(checkpoint['model'],strict=True);resumed.load_state_dict(checkpoint['optimizer'])
        replay,rg=update(restored,resumed,train[2],allowed)
        need(all(replay[k]==updates[2][k] for k in replay),'BRANCH_PROOF_CONTINUATION_UPDATE')
        continuation=persist(out/'continuation-step-3.pt',dict(model=restored.state_dict(),optimizer=resumed.state_dict(),
                                               gradients=rg,document=train[2]['document'],step=3))
        need(continuation==updates[2]['state'],'BRANCH_PROOF_CONTINUATION_STATE')
        final,payload=evaluate(model,items,target,modes);final_predictions=persist(out/'final-predictions.pt',payload)
        need(fingerprint(target.state_dict())==target_before,'BRANCH_PROOF_FROZEN_TARGET_CHANGED')
        results[name]=dict(parameters=sum(p.numel() for p in model.parameters()),checks=checks,controls=negative,
            initial_state=initial_state,initial_predictions=initial_predictions,final_predictions=final_predictions,
            initial=initial,final=final,updates=updates,continuation=continuation,
            final_state=fingerprint((model.state_dict(),optimizer.state_dict())))
        phases[name]=time.perf_counter()-start;print(json.dumps(dict(arm=name,parameters=results[name]['parameters'],seconds=phases[name])),flush=True)
    tick=time.perf_counter()
    result_pin=write_committed(a.output/'results.json',results)
    if previous:need(result_pin==previous['results_pin'],'BRANCH_PROOF_COLD_EXACT')
    need(not torch.cuda.is_initialized(),'BRANCH_PROOF_UNEXPECTED_CUDA')
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['readback']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(sum(phases.values())/seconds>=.9,'BRANCH_PROOF_PHASE_ACCOUNTING')
    report=dict(complete=True,binding=binding,files=files,results_pin=result_pin,previous_pin=a.previous_pin,
        documents=len(items),atoms=sum(len(i['frames']) for i in items),successors=sum(len(i['codes']) for i in items),
        split_targets={s:sum(len(i['codes']) for i in items if i['side']==s) for s in ['train','development','query']},
        updates_per_arm=len(train),old_fp32_label_changes=sum(i['old_label_changes'] for i in items),
        phases_seconds=phases,seconds=seconds,cpu_seconds=time.process_time()-process,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        scientific_admission=False,capability_demonstrated=False,placement='frozen_codebook_baseline',
        gpu_qualified=False,capacity_matched=False,number_status='measured',acceptance='author-checked')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),seconds=seconds)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous',type=Path)
    parser.add_argument('--previous-pin')
    args=parser.parse_args();need(bool(args.previous)==bool(args.previous_pin),'BRANCH_PROOF_PREVIOUS_ARGUMENTS');run(args)
