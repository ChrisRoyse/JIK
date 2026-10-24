"""Matched data-growth/conditioning experiments on authentic typed native windows."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit.legal_residual_codebook_readback import CONFIG, POLICY as ORIGINAL, pinned
from audit.legal_prefix_stream_readback import native_reader
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model.legal_learning_source import LegalLearningSource
from model.loop_training_checkpoint import tensor_fingerprint
from model.native_rq_conditioning import ConditionedNativeRQ, RECIPES
from model.native_rq_data import NativeRQData, fingerprint_frames, matched_donors, statistics, training_components
from senses.contract import digest
from text.constellation_corpus import file_sha, need
from text.legal_continuation_input import Continuations
from text.legal_reuse_input import ReuseInputs

POLICY = dict(ORIGINAL, recipes=list(RECIPES), updates=3306, tile=128, cache_bytes=32 << 20,
              read_bytes=16 << 20, baseline_first32_exact=True, sample_selection='source_hash_before_scores',
              normalized_loss_relative_error=5e-6,
              exposure='development_exposed_case_split_not_untouched')


def pins():
    base = Path(__file__).resolve().parents[1]
    return {n:file_sha(base/n) for n in ['audit/legal_rq_growth.py','model/native_rq_conditioning.py',
        'model/native_rq_data.py','model/native_residual_codebook.py','model/native_prefix_transformer.py',
        'model/attention.py','model/loop_training_checkpoint.py','model/legal_learning_source.py',
        'text/legal_native_batch.py']}


def make(dimensions, recipe, stats):
    torch.manual_seed(POLICY['seed']); model = ConditionedNativeRQ(dimensions, CONFIG, POLICY['sizes'], recipe, stats)
    opt = torch.optim.AdamW(model.parameters(), lr=POLICY['learning_rate'], weight_decay=POLICY['weight_decay'],
                           foreach=False, fused=False)
    return model,opt


def batch_indices(uids, step):
    return [(step*8+j) % len(uids) for j in range(8)]


def step(model, opt, reader, uids, components, number):
    model.check_statistics()
    positions = batch_indices(uids, number); ids = [uids[j] for j in positions]
    frames = reader.frames(ids); cases = [components[j] for j in positions]
    donors,eligible = matched_donors(frames,ids,cases,reader.roster)
    native_before = tensor_fingerprint(frames); before = digest(tensor_fingerprint(model.state_dict()))
    opt.zero_grad(set_to_none=True); result = model.training_loss(frames,donors,eligible); result['loss'].backward()
    gradients = {n:p.grad for n,p in model.named_parameters()}
    need(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients.values()), 'RQ_GROWTH_GRADIENT')
    branches = ['encoder.','books.','to_latent.']+['projectors.'+n+'.' for n in reader.roster]+['decoders.'+n+'.' for n in reader.roster]
    norms = {p:sum(float(g.double().square().sum()) for n,g in gradients.items() if n.startswith(p))**.5 for p in branches}
    gradient = digest(tensor_fingerprint(gradients)); opt.step()
    model.check_statistics()
    need(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'RQ_GROWTH_PARAMETER')
    need(native_before == tensor_fingerprint(frames), 'RQ_GROWTH_NATIVE_MUTATION')
    return dict(step=number+1, ids=ids, donors=donors.tolist(), eligible=eligible.tolist(), before=before,
                after=digest(tensor_fingerprint(model.state_dict())), gradient=gradient,
                optimizer=digest(tensor_fingerprint(opt.state_dict())), gradient_norms=norms,
                loss=float(result['loss'].detach()), reconstruction=float(result['reconstruction'].detach()),
                quantizer=float(result['quantizer_loss'].detach()), contrast=float(result['contrast'].detach()))


@torch.no_grad()
def initialize(model,reader,uids):
    latents = torch.cat([model.latent(frames) for _,frames in reader.tiles(uids)])
    need(latents.numel()*latents.element_size() <= POLICY['buffer_bytes'], 'RQ_GROWTH_LATENT_MEMORY')
    model.initialize_latents(latents)
    return digest(tensor_fingerprint(latents))


def components_for(data, uids, side):
    offsets = data.source.arrays['offsets']; loc = data.source.arrays['occurrences']; selected = []
    for uid in uids:
        candidates = loc[int(offsets[uid]):int(offsets[uid+1])]; candidates = candidates[data.sides[candidates] == side]
        need(len(candidates)>0,'RQ_GROWTH_SIDE_OCCURRENCE'); selected.append(int(candidates[0]))
    docs = np.searchsorted(data.documents[:,1],selected,side='right')
    return data.components[docs].astype('<i8').tolist()


@torch.no_grad()
def loss_reference(model, reader, uids, cases):
    ids=uids[:8];frames=reader.frames(ids);donors,eligible=matched_donors(frames,ids,cases[:8],reader.roster)
    out=model.training_loss(frames,donors,eligible)
    raw={n:float(np.mean([np.mean((d[n].numpy().astype(np.float64)-f[n].numpy().astype(np.float64))**2)
                         for d,f in zip(out['decoded'],frames)])) for n in reader.roster}
    raw_error=max(abs(raw[n]-float(out['per_lane'][n])) for n in reader.roster)
    need(raw_error<=POLICY['arithmetic_error'],'RQ_GROWTH_RAW_LOSS_REFERENCE')
    expected=float(np.mean([v if model.recipe=='baseline' else v/float(getattr(model,'variance_'+str(i)))
                            for i,(n,v) in enumerate(raw.items())]))
    contrast=0.
    if model.recipe=='normalized_contrast' and bool(eligible.any()):
        _,values,_=model.quantize(out['latent']);wrong=model.reconstruct(values[donors],[{n:len(f[n]) for n in reader.roster} for f in frames])
        terms=[]
        for i,n in enumerate(reader.roster):
            values=[]
            for ok,right,bad,frame in zip(eligible,out['decoded'],wrong,frames):
                if bool(ok):
                    target=frame[n].numpy().astype(np.float64)
                    delta=(np.mean((right[n].numpy().astype(np.float64)-target)**2)-np.mean((bad[n].numpy().astype(np.float64)-target)**2))/float(getattr(model,'variance_'+str(i)))
                    values.append(float(np.logaddexp(0,delta)))
            terms.append(float(np.mean(values)))
        contrast=float(np.mean(terms))
    errors={k:abs(actual-reference)/max(1.,abs(reference)) for k,actual,reference in
            [('reconstruction',float(out['reconstruction']),expected),('contrast',float(out['contrast']),contrast)]}
    need(max(errors.values())<=POLICY['normalized_loss_relative_error'],'RQ_GROWTH_NORMALIZED_LOSS_REFERENCE')
    return dict(raw_error=raw_error,normalized_relative_errors=errors,reference_reconstruction=expected,reference_contrast=contrast)


@torch.no_grad()
def evaluate(model,reader,uids,components,stats,root):
    root.mkdir(); roster = reader.roster; n = len(uids); k = len(roster)
    codes = np.zeros((n,len(model.sizes)),dtype='<i8'); errors = np.zeros((n,k,4),dtype='<f8')
    eligible_rows = np.zeros(n,dtype=np.bool_); counts = np.zeros((n,k),dtype='<i8')
    native_hash = hashlib.sha256(); decoded_hash = hashlib.sha256(); shuffled_hash = hashlib.sha256()
    latent_sum = torch.zeros(CONFIG['width'],dtype=torch.float64); latent_squared = torch.zeros_like(latent_sum)
    cursor=0
    for ids,frames in reader.tiles(uids):
        end=cursor+len(ids); out=model(frames); codes[cursor:end]=out['codes'].numpy()
        donors,eligible=matched_donors(frames,ids,components[cursor:end],roster)
        _,values,_=model.quantize(out['latent']); wrong=model.reconstruct(values[donors],[{x:len(f[x]) for x in roster} for f in frames])
        fingerprint_frames(native_hash,ids,frames,roster); fingerprint_frames(decoded_hash,ids,out['decoded'],roster)
        fingerprint_frames(shuffled_hash,ids,wrong,roster); eligible_rows[cursor:end]=eligible.numpy()
        latent_sum+=out['latent'].double().sum(0);latent_squared+=out['latent'].double().square().sum(0)
        for j,(frame,decoded,shuffled) in enumerate(zip(frames,out['decoded'],wrong)):
            for lane,name in enumerate(roster):
                target=frame[name].numpy().astype(np.float64); w=len(target); counts[cursor+j,lane]=w
                need(w<=len(stats['positions'][name]),'RQ_GROWTH_UNSEEN_BASELINE_POSITION')
                errors[cursor+j,lane]=[
                    np.mean((decoded[name].numpy().astype(np.float64)-target)**2),
                    np.mean((stats['centers'][name].numpy()-target)**2),
                    np.mean((stats['positions'][name][:w].numpy()-target)**2),
                    np.mean((shuffled[name].numpy().astype(np.float64)-target)**2)]
        cursor=end
    need(cursor==n and np.isfinite(errors).all(),'RQ_GROWTH_EVALUATION')
    for filename,value in [('codes.npy',codes),('errors.npy',errors),('eligible.npy',eligible_rows),('windows.npy',counts)]:
        np.save(root/filename,value,allow_pickle=False)
    summaries={}
    for label,mask in [('all',np.ones(n,dtype=bool)),('single_window',(counts==1).all(1)),('multiwindow',(counts>1).any(1))]:
        selected=errors[mask]; eligible=mask&eligible_rows
        summaries[label]=dict(rows=int(mask.sum()),contrast_rows=int(eligible.sum()),
            per_lane={name:dict(mse=float(selected[:,i,0].mean()),mean=float(selected[:,i,1].mean()),
                               position=float(selected[:,i,2].mean()),
                               shuffled_minus_correct=float((errors[eligible,i,3]-errors[eligible,i,0]).mean()) if eligible.any() else None)
                      for i,name in enumerate(roster)} if mask.any() else {})
    contingency={}
    for w,c in zip(counts,codes):
        key=','.join(map(str,w));contingency.setdefault(key,[0]*model.sizes[0]);contingency[key][c[0]]+=1
    result=dict(rows=n,uids=uids,distinct_codes=len(np.unique(codes,axis=0)),
                usage=[np.bincount(codes[:,j],minlength=s).tolist() for j,s in enumerate(model.sizes)],
                first_code_by_window_counts=contingency,slices=summaries,native_sha256=native_hash.hexdigest(),
                decoded_sha256=decoded_hash.hexdigest(),shuffled_sha256=shuffled_hash.hexdigest(),
                latent_variance=(latent_squared/n-(latent_sum/n).square()).tolist(),
                files={x:file_sha(root/x) for x in ['codes.npy','errors.npy','eligible.npy','windows.npy']})
    return dict(pin=write_committed(root/'report.json',result),**result)


def run(a):
    start=tick=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    baseline=pinned(a.baseline/'report.json',a.baseline_pin);prior=pinned(a.prior/'report.json',a.prior_pin)
    source=ReuseInputs(baseline['source']['path'],baseline['source']['pin'])
    continuation=Continuations(Path(baseline['continuations']['path']),baseline['continuations']['pin'])
    split,cold=continuation.report['split'],continuation.report['cold'];bank=baseline['bank'];proof=baseline['proof']
    data=LegalLearningSource(source,split['path'],split['pin'],cold['path'],cold['pin'],bank['path'],bank['pin'])
    need(data.binding==prior['binding'],'RQ_GROWTH_BINDING');reader=NativeRQData(data,Path(proof['path']),proof['pin'])
    small=prior['train_uids'];presence=continuation.presence
    populations=dict(small=small,large=sorted(set(np.flatnonzero((presence[:16384]&1)!=0).tolist()+small)))
    held=dict(development=np.flatnonzero(((presence[:16384]&1)==0)&((presence[:16384]&2)!=0))[:64].tolist(),
              query=np.flatnonzero(((presence[:16384]&3)==0)&((presence[:16384]&4)!=0))[:64].tolist())
    need(len(small)==136 and len(populations['large'])==13222 and all(len(x)==64 for x in held.values()),'RQ_GROWTH_POPULATION')
    selection=dict(populations=populations,held=held,source_binding=data.binding);selection_pin=write_committed(a.output/'selection.json',selection)
    cases={};stats={};stat_pins={}
    for name,uids in populations.items():
        _,cases[name]=training_components(data,uids,presence)
        if a.mode=='qualify':
            stats[name]=statistics(reader,uids,presence);torch.save(stats[name],a.output/(name+'-statistics.pt'))
        else:
            need(a.qualification is not None,'RQ_GROWTH_QUALIFICATION_REQUIRED')
            qualified=pinned(a.qualification/'report.json',a.qualification_pin)
            need(qualified['code']==pins() and qualified['selection_pin']==selection_pin and qualified['policy']==POLICY,
                 'RQ_GROWTH_UNQUALIFIED_CODE')
            path=a.qualification/(name+'-statistics.pt');need(file_sha(path)==qualified['statistic_pins'][name],'RQ_GROWTH_STATISTICS_PIN')
            stats[name]=torch.load(path,map_location='cpu',weights_only=True)
        need(stats[name]['uids']==uids,'RQ_GROWTH_STATISTICS_FIT_SCOPE')
        stat_pins[name]=file_sha((a.output if a.mode=='qualify' else a.qualification)/(name+'-statistics.pt'))
    dimensions={n:len(stats['small']['centers'][n]) for n in reader.roster}
    phases=dict(source_selection_statistics=time.perf_counter()-tick);tick=time.perf_counter();results={}
    if a.mode=='qualify':
        legacy,_=native_reader(data,Path(proof['path']),proof['pin']);left=reader.frames(small[:128])+reader.frames(small[128:])
        need(tensor_fingerprint(left)==tensor_fingerprint(legacy(small)),'RQ_GROWTH_BATCH_READER_CHANGED')
        for recipe in RECIPES:
            model,opt=make(dimensions,recipe,stats['small']);initialize(model,reader,small);records=[];saved=None
            loss_checks=[loss_reference(model,reader,small,cases['small'])]
            for i in range(32):
                if i==31:
                    path=a.output/(recipe+'-resume.pt');torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),path)
                records.append(step(model,opt,reader,small,cases['small'],i))
            if recipe=='baseline':
                need(all(records[i][key]==prior['results']['baseline']['ledger'][i][key]
                         for i in range(32) for key in ['gradient','loss','optimizer']),'RQ_GROWTH_BASELINE_DRIFT')
            final={n:x.detach().clone() for n,x in model.state_dict().items()}
            saved=torch.load(path,map_location='cpu',weights_only=True);cold_model,cold_opt=make(dimensions,recipe,stats['small'])
            cold_model.load_state_dict(saved['model']);cold_opt.load_state_dict(saved['optimizer'])
            replay=step(cold_model,cold_opt,reader,small,cases['small'],31)
            need(replay==records[-1] and tensor_fingerprint(final)==tensor_fingerprint(cold_model.state_dict()),'RQ_GROWTH_COLD_STEP')
            loss_checks.append(loss_reference(model,reader,small,cases['small']))
            results[recipe]=dict(steps=records,cold_step_exact=True,resume_pin=file_sha(path),loss_reference=loss_checks)
        results['controls']=dict(nontrain=rejected(lambda:statistics(reader,held['development'],presence)),
            wrong_recipe=rejected(lambda:make(dimensions,'wrong',stats['small'])),
            oversized_read=rejected(lambda:reader.frames(populations['large'])),
            same_case_donors=not bool(matched_donors(left[:8],small[:8],[0]*8,reader.roster)[1].any()),
            same_input_donors=not bool(matched_donors((left[0],)*8,[small[0]]*8,list(range(8)),reader.roster)[1].any()))
        need(results['controls']['same_case_donors'] and results['controls']['same_input_donors'],'RQ_GROWTH_DONOR_CONTROL')
        wrong=dict(left[0]);wrong.pop(reader.roster[0])
        results['controls']['missing_slot']=rejected(lambda:model.latent((wrong,)))
        with torch.no_grad():model.center_0[0]+=1
        results['controls']['statistics_mutation']=rejected(model.check_statistics)
    else:
        for population,uids in populations.items():
            for recipe in RECIPES:
                key=population+'-'+recipe;root=a.output/key;root.mkdir();began=time.perf_counter()
                model,opt=make(dimensions,recipe,stats[population]);initial=initialize(model,reader,uids)
                with (root/'updates.jsonl').open('x') as log:
                    for i in range(POLICY['updates']):
                        if i==POLICY['updates']-1:
                            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=i,
                                            selection_pin=selection_pin,statistic_pin=stat_pins[population]),root/'resume.pt')
                        record=step(model,opt,reader,uids,cases[population],i);log.write(json.dumps(record,sort_keys=True)+'\n')
                        if (i+1)%128==0:
                            log.flush();print(json.dumps(dict(arm=key,step=i+1,loss=record['loss'],seconds=time.perf_counter()-began)),flush=True)
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=POLICY['updates'],
                                selection_pin=selection_pin,statistic_pin=stat_pins[population]),root/'final.pt')
                measured={'train':evaluate(model,reader,uids,cases[population],stats[population],root/'train')}
                for name,side in [('development',1),('query',2)]:
                    measured[name]=evaluate(model,reader,held[name],components_for(data,held[name],side),stats[population],root/name)
                need(measured['train']['native_sha256']==stats[population]['native_sha256'],'RQ_GROWTH_TRAIN_BYTES_CHANGED')
                results[key]=dict(initial_latent_pin=initial,evaluation={n:dict(pin=v['pin'],rows=v['rows'],distinct_codes=v['distinct_codes'],slices=v['slices']) for n,v in measured.items()},
                    last_update=record,seconds=time.perf_counter()-began,files={n:file_sha(root/n) for n in ['resume.pt','final.pt','updates.jsonl']})
                write_committed(root/'report.json',results[key]);print(json.dumps(dict(arm=key,complete=True,seconds=results[key]['seconds'])),flush=True)
    phases['qualification_or_fits']=time.perf_counter()-tick;tick=time.perf_counter()
    data.fresh();source.fresh();continuation.fresh();phases['final_freshness']=time.perf_counter()-tick
    seconds=time.perf_counter()-start;need(sum(phases.values())/seconds>=.9,'RQ_GROWTH_PHASE_ATTRIBUTION')
    report=dict(schema='legal-rq-growth-v1',complete=True,mode=a.mode,policy=POLICY,code=pins(),selection_pin=selection_pin,
        statistic_pins=stat_pins,results=results,dimensions=dimensions,source_binding=data.binding,
        qualification=dict(path=str(a.qualification),pin=a.qualification_pin) if a.qualification else None,
        phases_seconds=phases,seconds=seconds,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        maximum_cached_native_bytes=reader.peak_cache_bytes,maximum_read_native_bytes=reader.maximum_read_bytes,
        native_reader_stats=data.native.stats,model_promotion=False,scientific_admission=False,
        acceptance='author-checked',number_status='measured')
    pin=write_committed(a.output/'report.json',report);print(json.dumps(dict(report_pin=pin,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['qualify','fit'],required=True)
    for name in ['baseline','prior']:
        p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-pin',required=True)
    p.add_argument('--qualification',type=Path);p.add_argument('--qualification-pin')
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
