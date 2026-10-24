"""Real-source CPU interface/gradient proof for target-withheld region prediction."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch

from audit import legal_region_predictor_readback as prior
from audit.legal_predictor_training import QUAL, COLD
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model.native_population_training import cpu, fingerprint
from model.native_reciprocal_region import NativeReciprocalRegion
from model.native_target_cache import FrozenLabels
from text.constellation_corpus import file_sha, need

POLICY = dict(device='cpu', threads=1, seed=20261025, atoms=64, optimizer_updates=0,
              output_error=5e-6, loss_recount_error=5e-7, scope='exposed_component_qualification')


def code():
    root = Path(__file__).resolve().parents[1]
    return dict(prior.code(), **{n:file_sha(root/n) for n in ['model/native_reciprocal_region.py',
        'model/native_target_cache.py', 'audit/legal_reciprocal_region_readback.py']})


def pinned(root, pin):
    return prior.training.pinned(root/'report.json', pin)


def array(root, report, name):
    ref = report['files'][name]; path = root/name
    need(path.stat().st_size == ref['bytes'] and file_sha(path) == ref['sha256'], 'RECIPROCAL_PROOF_SOURCE_ARRAY')
    return np.load(path, allow_pickle=False, mmap_mode='r')


def source(root):
    q, c = pinned(root/QUAL[0], QUAL[1]), pinned(root/COLD[0], COLD[1])
    need(q['complete'] and c['complete'] and c['previous_pin'] == QUAL[1]
         and q['binding'] == c['binding'] and q['results_pin'] == c['results_pin'], 'RECIPROCAL_PROOF_NATIVE_AUTHORITY')
    selection = prior.training.pinned(root/QUAL[0]/'selection.json', q['binding']['selection'])
    authority = selection['source']
    src = root/'legal-input-reuse-v1-20261022'; sr = pinned(src, authority['source'])
    split = root/'legal-case-split-v1-20261022'; sp = pinned(split, authority['split'])
    mapping = array(src, sr, 'mapping.npy'); documents = array(src, sr, 'documents.npy')
    components = array(split, sp, 'component.npy'); sides = array(split, sp, 'document-side.npy')
    rows = []
    for row in selection['documents']:
        name = f"document-{row['document']}/outputs.pt"; path = root/QUAL[0]/name
        need(file_sha(path) == q['files'][name], 'RECIPROCAL_PROOF_NATIVE_PIN')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        need(fingerprint(payload['frames']) == payload['native_pin'], 'RECIPROCAL_PROOF_NATIVE_VALUES')
        start, stop = map(int, documents[row['document']])
        need((start,stop) == (row['start'],row['stop']) and stop-start == len(payload['frames'])
             and row['side'] == ['train','development','query'][int(sides[row['document']])],
             'RECIPROCAL_PROOF_SOURCE_JOIN')
        for offset, frame in enumerate(payload['frames']):
            rows.append(dict(document=row['document'], side=row['side'], occurrence=start+offset,
                             uid=int(mapping[start+offset]), component=int(components[row['document']]), frame=frame))
    chosen = {}; ordered = sorted(rows, key=lambda r:hashlib.sha256(f"reciprocal-20261023:{r['occurrence']}".encode()).digest())
    for row in ordered:
        if any(len(x)>1 for x in row['frame'].values()): chosen.setdefault(row['uid'], row)
    for side in ['train','development','query']:
        count = 0
        for row in ordered:
            if row['side'] != side: continue
            chosen.setdefault(row['uid'], row); count += 1
            if count == 12: break
    need(len(chosen) <= POLICY['atoms'], 'RECIPROCAL_PROOF_REQUIRED_SCOPE')
    for row in ordered:
        chosen.setdefault(row['uid'], row)
        if len(chosen) == POLICY['atoms']: break
    need(len(chosen) == POLICY['atoms'], 'RECIPROCAL_PROOF_SCOPE')
    selected = sorted(chosen.values(), key=lambda r:r['occurrence'])
    metadata = [{k:v for k,v in r.items() if k != 'frame'} for r in selected]
    frames = tuple(r['frame'] for r in selected)
    return metadata, frames, dict(native_qualification=QUAL[1], native_cold=COLD[1], source=authority,
                                  proof_document_count=len(selection['documents']), proof_occurrences=len(rows))


def scalar(model, frames, names):
    result = []; encoder = model.encoder
    for frame in frames:
        origin = encoder.position(torch.tensor([0]))[0]; tokens = []; indices = []; cursor = 0
        for lane, name in enumerate(model.roster):
            if name in names:
                value = frame[name]
                local = encoder.position(torch.arange(len(value)))
                tokens.append(model.projectors[name](value)+origin+local+encoder.kind[lane]); cursor += len(value)
            indices.append(cursor); tokens.append((origin+encoder.kind[lane]+encoder.query[lane])[None]); cursor += 1
        value = torch.cat(tokens)[None]
        for block in encoder.blocks: value = block(value, torch.zeros((cursor,cursor), dtype=torch.bool))
        result.append(encoder.norm(value)[0, torch.tensor(indices)])
    return torch.stack(result)


def recount(logits, target):
    values = logits.detach().double().numpy(); high = values.max(1,keepdims=True)
    logp = values-high-np.log(np.exp(values-high).sum(1,keepdims=True))
    truth = target.numpy(); row = np.arange(len(truth)); probability = np.exp(logp)
    return dict(ce=-logp[row,truth], brier=np.square(probability-np.eye(values.shape[1])[truth]).sum(1),
                probabilities=probability)


def donor_map(metadata, frames, roster):
    layouts = [tuple(len(f[n]) for n in roster) for f in frames]
    donors = []
    for i,row in enumerate(metadata):
        eligible = [j for j,other in enumerate(metadata) if other['uid'] != row['uid']
                    and other['component'] != row['component'] and other['side'] == row['side'] and layouts[j] == layouts[i]]
        donors.append(min(eligible,key=lambda j:hashlib.sha256(f"reciprocal-donor:{row['uid']}:{metadata[j]['uid']}".encode()).digest())
                      if eligible else -1)
    return donors


def run(a):
    began = tick = time.perf_counter(); process = time.process_time(); a.output.mkdir(parents=True)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    for name,pin in code().items():
        path = a.output/'source'/name; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes())
        need(file_sha(path) == pin, 'RECIPROCAL_PROOF_SOURCE_COPY')
    metadata, frames, source_binding = source(a.root)
    stats_root = a.root/prior.initialization.STATS[0]
    stats_report = pinned(stats_root, prior.initialization.STATS[1])
    need(file_sha(stats_root/'statistics.pt') == stats_report['files']['statistics.pt'], 'RECIPROCAL_PROOF_STATS')
    stats = torch.load(stats_root/'statistics.pt',map_location='cpu',weights_only=True)
    target, fit, ref = prior.load_target(a.root, stats, {n:len(stats['centers'][n]) for n in frames[0]})
    need(fit['binding']['source'] == source_binding['source'], 'RECIPROCAL_PROOF_FIT_SOURCE')
    labels = FrozenLabels(target); label_codes, label_counts = labels.encode(frames, reference=True, canonical=True)
    codes = torch.from_numpy(label_codes.astype(np.int64)); counts = torch.from_numpy(label_counts.astype(np.int64))
    model = NativeReciprocalRegion(target,target_pin=ref['pin'],seed=POLICY['seed'])
    initial = fingerprint(model.state_dict()); native_pin = fingerprint(frames)
    donors = donor_map(metadata, frames, model.roster)
    selection = dict(atoms=metadata, donors=donors, native_pin=native_pin, labels=fingerprint((codes,counts)))
    selection_pin = write_committed(a.output/'selection.json',selection)
    binding = dict(code=code(), policy=POLICY, source=source_binding, target=ref,
                   target_runtime=labels.runtime_state, selection=selection_pin)
    previous = pinned(a.previous,a.previous_pin) if a.previous else None
    if previous:
        need(previous['complete'] and previous['binding'] == binding, 'RECIPROCAL_PROOF_COLD_BINDING')
        for name,pin in previous['files'].items():need(file_sha(a.previous/name)==pin,'RECIPROCAL_PROOF_COLD_FILE')
    torch.save(dict(frames=frames,codes=codes,counts=counts,initial=cpu(model.state_dict())),a.output/'inputs.pt')
    phases = dict(authentication=time.perf_counter()-tick); tick=time.perf_counter()
    records = {}; payloads = {}; masks = []
    for hidden in model.roster:
        other = tuple(n for n in model.roster if n != hidden)
        masks += [(hidden,other)]+[(hidden,(name,)) for name in other]
    if previous: masks.reverse()
    for hidden,names in masks:
        key = hidden+'--'+','.join(names); visible = tuple({n:f[n] for n in names} for f in frames)
        model.zero_grad(set_to_none=True)
        context = model.encode_visible(visible,hidden=hidden,visible=names)
        with torch.no_grad():
            reference = scalar(model,visible,names)
            reverse = model.encode_visible(visible[::-1],hidden=hidden,visible=names).flip(0)
            error = max(float((reference-context).abs().max()),float((reverse-context).abs().max()))
            need(error <= POLICY['output_error'], 'RECIPROCAL_PROOF_PACKED_SCALAR')
            forecast = cpu(model.forecast(context))
        lane = model.roster.index(hidden)
        loss = model.loss_withheld(context,codes,counts[:,lane],hidden=hidden)
        native_count = recount(loss['count_logits'],counts[:,lane]-1)
        region = [recount(score,codes[:,level]) for level,score in enumerate(loss['logits'])]
        ce_error = max(abs(float(r['ce'].mean())-float(x.detach())) for r,x in zip(region,loss['region']))
        ce_error = max(ce_error,abs(float(native_count['ce'].mean())-float(loss['count'].detach())))
        need(ce_error <= POLICY['loss_recount_error'], 'RECIPROCAL_PROOF_LOSS_RECOUNT')
        loss['loss'].backward()
        grads = cpu({n:p.grad for n,p in model.named_parameters()})
        for name in model.roster:
            values = [g for n,g in grads.items() if n.startswith('projectors.'+name+'.')]
            if name in names:
                need(all(g is not None and bool(torch.isfinite(g).all()) for g in values)
                     and any(bool(g.abs().max()>0) for g in values),'RECIPROCAL_PROOF_VISIBLE_GRADIENT')
            else: need(all(g is None for g in values),'RECIPROCAL_PROOF_HIDDEN_GRADIENT')
        need(all(g is None or bool(torch.isfinite(g).all()) for g in grads.values()),'RECIPROCAL_PROOF_NONFINITE_GRADIENT')
        eligible = [i for i,j in enumerate(donors) if j>=0]; shuffled = None
        with torch.no_grad():
            if eligible:
                wrong = tuple({n:frames[donors[i]][n] for n in names} for i in eligible)
                wrong_context = model.encode_visible(wrong,hidden=hidden,visible=names)
                wrong_loss = model.loss_withheld(wrong_context,codes[eligible],counts[eligible,lane],hidden=hidden)
                shuffled = dict(eligible=eligible, count=recount(wrong_loss['count_logits'],counts[eligible,lane]-1),
                                region=[recount(score,codes[eligible,level]) for level,score in enumerate(wrong_loss['logits'])])
            # Change the true hidden readings and counts only; the forward inputs
            # remain the same visible native objects, and forecasts must be exact.
            perturbed = [{**f,hidden:f[hidden].flip(0).neg()} for f in frames]
            again = tuple({n:f[n] for n in names} for f in perturbed)
            unchanged = model.encode_visible(again,hidden=hidden,visible=names)
            need(torch.equal(context.detach(),unchanged) and fingerprint(forecast)==fingerprint(cpu(model.forecast(unchanged))),
                 'RECIPROCAL_PROOF_TARGET_LEAK')
        payload = dict(context=context.detach(),forecast=forecast,gradients=grads,
                       correct_region=[{k:torch.from_numpy(v) for k,v in r.items()} for r in region],
                       correct_count={k:torch.from_numpy(v) for k,v in native_count.items()},
                       shuffled=None if shuffled is None else dict(eligible=eligible,
                           count={k:torch.from_numpy(v) for k,v in shuffled['count'].items()},
                           region=[{k:torch.from_numpy(v) for k,v in r.items()} for r in shuffled['region']]))
        path=a.output/(str(len(records))+'.pt');torch.save(payload,path)
        need(fingerprint(torch.load(path,map_location='cpu',weights_only=True))==fingerprint(payload),'RECIPROCAL_PROOF_PAYLOAD_READBACK')
        payloads[key]=fingerprint(payload)
        records[key]=dict(hidden=hidden,visible=list(names),rows=len(frames),scalar_error=error,ce_recount_error=ce_error,
                          loss=float(loss['loss'].detach()),shuffled_eligible=len(eligible),
                          correct_region_ce=[float(r['ce'].mean()) for r in region],
                          correct_hidden_count_ce=float(native_count['ce'].mean()),
                          eligible_correct_region_ce=[float(r['ce'][eligible].mean()) for r in region] if eligible else None,
                          shuffled_region_ce=[float(r['ce'].mean()) for r in shuffled['region']] if shuffled else None)
        print(json.dumps(dict(hidden=hidden,visible=len(names),rows=len(frames),eligible=len(eligible))),flush=True)
    phases['masked_forward_backward_and_readback']=time.perf_counter()-tick;tick=time.perf_counter()
    hidden=model.roster[0];names=tuple(n for n in model.roster if n!=hidden)
    visible=tuple({n:f[n] for n in names} for f in frames)
    controls=dict(target_visible=rejected(lambda:model.encode_visible(frames,hidden=hidden,visible=names)),
        hidden_in_names=rejected(lambda:model.encode_visible(frames,hidden=hidden,visible=model.roster)),
        missing_visibility=rejected(lambda:model.encode_visible(visible,hidden=hidden,visible=())),
        unknown_visibility=rejected(lambda:model.encode_visible(visible,hidden=hidden,visible=('unknown',))),
        missing_native=rejected(lambda:model.encode_visible(({n:x for n,x in visible[0].items() if n!=names[0]},),hidden=hidden,visible=names)),
        empty_native=rejected(lambda:model.encode_visible(({**visible[0],names[0]:visible[0][names[0]][:0]},),hidden=hidden,visible=names)),
        zero_target_count=rejected(lambda:model.loss_withheld(context,codes,torch.zeros(len(frames),dtype=torch.int64),hidden=hidden)),
        wrong_code_shape=rejected(lambda:model.loss_withheld(context,codes[:,:1],counts[:,0],hidden=hidden)))
    need(fingerprint(model.state_dict())==initial and fingerprint(frames)==native_pin,
         'RECIPROCAL_PROOF_STATE_CHANGED');labels.fresh()
    need(not torch.cuda.is_initialized(),'RECIPROCAL_PROOF_UNEXPECTED_CUDA')
    result_pin=write_committed(a.output/'results.json',dict(records=records,payloads=payloads,controls=controls))
    if previous:need(result_pin==previous['results_pin'],'RECIPROCAL_PROOF_COLD_RESULTS')
    phases['controls']=time.perf_counter()-tick;tick=time.perf_counter()
    files={str(f.relative_to(a.output)):file_sha(f) for f in a.output.rglob('*') if f.is_file()}
    phases['hashes']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(sum(phases.values())/seconds>=.9,'RECIPROCAL_PROOF_PHASE_COVERAGE')
    report=dict(complete=True,binding=binding,previous_pin=a.previous_pin,files=files,results_pin=result_pin,
        atoms=len(frames),masks=len(records),masked_targets=len(frames)*len(records),initial_state=initial,
        controls=controls,phases_seconds=phases,seconds=seconds,cpu_seconds=time.process_time()-process,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        optimizer_updates=0,gpu_context_created=False,scientific_admission=False,training_ready=False,
        acceptance='author-checked',number_status='measured')
    pin=write_committed(a.output/'report.json',report)
    print(json.dumps(dict(complete=True,pin=pin,seconds=seconds,masks=len(records))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path);p.add_argument('--previous-pin')
    args=p.parse_args()
    try:run(args)
    except Exception as error:
        if args.output.exists():write_committed(args.output/'failure.json',dict(error=repr(error),code=code()))
        raise
