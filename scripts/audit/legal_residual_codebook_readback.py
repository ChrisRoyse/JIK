"""Real native-window RQ component experiment and cold AdamW continuation.

An explicit one-thread CPU component fit; frozen instrument measurements are
reused from their CUDA publication. This is not legal prediction or admission.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import resource
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit.legal_stock_readback import rejected
from audit.legal_prefix_stream_readback import native_reader
from claims.process_conditions import write_committed
from model.legal_learning_source import LegalLearningSource
from model.loop_training_checkpoint import tensor_fingerprint
from model.native_residual_codebook import NativeResidualCodebook, AutoregressiveSemanticHead, FrozenNativeCodebook, save_frozen
from senses.contract import digest
from text.constellation_corpus import file_sha, need
from text.legal_continuation_input import Continuations
from text.legal_reuse_input import ReuseInputs

CONFIG = dict(width=32, heads=2, layers=1, max_native_tokens=4096,
              max_native_elements=1 << 22, max_attention_elements=1 << 20)
POLICY = dict(device='cpu', threads=1, seed=20261024, sizes=[4,16,64], updates=32,
              batch=8, learning_rate=1e-4, weight_decay=.01, buffer_bytes=64 << 20, arithmetic_error=5e-6)


def pins():
    base = Path(__file__).resolve().parents[1]
    names = ['audit/legal_residual_codebook_readback.py', 'model/native_residual_codebook.py',
             'model/native_prefix_transformer.py', 'model/attention.py', 'model/loop_training_checkpoint.py']
    return {n: file_sha(base/n) for n in names}


def pinned(path, pin):
    need(file_sha(path) == pin, 'RQ_AUDIT_INPUT_PIN'); return json.loads(path.read_bytes())


def fit_scope(uids, presence):
    need(uids and all(type(i) is int and 0 <= i < len(presence) and presence[i]&1 for i in uids),
         'RQ_AUDIT_NON_TRAIN_FIT')


def make(dimensions):
    torch.manual_seed(POLICY['seed'])
    model = NativeResidualCodebook(dimensions, CONFIG, POLICY['sizes'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=POLICY['learning_rate'],
        weight_decay=POLICY['weight_decay'], foreach=False, fused=False)
    return model, optimizer


def train_step(model, optimizer, frames, step):
    batch = tuple(frames[(step*POLICY['batch']+j) % len(frames)] for j in range(POLICY['batch']))
    before = digest(tensor_fingerprint(model.state_dict())); optimizer.zero_grad(set_to_none=True)
    output = model(batch); output['loss'].backward()
    gradients = {n: p.grad for n,p in model.named_parameters()}
    need(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients.values()), 'RQ_AUDIT_GRADIENT')
    for prefix in ['projectors.'+n for n in model.roster]+['decoders.'+n for n in model.roster]+['encoder.', 'books.', 'to_latent.']:
        need(any(bool(g.abs().max() > 0) for n,g in gradients.items() if n.startswith(prefix)), 'RQ_AUDIT_DISCONNECTED')
    grad = digest(tensor_fingerprint(gradients)); optimizer.step()
    need(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'RQ_AUDIT_NONFINITE_UPDATE')
    return dict(step=step+1, loss=float(output['loss'].detach()), gradient=grad, before=before,
                after=digest(tensor_fingerprint(model.state_dict())), optimizer=digest(tensor_fingerprint(optimizer.state_dict())))


@torch.no_grad()
def measure(model, frames):
    # Each observation is scored independently, with its complete attested windows.
    codes, errors, cosines = [], {n: [] for n in model.roster}, {n: [] for n in model.roster}
    for f in frames:
        out = model((f,)); codes.append(out['codes'][0])
        for n in model.roster:
            d = out['decoded'][0][n]
            need(d.shape == f[n].shape, 'RQ_AUDIT_LOST_OUTPUT_WINDOW')
            errors[n].append(float((d-f[n]).square().mean()))
            cosines[n].extend(torch.nn.functional.cosine_similarity(d, f[n], dim=-1).tolist())
    codes = torch.stack(codes)
    return dict(atoms=len(frames), per_lane_mse={n:float(np.mean(v)) for n,v in errors.items()},
                per_lane_mean_window_cosine={n:float(np.mean(v)) for n,v in cosines.items()},
                native_windows={n:len(v) for n,v in cosines.items()},
                usage=[torch.bincount(codes[:,l], minlength=c).tolist() for l,c in enumerate(model.sizes)],
                distinct_codes=int(torch.unique(codes, dim=0).shape[0])), codes


@torch.no_grad()
def reconstruction_baselines(groups, roster):
    # Atom-weighted TRAIN mean matches the experiment's per-atom MSE objective.
    mean={n:torch.stack([f[n].mean(0) for f in groups['train']]).mean(0) for n in roster}
    return {g:{n:dict(zero_mse=float(torch.stack([f[n].square().mean() for f in frames]).mean()),
                     train_mean_mse=float(torch.stack([(f[n]-mean[n]).square().mean() for f in frames]).mean()))
               for n in roster} for g,frames in groups.items()}


@torch.no_grad()
def independent_quantization(model, frames):
    latent = model.latent(frames); residual = latent.detach().numpy().astype(np.float64)
    rows = []; terms = []
    for book in model.books:
        values = book.detach().numpy().astype(np.float64)
        ids = np.asarray([min(range(len(values)), key=lambda i: float(np.dot(r-values[i], r-values[i]))) for r in residual])
        rows.append(ids); terms.append(values[ids]); residual = residual-values[ids]
    expected = np.stack(rows, -1); actual, straight, _ = model.quantize(latent)
    need(np.array_equal(expected, actual.numpy()), 'RQ_AUDIT_REFERENCE_CODES')
    error = float(np.max(np.abs(sum(terms)-straight.numpy())))
    need(error <= POLICY['arithmetic_error'], 'RQ_AUDIT_REFERENCE_VALUES')
    return dict(codes_exact=True, quantized_max_error=error), dict(latent=latent, codes=actual,
        books=[b.detach().clone() for b in model.books], straight=straight)


def head_check(model, frames, codes):
    head = AutoregressiveSemanticHead(len(model.roster), CONFIG['width'], model.sizes)
    # Actual typed context output, not native vector concatenation.
    context = model.encoder.encode(frames[:2], model.projectors).detach()
    targets = codes[:2]; logits,_ = head(context, targets); controls = []
    for level in range(len(model.sizes)):
        changed = targets.clone()
        for j in range(level, len(model.sizes)): changed[:,j] = (changed[:,j]+1)%model.sizes[j]
        other,_ = head(context, changed)
        need(all(torch.equal(logits[j], other[j]) for j in range(level+1)), 'RQ_AUDIT_FUTURE_CODE_LEAK')
        controls.append(level)
    changed = targets.clone(); changed[:,0] = (changed[:,0]+1)%model.sizes[0]
    other,_ = head(context, changed)
    effect = float((other[1]-logits[1]).abs().max().detach()); need(effect > 0, 'RQ_AUDIT_PREFIX_IGNORED')
    loss = sum(torch.nn.functional.cross_entropy(x, targets[:,i]) for i,x in enumerate(logits)); loss.backward()
    need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters()), 'RQ_AUDIT_HEAD_GRADIENT')
    sampled = head(context, generator=torch.Generator().manual_seed(9))[1]
    return dict(future_levels_invisible=controls, earlier_code_effect=effect, sampled_ids=sampled.tolist(),
                head_optimizer_updates=0), dict(weights=head.state_dict(), context=context, targets=targets,
                                               logits=[x.detach() for x in logits])


def run(a):
    start=tick=time.perf_counter(); a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    baseline=pinned(a.baseline/'report.json', a.baseline_pin)
    source=ReuseInputs(**{'root':baseline['source']['path'], 'pin':baseline['source']['pin']})
    continuation=Continuations(Path(baseline['continuations']['path']), baseline['continuations']['pin'])
    need(continuation.report['source']['pin']==source.pin, 'RQ_AUDIT_SOURCE_JOIN')
    split,cold=continuation.report['split'],continuation.report['cold']; bank=baseline['bank']; proof=baseline['proof']
    data=LegalLearningSource(source, split['path'], split['pin'], cold['path'], cold['pin'], bank['path'], bank['pin'])
    read,proofs=native_reader(data,Path(proof['path']),proof['pin']); presence=continuation.presence
    phases=dict(authenticate=time.perf_counter()-tick); tick=time.perf_counter()
    if a.previous is not None:
        previous=pinned(a.previous/'report.json',a.previous_pin)
        need(previous['code']==pins() and previous['binding']==data.binding,'RQ_AUDIT_PREVIOUS_BINDING')
        for n,ref in previous['files'].items(): need(file_sha(a.previous/n)==ref['sha256'],'RQ_AUDIT_PREVIOUS_FILE')
        saved=torch.load(a.previous/'resume.pt',map_location='cpu',weights_only=True)
        fit_scope(saved['uids'],presence); frames=read(saved['uids'])
        need(digest(tensor_fingerprint(frames))==saved['native_pin'],'RQ_AUDIT_COLD_NATIVE')
        model,opt=make(saved['dimensions']); model.load_state_dict(saved['weights']); opt.load_state_dict(saved['optimizer'])
        ledger=[train_step(model,opt,frames,i) for i in range(16,32)]
        need(ledger==previous['result']['ledger'][16:],'RQ_AUDIT_COLD_STEPS')
        r=previous['result']; all_uids=sorted(set(sum(r['groups'].values(),[])+r['multiwindow_uids']))
        all_frames=read(all_uids); actual=torch.load(a.previous/'arithmetic.pt',map_location='cpu',weights_only=True)
        frozen=FrozenNativeCodebook(a.previous/'codebook.pt',r['frozen_pin'],r['frozen_binding'],'cpu')
        codes=frozen.encode(all_frames); decoded=frozen.decode(codes,[{n:len(f[n]) for n in data.roster} for f in all_frames])
        need(torch.equal(codes,actual['codes']) and tensor_fingerprint(decoded)==tensor_fingerprint(actual['decoded']),
             'RQ_AUDIT_COLD_FROZEN_OUTPUT')
        result=dict(cold_native_exact=True,cold_updates_exact=16,final_weight_pin=ledger[-1]['after'],
                    final_optimizer_pin=ledger[-1]['optimizer'],previous_pin=a.previous_pin,cold_frozen_outputs_exact=True)
    else:
        groups={'train':np.flatnonzero(presence[:16384]&1)[:128].tolist(),
                'development':np.flatnonzero(((presence[:16384]&2)>0)&((presence[:16384]&1)==0))[:16].tolist(),
                'query':np.flatnonzero(((presence[:16384]&4)>0)&((presence[:16384]&3)==0))[:16].tolist()}
        canonical=proofs[data.roster[0]]
        multi=sorted({int(canonical.uids[i]) for i in range(len(canonical.uids)) if canonical.bounds[i+1]-canonical.bounds[i]>1})
        groups['train'] += [i for i in multi if presence[i]&1 and i not in groups['train']]
        fit_scope(groups['train'],presence); need(len(groups['development'])==len(groups['query'])==16,'RQ_AUDIT_SPLIT_SUPPORT')
        all_uids=sorted(set(sum(groups.values(),[])+multi)); loaded=read(all_uids); by_uid=dict(zip(all_uids,loaded))
        need(sum(x.numel()*x.element_size() for f in loaded for x in f.values())<=POLICY['buffer_bytes'],'RQ_AUDIT_BUFFER')
        frames={g:tuple(by_uid[i] for i in ids) for g,ids in groups.items()}
        dimensions={n:loaded[0][n].shape[1] for n in data.roster}; model,opt=make(dimensions)
        native_pin=digest(tensor_fingerprint(loaded)); model.initialize(frames['train']); before={g:measure(model,f)[0] for g,f in frames.items()}
        phases['load_initialize_and_before']=time.perf_counter()-tick; tick=time.perf_counter()
        ledger=[]
        for step in range(32):
            ledger.append(train_step(model,opt,frames['train'],step))
            if step==15:
                torch.save(dict(uids=groups['train'],dimensions=dimensions,weights=model.state_dict(),optimizer=opt.state_dict(),
                    native_pin=digest(tensor_fingerprint(frames['train']))), a.output/'resume.pt')
        phases['fit_32_steps']=time.perf_counter()-tick; tick=time.perf_counter()
        after={}; codes={}
        for g,f in frames.items(): after[g],codes[g]=measure(model,f)
        reference,arithmetic=independent_quantization(model,loaded)
        np.save(a.output/'fit-uids.npy',np.asarray(groups['train'],dtype='<i8'),allow_pickle=False)
        binding=dict(source_pin=source.pin,split_pin=split['pin'],bank_pin=bank['pin'],
                     fit_members_sha256=file_sha(a.output/'fit-uids.npy'),fitting='TRAIN_unique_inputs_only',
                     instruments={n:data.native.config['instruments'][n]['lens_id'] for n in data.roster},
                     scope='CPU_component_fit_not_full_source_prediction')
        artifact=a.output/'codebook.pt'; pin=save_frozen(model,artifact,binding)
        frozen=FrozenNativeCodebook(artifact,pin,binding,'cpu')
        actual=frozen.encode(loaded); need(torch.equal(actual,arithmetic['codes']),'RQ_AUDIT_FROZEN_CODES')
        counts=[{n:len(f[n]) for n in data.roster} for f in loaded]
        decoded=frozen.decode(actual,counts)
        with torch.no_grad():
            direct=model.reconstruct(arithmetic['straight'],counts)
            decode_error=max(float((d[n]-r[n]).abs().max()) for d,r in zip(decoded,direct) for n in data.roster)
        need(decode_error<=POLICY['arithmetic_error'],'RQ_AUDIT_FROZEN_RECONSTRUCTION')
        arithmetic['decoded']=decoded; reference['frozen_decode_error']=decode_error
        first=frozen.encode((loaded[0],)); mixed=frozen.encode((loaded[-1],loaded[0],loaded[0]))
        need(torch.equal(first[0],mixed[1]) and torch.equal(mixed[1],mixed[2]),'RQ_AUDIT_BATCH_SHAPE')
        head,head_tensors=head_check(model,frames['train'],codes['train'])
        controls=[]
        for name,fn in [('non_train',lambda:fit_scope(groups['development'],presence)),
            ('missing_slot',lambda:frozen.encode(({n:x for n,x in loaded[0].items() if n!=data.roster[0]},))),
            ('wrong_shape',lambda:frozen.encode(({**loaded[0],data.roster[0]:loaded[0][data.roster[0]][:,:-1]},))),
            ('empty_window',lambda:frozen.encode(({**loaded[0],data.roster[0]:loaded[0][data.roster[0]][:0]},))),
            ('wrong_pin',lambda:FrozenNativeCodebook(artifact,'0'*64,binding,'cpu')),
            ('invalid_code',lambda:frozen.decode(torch.full_like(actual,-1),counts)),
            ('wrong_binding',lambda:FrozenNativeCodebook(artifact,pin,{**binding,'source_pin':'0'*64},'cpu'))]:
            controls.append(dict(name=name,result=rejected(fn)))
        longest=by_uid[multi[-1]]; lane=next(n for n in data.roster if len(longest[n])>1)
        shortened={**longest,lane:longest[lane][:-1]}
        need(tensor_fingerprint(shortened)!=tensor_fingerprint(longest),'RQ_AUDIT_LOST_NATIVE_UNDETECTED')
        with torch.no_grad():
            original=model.latent((longest,)); reverse=model.latent(({n:x.flip(0) for n,x in longest.items()},))
            order_effect=float((original-reverse).abs().max()); need(order_effect>0,'RQ_AUDIT_ORDER_INVISIBLE')
            frozen.model.books[0][0,0].add_(1)
        controls.append(dict(name='changed_frozen_weight',result=rejected(lambda:frozen.encode((loaded[0],)))))
        need(native_pin==digest(tensor_fingerprint(loaded)),'RQ_AUDIT_NATIVE_CHANGED')
        torch.save(dict(**arithmetic,head=head_tensors),a.output/'arithmetic.pt')
        result=dict(groups=groups,multiwindow_uids=multi,before=before,after=after,ledger=ledger,reference=reference,
                    head=head,controls=controls,native_pin=native_pin,order_effect=order_effect,
                    parameters=sum(p.numel() for p in model.parameters()),frozen_pin=pin,frozen_binding=binding,
                    window_coverage='all_native_windows_reconstructed_per_slot',cold_continuation_pending=True,
                    reconstruction_baselines=reconstruction_baselines(frames,data.roster),
                    promotion='not_authorized_by_component_check')
    phases['readback_controls_and_publication']=time.perf_counter()-tick; tick=time.perf_counter()
    data.fresh(); continuation.fresh(); phases['freshness']=time.perf_counter()-tick
    files={p.name:dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in a.output.iterdir() if p.is_file()}
    seconds=time.perf_counter()-start; need(sum(phases.values())/seconds>=.9,'RQ_AUDIT_PHASE_ATTRIBUTION')
    report=dict(schema='legal-native-rq-component-v1',complete=True,binding=data.binding,policy=POLICY,config=CONFIG,
                code=pins(),result=result,files=files,phases_seconds=phases,seconds=seconds,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                acceptance='author-checked',number_status='measured',full_source_training_pending=True)
    pin=write_committed(a.output/'report.json',report); print(json.dumps(dict(pin=pin,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--baseline-pin',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path);p.add_argument('--previous-pin')
    run(p.parse_args())
