"""Real-source boundary/gradient proof for ordered native next-region prediction.

This CPU component qualification performs no optimizer updates and opens no CUDA
context. The actual target checkpoint is a completed full TRAIN fit. Its complete
CUDA replay is a separate requirement before a full predictor training run.
"""
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

from audit import legal_training_statistics as training, legal_population_training as population
from audit import legal_streaming_rq_readback as initialization
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model.legal_native_training_stream import CertifiedNativeFrames
from model.legal_predictive_stream import CertifiedLegalDocumentStream
from model.native_population_training import fingerprint, cpu
from model.native_region_predictor import NativeRegionPredictor, target_labels
from text.constellation_corpus import file_sha, need

FIT = ('legal-population-training-full-v1-20261023',
       '0c636c5f4d7ccbb3769a48435ce3168aee07dd02058fa85369253b65470c1f9d')
POLICY = dict(device='cpu', threads=1, seed=20261025, optimizer_updates=0,
              proof_tile=8, production_tile=32, cache_bytes=16 << 20,
              output_error=5e-6, gradient_error=2e-5, dense_prefix_atoms=32)


def code():
    root = Path(__file__).resolve().parents[1]
    return dict(population.code(), **{n: file_sha(root/n) for n in (
        'audit/legal_region_predictor_readback.py', 'model/native_region_predictor.py',
        'model/legal_predictive_stream.py', 'model/legal_document_stream.py',
        'model/native_prefix_stream.py', 'model/native_prefix_transformer.py')})


def load_target(root, stats, dimensions):
    fit = training.pinned(root/FIT[0]/'report.json', FIT[1])
    need(fit['complete'] and fit['scope'] == 'full' and fit['binding']['code'] == population.code(),
         'REGION_PROOF_FIT_AUTHORITY')
    ref = fit['checkpoints'][str(fit['updates'])]
    path = root/FIT[0]/f"checkpoint-{fit['updates']}.pt"
    need(file_sha(path) == ref['pin'], 'REGION_PROOF_CHECKPOINT_PIN')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    need(fingerprint(saved) == ref['state'] and saved['binding'] == fit['binding']
         and saved['ledger'] == fit['ledger_head'], 'REGION_PROOF_CHECKPOINT_STATE')
    model = population.probe.construct(population.PackedNativeRQ, stats, dimensions, saved['model'])
    model.eval().requires_grad_(False)
    return model, fit, ref


def selection(data, root):
    lengths = np.asarray(data.documents[:, 1]-data.documents[:, 0])
    sides = np.asarray(data.doc_sides)
    selected = {}
    for side in range(3):
        for lo, hi in ((4, 8), (17, 32)):
            eligible = np.flatnonzero((sides == side) & (lengths >= lo) & (lengths <= hi)).tolist()
            need(eligible, 'REGION_PROOF_DOCUMENT_STRATUM')
            ordered = sorted(eligible, key=lambda i: hashlib.sha256(f'next-region-20261025:{i}'.encode()).digest())
            for i in ordered[:2]: selected.setdefault(i, []).append(f'side{side}_length{lo}_{hi}')
    census = training.pinned(root/training.DEFAULTS['census'][0]/'report.json', training.DEFAULTS['census'][1])
    multi = training.pinned(root/training.DEFAULTS['census'][0]/'multiwindow.json', census['files']['multiwindow.json'])
    for side in range(3):
        candidates = set()
        for row in multi:
            uid = row['uid']; a, b = map(int, data.source.arrays['offsets'][uid:uid+2])
            occurrences = data.source.arrays['occurrences'][a:b]
            docs = np.searchsorted(data.documents[:, 1], occurrences, side='right')
            candidates.update(int(i) for i in docs if sides[i] == side and lengths[i] >= 4)
        need(candidates, 'REGION_PROOF_MULTIWINDOW_DOCUMENT')
        i = min(candidates, key=lambda i: (lengths[i], hashlib.sha256(str(i).encode()).digest()))
        selected.setdefault(i, []).append('native_multiwindow')
    rows = [dict(document=i, side=('train','development','query')[int(sides[i])],
                 start=int(data.documents[i, 0]), stop=int(data.documents[i, 1]), reasons=selected[i])
            for i in sorted(selected)]
    train = lengths[sides == 0]
    return dict(documents=rows, source=data.binding, score_blind=True,
                population=dict(train_documents=len(train), train_atoms=int(train.sum()),
                    train_successors=int((train-1).sum()), max_document_atoms=int(train.max()),
                    sum_document_lengths_squared=int(np.square(train).sum())))


def tensor_delta(a, b):
    need(a.keys() == b.keys(), 'REGION_PROOF_COMPARE_KEYS')
    return max(float((a[n]-b[n]).abs().max()) for n in a)


def grad(model):
    values = {n: p.grad for n, p in model.named_parameters()}
    need(all(x is not None and bool(torch.isfinite(x).all()) for x in values.values()),
         'REGION_PROOF_GRADIENT_MISSING')
    for prefix in ['projectors.'+n+'.' for n in model.roster]+['encoder.', 'region.', 'counts.']:
        need(any(bool(x.abs().max() > 0) for n, x in values.items() if n.startswith(prefix)),
             'REGION_PROOF_DISCONNECTED_BRANCH')
    return cpu(values)


def frozen_readings(data, reader, row):
    ids = data.mapping[row['start']:row['stop']].tolist(); frames = []
    for lo in range(0, len(ids), 128):
        part = ids[lo:lo+128]; actual = reader.frames(part)
        ordinary = data.observations(list(range(row['start']+lo, row['start']+lo+len(part))), side=row['side'])
        for j, frame in enumerate(actual):
            for name in data.roster:
                a, b = ordinary[name].offsets[j:j+2]
                need(np.array_equal(frame[name].numpy(), ordinary[name].values[a:b]), 'REGION_PROOF_NATIVE_READBACK')
        frames.extend(actual)
    return tuple(frames)


def case(model, target, data, reader, row, out):
    start = time.perf_counter(); frames = frozen_readings(data, reader, row); native_pin = fingerprint(frames)
    document = CertifiedLegalDocumentStream(data, reader, row['document'], side=row['side'])
    stream = model.stream(); hidden_parts = []; successors = []; target_frames = []; states = []
    while document.cursor < document.stop:
        hidden, tile = document.encode(stream, POLICY['proof_tile'], device='cpu', detach_history=True,
                                       memory_bytes=16 << 20)
        # Targets are retrieved only after that tile's permitted history is encoded.
        labels = document.targets(tile)
        hidden_parts.append(hidden.detach()); successors.extend(zip(labels['left'], labels['right']))
        target_frames.extend(labels['frames'])
        states.append(dict(stop=tile.stop, next_atom=document.state.next_atom))
    close = document.close(); hidden = torch.cat(hidden_parts)
    need(successors == [(i, i+1) for i in range(row['start'], row['stop']-1)]
         and fingerprint(tuple(target_frames)) == fingerprint(frames[1:]), 'REGION_PROOF_TARGET_COVERAGE')
    dense_count = min(len(frames), POLICY['dense_prefix_atoms'])
    with torch.no_grad():
        dense = model.encoder.encode(frames[:dense_count], model.projectors)
        dense_error = float((dense-hidden[:dense_count]).abs().max())
        prefix = model.encoder.encode(frames[:2], model.projectors)
        future_error = float((prefix-hidden[:2]).abs().max())
        need(max(dense_error, future_error) <= POLICY['output_error'], 'REGION_PROOF_CAUSAL_ARITHMETIC')
        # Independent singleton tile partition over the complete same history.
        other = model.stream(); state = None; singleton = []
        for i, frame in enumerate(frames):
            h, state = other.step((frame,), context=document.context, offset=i, state=state, detach_history=True)
            singleton.append(h)
        singleton_error = float((torch.cat(singleton)-hidden).abs().max())
        need(singleton_error <= POLICY['output_error'], 'REGION_PROOF_TILE_PARTITION')
        checkpoint = stream.export(document.state)
        torch.save(checkpoint, out/'context.pt')
        loaded = torch.load(out/'context.pt', map_location='cpu', weights_only=True)
        restored = stream.restore(loaded, context=document.context)
        need(fingerprint(stream.export(restored)) == fingerprint(checkpoint), 'REGION_PROOF_CONTEXT_READBACK')
        codes, counts = target_labels(target, frames[1:])
        predicted = model.forecast(hidden[:-1]); decoded = model.render(predicted, target)
        # Independent decode from the actually forecast values/counts.
        native_reference = tuple(target.reconstruct(target.values(predicted['codes'][i:i+1]),
            [{n: int(predicted['counts'][n][i]) for n in model.roster}])[0] for i in range(len(codes)))
        decode_error = max(float((a[n]-b[n]).abs().max()) for a,b in zip(decoded,native_reference) for n in model.roster)
        need(decode_error <= POLICY['output_error'], 'REGION_PROOF_FORECAST_DECODE')
        loss = model.loss(hidden[:-1], codes, counts)
        reference = [-score.double().log_softmax(-1).gather(1, codes[:, i:i+1]).mean()
                     for i, score in enumerate(loss['logits'])]
        need(max(abs(float(a-b)) for a,b in zip(reference, loss['region'])) <= 5e-7,
             'REGION_PROOF_CE_REFERENCE')
        # Teacher forcing never exposes the label currently being scored or later labels.
        teacher_controls = []
        for level, size in enumerate(model.sizes):
            changed = codes.clone(); changed[:, level:] = 0; changed[:, level] = (codes[:, level]+1) % size
            new = model.region(hidden[:-1], changed)[0]
            need(all(torch.equal(a,b) for a,b in zip(loss['logits'][:level+1], new[:level+1])),
                 'REGION_PROOF_CURRENT_LABEL_LEAK')
            teacher_controls.append(True)
        control_frames = tuple(reversed(frames[:dense_count-1]))+(frames[dense_count-1],)
        changed = model.encoder.encode(control_frames, model.projectors)[-1:]
        order_delta = float((dense[-1:]-changed).abs().max())
        need(order_delta > 0, 'REGION_PROOF_ORDER_INVISIBLE')
        correct_logits = model.forecast(dense[-1:])['logits']; wrong_logits = model.forecast(changed)['logits']
        order_logit_delta = max(float((a-b).abs().max()) for a,b in zip(correct_logits,wrong_logits))
        need(order_logit_delta > 0, 'REGION_PROOF_REGION_ORDER_INVISIBLE')
        generated = model.forecast(hidden[:-1], generator=torch.Generator().manual_seed(POLICY['seed']))
        again = model.forecast(hidden[:-1], generator=torch.Generator().manual_seed(POLICY['seed']))
        need(fingerprint(generated) == fingerprint(again), 'REGION_PROOF_SEEDED_GENERATION')
    gradients = None
    if len(frames) <= POLICY['dense_prefix_atoms']:
        model.zero_grad(set_to_none=True)
        full = model.encoder.encode(frames, model.projectors)
        model.loss(full[:-1], codes, counts)['loss'].backward(); expected = grad(model)
        model.zero_grad(set_to_none=True); retained = model.stream(); cache = None; parts = []
        for lo in range(0, len(frames), POLICY['proof_tile']):
            h, cache = retained.step(frames[lo:lo+POLICY['proof_tile']], context=document.context,
                                     offset=lo, state=cache, detach_history=False)
            parts.append(h)
        model.loss(torch.cat(parts)[:-1], codes, counts)['loss'].backward(); actual = grad(model)
        error = tensor_delta(actual, expected)
        need(error <= POLICY['gradient_error'], 'REGION_PROOF_GRADIENT_ARITHMETIC')
        gradients = dict(maximum_error=error, branches={n: fingerprint({k:v for k,v in actual.items() if k.startswith(n)})
            for n in ['encoder.', 'region.', 'counts.']+['projectors.'+n+'.' for n in model.roster]})
        model.zero_grad(set_to_none=True)
    need(fingerprint(frames) == native_pin, 'REGION_PROOF_NATIVE_CHANGED')
    payload = dict(frames=frames, hidden=hidden, codes=codes, counts=counts,
                   forecast=cpu(predicted), decoded=cpu(decoded), native_pin=native_pin)
    torch.save(payload, out/'outputs.pt')
    readback = torch.load(out/'outputs.pt', map_location='cpu', weights_only=True)
    need(fingerprint(payload) == fingerprint(readback), 'REGION_PROOF_OUTPUT_READBACK')
    result = dict(document=row['document'], side=row['side'], atoms=len(frames), successors=len(codes),
        native_windows={n:sum(len(f[n]) for f in frames) for n in model.roster}, all_native_exact=True,
        target_addresses=successors, native_pin=native_pin, output_state=fingerprint(payload),
        dense_compared_atoms=dense_count, dense_error=dense_error, future_exclusion_error=future_error,
        singleton_partition_error=singleton_error, order_hidden_delta=order_delta, order_logit_delta=order_logit_delta,
        singleton_decode_error=decode_error,
        current_future_code_withheld=teacher_controls, gradients=gradients, close=close, states=states,
        untrained_region_loss=[float(x) for x in loss['region']], untrained_count_loss={n:float(x) for n,x in loss['counts'].items()},
        autonomous_counts=True, placement='frozen_codebook_reconstruction_not_learned_stochastic_residual')
    write_committed(out/'result.json', result)
    return result, dict(seconds=time.perf_counter()-start)


def controls(model, data, reader, rows):
    row = rows[0]; doc = CertifiedLegalDocumentStream(data, reader, row['document'], side=row['side'])
    frames, args, tile = doc.prepare(1, device='cpu'); stream = model.stream()
    values = dict(target_before_context=rejected(lambda: doc.targets(tile)),
        cross_document=rejected(lambda: data.validate_next(np.asarray([row['stop']-1]), np.asarray([row['stop']]))),
        forbidden_held_fit=rejected(lambda: CertifiedLegalDocumentStream(data, reader,
            next(r['document'] for r in rows if r['side']=='query'), side='query', fit=True)))
    hidden, state = stream.step(frames, **args, detach_history=True); doc.accept(tile,state)
    values['missing_history'] = rejected(lambda: stream.step(frames,context=doc.context,offset=1,state=None,detach_history=True))
    values['wrong_context'] = rejected(lambda: stream.step(frames,context='other',offset=1,state=state,detach_history=True))
    bad = {n:x.clone() for n,x in frames[0].items()}; bad[model.roster[0]][0,0] = float('nan')
    values['nonfinite'] = rejected(lambda: stream.step((bad,),context='bad',offset=0,state=None,detach_history=True))
    try:model.forecast(hidden, target_codes=torch.zeros((1,3),dtype=torch.int64))
    except TypeError as error:values['forbidden_target_argument'] = str(error)
    else:raise AssertionError('REGION_PROOF_TARGET_ARGUMENT_ACCEPTED')
    wrong_target = deepcopy(model)
    values['wrong_frozen_target'] = rejected(lambda: model.render(model.forecast(hidden), wrong_target))
    saved = deepcopy(model.state_dict())
    with torch.no_grad(): next(model.encoder.parameters()).add_(1.)
    values['stale_context_weights'] = rejected(lambda: stream.step(frames,context=doc.context,offset=1,state=state,detach_history=True))
    model.load_state_dict(saved,strict=True)
    return values


def run(a):
    began = tick = time.perf_counter(); cpu_start = time.process_time()
    a.output.mkdir(parents=True, exist_ok=False)
    for name, pin in code().items():
        source = Path(__file__).resolve().parents[1]/name
        saved = a.output/'source'/name; saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(source.read_bytes()); need(file_sha(saved) == pin, 'REGION_PROOF_CODE_COPY')
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    _,_,data,c,_,_,_,_ = training.authenticate(a.root)
    root = a.root/initialization.STATS[0]; sr = training.pinned(root/'report.json', initialization.STATS[1])
    need(file_sha(root/'statistics.pt') == sr['files']['statistics.pt'], 'REGION_PROOF_STATISTICS_PIN')
    stats = torch.load(root/'statistics.pt', map_location='cpu', weights_only=True)
    dimensions = {n:len(stats['centers'][n]) for n in data.roster}
    target,fit,ref = load_target(a.root,stats,dimensions); target_before = fingerprint(target.state_dict())
    need(fit['binding']['source'] == data.binding, 'REGION_PROOF_SOURCE_BINDING')
    model = NativeRegionPredictor(target, target_pin=ref['pin'], seed=POLICY['seed'])
    before = fingerprint(model.state_dict()); reader = CertifiedNativeFrames(data,a.root/training.DEFAULTS['audit'][0],training.DEFAULTS['audit'][1])
    plan = selection(data,a.root); plan_pin = write_committed(a.output/'selection.json',plan)
    binding = dict(code=code(), policy=POLICY, source=data.binding, target=ref, fit=FIT[1], selection=plan_pin)
    previous = training.pinned(a.previous/'report.json', a.previous_pin) if a.previous else None
    if previous: need(previous['complete'] and previous['binding']==binding,'REGION_PROOF_COLD_BINDING')
    phases = dict(authentication_and_selection=time.perf_counter()-tick); tick=time.perf_counter()
    results=[]; timing=[]
    for row in plan['documents']:
        out=a.output/f"document-{row['document']}";out.mkdir()
        result, elapsed=case(model,target,data,reader,row,out);results.append(result);timing.append(elapsed)
        print(json.dumps(dict(document=row['document'],atoms=result['atoms'],seconds=elapsed['seconds'])),flush=True)
    phases['real_document_boundaries']=time.perf_counter()-tick;tick=time.perf_counter()
    negative=controls(model,data,reader,plan['documents'])
    need(before==fingerprint(model.state_dict()) and target_before==fingerprint(target.state_dict()),'REGION_PROOF_WEIGHTS_CHANGED')
    reader.close();c.fresh();data.fresh()
    need(not torch.cuda.is_initialized(), 'REGION_PROOF_UNEXPECTED_CUDA_CONTEXT')
    rp=write_committed(a.output/'results.json',dict(results=results,controls=negative))
    if previous:need(rp==previous['results_pin'],'REGION_PROOF_COLD_RESULTS')
    phases['controls_and_readback']=time.perf_counter()-tick;tick=time.perf_counter()
    files={str(f.relative_to(a.output)):file_sha(f) for f in a.output.rglob('*') if f.is_file()}
    phases['publication_hashes']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(sum(phases.values())/seconds>=.9,'REGION_PROOF_PHASE_ATTRIBUTION')
    result=dict(complete=True,binding=binding,results_pin=rp,files=files,controls=negative,
        documents=len(results),atoms=sum(r['atoms'] for r in results),successors=sum(r['successors'] for r in results),
        phases_seconds=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu_start,reader_stats=reader.stats,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,parameters=sum(p.numel() for p in model.parameters()),
        initial_state=before,target_state=target_before,previous_pin=a.previous_pin,
        optimizer_updates=0,gpu_context_created=False,scientific_admission=False,training_ready=False,
        full_target_cold_required=True,number_status='measured',acceptance='author-checked')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',result),documents=result['documents'],seconds=seconds)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous',type=Path);parser.add_argument('--previous-pin')
    a=parser.parse_args()
    try:run(a)
    except Exception as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error),code=code()))
        raise
