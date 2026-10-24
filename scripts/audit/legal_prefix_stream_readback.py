"""Real-source CPU check of the actual context encoder's carried-state path.

Native instrument outputs remain their saved CUDA measurements. This is an
explicit CPU component experiment, not a CPU fallback or a trained GPU model.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from torch import nn
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from kernel.legal_native_associations import BankLane, ProofLane
from model.attention import Linear
from model.legal_document_stream import LegalDocumentStream
from model.legal_learning_source import LegalLearningSource
from model.loop_training_checkpoint import tensor_fingerprint
from model.native_prefix_stream import NativePrefixStream, memory_tensors
from model.native_prefix_transformer import NativePrefixTransformer
from senses.contract import digest
from text.constellation_corpus import file_sha, need
from text.legal_continuations import SIDES
from text.legal_reuse_input import ReuseInputs

CONFIG = dict(width=32, heads=2, layers=2, max_native_tokens=4096,
              max_native_elements=1 << 22, max_attention_elements=1 << 20)
POLICY = dict(device='cpu', threads=1, seed=20261023, cache_bytes=16 << 20,
              output_error=5e-6, gradient_error=2e-5, optimizer_updates=0,
              graph_modes=['retained_for_gradient_equivalence', 'detached_for_bounded_backward'])


def pins():
    base = Path(__file__).parents[1]
    names = ('audit/legal_prefix_stream_readback.py', 'model/native_prefix_stream.py',
             'model/legal_document_stream.py', 'model/native_prefix_transformer.py', 'model/attention.py')
    return {name: file_sha(base/name) for name in names}


def component(roster, dimensions, saved=None):
    torch.manual_seed(POLICY['seed'])
    specs = [SimpleNamespace(name=n, dims=dimensions[n]) for n in roster]
    encoder = NativePrefixTransformer(SimpleNamespace(**CONFIG), specs)
    projectors = nn.ModuleDict({n: Linear(dimensions[n], CONFIG['width']) for n in roster})
    model = nn.ModuleDict(dict(encoder=encoder, projectors=projectors)).eval()
    if saved is not None: model.load_state_dict(saved, strict=True)
    return model, NativePrefixStream(encoder, projectors, cache_bytes=POLICY['cache_bytes'])


def native_reader(data, proof_root, proof_pin):
    proofs = {n: ProofLane(proof_root, proof_pin, data.source, n, data.native.config['instruments'][n]['lens_id'])
              for n in data.roster}
    def read(uids):
        rows = {n: [] for n in data.roster}
        for name in data.roster:
            for uid in uids:
                value = BankLane(data.native, name)([uid])[0] if uid < 16384 else proofs[name]([uid])[0]
                rows[name].append(torch.from_numpy(value.copy()))
        return tuple({n: rows[n][i] for n in data.roster} for i in range(len(uids)))
    return read, proofs


def gradient(model):
    result = {}
    for name, p in model.named_parameters():
        need(p.grad is not None and torch.isfinite(p.grad).all(), 'PREFIX_AUDIT_GRADIENT_MISSING')
        result[name] = p.grad.detach().clone()
    for prefix in ['projectors.'+n for n in model['projectors']]+['encoder.blocks.'+str(i) for i in range(CONFIG['layers'])]:
        need(any(bool(value.abs().max() > 0) for n, value in result.items() if n.startswith(prefix+'.')),
             'PREFIX_AUDIT_DISCONNECTED_BRANCH')
    return result


def check_case(model, stream, frames, context):
    before = stream.weight_pin()
    with torch.no_grad():
        dense = model['encoder'].encode(frames, model['projectors'])
        state = None; parts = []; footprints = []
        for i, frame in enumerate(frames):
            hidden, state = stream.step((frame,), context=context, offset=i, state=state, detach_history=True)
            parts.append(hidden); footprints.append(sum(t.numel()*t.element_size() for t in memory_tensors(state.layers)))
        joined = torch.cat(parts); error = float((joined-dense).abs().max())
        need(error <= POLICY['output_error'], 'PREFIX_AUDIT_DENSE_FORWARD')
        # Actual shorter prefix must not change when later source atoms are present in the same call.
        first = model['encoder'].encode(frames[:1], model['projectors'])
        causal_error = float((dense[:1]-first).abs().max())
        need(causal_error <= POLICY['output_error'], 'PREFIX_AUDIT_FUTURE_VISIBLE')
        # Keep the current atom fixed: only the earlier history changes order.
        reverse = model['encoder'].encode(tuple(reversed(frames[:-1]))+frames[-1:], model['projectors'])
        order_effect = float((dense[-1]-reverse[-1]).abs().max()) if len(frames) > 2 else None
        if len(frames) > 2: need(order_effect > POLICY['output_error'], 'PREFIX_AUDIT_ORDER_INVISIBLE')
        window_order_effect = None
        if any(len(x) > 1 for f in frames for x in f.values()):
            reversed_windows = tuple({n: x.flip(0) for n, x in f.items()} for f in frames)
            altered = model['encoder'].encode(reversed_windows, model['projectors'])
            window_order_effect = float((altered-dense).abs().max())
            need(window_order_effect > POLICY['output_error'], 'PREFIX_AUDIT_WINDOW_ORDER_INVISIBLE')
    # A fixed native-data cotangent tests the encoder Jacobian. No optimizer or task-fitting step.
    cotangent = torch.stack([frames[-1][n][0, :CONFIG['width']] for n in stream.names])
    model.zero_grad(set_to_none=True)
    expected = model['encoder'].encode(frames, model['projectors'])
    (expected[-1]*cotangent).sum().backward(); reference = gradient(model)
    model.zero_grad(set_to_none=True); state = None
    for i, frame in enumerate(frames):
        hidden, state = stream.step((frame,), context=context, offset=i, state=state, detach_history=False)
    (hidden[-1]*cotangent).sum().backward(); actual = gradient(model)
    gradient_error = max(float((actual[n]-reference[n]).abs().max()) for n in actual)
    need(gradient_error <= POLICY['gradient_error'], 'PREFIX_AUDIT_DENSE_GRADIENT')
    # Detached mode supports a backward per chunk without retaining the earlier graph.
    model.zero_grad(set_to_none=True); state = None
    for i, frame in enumerate(frames):
        hidden, state = stream.step((frame,), context=context, offset=i, state=state, detach_history=True)
        (hidden[-1]*cotangent).sum().backward()
    detached = gradient(model); model.zero_grad(set_to_none=True)
    need(stream.weight_pin() == before, 'PREFIX_AUDIT_WEIGHTS_CHANGED')
    return dict(atoms=len(frames), native_windows={n: sum(len(f[n]) for f in frames) for n in stream.names},
                max_forward_error=error, max_causal_error=causal_error, max_gradient_error=gradient_error,
                order_effect=order_effect, window_order_effect=window_order_effect, cache_bytes_after_each_atom=footprints,
                every_projector_and_block_connected=True, detached_backward_per_chunk=True,
                weight_pin=before), dict(dense=dense, chunked=joined, gradients=actual, reference_gradients=reference,
                                        detached_gradients=detached)


def controls(model, stream, frames):
    context = 'source-control'; _, state = stream.step(frames[:1], context=context, offset=0, detach_history=True)
    payload = stream.export(state)
    result = dict(wrong_context=rejected(lambda: stream.step(frames[1:2], context='wrong', offset=1, state=state, detach_history=True)),
                  wrong_offset=rejected(lambda: stream.step(frames[1:2], context=context, offset=2, state=state, detach_history=True)),
                  missing_history=rejected(lambda: stream.step(frames[1:2], context=context, offset=1, detach_history=True)))
    changed = stream.restore(payload, context=context); changed.layers[0][0].key.add_(.01)
    result['changed_cache'] = rejected(lambda: stream.step(frames[1:2], context=context, offset=1, state=changed, detach_history=True))
    corrupt = deepcopy(payload); corrupt['layers'][0][0]['value'].add_(.01)
    result['corrupt_checkpoint'] = rejected(lambda: stream.restore(corrupt, context=context))
    small = NativePrefixStream(model['encoder'], model['projectors'], cache_bytes=1)
    result['insufficient_cache'] = rejected(lambda: small.step(frames[:1], context=context, offset=0, detach_history=True))
    cloned, other = component(stream.names, {n: frames[0][n].shape[1] for n in stream.names}, model.state_dict())
    _, old = other.step(frames[:1], context=context, offset=0, detach_history=True)
    with torch.no_grad(): next(cloned.parameters()).add_(.001)
    result['changed_weights'] = rejected(lambda: other.step(frames[1:2], context=context, offset=1, state=old, detach_history=True))
    result['old_checkpoint_new_weights'] = rejected(lambda: other.restore(payload, context=context))
    return result


def source_session(data, model, stream):
    starts = data.documents[:, 0]; enough = data.documents[:, 1]-starts >= 2
    available = (data.mapping[starts] < 16384) & (data.mapping[np.minimum(starts+1, len(data.mapping)-1)] < 16384)
    choices = np.flatnonzero(enough & available); need(len(choices) > 0, 'PREFIX_AUDIT_DOCUMENT_SUPPORT')
    doc = int(choices[0]); side = SIDES[int(data.doc_sides[doc])]
    session = LegalDocumentStream(data, doc, side=side)
    frames, args, first = session.prepare(1, device='cpu')
    before_target = rejected(lambda: session.targets(first))
    double_prepare = rejected(lambda: session.prepare(1, device='cpu'))
    first_hidden, cache = stream.step(frames, **args, detach_history=True); session.accept(first, cache)
    target = session.targets(first)
    second_frames, args, second = session.prepare(1, device='cpu')
    need(target['right'] == (second.start,) and all(np.array_equal(target['observations'][n].values, second_frames[0][n].numpy())
         for n in data.roster), 'PREFIX_AUDIT_TARGET_INPUT_JOIN')
    second_hidden, cache = stream.step(second_frames, **args, detach_history=True); session.accept(second, cache)
    dense = model['encoder'].encode(frames+second_frames, model['projectors'])
    error = float((dense-torch.cat((first_hidden, second_hidden))).detach().abs().max())
    need(error <= POLICY['output_error'], 'PREFIX_AUDIT_DOCUMENT_CARRY')
    incomplete = rejected(session.close)
    wired = LegalDocumentStream(data, doc, side=side)
    wired_first, _ = wired.encode(stream, 1, device='cpu', detach_history=True)
    wired_second, _ = wired.encode(stream, 1, device='cpu', detach_history=True)
    need(torch.equal(wired_first, first_hidden) and torch.equal(wired_second, second_hidden),
         'PREFIX_AUDIT_WIRED_SOURCE_PATH')
    altered = LegalDocumentStream(data, doc, side=side)
    changed_frames, arguments, tile = altered.prepare(1, device='cpu')
    _, retained = stream.step(changed_frames, **arguments, detach_history=True)
    changed_frames[0][data.roster[0]].add_(.01)
    native_changed = rejected(lambda: altered.accept(tile, retained))
    return dict(document=doc, side=side, document_atoms=session.stop-session.start, observed_atoms=2,
                context_scope='true_document_prefix_incomplete', max_forward_error=error,
                targets_separate_until_context_accepted=True, target_matches_next_observation=True,
                wired_source_encoder_path_exact=True,
                controls=dict(early_target=before_target, repeated_prepare=double_prepare, incomplete_close=incomplete,
                              changed_native_before_accept=native_changed),
                full_document_execution_pending=True)


def run(a):
    start = tick = time.perf_counter(); a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    source = ReuseInputs(a.source, a.source_pin)
    data = LegalLearningSource(source, a.split, a.split_pin, a.cold, a.cold_pin, a.bank, a.bank_pin)
    read, proofs = native_reader(data, a.proof, a.proof_pin)
    phases = dict(authenticate=time.perf_counter()-tick); tick = time.perf_counter()
    if a.previous is not None:
        need(file_sha(a.previous/'report.json') == a.previous_pin, 'PREFIX_AUDIT_PREVIOUS_PIN')
        previous = json.loads((a.previous/'report.json').read_bytes())
        need(previous['code'] == pins() and previous['binding'] == data.binding, 'PREFIX_AUDIT_PREVIOUS_BINDING')
        for name, ref in previous['files'].items(): need(file_sha(a.previous/name) == ref['sha256'], 'PREFIX_AUDIT_PREVIOUS_FILE')
        saved = torch.load(a.previous/'continuation.pt', map_location='cpu', weights_only=True)
        frames = read(saved['uids'])
        need(digest(tensor_fingerprint(frames)) == saved['native_pin'], 'PREFIX_AUDIT_COLD_NATIVE')
        model, stream = component(data.roster, saved['dimensions'], saved['weights'])
        cache = stream.restore(saved['cache'], context=saved['context'])
        with torch.no_grad():
            actual, final = stream.step(frames[1:], context=saved['context'], offset=1, state=cache, detach_history=True)
        need(torch.equal(actual, saved['expected']) and stream.export(final)['payload_pin'] == saved['final_cache_pin'],
             'PREFIX_AUDIT_COLD_CONTINUATION')
        result = dict(cold_native_exact=True, cold_continuation_byte_exact=True, cold_final_cache_exact=True,
                      weight_pin=stream.weight_pin(), previous=dict(path=str(a.previous), pin=a.previous_pin))
        files = {}
    else:
        need(file_sha(a.shapes) == a.shapes_pin, 'PREFIX_AUDIT_SHAPE_PIN')
        shapes = json.loads(a.shapes.read_bytes()); need(shapes['source_pin'] == source.pin, 'PREFIX_AUDIT_SHAPE_SOURCE')
        cases = [r for r in shapes['longest_available_runs'] if r['atoms'] == r['distinct_inputs'] == 3]
        need(len(cases) == 3, 'PREFIX_AUDIT_CONTIGUOUS_SUPPORT')
        first_ids = data.mapping[cases[0]['start']:cases[0]['stop']].tolist(); first_frames = read(first_ids)
        dimensions = {n: first_frames[0][n].shape[1] for n in data.roster}; model, stream = component(data.roster, dimensions)
        records = []; tensors = {}; before = stream.weight_pin()
        for i, case in enumerate(cases):
            need(int(data.documents[case['document'], 0]) <= case['start'] < case['stop'] <= int(data.documents[case['document'], 1]),
                 'PREFIX_AUDIT_SOURCE_SEQUENCE')
            ids = data.mapping[case['start']:case['stop']].tolist(); frames = read(ids)
            record, raw = check_case(model, stream, frames, 'source-subsequence:'+str(case['start']))
            records.append(dict(source=case, uids=ids, scope='contiguous_source_subsequence_not_full_document', **record))
            tensors[str(i)] = raw
        longest = int(proofs[data.roster[0]].uids[int(np.argmax(np.diff(proofs[data.roster[0]].bounds)))])
        native = read([longest]); multi, raw = check_case(model, stream, native, 'multiwindow:'+str(longest))
        need(max(multi['native_windows'].values()) > 1, 'PREFIX_AUDIT_MULTIWINDOW_SUPPORT')
        tensors['multiwindow'] = raw
        negative = controls(model, stream, first_frames)
        with torch.no_grad(): document = source_session(data, model, stream)
        context = 'cold-source-subsequence:'+str(cases[0]['start'])
        with torch.no_grad():
            _, cache = stream.step(first_frames[:1], context=context, offset=0, detach_history=True)
            payload = stream.export(cache)
            expected, final = stream.step(first_frames[1:], context=context, offset=1, state=cache, detach_history=True)
        continuation = dict(uids=first_ids, dimensions=dimensions, weights=model.state_dict(), context=context,
                            native_pin=digest(tensor_fingerprint(first_frames)), cache=payload, expected=expected,
                            final_cache_pin=stream.export(final)['payload_pin'])
        torch.save(continuation, a.output/'continuation.pt'); torch.save(tensors, a.output/'arithmetic.pt')
        need(stream.weight_pin() == before, 'PREFIX_AUDIT_FINAL_WEIGHT_PIN')
        files = {n: dict(bytes=(a.output/n).stat().st_size, sha256=file_sha(a.output/n)) for n in ('continuation.pt', 'arithmetic.pt')}
        result = dict(cases=records, multiwindow=dict(uid=longest, **multi), controls=negative, document=document,
                      initial_weight_pin=before, final_weight_pin=stream.weight_pin(),
                      parameters=sum(p.numel() for p in model.parameters()), cold_continuation_pending=True)
    phases['actual_component_and_state']=time.perf_counter()-tick; tick=time.perf_counter()
    data.fresh(); phases['final_source_freshness']=time.perf_counter()-tick; seconds=time.perf_counter()-start
    need(sum(phases.values())/seconds >= .9, 'PREFIX_AUDIT_PHASE_ATTRIBUTION')
    report = dict(schema='legal-prefix-stream-component-v1', complete=True, binding=data.binding, code=pins(),
                  policy=POLICY, config=CONFIG, result=result, files=files, phases_seconds=phases, seconds=seconds,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  new_encoder_calls=0, optimizer_updates=0, full_model_training=False, scientific_admission=False,
                  acceptance='author-checked', number_status='measured')
    pin=write_committed(a.output/'report.json', report); print(json.dumps(dict(pin=pin, seconds=seconds, result=result)), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'split', 'cold', 'bank', 'proof'):
        p.add_argument('--'+name, type=Path, required=True); p.add_argument('--'+name+'-pin', required=True)
    p.add_argument('--shapes', type=Path); p.add_argument('--shapes-pin')
    p.add_argument('--previous', type=Path); p.add_argument('--previous-pin')
    p.add_argument('--output', type=Path, required=True); run(p.parse_args())
