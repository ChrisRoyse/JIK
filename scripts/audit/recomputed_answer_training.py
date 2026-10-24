"""Real complete-answer arithmetic and memory qualification for attention recomputation."""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import full_history_source as source
from audit.answer_surface_stream import rejected
from audit.predicted_surface_training import persist
from claims.process_conditions import write_committed
from model import answer_full_history as learner
from model.native_population_training import cpu, fingerprint
from model.native_request_input import NativeRequest, legal
from model.recomputed_answer_attention import recompute_attention
from model.saved_gradient_storage import SavedStorage
from text.constellation_corpus import file_sha, need

CONSUMER = ('legal-complete-answer-consumer-v2-20261024',
            '5ea64522dde1aa415c84bcacdac263c36f15e8177f5b48dfaf37c9e44be65ea2')
CONSUMER_COLD = ('legal-complete-answer-consumer-cold-v2-20261024',
                 'c2b6c3534393239073db8e81e852f7ddc4e6deab8e1e067f37b9415632491268')
NEW = ('model/recomputed_answer_attention.py', 'audit/recomputed_answer_training.py')
PHYSICAL_REFUSALS = {'MNLM_TEXT_CORPUS_GRADIENT_STORAGE_MEMORY', 'MNLM_CODATA_PREFIX_STREAM_CACHE_MEMORY'}


def numerical(payload):
    row = payload['record']
    return {**{n:payload[n] for n in ('gradients', 'early', 'model', 'optimizer')},
            'record':{n:v for n,v in row.items() if n not in ('storage', 'phases')}}


def exact(left, right):
    need(fingerprint(numerical(left)) == fingerprint(numerical(right)), 'RECOMPUTE_NUMERICAL_MISMATCH')


def controls(state):
    model, opt = source.prior.make(state, 'shared'); item, request = source.prior.item(state, state['train'][0])
    origin = fingerprint((model.state_dict(), opt.state_dict())); out = {}
    def run(value=item, q=request, policy=None):
        with recompute_attention(model):
            return learner.update(model, opt, value, 5, policy or state['policy'], request=q,
                                  target=state['target'], full=True)
    held = next(k for k,d in state['items'].items() if d['side'] != 'train')
    out['held_out_fit'] = rejected(lambda:run(source.prior.item(state, held)[0]), 'CONDITIONED_NONTRAIN')
    out['request'] = rejected(lambda:run(q=NativeRequest(dict(legal(2), operation='respond_to_source_question'))),
                              'FULL_HISTORY_REQUEST')
    for field, code in (('native','CONDITIONED_NATIVE_PIN'), ('label','CONDITIONED_LABEL_PIN'),
                        ('binding','CONDITIONED_SOURCE_BINDING')):
        bad = deepcopy(item)
        if field == 'native': next(iter(bad['answers'][0].values())).add_(1)
        elif field == 'label': bad['codes'][0,0] = (bad['codes'][0,0]+1)%4
        else: bad['provenance']['observed'] += 1
        out[field] = rejected(lambda:run(bad), code)
    out['physical_memory'] = rejected(lambda:run(policy=dict(state['policy'], working_bytes=1)),
                                      'GRADIENT_STORAGE_MEMORY')
    need(out['physical_memory'] in PHYSICAL_REFUSALS, 'RECOMPUTE_PHYSICAL_ERROR_CONTRACT')
    need('stream' not in vars(model) and all(p.grad is None for p in model.parameters()), 'RECOMPUTE_REFUSAL_STATE')
    with recompute_attention(model):
        stream = model.stream(request)
        hidden, cache = stream.step(item['context'][:1], context='controls', offset=0, detach_history=False)
        def next_step(**kwargs):
            args = dict(context='controls', offset=1, state=cache, detach_history=False); args.update(kwargs)
            return stream.step(item['context'][1:2], **args)
        out['context'] = rejected(lambda:next_step(context='wrong'), 'PREFIX_STREAM_WRONG_CONTEXT')
        out['offset'] = rejected(lambda:next_step(offset=2), 'PREFIX_STREAM_OFFSET')
        out['detach'] = rejected(lambda:next_step(detach_history=True), 'PREFIX_STREAM_OFFSET')
        with torch.no_grad(): cache.layers[0][0].key.add_(1)
        out['cache_mutation'] = rejected(next_step, 'PREFIX_STREAM_CACHE_CHANGED')
        del hidden, cache, stream
    need(origin == fingerprint((model.state_dict(), opt.state_dict())), 'RECOMPUTE_CONTROL_CHANGED_WEIGHTS')
    gc.collect(); out['no_update_and_policy_restored'] = True
    return out


def long_item(root, document):
    reports = []
    for name,pin in (CONSUMER, CONSUMER_COLD):
        path = root/name/'report.json'; need(file_sha(path) == pin, 'RECOMPUTE_CONSUMER_PIN')
        reports.append(json.loads(path.read_bytes()))
    first, cold = reports
    need(first['complete'] and cold['complete'] and cold['previous_pin'] == CONSUMER[1]
         and first['rows'] == cold['rows'] and first['binding'] == cold['binding'], 'RECOMPUTE_CONSUMER_COLD')
    name = f'document-{document}.pt'; values = []
    for (directory,_),report in zip((CONSUMER, CONSUMER_COLD), reports):
        path = root/directory/name; need(file_sha(path) == report['files'][name], 'RECOMPUTE_ITEM_PIN')
        values.append(torch.load(path, map_location='cpu', weights_only=True))
    need(fingerprint(values[0]) == fingerprint(values[1]), 'RECOMPUTE_ITEM_COLD')
    value = values[0]; item = value['item']; request = NativeRequest(value['descriptor'])
    learner.check(item)
    need(request.pin == item['provenance']['request'] and torch.equal(request.features, value['features']),
         'RECOMPUTE_ITEM_REQUEST')
    return item, request


def memory_probe(state, model, opt, item, request, output, mode):
    """Whole-document forward/backward, with no optimizer mutation or truncated success."""
    origin = fingerprint((model.state_dict(), opt.state_dict())); calls = [0]
    storage = SavedStorage(state['policy']['working_bytes'], 'cpu'); progress = []
    def hook(module, args, value):
        calls[0] += 1
        if calls[0] % 128 == 0:
            row = dict(projected_atoms=calls[0], saved=storage.report(),
                       rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            progress.append(row); print(json.dumps(row), flush=True)
    handle = model.forecaster.predictor.projectors[model.roster[0]].register_forward_hook(hook)
    phase = {}; result = {}; start = time.perf_counter()
    def attempt():
        nonlocal phase
        with storage.hooks():
            total, end, records = learner.forward(model, item, 5, state['policy'], request, state['target'])
            phase['forward'] = time.perf_counter()-start; tick = time.perf_counter()
            total.backward(); phase['backward'] = time.perf_counter()-tick
            gradients = cpu({n:p.grad for n,p in model.named_parameters()})
            need(all(v is not None and bool(v.isfinite().all()) for v in gradients.values()), 'RECOMPUTE_LONG_GRADIENTS')
            persist(output/'full-gradient.pt', dict(records=records, gradients=gradients,
                                                   loss=float(total.detach()), end=float(end.detach())))
            return dict(status='complete_forward_backward', decisions=len(records), loss=float(total.detach()),
                        gradient_pin=fingerprint(gradients), records_pin=fingerprint(records))
    try:
        with recompute_attention(model) if mode == 'recomputed' else nullcontext(): result = attempt()
    except (ValueError, RuntimeError) as exc:
        code = str(exc)
        need(code in PHYSICAL_REFUSALS, 'RECOMPUTE_UNEXPECTED_FAILURE:'+code)
        result = dict(status='physical_refusal', code=code, decisions_completed=None)
    finally:
        handle.remove(); opt.zero_grad(set_to_none=True)
    gc.collect()
    need(storage.current == 0 and origin == fingerprint((model.state_dict(), opt.state_dict())), 'RECOMPUTE_LONG_ATOMICITY')
    return dict(**result, projected_atoms=calls[0], total_atoms=len(item['context'])+len(item['answers']),
                storage=storage.report(), phases=phase, seconds=time.perf_counter()-start, progress=progress,
                optimizer_updates=0, no_truncation=True, original_state_unchanged=True)


def run(a):
    a.output.mkdir(exist_ok=False); began = tick = time.perf_counter(); phases = {}
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    state = source.load(a.root); base = Path(__file__).resolve().parents[1]
    code = dict(state['code'], **{n:file_sha(base/n) for n in NEW}); previous = None
    if a.previous:
        need(file_sha(a.previous/'report.json') == a.previous_pin, 'RECOMPUTE_PREVIOUS_PIN')
        previous = json.loads((a.previous/'report.json').read_bytes())
        need(previous['code'] == code and previous['architecture'] == a.architecture
             and previous['mode'] == a.mode, 'RECOMPUTE_PREVIOUS_BINDING')
    reference = None
    if a.reference:
        need(file_sha(a.reference/'report.json') == a.reference_pin, 'RECOMPUTE_REFERENCE_PIN')
        reference = json.loads((a.reference/'report.json').read_bytes())
        need(reference['architecture'] == a.architecture and reference['mode'] == 'reference', 'RECOMPUTE_REFERENCE_BINDING')
    model, opt = source.prior.make(state, a.architecture); model.train()
    persist(a.output/'initial.pt', dict(model=model.state_dict(), optimizer=opt.state_dict()))
    initial = fingerprint((model.state_dict(), opt.state_dict()))
    phases['authentication'] = time.perf_counter()-tick; tick = time.perf_counter(); rows = []; negative = {}
    if a.long_document is not None:
        item, request = long_item(a.root, a.long_document)
        phases['consumer_readback'] = time.perf_counter()-tick; tick = time.perf_counter()
        long = memory_probe(state, model, opt, item, request, a.output, a.mode)
        phases['complete_document_memory_probe'] = time.perf_counter()-tick
    else:
        long = None
        for cursor, doc in enumerate(state['train']):
            item, request = source.prior.item(state, doc)
            with recompute_attention(model) if a.mode == 'recomputed' else nullcontext() as streams:
                row, gradients, credit = learner.update(model, opt, item, 5+cursor, state['policy'],
                                                        request=request, target=state['target'], full=True)
                recomputation = [] if streams is None else [dict(forward=s.forward_folds, calls=s.fold_calls) for s in streams]
            payload = dict(record=row, gradients=gradients, early=credit, model=cpu(model.state_dict()),
                           optimizer=cpu(opt.state_dict()))
            name = f'update-{6+cursor}.pt'; persist(a.output/name, payload)
            reread = torch.load(a.output/name, map_location='cpu', weights_only=True); exact(payload, reread)
            for directory,report in ((a.reference,reference), (a.previous,previous)):
                if directory is not None:
                    need(file_sha(directory/name) == report['files'][name], 'RECOMPUTE_COMPARISON_PIN')
                    exact(payload, torch.load(directory/name, map_location='cpu', weights_only=True))
            rows.append(dict(document=doc, decisions=row['decisions'], storage=row['storage'], phases=row['phases'],
                             numerical_pin=fingerprint(numerical(payload)), recomputation=recomputation))
            print(json.dumps(dict(architecture=a.architecture, mode=a.mode, **rows[-1])), flush=True)
            del payload, reread, gradients, credit
        phases['updates_and_readback'] = time.perf_counter()-tick; tick = time.perf_counter()
        if a.mode == 'recomputed' and a.architecture == 'shared': negative = controls(state)
        phases['negative_controls'] = time.perf_counter()-tick
    tick = time.perf_counter(); state['fresh']()
    need(all(file_sha(base/n) == pin for n,pin in code.items()), 'RECOMPUTE_CODE_CHANGED')
    need(not torch.cuda.is_initialized() and 'stream' not in vars(model), 'RECOMPUTE_RUNTIME_SCOPE')
    if previous: need(negative == previous['controls'], 'RECOMPUTE_COLD_CONTROLS')
    files = {p.name:file_sha(p) for p in sorted(a.output.iterdir()) if p.is_file()}
    phases['final_authentication'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
    need(.9 <= sum(phases.values())/seconds <= 1.001, 'RECOMPUTE_PHASE_ATTRIBUTION')
    report = dict(complete=True, architecture=a.architecture, mode=a.mode, source=source.prior.PARENT,
        previous_pin=a.previous_pin, reference_pin=a.reference_pin, code=code, files=files, initial=initial,
        rows=rows, controls=negative, long=long, phases=phases, seconds=seconds,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        torch_version=torch.__version__, cpu_threads=1, gpu_used=False, native_encoder_calls=0,
        acceptance='author-checked', promoted=False, capability_demonstrated=False,
        scope='CPU arithmetic and memory qualification; no GPU or capability qualification')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json', report), seconds=seconds)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, default=source.ROOT)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--architecture', choices=source.prior.ARCHITECTURES, required=True)
    p.add_argument('--mode', choices=('reference','recomputed'), required=True)
    p.add_argument('--reference', type=Path); p.add_argument('--reference-pin')
    p.add_argument('--previous', type=Path); p.add_argument('--previous-pin'); p.add_argument('--long-document', type=int)
    a = p.parse_args()
    try: run(a)
    except BaseException as exc:
        if a.output.exists(): write_committed(a.output/'failure.json', dict(error=type(exc).__name__, detail=str(exc)))
        raise
