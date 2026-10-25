"""Real-source CPU mechanics for autonomous answer archival and later scoring."""
import argparse
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import resource
import shutil
import struct
import subprocess
import sys
import time

if '--cpu-proof' in sys.argv: os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import legal_answer_epoch_cuda as source
from claims.process_conditions import write_committed
from model.complete_answer_scores import score
from model.complete_answer_trace import collect, replay, PHYSICAL
from model.native_open_window_answer import answer_model
from model.native_population_training import cpu, fingerprint
from model.task_answer_training import extension
from text.constellation_corpus import file_sha, need

NEW = ('model/complete_answer_trace.py', 'model/complete_answer_scores.py', 'audit/complete_answer_trace.py')
CACHE = 64 << 10


def reject(fn, code):
    try: fn()
    except (ValueError, TypeError) as exc:
        need(code in str(exc), 'TRACE_PROOF_WRONG_REFUSAL:'+str(exc)); return str(exc)
    raise ValueError('TRACE_PROOF_NEGATIVE_ACCEPTED')


def independent(output):
    """Direct read of framed torch payloads, separate from replay()."""
    rows = []
    with (output/'events.bin').open('rb') as f:
        while header := f.read(8):
            need(len(header) == 8, 'TRACE_PROOF_HEADER')
            size = struct.unpack('>Q', header)[0]
            rows.append(torch.load(io.BytesIO(f.read(size)), map_location='cpu', weights_only=True))
    return rows


def reference_observe(model, target, prefix, request, proposal, seed, count):
    generator = model.rollout(prefix, request=request, target=target, proposal=proposal,
                              context='trace-proof', seed=seed)
    rows = []; refusal = None
    try:
        for _ in range(count):
            row = cpu(next(generator)); rows.append(row)
            if row['kind'] == 'end': break
    except ValueError as exc:
        need(str(exc) in PHYSICAL, 'TRACE_PROOF_UNEXPECTED_REFUSAL'); refusal = str(exc)
    finally: generator.close()
    return rows, refusal


def outcome(output, pin, item, roster, copy_frame):
    # The legacy scorer is imported only inside this explicitly CPU proof.
    from audit.legal_answer_sequence import metrics as legacy
    events = independent(output); actual = list(replay(output, pin))
    need(fingerprint(events) == fingerprint(actual), 'TRACE_PROOF_DIRECT_EVENTS')
    result = score(output, pin, future=item['answers'], roster=roster, copy_frame=copy_frame)
    if events:
        old = legacy(events, item['answers'], roster, copy_frame)
        need(all(result[n] == v for n,v in old.items()), 'TRACE_PROOF_METRIC_PARITY')
    generated = sum(e['kind'] == 'constellation' for e in events)
    need(result['generated_atoms'] == generated and result['missing'] == max(0, len(item['answers'])-generated)
         and result['extra'] == max(0, generated-len(item['answers'])), 'TRACE_PROOF_DENOMINATOR')
    return dict(metrics=result, event_pins=[fingerprint(e) for e in events])


def run(a):
    need(a.cpu_proof and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'TRACE_EXPLICIT_CPU_PROOF')
    a.output.mkdir(exist_ok=False); began = tick = time.perf_counter(); phases = {}; corpus = None
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    try:
        reports, code, fresh = source.authenticate(a.root)
        corpus, labels, continuation, bundle, inherited = source.prepare(a.root)
        need(all(code.get(n) == h for n,h in inherited.items()), 'TRACE_PROOF_SOURCE_CODE')
        code.update({n:file_sha(source.BASE/n) for n in NEW})
        target, target_pin = extension(bundle); frozen = fingerprint(target.state_dict())
        documents = sorted(v['document'] for v in bundle['items'] if v['side'] != 'train')
        doc = documents[0]; selected = corpus.plan._row(doc)
        donor = min(v['document'] for v in bundle['items'] if v['side'] == 'train'
                    and corpus.plan._row(v['document'])['component'] != selected['component'])
        prefix, request = corpus.public(doc); donor_prefix, donor_request = corpus.public(donor)
        need(len(prefix) == len(donor_prefix), 'TRACE_PROOF_DONOR_PREFIX')
        binding = dict(source=corpus.binding, code=code, checkpoint=source.PROOFS['initial'][1],
            document=doc, donor=donor, cache_bytes=CACHE, device='cpu', torch=str(torch.__version__),
            role='memory_pressure_mechanical_proof_not_quality', exposure='previously_exposed_group_holdout')
        binding = json.loads(json.dumps(binding))
        previous = source.read(a.previous/'report.json', a.previous_pin) if a.previous else None
        if previous: need(previous['complete'] and previous['binding'] == binding, 'TRACE_PROOF_PREVIOUS')
        phases['authentication'] = time.perf_counter()-tick; results = {}; controls = {}; proofs = []
        for architecture in source.ARCHITECTURES:
            tick = time.perf_counter(); directory = a.output/architecture; directory.mkdir()
            initial_path = a.root/source.PROOFS['initial'][0]/architecture/'update-5.pt'
            initial = torch.load(initial_path, map_location='cpu', weights_only=True)
            model = answer_model(target, bundle, architecture, target_pin=target_pin)
            model.load_state_dict(initial['model'], strict=True); model.eval()
            model.forecaster.predictor.cache_bytes = CACHE
            before = fingerprint(model.state_dict()); entries = []
            for mode, given, req in (('correct', prefix, request), ('reversed', prefix[::-1], request),
                                     ('train_donor', donor_prefix, donor_request)):
                for ordinal, proposal in enumerate(model.proposals):
                    output = directory/(mode+'-'+str(ordinal)); seed = 20261024+doc
                    pin = collect(model, given, request=req, target=target, proposal=proposal,
                        context='trace-proof', seed=seed, output=output, authority=binding)
                    # Private references are read only after generation has terminated.
                    item, _, _ = corpus.example(doc, fitting=False)
                    scored = outcome(output, pin, item, model.roster, prefix[-1])
                    old, refusal = reference_observe(model, target, given, req, proposal, seed, len(item['answers'])+1)
                    actual = independent(output)
                    need(fingerprint(old) == fingerprint(actual[:len(old)]), 'TRACE_PROOF_OLD_PREFIX_CHANGED')
                    terminal = json.loads((output/'report.json').read_bytes())
                    if refusal: need(terminal['refusal'] == refusal, 'TRACE_PROOF_REFUSAL_CHANGED')
                    entries.append(dict(mode=mode, proposal=proposal, path=str(output.relative_to(a.output)),
                        report=pin, result=scored, termination=terminal['status'], old_observed_events=len(old)))
                    proofs.append((output, pin)); del item, actual, old
            need(fingerprint(model.state_dict()) == before, 'TRACE_PROOF_MODEL_CHANGED')
            # Stop-head intervention is an explicit negative/control arm, never a learned result.
            probe = deepcopy(model)
            with torch.no_grad():
                for group in probe.ending.values():
                    for head in group.values(): head.weight.zero_(); head.bias.copy_(torch.tensor([-20., 20.]))
            output = directory/'forced-end-control'; proposal = probe.proposals[0]
            pin = collect(probe, prefix, request=request, target=target, proposal=proposal,
                context='trace-proof', seed=1, output=output, authority=dict(binding, intervention='forced_END'))
            ended = independent(output); need(len(ended) == 1 and ended[0]['kind'] == 'end', 'TRACE_PROOF_END_CONTROL')
            proofs.append((output,pin))
            kwargs = dict(request=request, target=target, proposal=proposal, context='trace-proof', seed=1)
            controls[architecture] = dict(
                target_length=reject(lambda:probe.rollout(prefix, **kwargs, target_length=1), 'target_length'),
                target_frames=reject(lambda:probe.rollout(prefix, **kwargs, target_frames=prefix), 'target_frames'))
            # Physical refusal before the first event must remain an empty failed-to-complete answer.
            probe.forecaster.predictor.cache_bytes = 6
            output = directory/'physical-control'
            pin = collect(probe, prefix, request=request, target=target, proposal=proposal,
                context='trace-proof', seed=1, output=output, authority=dict(binding, intervention='six_byte_cache'))
            refused = json.loads((output/'report.json').read_bytes())
            need(refused['status'] == 'physical_refusal' and not refused['model_ended'] and refused['events'] == 0,
                 'TRACE_PROOF_PHYSICAL_AS_END'); proofs.append((output,pin))
            controls[architecture]['physical'] = refused['refusal']
            controls[architecture]['forced_end'] = fingerprint(ended)
            results[architecture] = entries; phases[architecture] = time.perf_counter()-tick
            print(json.dumps(dict(architecture=architecture, trajectories=len(entries),
                ends=sum(x['termination']=='ended' for x in entries))), flush=True)
        tick = time.perf_counter(); original, pin = proofs[0]
        broken = a.output/'corrupted-control'; shutil.copytree(original, broken)
        with (broken/'events.bin').open('r+b') as f:
            first = f.read(1); need(first != b'', 'TRACE_PROOF_NO_EVENT'); f.seek(0); f.write(bytes([first[0]^1]))
        controls['corruption'] = reject(lambda:list(replay(broken,pin)), 'ANSWER_TRACE_BYTES')
        controls['wrong_report'] = reject(lambda:list(replay(original,'0'*64)), 'ANSWER_TRACE_REPORT')
        # Changing the private reference changes only the later denominator, never the saved trajectory.
        item, _, _ = corpus.example(doc, fitting=False); original_file = file_sha(original/'events.bin')
        shorter = score(original,pin,future=item['answers'][:1],roster=corpus.roster,copy_frame=prefix[-1])
        longer = score(original,pin,future=(*item['answers'],item['answers'][-1]),roster=corpus.roster,copy_frame=prefix[-1])
        need(shorter['generated_atoms'] == longer['generated_atoms'] and file_sha(original/'events.bin') == original_file,
             'TRACE_PROOF_REFERENCE_ALTERED_GENERATION')
        controls['private_reference'] = dict(short_expected=shorter['expected_atoms'],long_expected=longer['expected_atoms'],
            generated=shorter['generated_atoms'],events_unchanged=True)
        numerical = {arch:[{k:v for k,v in row.items() if k not in ('report','path')} for row in rows]
                     for arch,rows in results.items()}
        if previous:
            need(previous['numerical'] == numerical and previous['controls'] == controls, 'TRACE_PROOF_COLD_NUMERICAL')
            for arch, rows in results.items():
                for row, prior in zip(rows,previous['results'][arch],strict=True):
                    old = independent(a.previous/prior['path']); current = independent(a.output/row['path'])
                    need(fingerprint(old) == fingerprint(current), 'TRACE_PROOF_COLD_EVENTS')
        fresh(); corpus.close(); corpus.cache.close(); corpus.reader.close(); labels.fresh(); continuation.fresh()
        need(fingerprint(target.state_dict()) == frozen and not torch.cuda.is_initialized(), 'TRACE_PROOF_FROZEN_DEVICE')
        for n,h in code.items():
            need(file_sha(source.BASE/n) == h, 'TRACE_PROOF_CODE_CHANGED')
            dest = a.output/'source'/n; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes((source.BASE/n).read_bytes())
        phases['controls_and_seal'] = time.perf_counter()-tick; tick = time.perf_counter()
        files = {str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
        phases['hashes'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
        need(.9 <= sum(phases.values())/seconds <= 1.001, 'TRACE_PROOF_PHASES')
        report = dict(complete=True,binding=binding,code=code,results=results,numerical=numerical,controls=controls,files=files,
            previous_pin=a.previous_pin,cold_reproduced=previous is not None,phases=phases,seconds=seconds,
            optimizer_updates=0,frozen_encoder_calls=0,gpu_used=False,capability_demonstrated=False,
            architecture_winner=False,scientific_admission=False,acceptance='author-checked',
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        print(json.dumps(dict(complete=True,pin=write_committed(a.output/'report.json',report),seconds=seconds)),flush=True)
    finally:
        if corpus is not None and not corpus.closed: corpus.close(); corpus.cache.close(); corpus.reader.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,default=source.ROOT)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--cpu-proof',action='store_true')
    p.add_argument('--previous',type=Path); p.add_argument('--previous-pin'); a = p.parse_args()
    try: run(a)
    except BaseException as exc:
        if a.output.exists(): write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise
