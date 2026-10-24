"""Train the native ordered predictor with canonical cached labels and physical replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch

from audit import legal_breadth_schedule_readback as breadth
from audit import legal_predictor_training as small
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from lenscommon import take_lease, release_lease
from model import legal_cached_documents as documents, legal_prediction_metrics as metrics
from model.native_population_training import cpu, fingerprint
from senses.contract import canonical, digest
from text.constellation_corpus import file_sha, need

POLICY = dict(small.POLICY, tile=32, epochs=1, checkpoint_interval=1024,
              scope='canonical_cached_label_document_prediction', target_runtime='singleton_fp64_v2')
LONG = ('legal-long-document-training-qualification-v1-20261023',
        '0916ae94ced1640727dbf1ad2193d212bd646f5007210102f7bd6e83614b62ed')


def code():
    root = Path(__file__).resolve().parents[1]
    return dict(breadth.code(), **{n:file_sha(root/n) for n in ['model/legal_cached_documents.py',
        'model/legal_prediction_metrics.py', 'audit/legal_cached_predictor_training.py']})


def tensorize(value):
    if isinstance(value, np.ndarray): return torch.from_numpy(value.copy())
    if isinstance(value, dict): return {k:tensorize(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return type(value)(tensorize(v) for v in value)
    return value


def pinned(root, pin):
    return small.qualification.training.pinned(root/'report.json', pin)


def physical(root, report):
    for name, pin in report['files'].items(): need(file_sha(root/name) == pin, 'CACHED_PREDICTOR_PHYSICAL_PIN')


def load_schedule(a, data, cache, meta):
    q, c = pinned(a.schedule, a.schedule_pin), pinned(a.schedule_cold, a.schedule_cold_pin)
    need(q['complete'] and c['complete'] and q['mode'] == 'publish' and c['mode'] == 'cold'
         and c['previous_pin'] == a.schedule_pin and q['binding'] == c['binding']
         and q['summary'] == c['summary'] and q['binding']['code'] == breadth.code()
         and q['binding']['cache'] == cache.binding and q['binding']['metadata'] == meta.binding,
         'CACHED_PREDICTOR_SCHEDULE_AUTHORITY')
    physical(a.schedule, q); physical(a.schedule_cold, c)
    for root, pin in [(a.schedule, a.schedule_pin), (a.schedule_cold, a.schedule_cold_pin)]:
        need((root/'COMMITTED').read_text().strip() == pin, 'CACHED_PREDICTOR_SCHEDULE_COMMIT')
    value = np.load(a.schedule/'schedule.npy', allow_pickle=False)
    need(len(value) == q['summary']['documents'] == data.report['support']['train']['documents'],
         'CACHED_PREDICTOR_SCHEDULE_COVERAGE')
    return value, q


def selection(a, data, meta, schedule):
    exposed = small.qualification.selection(data, a.root)['documents']
    if a.scope == 'qualify':
        report = pinned(a.root/LONG[0], LONG[1]); physical(a.root/LONG[0], report)
        plan = small.qualification.training.pinned(a.root/LONG[0]/'selection.json', report['binding']['selection'])
        train = [r['document'] for r in plan['documents']]
        return dict(train_documents=train, evaluations=[dict(r, exposure='existing_development_probe') for r in exposed],
                    scope='five_complete_TRAIN_document_execution_proof', prior_long_proof=LONG[1])
    excluded = {r['document'] for r in exposed}; chosen = []
    excluded_cases = {int(data.components[i]) for i in excluded}; chosen_cases = set()
    for side in [1, 2]:
        for source in sorted({r['source_path'] for r in meta.rows}):
            candidates = [i for i,r in enumerate(meta.rows) if int(data.doc_sides[i]) == side
                          and r['source_path'] == source and i not in excluded
                          and int(data.components[i]) not in excluded_cases]
            candidates.sort(key=lambda i: hashlib.sha256(f'legal-predictor-evaluation-20261023:{i}'.encode()).digest())
            unique = []
            for number in candidates:
                component = int(data.components[number])
                if component in chosen_cases: continue
                unique.append(number); chosen_cases.add(component)
                if len(unique) == 32: break
            need(len(unique) == 32, 'CACHED_PREDICTOR_EVALUATION_STRATUM')
            for number in unique:
                start, stop = map(int, data.documents[number])
                chosen.append(dict(document=number, side=['train','development','query'][side], start=start, stop=stop,
                                   exposure='new_predictor_probe_existing_source_split', case_component=int(data.components[number]),
                                   reasons=['hash_selected_per_source_and_side_distinct_case_component']))
    need(len({r['document'] for r in chosen}) == len(chosen), 'CACHED_PREDICTOR_EVALUATION_DUPLICATE')
    key = 'full' if a.arm == 'breadth' else 'repeated'
    return dict(train_documents=schedule[key].tolist(), evaluations=chosen, scope='complete_TRAIN_one_pass',
                shared_evaluation_selection=True, score_blind=True, untouched_external_test=False)


def evaluate(model, target, plan, data, reader, cache, continuation, stats, meta, output):
    output.mkdir(); results = []; native_seconds = metric_seconds = 0.
    for row in plan['evaluations']:
        item, timing = documents.read_document(data, reader, cache, row['document'], side=row['side'], fit=False)
        native_seconds += sum(v for k,v in timing.items() if k.endswith('_seconds'))
        need((item['source_start'], item['source_stop']) == (row['start'], row['stop']), 'CACHED_PREDICTOR_EVALUATION_BOUNDARY')
        tick = time.perf_counter(); value = metrics.document(model, target, item, cache, continuation, stats, POLICY['tile'])
        value.update(source_collection=meta.rows[row['document']]['source_path'], exposure=row['exposure'])
        payload = tensorize(value); path = output/f"document-{row['document']}.pt"; torch.save(payload, path)
        loaded = torch.load(path, map_location='cpu', weights_only=True)
        need(fingerprint(loaded) == fingerprint(payload), 'CACHED_PREDICTOR_EVALUATION_READBACK')
        metric_seconds += time.perf_counter()-tick; results.append(value)
    summary = metrics.summary(results, data.roster)
    payload_pin = fingerprint([tensorize(r) for r in results])
    write_committed(output/'summary.json', dict(summary=summary, payload_state=payload_pin,
        full_train_baselines=True, native_distance_order=['model','copy_current','train_mean'],
        probabilities='teacher_forced_earlier_code_levels_only', scientific_admission=False))
    return dict(summary=summary, payload_state=payload_pin), dict(native_read=native_seconds, metrics=metric_seconds)


def qualification_gate(a, cache, schedule_report):
    if a.scope != 'full': return None
    need(a.full_qualification and a.full_cold, 'CACHED_PREDICTOR_FULL_REQUIRES_PROOF')
    q, c = pinned(a.full_qualification, a.full_qualification_pin), pinned(a.full_cold, a.full_cold_pin)
    need(q['complete'] and c['complete'] and q['scope'] == c['scope'] == 'qualify'
         and c['previous_pin'] == a.full_qualification_pin and c['binding'] == q['binding']
         and q['binding']['code'] == code() and q['binding']['cache'] == cache.binding
         and q['binding']['schedule'] == a.schedule_pin and q['updates'] == 5 and q['target_presentations'] == 4104
         and q['ledger_head'] == c['ledger_head'] and q['evaluations'] == c['evaluations']
         and q['all_updates_equal_prior_native_proof'], 'CACHED_PREDICTOR_FULL_PROOF_GATE')
    physical(a.full_qualification, q); physical(a.full_cold, c)
    return dict(qualification=a.full_qualification_pin, cold=a.full_cold_pin)


def run(a):
    began = tick = time.perf_counter(); cpu_start = time.process_time(); a.output.mkdir(parents=True, exist_ok=False)
    for name, pin in code().items():
        dest = a.output/'source'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes())
        need(file_sha(dest) == pin, 'CACHED_PREDICTOR_SOURCE_COPY')
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'], text=True)
    need(not active.strip() and torch.cuda.is_available(), 'CACHED_PREDICTOR_GPU_BUSY')
    take_lease('29', 'cached-native-document-predictor')
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    data, continuation, reader, labels, cache, meta = breadth.prepare(a)
    schedule, schedule_report = load_schedule(a, data, cache, meta)
    admission = qualification_gate(a, cache, schedule_report)
    target = labels.model
    # The cache binds both the physical frozen checkpoint and its exact fp64 label runtime.
    target_ref = cache.report['binding']['target_checkpoint']; target_before = fingerprint(target.state_dict())
    stats_path = a.root/small.qualification.initialization.STATS[0]
    stats_report = pinned(stats_path, small.qualification.initialization.STATS[1])
    need(file_sha(stats_path/'statistics.pt') == stats_report['files']['statistics.pt'], 'CACHED_PREDICTOR_STATISTICS_PIN')
    stats = torch.load(stats_path/'statistics.pt', map_location='cpu', weights_only=True)
    plan = selection(a, data, meta, schedule); plan_pin = write_committed(a.output/'selection.json', plan)
    binding = dict(code=code(), policy=POLICY, scope=a.scope, arm=a.arm, source=data.binding, cache=cache.binding,
                   metadata=meta.binding, schedule=a.schedule_pin, schedule_cold=a.schedule_cold_pin,
                   selection=plan_pin, full_authority=admission, target=target_ref,
                   torch_version=str(torch.__version__), device=torch.cuda.get_device_name(0))
    previous = pinned(a.previous, a.previous_pin) if a.previous else None
    if previous: need(previous['complete'] and previous['binding'] == binding, 'CACHED_PREDICTOR_COLD_BINDING')
    model, optimizer = small.make(target, target_ref['pin']); chain = digest(binding); step = rows = 0
    phases = dict(authentication_and_plan=time.perf_counter()-tick); tick = time.perf_counter()
    evaluations = {}; profiles = {}
    evaluations['initial'], profiles['initial_evaluation'] = evaluate(model, target, plan, data, reader, cache, continuation, stats, meta, a.output/'initial-evaluation')
    checkpoints = {'0':small.steps.checkpoint(a.output/'checkpoint-0.pt', model, optimizer, binding, 0, 0, chain)}
    if previous:
        need(checkpoints['0']['state'] == previous['checkpoints']['0']['state'], 'CACHED_PREDICTOR_COLD_INITIAL_STATE')
        physical(a.previous, previous)
    phases['initial_evaluation_checkpoint'] = time.perf_counter()-tick; tick = time.perf_counter()
    old = (a.previous/'steps.jsonl').open() if previous else None
    prior = (a.root/LONG[0]/'steps.jsonl').open() if a.scope == 'qualify' else None
    train = plan['train_documents']; native_windows = {n:0 for n in data.roster}
    times = dict(native_read=0., target_join=0., fingerprints=0., updates=0., checkpoints=0.)
    with (a.output/'steps.jsonl').open('x') as ledger:
        for cursor, number in enumerate(train):
            item, read = documents.read_document(data, reader, cache, number, side='train', fit=True)
            times['native_read'] += read['native_read_seconds']; times['target_join'] += read['target_join_seconds']
            times['fingerprints'] += read['native_and_label_fingerprints_seconds']
            for n in native_windows: native_windows[n] += read['native_windows'][n]
            if prior:
                proof = torch.load(a.root/LONG[0]/f'labels-{number}.pt', map_location='cpu', weights_only=True)
                need(item['native_pin'] == proof['native_pin'] and item['uids'] == proof['uids']
                     and torch.equal(item['codes'], proof['codes'][1:]), 'CACHED_PREDICTOR_PRIOR_NATIVE_LABELS')
            torch.cuda.synchronize(); start = time.perf_counter()
            update = small.steps.update(model, optimizer, item, step, POLICY); torch.cuda.synchronize()
            times['updates'] += time.perf_counter()-start
            if prior: need(update == json.loads(prior.readline())['update'], 'CACHED_PREDICTOR_PRIOR_UPDATE_EXACT')
            step += 1; rows += update['targets']
            record = dict(parent=chain, cursor=cursor, source_start=item['source_start'], source_stop=item['source_stop'],
                          uid_order=fingerprint(torch.tensor(item['uids'], dtype=torch.int64)), update=update)
            chain = digest(record); record['hash'] = chain
            if old: need(record == json.loads(old.readline()), 'CACHED_PREDICTOR_COLD_UPDATE')
            ledger.write(canonical(record).decode()+'\n'); ledger.flush(); os.fsync(ledger.fileno())
            if a.scope == 'qualify' or step == 1 or step % POLICY['checkpoint_interval'] == 0 or step == len(train):
                start = time.perf_counter(); path = a.output/f'checkpoint-{step}.pt'
                checkpoints[str(step)] = small.steps.checkpoint(path, model, optimizer, binding, step, rows, chain)
                if previous:
                    expected = previous['checkpoints'][str(step)]
                    need(checkpoints[str(step)]['state'] == expected['state'], 'CACHED_PREDICTOR_COLD_CHECKPOINT')
                    model, optimizer = small.make(target, target_ref['pin'])
                    saved = small.steps.load(a.previous/path.name, expected['pin'], model, optimizer, binding, step, rows, POLICY)
                    need(saved['ledger'] == chain, 'CACHED_PREDICTOR_COLD_LEDGER')
                times['checkpoints'] += time.perf_counter()-start
            if step % 128 == 0 or a.scope == 'qualify':
                print(json.dumps(dict(step=step, total=len(train), targets=rows, last_document=number, times=times)), flush=True)
    for handle in [old, prior]:
        if handle: need(not handle.read(), 'CACHED_PREDICTOR_EXTRA_LEDGER_STEP'); handle.close()
    if a.scope == 'full':
        need(step == schedule_report['summary']['documents'] and rows == schedule_report['summary']['targets']
             and native_windows == schedule_report['summary']['native_windows'], 'CACHED_PREDICTOR_FULL_COVERAGE')
    phases['complete_document_training'] = time.perf_counter()-tick; tick = time.perf_counter()
    evaluations['final'], profiles['final_evaluation'] = evaluate(model, target, plan, data, reader, cache, continuation, stats, meta, a.output/'final-evaluation')
    checkpoint = a.output/f'checkpoint-{step}.pt'; cp_pin = checkpoints[str(step)]['pin']; before = fingerprint(model.state_dict())
    held = int(np.flatnonzero(data.doc_sides == 2)[0])
    controls = dict(held_fit=rejected(lambda: cache.document(held, side='query', fit=True)),
        changed_label=rejected(lambda: small.steps.update(model, optimizer, dict(item, label_pin='0'*64), step, POLICY)),
        changed_checkpoint=rejected(lambda: small.steps.load(checkpoint, '0'*64, model, optimizer, binding, step, rows, POLICY)),
        changed_cursor=rejected(lambda: small.steps.load(checkpoint, cp_pin, model, optimizer, binding, step+1, rows, POLICY)))
    need(fingerprint(model.state_dict()) == before and fingerprint(target.state_dict()) == target_before,
         'CACHED_PREDICTOR_CONTROL_CHANGED_STATE')
    if previous: need(evaluations == previous['evaluations'] and controls == previous['controls'], 'CACHED_PREDICTOR_COLD_EVALUATION')
    cache.close(); reader.close(); continuation.fresh(); meta.fresh()
    phases['final_evaluation_controls'] = time.perf_counter()-tick; tick = time.perf_counter()
    files = {str(f.relative_to(a.output)):file_sha(f) for f in a.output.rglob('*') if f.is_file()}
    phases['hashes'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
    need(sum(phases.values())/seconds >= .9, 'CACHED_PREDICTOR_PHASE_ATTRIBUTION')
    result = dict(complete=True, mode='cold' if previous else 'train', scope=a.scope, arm=a.arm, binding=binding,
        previous_pin=a.previous_pin, files=files, checkpoints=checkpoints, updates=step, target_presentations=rows,
        native_windows=native_windows, ledger_head=chain, evaluations=evaluations, controls=controls,
        all_updates_equal_prior_native_proof=a.scope == 'qualify', update_phases=times, evaluation_phases=profiles,
        reader=reader.stats, phases_seconds=phases, seconds=seconds, cpu_seconds=time.process_time()-cpu_start,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        target_state=target_before, model_promotion=False, scientific_admission=False, acceptance='author-checked', number_status='measured')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json', result), updates=step, seconds=seconds)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output', type=Path, required=True); p.add_argument('--scope', choices=['qualify','full'], required=True)
    p.add_argument('--arm', choices=['breadth','repeated'], default='breadth')
    for name in ['qualification','cold','cache','cache-cold','schedule','schedule-cold']:
        p.add_argument('--'+name, type=Path, required=True); p.add_argument('--'+name+'-pin', required=True)
    for name in ['previous','full-qualification','full-cold']:
        p.add_argument('--'+name, type=Path); p.add_argument('--'+name+'-pin')
    a = p.parse_args()
    try: run(a)
    except BaseException as error:
        if a.output.exists(): write_committed(a.output/'failure.json', dict(error=repr(error), code=code()))
        raise
    finally: release_lease()
