"""Cold predictor/gradient replay plus independent source, union and byte readback.

The journal reader never calls the live memory producer. NumPy independently
checks vector-derived region labels; the separate CUDA process replays the
retained forward and full byte objective, including the predictor VJP.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1] / 'lens')]
import numpy as np
import torch
from torch import nn

from audit.capability_live_readback import LiveReader
from audit.fsv_identity_loop import configure
from claims.capability_population import CapabilityPopulation, need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import run_cli, take_lease
from model.capability_objective import PredictedSurfaceDecoder, next_constellation_loss
from model.constellation_world import ConstellationWorld
from model.contracts import Instrument, ModelConfig
from model.field_world import FieldEvidence
from model.identity_loop import IdentityLoopPredictor
from model.native import NativeInput
from model.native_world import NativeEvidence
from model.repository_relation_world import RepositoryEvidence


def read(path): return json.loads(Path(path).read_bytes())


def arrays(path):
    with np.load(path, allow_pickle=False) as z: return {n: z[n].copy() for n in z.files}


def equal(actual, expected, code):
    need(set(actual) == set(expected), code + ':roster')
    for name, value in actual.items():
        a = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
        b = expected[name]
        need(a.shape == b.shape and np.array_equal(a, b), code + ':' + name)


def load_view(root, metadata):
    config = ModelConfig(**metadata['config'])
    specs = tuple(Instrument(**s) for s in metadata['specs'])
    weights = torch.load(root / 'view.pt', map_location='cpu', weights_only=True)
    def bank(prefix, names):
        return {s.name: NativeInput(s.identity, tuple(names[s.name]),
            weights[prefix + f'values_{i}'], weights[prefix + f'padding_{i}'],
            weights[prefix + f'absent_{i}']) for i, s in enumerate(specs)}
    need(metadata['core_type'] == 'IdentityLoopPredictor', 'PREDICTION_READER_CORE')
    core = IdentityLoopPredictor(config, specs, bank('core.ref_', metadata['core_reference_rows']),
                                 source_id=metadata['source_id'])
    view = ConstellationWorld(core, bank('world_', metadata['reference_rows']),
                              world_source_id=metadata['world_source_id'])
    # Retain the actual legacy auxiliary heads too; never ignore unknown weights.
    for namespace, factory in (('_native', NativeEvidence), ('_fields', FieldEvidence),
                               ('_repository', RepositoryEvidence)):
        if any(n.startswith('query.' + namespace + '.') for n in weights):
            view.query[namespace] = nn.ModuleDict({s.name: factory(s.dims, config.width) for s in specs})
    view.load_state_dict(weights, strict=True)
    view.to('cuda').eval().requires_grad_(True)
    view.check_reference()
    need(view.standard_id == metadata['standard_id'] and view.world_id == metadata['world_id'],
         'PREDICTION_READER_BANK')
    return view


def slots(values, specs, row):
    return {s.name: NativeInput(s.identity, (row,), torch.from_numpy(values[s.name]).to('cuda'),
        torch.zeros(values[s.name].shape[:2], dtype=torch.bool, device='cuda'),
        torch.zeros(1, dtype=torch.bool, device='cuda')) for s in specs}


def check_reading(value, receipt):
    need(value.shape == (receipt['windows'], receipt['dimensions']) and value.dtype == np.float32
         and sha(value.astype('<f4').tobytes()) == receipt['sha256'], 'PREDICTION_READER_NATIVE')


def check_schedule(root, raw, config):
    windows = read(root / 'schedule.json'); cursor = 0; covered = []
    tokens = [256, *raw, 257]
    for w in windows:
        lo, (start, stop) = w['offset'], w['scored']
        need(start == cursor and 0 <= lo <= start < stop <= len(raw) + 1
             and len(w['prefix']) <= config.max_prefix, 'PREDICTION_READER_SCHEDULE')
        need(w['prefix'] == tokens[lo:stop] and w['labels'] ==
             [-100] * (start-lo) + tokens[start+1:stop+1], 'PREDICTION_READER_CAUSAL_BYTES')
        covered.extend(w['labels'][start-lo:]); cursor = stop
    need(covered == [*raw, 257], 'PREDICTION_READER_BYTE_COVERAGE')
    return len(windows)


def numpy_labels(view, target, recorded):
    scores = []
    for i, spec in enumerate(view.specs):
        t = target[spec.name][0].astype(np.float64)
        t /= np.linalg.norm(t, axis=-1, keepdims=True)
        values = getattr(view, f'world_values_{i}').detach().cpu().numpy()
        padding = getattr(view, f'world_padding_{i}').cpu().numpy()
        lane = []
        for x, p in zip(values, padding):
            v = x[~p].astype(np.float64); v /= np.linalg.norm(v, axis=-1, keepdims=True)
            matrix = np.einsum('id,jd->ij', t, v)
            lane.append((matrix.max(0).mean() + matrix.max(1).mean()) / 2)
        need(np.allclose(lane, recorded['assignment_scores'][spec.name], atol=2e-6, rtol=0),
             'PREDICTION_READER_VECTOR_LABEL_SCORES')
        scores.append(lane)
    label = int(np.mean(scores, axis=0).argmax())
    need(label == recorded['label'], 'PREDICTION_READER_VECTOR_LABEL')
    return label


def source_check(root, record, reader, populations, receipts, kernel, view):
    desc, route = read(root/'definition.json'), read(root/'route.json')
    tasks = reader.base['tasks']; task = tasks[record['key']]
    need(desc['head'] == reader_result_head(reader) and desc['kernel'] == route['kernel']
         == reader.active['kernel']['pin'] and desc['query'] == route['query'] == record['key'],
         'PREDICTION_READER_SOURCE_JOIN')
    context, target = arrays(root/'context.npz'), arrays(root/'target.npz')
    selection = task['selections']['context']
    need(desc['context']['selection'] == selection, 'PREDICTION_READER_CONTEXT_SELECTION')
    field_receipts = receipts[task['population']]
    for name, x in context.items():
        cursor = 0
        for f in selection['fields']:
            receipt = field_receipts[f['id']][name]; end = cursor + receipt['windows']
            check_reading(x[0, cursor:end], receipt); cursor = end
        need(cursor == x.shape[1], 'PREDICTION_READER_CONTEXT_COMPLETE')
    source = read(root/'source.json'); field = source['field']
    need(field == record['field'] and source['binding'] == task['binding']
         and field in [f['id'] for f in task['selections']['targets']['fields']],
         'PREDICTION_READER_TARGET_SELECTION')
    raw = populations[task['population']].fields[field]['text'].encode()
    need(raw == source['text'].encode() and sha(raw) == record['surface']['sha256'],
         'PREDICTION_READER_SURFACE_BYTES')
    for name, x in target.items(): check_reading(x[0], field_receipts[field][name])
    support = {k for trace in route['traces'].values() for k, _ in trace['hits']}
    need(support == set(route['targets']) and record['key'] not in support,
         'PREDICTION_READER_UNION')
    for name, trace in route['traces'].items():
        matches = [r for r in kernel['calibration'][name]['rows'] if r['query'] == record['key']]
        need(len(matches) == 1 and matches[0]['route'] == trace, 'PREDICTION_READER_ROUTING')
    expected = {}
    for other in sorted(support):
        t = tasks[other]
        need(t['split_group'] != task['split_group'] and other in kernel['ids']
             and route['targets'][other]['targets'] == t['selections']['targets'],
             'PREDICTION_READER_FUTURE')
        for f in t['selections']['targets']['fields']:
            key = digest(dict(binding=t['binding'], field=f['id']))
            expected.setdefault(key, dict(binding=t['binding'], field=f, tasks=[]))['tasks'].append(other)
    need(expected == desc['regions'] and all(tuple(sorted(expected)) == rows
         for rows in view.reference_rows.values()), 'PREDICTION_READER_REGION_UNION')
    for j, key in enumerate(sorted(expected)):
        region = expected[key]; t = tasks[region['tasks'][0]]
        for i, spec in enumerate(view.specs):
            v = getattr(view, f'world_values_{i}')[j].detach().cpu().numpy()
            p = getattr(view, f'world_padding_{i}')[j].cpu().numpy()
            check_reading(v[~p], receipts[t['population']][region['field']['id']][spec.name])
    partial = {k: [n for n, trace in route['traces'].items() if k in {x for x, _ in trace['hits']}]
               for k in support}
    partial = {k: names for k, names in partial.items() if len(names) == 1}
    need(partial == record['single_instrument_neighbors'], 'PREDICTION_READER_PARTIAL')
    return raw, context, target, desc, partial


def reader_result_head(reader):
    # run() returns the checked journal head; keep that receipt separate from snapshots.
    return reader.checked_head


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--proofs', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args(); need(not a.output.exists(), 'PREDICTION_READER_EXISTS')
    started = time.perf_counter(); report = read(a.root/'report.json')
    source = Path(report['source']); config = read(source/'config.json'); proofs = read(a.proofs)
    reader = LiveReader(config, proofs); state = reader.run(source/'experience.sqlite3')
    reader.checked_head = state['head']
    need(state['head'] == report['head'] and state['pending'] is None
         and state['active'] == report['kernel']
         and sha((source/'experience.sqlite3').read_bytes()) == report['journal'],
         'PREDICTION_READER_JOURNAL')
    need(report['optimizer_steps'] == 0 and report['parameters_unchanged']
         and all(report[k] is False for k in ('scientific_admission', 'learned_capability',
                 'serving', 'generated_surface_guard_complete')), 'PREDICTION_READER_SCOPE')
    populations, receipts = {}, {}
    for spec in read(Path(config['source'])/'config.json')['populations']:
        p = CapabilityPopulation(spec['population'], spec['population_pin']); populations[p.pin] = p
        receipts[p.pin] = read(Path(proofs[p.family]['path'])/'field-receipts.json')
    kernel = read(Path(state['active']['kernel']['path'])/'manifest.json')
    gpu = subprocess.check_output(['nvidia-smi'], text=True)
    take_lease('29', 'capability-prediction-cold-readback'); configure()
    cases = []; files = {}
    for path in sorted(a.root.rglob('*')):
        if path.is_file(): files[str(path.relative_to(a.root))] = sha(path.read_bytes())
    for i, record in enumerate(report['cases']):
        root = a.root/str(i); metadata = read(root/'view.json')
        for info in record['files'].values():
            need(sha((root/info['name']).read_bytes()) == info['sha256'], 'PREDICTION_READER_FILE')
        view = load_view(root, metadata)
        raw, context, target, desc, partial = source_check(root, record, reader, populations,
                                                         receipts, kernel, view)
        native = slots(context, view.specs, digest(desc['context']))
        target_native = slots(target, view.specs, 'loss-only-target')
        allowed = torch.ones(len(desc['regions']), dtype=torch.bool, device='cuda')
        def predict():
            return view(native, (view.start('capability:' + record['key']),), allowed=allowed,
                        phase=0, event_ids=[metadata['event']], evidence_origin=['source_observation'],
                        advance_phase=False)
        output = predict()
        equal({n: x.values for n, x in output['predicted_constellation'].slots.items()},
              arrays(root/'prediction.npz'), 'PREDICTION_READER_COLD_FORWARD')
        label = numpy_labels(view, target, record['meaning'])
        loss, meaning = next_constellation_loss(view, output, target_native)
        need(meaning == record['meaning'], 'PREDICTION_READER_MEANING_LOSS')
        decoder = PredictedSurfaceDecoder(view.config, view.specs).to('cuda').eval()
        decoder.load_state_dict(torch.load(a.root/'decoder.pt', map_location='cuda', weights_only=True), strict=True)
        windows = check_schedule(root, raw, view.config)
        result, bridge = decoder.render_loss(raw, output['predicted_constellation'], backward=True,
                                             meaning_loss=loss)
        need(result == record['surface'], 'PREDICTION_READER_SURFACE_LOSS')
        equal(bridge, arrays(root/'bridge.npz'), 'PREDICTION_READER_BRIDGE_GRADIENT')
        equal({n: p.grad for n, p in view.named_parameters() if p.grad is not None},
              arrays(root/'model-gradients.npz'), 'PREDICTION_READER_MODEL_GRADIENT')
        equal({n: p.grad for n, p in decoder.named_parameters() if p.grad is not None},
              arrays(root/'decoder-gradients.npz'), 'PREDICTION_READER_DECODER_GRADIENT')
        # Counterfactual target cannot enter the forward input, even after supervision.
        for x in target_native.values(): x.values.neg_()
        with torch.no_grad(): again = predict()
        equal({n: x.values for n, x in again['predicted_constellation'].slots.items()},
              arrays(root/'prediction.npz'), 'PREDICTION_READER_TARGET_LEAK')
        cases.append(dict(case=i, family=record['family'], source_bytes=len(raw), windows=windows,
            single_instrument_neighbors=len(partial), mathematical_label=label,
            cold_forward_and_gradients_exact=True, source_and_region_bytes_exact=True,
            target_counterfactual_no_effect=True))
        print(json.dumps(cases[-1]), flush=True)
        del view, decoder, output, again, bridge, loss
    need({c['family'] for c in cases} == {'coding', 'conversation', 'reasoning'}, 'PREDICTION_READER_FAMILIES')
    result = dict(schema='capability-prediction-readback-v1', source_report=files['report.json'],
        files=files, journal=report['journal'], head=state['head'], cases=cases, gpu_before=gpu,
        elapsed=time.perf_counter()-started, number_status='measured', acceptance='author-checked',
        scientific_admission=False, learned_capability=False, generated_surface_guard_complete=False,
        metric_mapping=['T3/spec27 meaning prediction', 'T5/spec27 continual consistency'])
    pin = write_committed(a.output, result)
    print(json.dumps(dict(readback=pin, cases=len(cases))), flush=True)


if __name__ == '__main__': run_cli(main)
