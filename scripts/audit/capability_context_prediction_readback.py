"""Cold forward/gradient replay and independent source joins for newly arrived capability data."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'lens')]
import torch
from audit.capability_context_live_readback import ContextReader, read
from audit.capability_prediction_readback import load_view, arrays, source_check, slots, equal, numpy_labels, check_schedule
from audit.fsv_identity_loop import configure
from claims.capability_population import CapabilityPopulation, need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import take_lease, run_cli
from model.capability_objective import PredictedSurfaceDecoder, next_constellation_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'proofs', 'output'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); need(not a.output.exists(), 'GROWTH_PREDICTION_EXISTS')
    started = time.perf_counter(); config = read(a.config); root = Path(config['root'])
    proof = read(a.proofs); reader = ContextReader(config, proof)
    state = reader.run_context(root / 'experience.sqlite3')
    saved = read(root / 'snapshot-final.json')
    need(state['head'] == saved['head'] and state['active'] == saved['task_kernel']['checkpoint']
         and state['pending'] is None, 'GROWTH_PREDICTION_STATE')
    reader_view = SimpleNamespace(base=dict(tasks=reader.tasks), active=state['active'], checked_head=state['head'])
    populations, receipts = dict(reader.populations), dict(reader.receipts)
    delta = reader.delta_config
    original = read(Path(delta['source']) / 'config.json')
    for spec in read(Path(original['source']) / 'config.json')['populations']:
        pop = CapabilityPopulation(spec['population'], spec['population_pin'])
        populations[pop.pin] = pop
        receipts[pop.pin] = read(Path(proof[pop.family]['path']) / 'field-receipts.json')
    kernel = read(Path(state['active']['kernel']['path']) / 'manifest.json')
    destination = root / 'prediction'; report = read(destination / 'report.json')
    need(report['parameters_unchanged'] and report['optimizer_steps'] == 0, 'GROWTH_PREDICTION_SCOPE')
    files = {str(f.relative_to(destination)): sha(f.read_bytes()) for f in destination.rglob('*') if f.is_file()}
    gpu = subprocess.check_output(['nvidia-smi'], text=True)
    take_lease('29', 'capability-growth-prediction-cold'); configure()
    cases = []
    for i, record in enumerate(report['cases']):
        case = destination / str(i); metadata = read(case / 'view.json')
        for info in record['files'].values():
            need(sha((case / info['name']).read_bytes()) == info['sha256'], 'GROWTH_PREDICTION_FILE')
        view = load_view(case, metadata)
        raw, context, target, desc, partial = source_check(case, record, reader_view, populations, receipts, kernel, view)
        native = slots(context, view.specs, digest(desc['context']))
        target_native = slots(target, view.specs, 'loss-only-target')
        allowed = torch.ones(len(desc['regions']), dtype=torch.bool, device='cuda')
        def predict():
            return view(native, (view.start('capability:'+record['key']),), allowed=allowed,
                phase=0, event_ids=[metadata['event']], evidence_origin=['source_observation'], advance_phase=False)
        output = predict()
        equal({n: x.values for n, x in output['predicted_constellation'].slots.items()},
              arrays(case / 'prediction.npz'), 'GROWTH_PREDICTION_COLD_FORWARD')
        label = numpy_labels(view, target, record['meaning'])
        loss, meaning = next_constellation_loss(view, output, target_native)
        need(meaning == record['meaning'], 'GROWTH_PREDICTION_MEANING_LOSS')
        decoder = PredictedSurfaceDecoder(view.config, view.specs).to('cuda').eval()
        decoder.load_state_dict(torch.load(destination / 'decoder.pt', map_location='cuda', weights_only=True), strict=True)
        windows = check_schedule(case, raw, view.config)
        result, bridge = decoder.render_loss(raw, output['predicted_constellation'], backward=True, meaning_loss=loss)
        need(result == record['surface'], 'GROWTH_PREDICTION_SURFACE_LOSS')
        equal(bridge, arrays(case / 'bridge.npz'), 'GROWTH_PREDICTION_BRIDGE')
        equal({n: p.grad for n, p in view.named_parameters() if p.grad is not None},
              arrays(case / 'model-gradients.npz'), 'GROWTH_PREDICTION_MODEL_GRADIENT')
        equal({n: p.grad for n, p in decoder.named_parameters() if p.grad is not None},
              arrays(case / 'decoder-gradients.npz'), 'GROWTH_PREDICTION_DECODER_GRADIENT')
        for x in target_native.values(): x.values.neg_()
        with torch.no_grad(): again = predict()
        equal({n: x.values for n, x in again['predicted_constellation'].slots.items()},
              arrays(case / 'prediction.npz'), 'GROWTH_PREDICTION_TARGET_LEAK')
        cases.append(dict(case=i, family=record['family'], source_bytes=len(raw), windows=windows,
            single_instrument_neighbors=len(partial), mathematical_label=label,
            cold_forward_and_gradients_exact=True, source_and_region_bytes_exact=True,
            target_counterfactual_no_effect=True))
        print(json.dumps(cases[-1]), flush=True)
        del view, decoder, output, again, bridge, loss
    need(len(cases) == 1 and {c['family'] for c in cases} == {'coding'}, 'CONTEXT_PREDICTION_CASES')
    result = dict(schema='capability-context-prediction-readback-v1', files=files, cases=cases,
        head=state['head'], journal=sha((root / 'experience.sqlite3').read_bytes()), gpu_before=gpu,
        elapsed=time.perf_counter()-started, number_status='measured', acceptance='author-checked',
        scientific_admission=False, learned_capability=False, generated_surface_guard_complete=False,
        metric_mapping=['T3/spec27 meaning prediction', 'T5/spec27 continual consistency'])
    pin = write_committed(a.output, result)
    print(json.dumps(dict(pin=pin, cases=len(cases))), flush=True)


if __name__ == '__main__': run_cli(main)
