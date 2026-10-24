"""Actual live-kernel -> retained predictor -> predicted-slot decoder connectivity."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1] / 'lens')]
from audit.fsv_identity_loop import configure
from audit.repository_demonstration_memory import authenticate
from audit.capability_live_open import open_live
from claims.capability_population import need, sha
from claims.process_conditions import write_committed
from kernel.repository_live_population import LiveRepositoryPopulation
from lens.lenscommon import run_cli, take_lease
from model.capability_prediction import CapabilityPrediction
from model.capability_objective import PredictedSurfaceDecoder, next_constellation_loss
from model.reciprocal_constellation import ReciprocalConstellation
from model.stream_surface import schedule
from model.tool_growth import parameters
import numpy as np
import torch


def read(path): return json.loads(Path(path).read_bytes())


def save_arrays(path, arrays):
    np.savez_compressed(path, **{n: x.detach().cpu().numpy() for n, x in arrays.items()})
    return dict(name=path.name, sha256=sha(path.read_bytes()))


def refuse(code, operation):
    try: operation()
    except ValueError as exc:
        need(code in str(exc), 'PREDICTION_FSV_WRONG_REFUSAL:' + str(exc))
        return dict(expected=code, actual=str(exc))
    raise AssertionError('PREDICTION_FSV_ACCEPTED:' + code)


def run(session, cases, root):
    original = parameters(session.model)
    flags = {n: p.requires_grad for n, p in session.model.named_parameters()}
    session.model.requires_grad_(True)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(20261022)
        decoder = PredictedSurfaceDecoder(session.model.config, session.model.specs).to(session.device)
        reciprocal = ReciprocalConstellation(session.model.core).to(session.device)
    decoder.eval(); reciprocal.eval()
    initial_decoder = parameters(decoder)
    torch.save(decoder.state_dict(), root / 'decoder.pt')
    torch.save(reciprocal.state_dict(), root / 'reciprocal.pt')
    records = []
    try:
        for i, case in enumerate(cases):
            started = time.perf_counter(); dest = root / str(i); dest.mkdir()
            connection = CapabilityPrediction(session, case['key'], kernel=session.task_checkpoint['kernel']['pin'])
            controls = [refuse('CAPABILITY_PREDICTION_TARGET_BEFORE_FORWARD', connection.targets)]
            output = connection.forward()
            opened = connection.targets()
            need(len(opened) == 1, 'PREDICTION_FSV_TARGET_CARDINALITY')
            field, raw, target = opened[0]
            predicted = output['predicted_constellation']
            meaning_loss, meaning = next_constellation_loss(connection.view, output, target)
            session.model.zero_grad(set_to_none=True); decoder.zero_grad(set_to_none=True)
            surface, bridge = decoder.render_loss(raw, predicted, backward=True, meaning_loss=meaning_loss)
            gradients = {n: p.grad for n, p in session.model.named_parameters() if p.grad is not None}
            norms = {}
            for s in session.model.specs:
                for group in ('core.projectors.' + s.name + '.', 'query.' + s.name + '.', 'place.' + s.name + '.'):
                    norm = sum(float(g.double().square().sum()) for n, g in gradients.items() if n.startswith(group))
                    need(norm > 0, 'PREDICTION_FSV_DISCONNECTED:' + group); norms[group] = norm
                need(bool(torch.isfinite(bridge[s.name]).all()) and bool(bridge[s.name].abs().any()),
                     'PREDICTION_FSV_SURFACE_BRIDGE')
            need(all(bool(torch.isfinite(g).all()) for g in gradients.values()), 'PREDICTION_FSV_GRADIENT')
            files = {}
            for name, arrays in (
                ('context', {n: x.values for n, x in connection.native.items()}),
                ('target', {n: x.values for n, x in target.items()}),
                ('prediction', {n: x.values for n, x in predicted.slots.items()}),
                ('bridge', bridge), ('model-gradients', gradients),
                ('decoder-gradients', {n: p.grad for n, p in decoder.named_parameters() if p.grad is not None})):
                files[name] = save_arrays(dest / (name + '.npz'), arrays)
            write_committed(dest / 'schedule.json', list(schedule(raw, width=decoder.config.max_prefix)))
            write_committed(dest / 'definition.json', connection.definition)
            write_committed(dest / 'route.json', connection.route)
            write_committed(dest / 'source.json', dict(field=field, text=raw.decode('utf-8'),
                binding=session.capability_tasks[case['key']]['binding']))
            # Save the actual frozen identity and native region banks alongside
            # current parameter objects, so another process can replay this view.
            torch.save(connection.view.state_dict(), dest / 'view.pt')
            write_committed(dest / 'view.json', dict(core_type=type(connection.view.core).__name__,
                source_id=connection.view.core.source_id, world_source_id=connection.view.world_source_id,
                standard_id=connection.view.standard_id, world_id=connection.view.world_id,
                reference_rows=connection.view.reference_rows, core_reference_rows=connection.view.core.reference_rows,
                config=asdict(connection.view.config), specs=[asdict(s) for s in connection.view.specs],
                prediction_regions=predicted.regions, meaning=meaning, event=connection.event,
                model_gradient_norms=norms, surface=surface))
            mask_records = []
            # Same observed context, each instrument hidden in turn. The only
            # targets are frozen numbers; there are no hand-authored meaning IDs.
            for spec in session.model.specs:
                reciprocal.zero_grad(set_to_none=True)
                hidden = spec.name; truth = connection.native[hidden].values
                visible = {n: x for n, x in connection.native.items() if n != hidden}
                reconstruction = reciprocal(visible, hidden=hidden, windows=truth.shape[1])
                loss = reciprocal.loss(reconstruction['prediction'], truth); loss.backward()
                branches = {n: float(reciprocal.projectors[n].weight.grad.double().square().sum())
                            for n in visible}
                need(all(v > 0 for v in branches.values()) and reciprocal.projectors[hidden].weight.grad is None,
                     'PREDICTION_FSV_RECIPROCAL_MASK')
                mask_records.append(dict(hidden=hidden, visible=sorted(visible), loss=float(loss.detach()),
                    branch_gradients=branches, target_window_count=reconstruction['target_window_count']))
            old = connection.native
            connection.native = dict(old); connection.native.pop(session.model.specs[0].name)
            controls.append(refuse('CONSTELLATION_ROSTER', connection.forward))
            connection.native = old
            prior_head = connection.head; connection.head = '0' * 64
            controls.append(refuse('CAPABILITY_PREDICTION_STALE', connection.forward))
            connection.head = prior_head
            votes = {k: [n for n, t in connection.route['traces'].items() if k in {h for h, _ in t['hits']}]
                     for k in connection.route['targets']}
            partial = {k: v for k, v in votes.items() if len(v) == 1}
            record = dict(family=case['family'], key=case['key'], field=field, files=files,
                surface=surface, meaning=meaning, gradients=norms, reciprocal=mask_records,
                controls=controls, single_instrument_neighbors=partial,
                seconds=time.perf_counter() - started, native_prediction_is_decoder_condition=True)
            write_committed(dest / 'report.json', record); records.append(record)
            print(json.dumps(dict(case=i, family=case['family'], bytes=len(raw), regions=len(connection.regions),
                partial_neighbors=len(partial), seconds=record['seconds'])), flush=True)
            del connection, output, predicted, meaning_loss, target, bridge, gradients
        need(parameters(session.model) == original and parameters(decoder) == initial_decoder,
             'PREDICTION_FSV_UNREQUESTED_TRAINING')
        return dict(cases=records, parameters_unchanged=True, model=original, decoder=initial_decoder,
                    optimizer_steps=0, standard_id=session.model.standard_id)
    finally:
        for n, p in session.model.named_parameters(): p.requires_grad_(flags[n])
        session.model.zero_grad(set_to_none=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--census', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args(); need(not a.output.exists(), 'PREDICTION_FSV_EXISTS')
    config = read(a.source / 'config.json'); memory = read(Path(config['source']) / 'config.json')
    repository = Path(memory['source']); live = read(repository / 'config.json')
    recipes = authenticate(read(Path(live['source']) / 'config.json'))
    before = sha((a.source / 'experience.sqlite3').read_bytes())
    a.output.mkdir()
    producer = a.output / 'producer'; producer.mkdir()
    scripts = Path(__file__).resolve().parents[1]
    for name in ('audit/capability_prediction_fsv.py', 'model/capability_prediction.py',
                 'model/capability_objective.py'):
        (producer / Path(name).name).write_bytes((scripts / name).read_bytes())
    gpu = subprocess.check_output(['nvidia-smi'], text=True)
    take_lease('29', 'capability-prediction-connectivity'); configure()
    (a.output / 'gpu-before.txt').write_text(gpu)
    started = time.perf_counter()
    population = LiveRepositoryPopulation(Path(live['base']) / 'experience.sqlite3',
        live['base_pin'], repository / 'experience.sqlite3')
    session, sessions = open_live(recipes, a.source, population)
    restored = time.perf_counter(); head = session.head
    print(json.dumps(dict(restored=True, seconds=restored-started, head=head)), flush=True)
    try:
        result = run(session, read(a.census)['cases'], a.output)
        need(session.head == head and sha((a.source / 'experience.sqlite3').read_bytes()) == before,
             'PREDICTION_FSV_JOURNAL_CHANGED')
        result.update(schema='capability-prediction-connectivity-v1', source=str(a.source), journal=before,
            head=head, kernel=session.task_checkpoint, phases=dict(restore=restored-started,
            work=time.perf_counter()-restored), mathematical_supervision=True,
            scientific_admission=False, learned_capability=False, serving=False,
            generated_surface_guard_complete=False, acceptance='author-checked', number_status='measured')
        pin = write_committed(a.output / 'report.json', result)
        print(json.dumps(dict(report=pin, cases=len(result['cases']))), flush=True)
    except BaseException as exc:
        write_committed(a.output / 'failure.json', dict(error=str(exc), trace=traceback.format_exc()))
        raise
    finally:
        for s in sessions: s.close()


if __name__ == '__main__': run_cli(main)
