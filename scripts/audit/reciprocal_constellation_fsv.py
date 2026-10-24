"""Actual retained predictor branches receive reciprocal supervision from real native targets."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'lens'))
import numpy as np
import torch
from audit.fsv_identity_loop import configure
from audit.repository_demonstration_memory import authenticate
from audit.repository_live_bootstrap import open_current
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from kernel.repository_live_population import LiveRepositoryPopulation
from lens.lenscommon import run_cli, take_lease
from model.capability_memory import CapabilityMemoryLoop
from model.reciprocal_constellation import ReciprocalConstellation
from model.stream_surface import ConnectedSurface
from senses.contract import SenseError


def read(path): return json.loads(Path(path).read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); config = read(a.config); output = a.output
    need(not output.exists(), 'RECIPROCAL_FSV_EXISTS')
    root, parent = Path(config['root']), Path(config['source'])
    live = read(parent/'config.json')
    recipes = authenticate(read(Path(live['source'])/'config.json'))
    population = LiveRepositoryPopulation(Path(live['base'])/'experience.sqlite3', live['base_pin'],
                                          parent/'experience.sqlite3')
    gpu = subprocess.check_output(['nvidia-smi'], text=True)
    take_lease('29', 'reciprocal-constellation'); output.mkdir()
    (output/'gpu.txt').write_text(gpu)
    configure(); start = time.perf_counter()
    journal_before = sha((root/'experience.sqlite3').read_bytes())
    session, sessions = open_current(recipes, root, population, loop_type=CapabilityMemoryLoop)
    restored = time.perf_counter()
    print(json.dumps(dict(restored_events=session.journal.count, seconds=restored-start)), flush=True)
    try:
        snapshot = digest(session.snapshot())
        core = session.model.core
        freeze_flags = {n: p.requires_grad for n, p in core.named_parameters()}
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(20261022)
            model = ReciprocalConstellation(core).to(session.device)
        # Prior experiments deliberately froze these branches. This diagnostic
        # differentiates through them without updating any retained parameter.
        for parameter in model.projectors.parameters(): parameter.requires_grad_(True)
        before = {n: sha(p.detach().cpu().numpy().tobytes()) for n, p in core.named_parameters()}
        np.savez_compressed(output/'initial.npz', **{n: p.detach().cpu().numpy() for n, p in model.named_parameters()})
        cases, controls = [], []
        surface_root = root.parent/'capability-surfaces-v1-20261022'
        prior = read(surface_root/'report.json')
        # Read the exact previously covered field roster, not a newly cherry-picked set.
        selected = prior['cases']
        need(isinstance(selected, list) and len(selected) == 6, 'RECIPROCAL_FSV_FIELD_ROSTER')
        for i, entry in enumerate(selected):
            task = session.capability_tasks[entry['task']]
            source = ConnectedSurface(session, entry['task'], binding=task['binding'])
            _, native = source.field(entry['field'], core.config, session.device)
            arrays = {n: x.values.detach().cpu().numpy() for n, x in native.items()}
            np.savez_compressed(output/('native-'+str(i)+'.npz'), **arrays)
            for hidden in native:
                model.zero_grad(set_to_none=True); core.zero_grad(set_to_none=True)
                visible = {n: x for n, x in native.items() if n != hidden}
                target = native[hidden].values
                actual = model(visible, hidden=hidden, windows=target.shape[1])
                loss = model.loss(actual['prediction'], target); loss.backward()
                gradients = {n: p.grad.detach().cpu().numpy() for n, p in model.named_parameters() if p.grad is not None}
                norms = {}
                for name in native:
                    group = 'projectors.'+name+'.'
                    total = sum(float(np.square(v.astype(np.float64)).sum()) for n, v in gradients.items() if n.startswith(group))
                    need(total == 0 if name == hidden else total > 0, 'RECIPROCAL_FSV_CONNECTION:'+name)
                    norms[name] = total
                need(all(np.isfinite(v).all() for v in gradients.values()), 'RECIPROCAL_FSV_GRADIENT')
                with torch.no_grad():
                    repeat = model(visible, hidden=hidden, windows=target.shape[1])['prediction']
                    changed = model.loss(repeat, -target)
                    need(torch.equal(repeat, actual['prediction']) and abs(float(changed-loss)) > 1e-7,
                         'RECIPROCAL_FSV_TARGET_ISOLATION')
                prefix = str(i)+'-'+hidden
                np.savez_compressed(output/(prefix+'-gradients.npz'), **gradients)
                np.savez_compressed(output/(prefix+'-prediction.npz'), prediction=repeat.cpu().numpy())
                cases.append(dict(field=entry['field'], task=entry['task'], family=entry['family'],
                    native='native-'+str(i)+'.npz', hidden=hidden, visible=list(visible),
                    prefix=prefix, loss=float(loss.detach()), changed_target_loss=float(changed),
                    source_projector_gradients=norms, predicted_shape=list(repeat.shape)))
                try: model(native, hidden=hidden, windows=target.shape[1])
                except ValueError as exc: need('RECIPROCAL_MASK' in str(exc), 'RECIPROCAL_FSV_ERROR')
                else: raise AssertionError('RECIPROCAL_FSV_HIDDEN_VISIBLE')
                controls.append(dict(case=prefix, hidden_input_refused=True, target_only_changes_loss=True))
                missing = dict(visible); missing.pop(next(iter(missing)))
                try: model(missing, hidden=hidden, windows=target.shape[1])
                except ValueError as exc: need('RECIPROCAL_MASK' in str(exc), 'RECIPROCAL_FSV_ERROR')
                else: raise AssertionError('RECIPROCAL_FSV_MISSING_ACCEPTED')
                swapped = dict(visible); names = list(swapped)
                swapped[names[0]], swapped[names[1]] = swapped[names[1]], swapped[names[0]]
                try: model(swapped, hidden=hidden, windows=target.shape[1])
                except SenseError: pass
                else: raise AssertionError('RECIPROCAL_FSV_SWAPPED_ACCEPTED')
                controls[-1].update(missing_input_refused=True, swapped_instrument_refused=True)
                print(json.dumps(dict(case=prefix, loss=cases[-1]['loss'])), flush=True)
        need(before == {n: sha(p.detach().cpu().numpy().tobytes()) for n, p in core.named_parameters()},
             'RECIPROCAL_FSV_MODEL_CHANGED')
        for name, parameter in core.named_parameters():
            parameter.requires_grad_(freeze_flags[name]); parameter.grad = None
        need(freeze_flags == {n: p.requires_grad for n, p in core.named_parameters()},
             'RECIPROCAL_FSV_FREEZE_RESTORE')
        core.check_reference()
        need(digest(session.snapshot()) == snapshot and sha((root/'experience.sqlite3').read_bytes()) == journal_before,
             'RECIPROCAL_FSV_MEMORY_CHANGED')
        report = dict(cases=cases, controls=controls, core_standard=core.standard_id, journal=journal_before,
            model_updated=False, optimizer_steps=0, teacher_window_count=True,
            trainable_projectors_during_diagnostic=True, original_freeze_flags=freeze_flags,
            original_flags_restored=True,
            scientific_admission=False, generation_ready=False, learned_capability=False,
            phases=dict(restore=restored-start, checks=time.perf_counter()-restored),
            code={n: sha((Path(__file__).parents[1]/n).read_bytes()) for n in
                  ('model/reciprocal_constellation.py', 'audit/reciprocal_constellation_fsv.py')},
            files={p.name:dict(bytes=p.stat().st_size, sha256=sha(p.read_bytes())) for p in output.iterdir()},
            acceptance='author-checked', number_status='measured')
        pin = write_committed(output/'report.json', report)
        print(json.dumps(dict(report=pin, cases=len(cases))), flush=True)
    finally:
        if 'freeze_flags' in locals():
            for name, parameter in core.named_parameters():
                parameter.requires_grad_(freeze_flags[name]); parameter.grad = None
        for item in sessions: item.close()


if __name__ == '__main__': run_cli(main)
