"""Real journal targets -> complete causal bytes -> typed decoder loss and gradients."""
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch

from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from model.contracts import ModelConfig
from model.stream_surface import ConnectedSurface, StreamSurfaceDecoder, schedule, whole_surface_loss


def check_schedule(raw, items):
    tokens = [256, *raw, 257]
    recovered, end = [], 0
    for item in items:
        a, b = item['scored']
        lo = item['offset']
        need(a == end and lo <= a < b and len(item['prefix']) <= 32, 'SURFACE_FSV_RANGES')
        need(item['prefix'] == tokens[lo:b], 'SURFACE_FSV_CAUSAL_PREFIX')
        need(item['labels'] == [-100] * (a - lo) + tokens[a+1:b+1], 'SURFACE_FSV_LABELS')
        recovered.extend(x for x in item['labels'] if x != -100)
        end = b
    need(recovered == [*raw, 257] and end == len(raw)+1, 'SURFACE_FSV_BYTES')


def run(session, output):
    output = Path(output)
    output.mkdir(exist_ok=False)
    config = ModelConfig()
    proofs, selected = [], []
    for family in ('coding', 'conversation', 'reasoning'):
        keys = [k for k, t in session.capability_tasks.items() if t['family'] == family]
        def width(key):
            t = session.capability_tasks[key]
            e = next(e for e in session.capability_sources.values() if e.native.pin == t['binding'])
            return max(len(e.population.fields[f]['text'].encode()) for f in e.population.tasks[key]['targets'])
        for key in dict.fromkeys([keys[0], max(keys, key=width)]):
            task = session.capability_tasks[key]
            source = ConnectedSurface(session, key, binding=task['binding'])
            for field in source.fields:
                selected.append((family, key, field, source))
    specs = selected[0][-1].evidence.native.specs
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(20261022)
        model = StreamSurfaceDecoder(config, specs).to(session.device)
    model.train()
    before = {n: sha(p.detach().cpu().numpy().tobytes()) for n, p in model.named_parameters()}
    np.savez_compressed(output / 'decoder-initial.npz',
                        **{n: p.detach().cpu().numpy() for n, p in model.named_parameters()})
    for i, (family, key, field, source) in enumerate(selected):
        raw, native = source.field(field, config, session.device)
        items = list(schedule(raw))
        check_schedule(raw, items)
        write_committed(output / ('schedule-' + str(i) + '.json'), items)
        model.zero_grad(set_to_none=True)
        loss = whole_surface_loss(model, raw, native, backward=True)
        need(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
             'SURFACE_FSV_GRADIENT_FINITE')
        norms = {}
        for spec in specs:
            for group in ('projectors.' + spec.name + '.', 'blocks.0.branches.' + spec.name + '.'):
                norm = sum(float(p.grad.detach().square().sum()) for n, p in model.named_parameters()
                           if n.startswith(group) and p.grad is not None)
                need(norm > 0, 'SURFACE_FSV_DISCONNECTED:' + group)
                norms[group] = norm
        need(model.output.weight.grad is not None and bool(model.output.weight.grad.abs().any())
             and model.tokens.weight.grad is not None and bool(model.tokens.weight.grad.abs().any()),
             'SURFACE_FSV_TEXT_DISCONNECTED')
        arrays = {n: x.values.detach().cpu().numpy() for n, x in native.items()}
        np.savez_compressed(output / ('native-' + str(i) + '.npz'), **arrays)
        gradient_path = output / ('gradients-' + str(i) + '.npz')
        np.savez_compressed(gradient_path, **{n: p.grad.detach().cpu().numpy()
            for n, p in model.named_parameters() if p.grad is not None})
        proofs.append(dict(family=family, task=key, field=field, binding=source.binding,
            episode=session.capability_tasks[key]['episode'], loss=loss, gradients=norms,
            gradient_file=dict(name=gradient_path.name, sha256=sha(gradient_path.read_bytes())),
            native={n: dict(shape=list(x.shape), sha256=sha(x.astype('<f4').tobytes())) for n, x in arrays.items()},
            teacher_forcing='prior_bytes_only', decoder_conditioning='true_target_native_windows',
            no_predictor_target_leakage_claim_from_this_component=True))
        print(dict(surface=i, family=family, bytes=len(raw), windows=loss['windows'], loss=loss['loss']), flush=True)
    need(before == {n: sha(p.detach().cpu().numpy().tobytes()) for n, p in model.named_parameters()},
         'SURFACE_FSV_UNREQUESTED_UPDATE')
    controls = []
    import copy
    for name in ('future_prefix', 'missing_tail', 'changed_target'):
        bad = copy.deepcopy(items)
        if name == 'future_prefix': bad[0]['prefix'][0] = raw[0]
        elif name == 'missing_tail': bad.pop()
        else: bad[0]['labels'][0] = (raw[0] + 1) % 256
        try: check_schedule(raw, bad)
        except ValueError as exc: controls.append(dict(name=name, error=str(exc)))
        else: raise AssertionError('SURFACE_FSV_CONTROL_ACCEPTED:' + name)
    report = dict(schema='complete-capability-surface-connectivity-v1', cases=proofs, controls=controls,
        config=asdict(config), instruments=[asdict(s) for s in specs],
        initial_file_sha256=sha((output / 'decoder-initial.npz').read_bytes()),
        code={n: sha((Path(__file__).resolve().parents[1] / n).read_bytes()) for n in
              ('model/stream_surface.py', 'model/network.py', 'audit/capability_surface_fsv.py')},
        model_parameters=digest(before), journal_head=session.head, optimizer_steps=0,
        complete_target_bytes=True, bounded_gradients_measured=True, numbers='measured',
        acceptance='author-checked', scientific_admission=False, learned_capability=False, serving=False,
        predictor_kernel_guard_connected=False)
    pin = write_committed(output / 'report.json', report)
    return dict(path=str(output), pin=pin)
