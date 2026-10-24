"""One complete kernel curriculum pass with durable optimizer and branch evidence."""
import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'lens')]
import torch
from torch import nn
from audit.capability_prediction_readback import load_view
from audit.fsv_identity_loop import configure
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import take_lease, run_cli
from model.capability_kernel_learning import KernelLearningCorpus, RequestedOperation, predict, pinned
from model.capability_objective import PredictedSurfaceDecoder, next_constellation_loss
from model.loop_training_checkpoint import tensor_fingerprint
from model.reciprocal_constellation import ReciprocalConstellation


def parameter_hash(model): return digest(tensor_fingerprint(dict(model.named_parameters())))


def shuffled(corpus):
    groups = defaultdict(list)
    for key in corpus.train: groups[corpus.tasks[key]['kind']].append(key)
    result = {}
    for kind, keys in sorted(groups.items()):
        # Rotate by a fixed half population; persist same-group residuals openly.
        need(len(keys) > 1, 'LEARNING_SHUFFLE_SUPPORT')
        shift = max(1, len(keys) // 2)
        for i, key in enumerate(keys): result[key] = keys[(i + shift) % len(keys)]
    need(set(result) == set(result.values()) == set(corpus.train)
         and all(k != v for k, v in result.items()), 'LEARNING_SHUFFLE_PERMUTATION')
    return result


def system(config, corpus):
    for ref in config['initial']['files'].values():
        need(sha(Path(ref['path']).read_bytes()) == ref['pin'], 'LEARNING_INITIAL_PIN')
    root = Path(config['initial']['view'])
    metadata = json.loads((root / 'view.json').read_bytes())
    parent = load_view(root, metadata)
    world = corpus.world(parent)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(config['seed'])
        decoder = PredictedSurfaceDecoder(world.config, world.specs).to('cuda').eval()
        decoder.load_state_dict(torch.load(config['initial']['decoder'], map_location='cuda', weights_only=True))
        request = RequestedOperation(world.specs, world.config.width).to('cuda')
        reciprocal = ReciprocalConstellation(world.core).to('cuda').eval()
    result = nn.ModuleDict(dict(world=world, decoder=decoder, request=request, reciprocal=reciprocal))
    result.requires_grad_(True).eval()
    need(sum(p.numel() for p in result.parameters()) <= world.config.max_parameters, 'LEARNING_PARAMETERS')
    return result


def branch_evidence(model):
    groups, parameters = defaultdict(dict), defaultdict(dict)
    for name, p in model.named_parameters():
        branch = next((s.name for s in model['world'].specs if '.' + s.name + '.' in name), 'shared')
        parameters[branch][name] = p
        if p.grad is not None: groups[branch][name] = p.grad
    return {branch: dict(sha256=digest(tensor_fingerprint(values)),
                parameters=digest(tensor_fingerprint(parameters[branch])),
                squared_norm=sum(float(v.double().square().sum()) for v in values.values()))
            for branch, values in sorted(groups.items())}


def step(model, corpus, key, target_key, optimizer):
    world, decoder, request, reciprocal = (model[n] for n in ('world', 'decoder', 'request', 'reciprocal'))
    need(key in corpus.train and target_key in corpus.train, 'LEARNING_FIT_QUERY')
    need(corpus.tasks[key]['kind'] == corpus.tasks[target_key]['kind'], 'LEARNING_TARGET_OPERATION')
    native = corpus.batch(key, 'context', world.config, 'cuda')
    optimizer.zero_grad(set_to_none=True)
    output = predict(world, request, native, key=key, operation=corpus.tasks[key]['kind'])
    raw, target = corpus.target(target_key, world.config, 'cuda', fitting=True)
    meaning_loss, meaning = next_constellation_loss(world, output, target)
    surface, _ = decoder.render_loss(raw, output['predicted_constellation'], backward=True, meaning_loss=meaning_loss)
    reciprocal_rows = []
    # The null breaks whole constellation co-observation as well as sequence targets.
    donor = native if key == target_key else corpus.batch(target_key, 'context', world.config, 'cuda')
    for spec in world.specs:
        hidden = spec.name; truth = donor[hidden].values
        visible = {n: x for n, x in native.items() if n != hidden}
        prediction = reciprocal(visible, hidden=hidden, windows=truth.shape[1])
        loss = reciprocal.loss(prediction['prediction'], truth) / len(world.specs)
        loss.backward()
        reciprocal_rows.append(dict(hidden=hidden, loss=float(loss.detach()) * len(world.specs),
                                    target_windows=truth.shape[1]))
    need(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
         'LEARNING_GRADIENT_NONFINITE')
    gradients = branch_evidence(model)
    before = parameter_hash(model)
    optimizer.step()
    after = parameter_hash(model)
    final_branches = branch_evidence(model)
    for name, evidence in gradients.items():
        evidence['parameters_after'] = final_branches[name]['parameters']
    need(before != after, 'LEARNING_NO_UPDATE')
    need(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'LEARNING_PARAMETER_NONFINITE')
    return dict(task=key, target=target_key, family=corpus.tasks[key]['family'],
        operation=corpus.tasks[key]['kind'], before=before, after=after,
        gradients=gradients, meaning=meaning, surface=surface, reciprocal=reciprocal_rows)


def save_checkpoint(path, model, optimizer, count, source, previous):
    need(not path.exists(), 'LEARNING_CHECKPOINT_EXISTS')
    temporary = path.with_suffix('.pending')
    payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), updates=count,
                   source=source, previous=previous, parameter_hash=parameter_hash(model),
                   optimizer_hash=digest(tensor_fingerprint(optimizer.state_dict())))
    torch.save(payload, temporary)
    with temporary.open('rb') as f: os.fsync(f.fileno())
    temporary.replace(path)
    # Serialization readback is a boundary check, not the separate cold replay.
    loaded = torch.load(path, map_location='cpu', weights_only=True)
    need(loaded['updates'] == count and loaded['source'] == source
         and digest(tensor_fingerprint(loaded['model'])) == digest(tensor_fingerprint(payload['model']))
         and digest(tensor_fingerprint(loaded['optimizer'])) == payload['optimizer_hash'],
         'LEARNING_CHECKPOINT_READBACK')
    return sha(path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=('prepare', 'fit'), required=True)
    parser.add_argument('--arm', choices=('correct', 'shuffled'), default='correct')
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); config = json.loads(a.config.read_bytes())
    need(not a.output.exists(), 'LEARNING_OUTPUT_EXISTS'); a.output.mkdir()
    started = tick = time.perf_counter(); phases = {}
    try:
        corpus = KernelLearningCorpus(config)
        phases['authenticate_sources'] = time.perf_counter() - tick; tick = time.perf_counter()
        mapping = shuffled(corpus)
        source = dict(config=sha(a.config.read_bytes()), snapshot=config['snapshot'], kernel=config['kernel'],
            journal=config['journal'], code={name:sha((Path(__file__).resolve().parents[1] / name).read_bytes())
            for name in ('model/capability_kernel_learning.py', 'audit/capability_kernel_learning.py')})
        plan = dict(source=source, train=corpus.train, queries=corpus.query, regions=corpus.regions,
            counts=dict(train=len(corpus.train), queries=len(corpus.query), regions=len(corpus.regions)),
            families=dict(Counter(corpus.tasks[k]['family'] for k in corpus.train)), shuffled=mapping,
            same_group_shuffle=sum(corpus.tasks[k]['split_group'] == corpus.tasks[v]['split_group']
                                   for k, v in mapping.items()), scientific_admission=False,
            exposure='development_exposed', serving=False)
        write_committed(a.output / 'plan.json', plan)
        phases['plan'] = time.perf_counter() - tick; tick = time.perf_counter()
        print(json.dumps(dict(stage='prepared', **plan['counts'], families=plan['families'])), flush=True)
        if a.stage == 'prepare':
            write_committed(a.output / 'report.json', dict(plan=plan, phases=phases, number_status='measured'))
            return
        gpu = subprocess.check_output(['nvidia-smi'], text=True)
        take_lease('29', 'capability-kernel-learning-' + a.arm); configure()
        (a.output / 'gpu-before.txt').write_text(gpu)
        model = system(config, corpus)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                      weight_decay=config['weight_decay'], foreach=False)
        initial = parameter_hash(model); frozen = digest(tensor_fingerprint(dict(model.named_buffers())))
        previous = save_checkpoint(a.output / 'initial.pt', model, optimizer, 0, source, None)
        phases['initialize'] = time.perf_counter() - tick; tick = time.perf_counter()
        print(json.dumps(dict(stage='initialized', parameters=initial, frozen=frozen)), flush=True)
        with (a.output / 'steps.jsonl').open('x') as ledger:
            for index, key in enumerate(corpus.train):
                boundary = index in (0, 1, len(corpus.train) - 1)
                if boundary and index:
                    save_checkpoint(a.output / f'before-{index}.pt', model, optimizer, index, source, previous)
                before = time.perf_counter(); cpu = time.process_time()
                record = step(model, corpus, key, mapping[key] if a.arm == 'shuffled' else key, optimizer)
                record.update(index=index, cpu_seconds=time.process_time()-cpu, seconds=time.perf_counter()-before)
                record['previous'] = previous
                previous = digest(record)
                ledger.write(json.dumps(dict(record, sha256=previous), sort_keys=True) + '\n')
                ledger.flush(); os.fsync(ledger.fileno())
                if boundary:
                    save_checkpoint(a.output / f'after-{index}.pt', model, optimizer, index+1, source, previous)
                print(json.dumps(dict(stage='step', index=index, family=record['family'],
                    surface_loss=record['surface']['loss'], region_loss=record['meaning']['region_loss'],
                    seconds=record['seconds'])), flush=True)
        phases['learning'] = time.perf_counter() - tick; tick = time.perf_counter()
        final_pin = save_checkpoint(a.output / 'final.pt', model, optimizer, len(corpus.train), source, previous)
        model['world'].check_reference()
        need(digest(tensor_fingerprint(dict(model.named_buffers()))) == frozen, 'LEARNING_FROZEN_DRIFT')
        need(sha(Path(config['journal']['path']).read_bytes()) == config['journal']['pin'], 'LEARNING_JOURNAL_CHANGED')
        phases['finalize'] = time.perf_counter() - tick
        wall = time.perf_counter() - started
        need(sum(phases.values()) / wall >= .9, 'LEARNING_PHASE_COVERAGE')
        report = dict(schema='capability-kernel-learning-v1', arm=a.arm, source=source,
            initial_parameters=initial, final_parameters=parameter_hash(model), final_checkpoint=final_pin,
            optimizer_updates=len(corpus.train), ledger=sha((a.output / 'steps.jsonl').read_bytes()),
            frozen_buffers_unchanged=True, journal_unchanged=True, phases=phases, seconds=wall,
            scientific_admission=False, learned_capability=False, serving=False,
            evaluation_pending=True, cold_replay_pending=True, number_status='measured', acceptance='author-checked')
        print(json.dumps(dict(stage='complete', pin=write_committed(a.output / 'report.json', report))), flush=True)
    except BaseException as exc:
        write_committed(a.output / 'failure.json', dict(error=str(exc), trace=traceback.format_exc()))
        raise


if __name__ == '__main__': run_cli(main)
