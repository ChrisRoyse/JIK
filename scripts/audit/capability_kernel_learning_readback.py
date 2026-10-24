"""Cold loss/gradient/AdamW replay from retained native sources and checkpoints."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'lens')]
import torch
from audit.capability_kernel_learning import system
from audit.fsv_identity_loop import configure
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import take_lease, run_cli
from model.capability_kernel_learning import KernelLearningCorpus, predict
from model.capability_objective import next_constellation_loss
from model.loop_training_checkpoint import tensor_fingerprint


def read(path): return json.loads(Path(path).read_bytes())
def fingerprint(x): return digest(tensor_fingerprint(x))


def refused(code, call):
    try: call()
    except ValueError as exc:
        need(code in str(exc), 'LEARNING_READER_WRONG_REFUSAL:' + str(exc))
        return dict(expected=code, actual=str(exc))
    raise AssertionError('LEARNING_READER_ACCEPTED:' + code)


def checked_checkpoint(path, source):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    need(saved['source'] == source and fingerprint(saved['optimizer']) == saved['optimizer_hash'],
         'LEARNING_READER_CHECKPOINT_BINDING')
    return saved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--report-pin', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); need(not a.output.exists(), 'LEARNING_READER_EXISTS')
    started = time.perf_counter(); report = read(a.root / 'report.json')
    need(sha((a.root / 'report.json').read_bytes()) == a.report_pin, 'LEARNING_READER_REPORT')
    config = read(a.config)
    need(sha(a.config.read_bytes()) == report['source']['config'], 'LEARNING_READER_CONFIG')
    for name, pin in report['source']['code'].items():
        need(sha((Path(__file__).resolve().parents[1] / name).read_bytes()) == pin, 'LEARNING_READER_CODE')
    need(sha((a.root / 'steps.jsonl').read_bytes()) == report['ledger'], 'LEARNING_READER_LEDGER')
    corpus = KernelLearningCorpus(config); plan = read(a.root / 'plan.json')
    need(plan['train'] == list(corpus.train) and plan['queries'] == list(corpus.query)
         and plan['regions'] == [list(x) for x in corpus.regions], 'LEARNING_READER_POPULATION')
    records = [json.loads(x) for x in (a.root / 'steps.jsonl').read_bytes().splitlines()]
    need(len(records) == len(corpus.train) == report['optimizer_updates'], 'LEARNING_READER_COUNT')
    previous = sha((a.root / 'initial.pt').read_bytes()); current = report['initial_parameters']
    for index, record in enumerate(records):
        body = {k:v for k,v in record.items() if k != 'sha256'}
        expected_target = record['task'] if report['arm'] == 'correct' else plan['shuffled'][record['task']]
        need(record['index'] == index and record['task'] == corpus.train[index]
             and record['target'] == expected_target and record['previous'] == previous
             and record['before'] == current and digest(body) == record['sha256'], 'LEARNING_READER_CHAIN')
        previous, current = record['sha256'], record['after']
    need(current == report['final_parameters'] and sha((a.root / 'final.pt').read_bytes())
         == report['final_checkpoint'], 'LEARNING_READER_FINAL')
    gpu = subprocess.check_output(['nvidia-smi'], text=True)
    take_lease('29', 'capability-learning-cold-' + report['arm']); configure()
    model = system(config, corpus); cases = []; controls = []
    frozen = fingerprint(dict(model.named_buffers()))
    first_families = {family:next(i for i,r in enumerate(records) if r['family'] == family)
                      for family in ('coding', 'conversation', 'reasoning')}
    indices = list(range(max(first_families.values()) + 1)) + [len(records) - 1]
    for index in indices:
        record = records[index]
        before_path = a.root / ('initial.pt' if index == 0 else f'before-{index}.pt')
        expected_path = a.root / f'after-{index}.pt'
        expected = checked_checkpoint(expected_path, report['source']) if expected_path.exists() else None
        if before_path.exists():
            saved = checked_checkpoint(before_path, report['source'])
            need(saved['updates'] == index, 'LEARNING_READER_CURSOR')
            model.load_state_dict(saved['model'], strict=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                          weight_decay=config['weight_decay'], foreach=False)
            optimizer.load_state_dict(saved['optimizer'])
        else:
            need(index > 1 and cases[-1]['index'] == index-1, 'LEARNING_READER_RESUME_GAP')
        optimizer.zero_grad(set_to_none=True)
        need(fingerprint(dict(model.named_parameters())) == record['before'], 'LEARNING_READER_BEFORE')
        world = model['world']; key = record['task']; target_key = record['target']
        native = corpus.batch(key, 'context', world.config, 'cuda')
        operation = corpus.tasks[key]['kind']
        output = predict(world, model['request'], native, key=key, operation=operation)
        raw, target = corpus.target(target_key, world.config, 'cuda', fitting=True)
        loss, meaning = next_constellation_loss(world, output, target)
        surface, _ = model['decoder'].render_loss(raw, output['predicted_constellation'],
                                                 backward=True, meaning_loss=loss)
        need(meaning == record['meaning'] and surface == record['surface'], 'LEARNING_READER_FORWARD_LOSS')
        donor = native if key == target_key else corpus.batch(target_key, 'context', world.config, 'cuda')
        for spec, row in zip(world.specs, record['reciprocal']):
            visible = {n:x for n,x in native.items() if n != spec.name}; truth = donor[spec.name].values
            prediction = model['reciprocal'](visible, hidden=spec.name, windows=truth.shape[1])
            auxiliary = model['reciprocal'].loss(prediction['prediction'], truth) / len(world.specs)
            need(row == dict(hidden=spec.name, loss=float(auxiliary.detach()) * len(world.specs),
                             target_windows=truth.shape[1]), 'LEARNING_READER_RECIPROCAL')
            auxiliary.backward()
        for branch, evidence in record['gradients'].items():
            grads = {}
            for name, parameter in model.named_parameters():
                owner = next((s.name for s in world.specs if '.'+s.name+'.' in name), 'shared')
                if owner == branch and parameter.grad is not None: grads[name] = parameter.grad
            need(fingerprint(grads) == evidence['sha256'], 'LEARNING_READER_BRANCH_GRADIENT')
        optimizer.step()
        need(fingerprint(dict(model.named_parameters())) == record['after'], 'LEARNING_READER_ADAMW')
        if expected is not None:
            need(expected['updates'] == index+1
                 and fingerprint(model.state_dict()) == fingerprint(expected['model'])
                 and fingerprint(optimizer.state_dict()) == expected['optimizer_hash'], 'LEARNING_READER_MOMENTS')
        need(fingerprint(dict(model.named_buffers())) == frozen, 'LEARNING_READER_FROZEN')
        cases.append(dict(index=index, task=key, family=record['family'], full_surface_positions=len(raw)+1,
            native_forward_loss_exact=True, branch_gradients_exact=True, parameters_exact=True,
            moments_equal_persisted_checkpoint=expected is not None,
            optimizer_fingerprint=fingerprint(optimizer.state_dict())))
        if index == 0:
            controls.append(refused('LEARNING_OPERATION_REQUIRED', lambda: predict(world, model['request'],
                native, key=key, operation='unknown-operation')))
            controls.append(refused('LEARNING_TARGET_SPLIT', lambda: corpus.target(corpus.query[0],
                world.config, 'cuda', fitting=True)))
            missing = dict(native); missing.pop(world.specs[0].name)
            controls.append(refused('CONSTELLATION_ROSTER', lambda: predict(world, model['request'],
                missing, key=key, operation=operation)))
            # A changed moment must change the checkpoint's independently read identity.
            corrupted = copy.deepcopy(expected['optimizer'])
            moment = next(iter(corrupted['state'].values()))['exp_avg']
            moment.reshape(-1)[0] += 1
            need(fingerprint(corrupted) != expected['optimizer_hash'], 'LEARNING_READER_MOMENT_CONTROL')
            controls.append(dict(expected='changed_optimizer_moment_hash', rejected=True))
        print(json.dumps(cases[-1]), flush=True)
    need(sha(Path(config['journal']['path']).read_bytes()) == config['journal']['pin'], 'LEARNING_READER_JOURNAL')
    result = dict(schema='capability-kernel-learning-cold-readback-v2', source_report=a.report_pin,
        arm=report['arm'], ledger=report['ledger'], cases=cases, controls=controls, gpu_before=gpu,
        all_ledger_rows_bound=True, frozen_buffers_unchanged=True, seconds=time.perf_counter()-started,
        first_by_family=first_families, producer_sha256=sha(Path(__file__).read_bytes()),
        number_status='measured', acceptance='author-checked', scientific_admission=False,
        learned_capability=False, heldout_evaluation_pending=True)
    print(json.dumps(dict(pin=write_committed(a.output, result))), flush=True)


if __name__ == '__main__': run_cli(main)
