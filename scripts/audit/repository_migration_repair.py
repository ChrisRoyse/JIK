"""Exercise explicit migration repair under joint-bank retention and new-loss descent."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.proposal_recognition import read
from audit.repository_cohort_open import authenticate, open_memory
from audit.repository_relation_learning import retained
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import run_cli, take_lease
from model.loop_training_checkpoint import fingerprint, tensor_fingerprint
from model.repository_relation_data import RelationData
from model import repository_relation_checkpoint as parent
from model import repository_retention_curvature_checkpoint as previous_checkpoint
from model.repository_retention_curvature import Retention as PreviousRetention
from model.repository_bank_retention import compare, measurements, joined
from model.repository_migration_repair import Retention, POLICY, update, accepted
from model import repository_migration_repair_checkpoint as checkpoint
from senses.contract import SenseError
from senses.trajectory_population import read_pinned

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--stage', choices=('learn', 'cold'), required=True)
    args = p.parse_args(); started = time.perf_counter()
    c = read(args.config); root = Path(c['root']); previous = Path(c['previous'])
    prior = read_pinned(previous/'result.json', c['previous_pin'])
    pc = read(previous/'config.json'); relation = Path(pc['previous'])
    earlier = read_pinned(relation/'result.json', pc['previous_pin'])
    config = read(relation/'config.json'); source = Path(config['source'])
    proof = read_pinned(Path(config['proof']), config['proof_pin'])
    check(sha((source/'experience.sqlite3').read_bytes()) == proof['journal'] and
          sha((source/'snapshot-final.json').read_bytes()) == proof['snapshot'] and proof['cold_replay_equal'],
          'BANK_RETENTION_SOURCE_PIN')
    for name, pin in proof['immutable_after'].items():
        check(sha(Path(name).read_bytes()) == pin, 'BANK_RETENTION_ANCESTOR_PIN')
    recipes = authenticate(read(source/'config.json'))
    if args.stage == 'learn':
        check(not root.exists(), 'BANK_RETENTION_OUTPUT_EXISTS'); root.mkdir()
        write_committed(root/'config.json', c)
    else:
        check(read(root/'config.json') == c, 'BANK_RETENTION_CONFIG_PIN')
    gpu = subprocess.check_output(['nvidia-smi'], text=True)
    take_lease('28', 'repository-migration-repair-'+args.stage)
    (root/('gpu-'+args.stage+'.txt')).open('x').write(gpu)
    configure(); authenticated = time.perf_counter()
    session, sessions = open_memory(recipes, source, read(source/'snapshot-final.json'))
    snapshot = digest(session.snapshot()); restored = time.perf_counter()
    print(dict(stage=args.stage, restored=session.journal.count), flush=True)
    data = RelationData(session, config['native'], config['native_proof'], config['native_proof_pin'])
    model, optimizer, _ = parent.load(relation/'learn_1-final', data, earlier['checkpoints']['learn_1-final'])
    keys = data.stage('learn_1', model)['keys']
    references = dict(learn_1=earlier['stages']['learn_1']['after'],
                      learn_2={k:prior['before'][k] for k in keys})
    predecessor = dict(root=str(relation), result=pc['previous_pin'],
                       checkpoint=earlier['checkpoints']['learn_1-final'])
    retention = Retention(data, model, references, predecessor)
    previous_retention = PreviousRetention(data, model, references['learn_1'], predecessor)
    bad, _, _ = previous_checkpoint.load(previous/'attempt-08', previous_retention,
                                         prior['checkpoints']['attempt-08'])
    bad_views = retention.data.views(bad)
    bad_joint = measurements(joined(bad_views))
    old_violations = compare(measurements(references['learn_1']), measurements(bad_views['learn_1']))
    joint_violations = compare(retention.ceiling, bad_joint)
    check(not old_violations and joint_violations, 'BANK_RETENTION_NEGATIVE_NOT_EXPOSED')
    old = retained(session, model)
    check(old == earlier['parent_retained'], 'BANK_RETENTION_LEGACY_PARENT')
    controls = []
    def reject(name, code, fn):
        try:
            fn()
        except (ValueError, SenseError) as exc:
            check(code in str(exc), 'BANK_RETENTION_CONTROL_CODE:'+str(exc))
        else:
            raise AssertionError('BANK_RETENTION_CONTROL_ACCEPTED:'+name)
        controls.append(dict(name=name, code=code))
    reject('previous_candidate_expanded_bank', 'RETENTION_EXISTING_REGRESSION', lambda:retention.scores(bad))
    missing = dict(retention.ceiling); missing.pop(next(iter(missing)))
    reject('missing_bank_score', 'RETENTION_SCORE_ROSTER', lambda:compare(retention.ceiling, missing))
    saved = copy.deepcopy(retention.contract)
    retention.contract['banks']['learn_2']['labels'][0] = 9
    reject('forged_bank_label', 'REPAIR_CONTRACT_CHANGED', retention.checked)
    retention.contract = saved
    for name, values in [('nondecreasing_new_objective', (1., 1., 1., .5)),
                         ('nondecreasing_migration_deficit', (1., .5, 1., 1.))]:
        passed, _ = accepted(retention.ceiling, retention.ceiling, *values, True)
        check(not passed, 'REPAIR_CONTROL_ACCEPTED:'+name)
        controls.append(dict(name=name, code='REPAIR_GATE_REFUSED'))
    del bad
    materialized = time.perf_counter()
    phases = dict(authentication=authenticated-started, lineage_restore=restored-authenticated,
                  materialization_controls=materialized-restored)
    cursor = time.perf_counter()
    if args.stage == 'learn':
        result = dict(contract=retention.contract, code=checkpoint.code_pins(), controls=controls,
            previous_candidate=dict(views=bad_views, old_regressions=old_violations, joint_regressions=joint_violations),
            references=references, checkpoints={}, attempts=[], legacy=old,
            initial_parameters=tensor_fingerprint(dict(model.named_parameters())),
            before=data.evaluate(model, 'learn_2'), before_views=retention.data.views(model),
            migration_before=retention.migration(model))
        pin = checkpoint.save(root/'initial', model, optimizer, retention, 8, 0, predecessor)
        result['checkpoints']['initial'] = pin
        forced = update(retention, model, optimizer, force_reject=True)
        check(not forced['accepted'] and forced['before'] == forced['after'] and
              forced['proposal_fingerprint'] != forced['before'], 'BANK_RETENTION_FORCED_ROLLBACK')
        result['forced_rejection'] = forced
        clock = 8
        for attempt in range(1, POLICY['attempts']+1):
            row = update(retention, model, optimizer)
            if row['accepted']: clock += 1
            row.update(attempt=attempt, optimizer_step=clock)
            result['attempts'].append(row)
            check(retained(session, model) == old, 'BANK_RETENTION_LEGACY_DRIFT')
            name = f'attempt-{attempt:02}'
            pin = checkpoint.save(root/name, model, optimizer, retention, clock, attempt, pin)
            result['checkpoints'][name] = pin
            write_committed(root/(name+'.json'), row)
            print(dict(attempt=attempt, accepted=row['accepted'], projection=row['projection']['status'],
                       trials=len(row['trials']), seconds=row['wall_seconds']), flush=True)
            if not row['accepted']: break
        result.update(after=data.evaluate(model, 'learn_2'), after_views=retention.data.views(model),
                      migration_after=retention.migration(model),
                      withheld_action=data.evaluate(model, 'learn_2', intervention=True),
                      final_parameters=tensor_fingerprint(dict(model.named_parameters())),
                      final_fingerprint=fingerprint(model, optimizer))
        result['changed_parameters'] = [n for n,v in result['final_parameters'].items()
                                        if v != result['initial_parameters'][n]]
        check(all(n.startswith('query._repository.') for n in result['changed_parameters']),
              'BANK_RETENTION_PARAMETER_SCOPE')
        rollback, _, _ = checkpoint.load(root/'initial', retention, result['checkpoints']['initial'])
        check(data.evaluate(rollback, 'learn_2') == result['before'] and retained(session, rollback) == old,
              'BANK_RETENTION_ROLLBACK')
        reject('wrong_checkpoint_pin', 'RETENTION_CHECKPOINT_PIN',
               lambda:checkpoint.load(root/'initial', retention, '0'*64))
        write_committed(root/'result.json', result)
        phases['learning_controls_persistence'] = time.perf_counter()-cursor
    else:
        expected = read(root/'result.json')
        check(expected['contract'] == retention.contract and expected['code'] == checkpoint.code_pins(),
              'BANK_RETENTION_COLD_CONTRACT')
        first = expected['attempts'][0]
        resume = 'attempt-01' if first['accepted'] else 'initial'
        model, optimizer, _ = checkpoint.load(root/resume, retention, expected['checkpoints'][resume])
        rows = expected['attempts'][1:] if first['accepted'] else expected['attempts']
        for row in rows:
            actual = update(retention, model, optimizer)
            for key in ('accepted','code','before','after','before_scores','after_scores','trials','projection','migration_debt_before','migration_debt_after','migration_deficits_before','new_loss_before'):
                check(actual[key] == row[key], 'BANK_RETENTION_RESUMED_UPDATE:'+key)
        check(fingerprint(model, optimizer) == expected['final_fingerprint'] and
              data.evaluate(model, 'learn_2') == expected['after'] and
              retention.data.views(model) == expected['after_views'] and retained(session, model) == old and
              retention.migration(model) == expected['migration_after'], 'BANK_RETENTION_COLD_FINAL')
        name = f"attempt-{len(expected['attempts']):02}"
        loaded, loaded_opt, _ = checkpoint.load(root/name, retention, expected['checkpoints'][name])
        check(fingerprint(loaded, loaded_opt) == expected['final_fingerprint'], 'BANK_RETENTION_FINAL_LOAD')
        rollback, _, _ = checkpoint.load(root/'initial', retention, expected['checkpoints']['initial'])
        check(data.evaluate(rollback, 'learn_2') == expected['before'] and retained(session, rollback) == old,
              'BANK_RETENTION_COLD_ROLLBACK')
        result = dict(resumed_exact=True, final_load_equal=True, rollback_equal=True, legacy_equal=True)
        phases['restart_controls'] = time.perf_counter()-cursor
    data.fresh()
    check(digest(session.snapshot()) == snapshot and
          sha((source/'experience.sqlite3').read_bytes()) == proof['journal'], 'BANK_RETENTION_SOURCE_CHANGED')
    for s in sessions: s.close()
    ended = time.perf_counter(); phases['final_source_readback'] = ended-started-sum(phases.values())
    report = dict(stage=args.stage, wall_seconds=ended-started, phases=phases, controls=controls,
        source_unchanged=True, serving=False, scientific_admission=False, heldout=False,
        useful_new_capability_demonstrated=False, acceptance='author-checked', number_status='measured',
        result=sha((root/'result.json').read_bytes()), cold=result if args.stage == 'cold' else None,
        peak_cuda_allocated=torch.cuda.max_memory_allocated(), peak_cuda_reserved=torch.cuda.max_memory_reserved())
    write_committed(root/(args.stage+'-readback.json'), report)
    print(report, flush=True)


if __name__ == '__main__': run_cli(main)
