"""Independent source, metric and checkpoint readback for retention across bank growth."""
import argparse
import copy
import io
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.repository_cohort_native import reconstruct
from audit.repository_relation_readback import check_stage, summary
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from claims.repository_cohort_population import CohortPopulation
from model.loop_training_checkpoint import tensor_fingerprint
from senses.contract import SenseError
from senses.trajectory_population import read_pinned

import torch


def read(p): return json.loads(p.read_bytes())


def scores(rows):
    return {key+'|'+lane+'|'+metric:float(value[metric])
            for key,row in rows.items() for lane,value in row['lanes'].items()
            for metric in ('brier', 'native_expected_loss', 'native_placement_loss') if metric in value}


def combined(views):
    check(set(views) == {'learn_1', 'learn_2'}, 'BANK_READER_VIEWS')
    return {view+'|'+key:value for view,rows in views.items() for key,value in scores(rows).items()}


def differences(before, after):
    check(before.keys() == after.keys(), 'BANK_READER_ROSTER')
    check(all(math.isfinite(v) for v in (*before.values(), *after.values())), 'BANK_READER_FINITE')
    return {k:dict(before=v, after=after[k], delta=after[k]-v) for k,v in before.items() if after[k] > v}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); started = time.perf_counter(); root = a.root
    c, result = read(root/'config.json'), read(root/'result.json')
    previous = Path(c['previous']); prior = read_pinned(previous/'result.json', c['previous_pin'])
    pc = read(previous/'config.json'); relation = Path(pc['previous'])
    earlier = read_pinned(relation/'result.json', pc['previous_pin']); config = read(relation/'config.json')
    native = config['native']; proof = read_pinned(Path(config['native_proof']), config['native_proof_pin'])
    population = CohortPopulation(native['population'], native['population_pin'])
    examples = earlier['binding']['examples']; contract = result['contract']
    old = sorted(k for k,e in examples.items() if e['partition'] == 'learn_1')
    new = sorted(k for k,e in examples.items() if e['partition'] == 'learn_2')
    check(contract['keys'] == old and contract['new'] == new, 'BANK_READER_PARTITIONS')
    for key,e in examples.items():
        actual = {side:reconstruct(population, Path(native['native']), native['native_pin'], key, side)
                  for side in ('input', 'echo')}
        check(actual == proof['receipts'][key] and
              e['outcome'] == population.observations.entries[key]['outcome'] and
              e['partition'] == population.fields[key]['partition'], 'BANK_READER_NATIVE_TARGET')
        absent = all(v['absent'] for v in actual['echo']['slots'].values())
        check(e['absent'] == absent and e['region'] == (None if absent else digest(actual['echo']['slots'])),
              'BANK_READER_NATIVE_REGION')
    expected = dict(learn_1=earlier['stages']['learn_1']['after'],
                    learn_2={k:prior['before'][k] for k in old})
    check(result['references'] == expected == result['before_views'], 'BANK_READER_BASELINE')
    ceilings = combined(expected)
    check(contract['ceiling'] == ceilings and contract['original_ceiling'] == scores(expected['learn_1']) and
          contract['binding'] == digest(earlier['binding']) and contract['policy']['tolerance'] == 0.,
          'BANK_READER_CONTRACT')
    for view,bank in contract['banks'].items():
        keys = sorted(old) if view == 'learn_1' else sorted(old)+sorted(new)
        regions = list(dict.fromkeys(examples[k]['region'] for k in keys if not examples[k]['absent']))
        labels = []
        for region in regions:
            values = {examples[k]['outcome'] for k in keys if examples[k]['region'] == region}
            labels.append(next(iter(values)) if len(values) == 1 else 2)
        check(bank['description'] == dict(binding=contract['binding'], partition=view, keys=keys, regions=regions)
              and bank['labels'] == labels, 'BANK_READER_REFERENCE_ROSTER')
    stages = dict(learn_2=dict(description=earlier['stages']['learn_2']['description'],
        before=result['before'], after=result['after'], withheld_action=result['withheld_action']))
    ordered = {k:examples[k] for k in sorted(examples, key=lambda k:(examples[k]['partition'], k))}
    checked_summary = check_stage(dict(stages=stages), ordered, 'learn_2')
    known = result['previous_candidate']
    check(known['views'] == dict(learn_1=prior['retained_after'], learn_2={k:prior['after'][k] for k in old}),
          'BANK_READER_KNOWN_SOURCE')
    check(not differences(scores(expected['learn_1']), scores(known['views']['learn_1'])) and
          differences(ceilings, combined(known['views'])) == known['joint_regressions'] and
          known['joint_regressions'] and not known['old_regressions'], 'BANK_READER_KNOWN_NEGATIVE')
    for when in ('before', 'after'):
        verdict = result['migration_'+when]
        actual = scores({k:result[when][k] for k in old})
        changes = differences(contract['original_ceiling'], actual)
        check(verdict['original'] == contract['original_ceiling'] and verdict['enlarged'] == actual and
              verdict['regressions'] == changes and verdict['retention_passed'] == (not changes) and
              not verdict['serving'], 'BANK_READER_MIGRATION')
    snapshots = {}
    for name,pin in result['checkpoints'].items():
        d = root/name; m = read_pinned(d/'manifest.json', pin)
        check((d/'COMMITTED').read_text() == pin and m['contract'] == contract and m['code'] == result['code'] and
              m['schema'] == 'repository-bank-retention-checkpoint-v1' and not m['serving'] and
              not m['scientific_admission'], 'BANK_READER_CHECKPOINT')
        for code,code_pin in m['code'].items():
            check(sha((Path(__file__).resolve().parents[1]/'model'/code).read_bytes()) == code_pin,
                  'BANK_READER_CODE')
        raw = (d/'delta.pt').read_bytes()
        check(dict(bytes=len(raw), sha256=sha(raw)) == m['payload'], 'BANK_READER_BYTES')
        state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        check(all(int(v['step']) == m['step'] for v in state['optimizer']['state'].values()), 'BANK_READER_CLOCK')
        snapshots[name] = dict(manifest=m, state=state)
    check(set(snapshots) == {'initial'}|{f'attempt-{i:02}' for i in range(1, len(result['attempts'])+1)},
          'BANK_READER_CHECKPOINT_ROSTER')
    forced = result['forced_rejection']
    fp = snapshots['initial']['manifest']['fingerprint']; cursor = ceilings; clock = 8; accepted = 0
    check(forced['before'] == forced['after'] == fp and not forced['accepted'] and
          forced['proposal_fingerprint'] != fp, 'BANK_READER_FORCED_REJECTION')
    predecessor = contract['predecessor']
    check(snapshots['initial']['manifest']['predecessor'] == predecessor, 'BANK_READER_PARENT')
    predecessor = result['checkpoints']['initial']
    for index,row in enumerate(result['attempts'], 1):
        check(row['attempt'] == index and row['before'] == fp and row['before_scores'] == cursor and
              not differences(cursor, row['after_scores']), 'BANK_READER_UPDATE')
        if row['accepted']:
            last = row['trials'][-1]
            check(last['gate_passed'] and last['scores'] == row['after_scores'] and
                  last['new_loss'] < row['new_loss_before'] and row['after'] != fp, 'BANK_READER_GAIN')
            accepted += 1; clock += 1
        else:
            check(row['after'] == fp and row['after_scores'] == cursor and index == len(result['attempts']),
                  'BANK_READER_REFUSAL')
        for trial in row['trials']:
            violations = differences(cursor, trial['scores'])
            check(violations == trial['regressions'], 'BANK_READER_TRIAL')
            if 'next_scale' in trial:
                roots = []; possible = True
                for key,change in violations.items():
                    slope = row['directional'].get(key)
                    if slope is None or slope >= 0: possible = False; break
                    curvature = (change['delta']-trial['scale']*slope)/trial['scale']**2
                    if not math.isfinite(curvature) or curvature <= 0: possible = False; break
                    roots.append(-slope/curvature)
                target = (.5*min(trial['scale'], *roots) if roots else .5*trial['scale']) if possible else None
                check(target == trial['next_scale'], 'BANK_READER_CONTRACTION')
        name = f'attempt-{index:02}'; m = snapshots[name]['manifest']
        check(m['step'] == row['optimizer_step'] == clock and m['fingerprint'] == row['after'] and
              m['predecessor'] == predecessor, 'BANK_READER_CHAIN')
        check(abs(sum(row['phases'].values())-row['wall_seconds']) < 1e-6, 'BANK_READER_TIMING')
        predecessor = result['checkpoints'][name]; fp = row['after']; cursor = row['after_scores']
    check(cursor == combined(result['after_views']) and fp == result['final_fingerprint'], 'BANK_READER_FINAL')
    changed = [n for n,v in result['final_parameters'].items() if v != result['initial_parameters'][n]]
    check(set(changed) == set(result['changed_parameters']) and
          all(n.startswith('query._repository.') for n in changed) and bool(changed) == bool(accepted),
          'BANK_READER_CHANGED_SCOPE')
    for tag,name in (('initial', 'initial'), ('final', f"attempt-{len(result['attempts']):02}")):
        for key,value in snapshots[name]['state']['branch'].items():
            check(tensor_fingerprint(value) == result[tag+'_parameters']['query._repository.'+key],
                  'BANK_READER_PARAMETERS')
    for stage in ('learn', 'cold'):
        receipt = read(root/(stage+'-readback.json'))
        check(receipt['source_unchanged'] and receipt['result'] == sha((root/'result.json').read_bytes()) and
              abs(sum(receipt['phases'].values())-receipt['wall_seconds']) < 1e-6, 'BANK_READER_RECEIPT')
        if stage == 'cold': check(all(receipt['cold'].values()), 'BANK_READER_COLD')
    controls = []
    def reject(name, code, fn):
        try: fn()
        except (ValueError, SenseError) as exc: check(code in str(exc), 'BANK_READER_CONTROL_CODE')
        else: raise AssertionError('BANK_READER_CONTROL_ACCEPTED')
        controls.append(dict(name=name, code=code))
    missing = dict(cursor); missing.pop(next(k for k in missing if k.startswith('learn_2|')))
    reject('missing_enlarged_bank_score', 'BANK_READER_ROSTER', lambda:differences(ceilings, missing))
    bad = copy.deepcopy(stages); next(iter(bad['learn_2']['after'].values()))['outcome'] = 2
    reject('forged_outcome', 'RELATION_READER_TARGET', lambda:check_stage(dict(stages=bad), ordered, 'learn_2'))
    reject('wrong_checkpoint_pin', 'MNLM_UCR_INPUT_DIGEST', lambda:read_pinned(root/'initial/manifest.json', '0'*64))
    source = Path(config['source']); source_proof = read_pinned(Path(config['proof']), config['proof_pin'])
    check(sha((source/'experience.sqlite3').read_bytes()) == source_proof['journal'] and
          sha((source/'snapshot-final.json').read_bytes()) == source_proof['snapshot'], 'BANK_READER_SOURCE')
    report = dict(number_status='measured', acceptance='author-checked', result=sha((root/'result.json').read_bytes()),
        native_joins=len(examples), retained_checks=len(ceilings), regressions=differences(ceilings, cursor),
        known_candidate_regressions=known['joint_regressions'], accepted_updates=accepted, attempts=len(result['attempts']),
        migration_before=result['migration_before'], migration_after=result['migration_after'], summary=checked_summary,
        new_example_summary={when:summary({k:result[when][k] for k in new}) for when in ('before', 'after')},
        changed_parameter_tensors=len(changed), controls=controls, source_unchanged=True, serving=False,
        scientific_admission=False, heldout=False, useful_new_capability_demonstrated=False,
        wall_seconds=time.perf_counter()-started)
    write_committed(a.output, report)
    print(json.dumps(dict(path=str(a.output), sha256=sha(a.output.read_bytes()), accepted=accepted,
                         retained=len(ceilings), migration_regressions=len(result['migration_after']['regressions']))))


if __name__ == '__main__': main()
