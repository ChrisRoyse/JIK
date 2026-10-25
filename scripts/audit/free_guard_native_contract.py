"""Tensor-free source, ownership, and predecessor gates for native output guarding."""
import fcntl
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit import paired_surface_contract as prior
from audit.surface_device_queue import check_owner, identity, need, sha, stamp, write

P = prior.P; REPO = prior.REPO; read = prior.read
LOCK = REPO / '.lease/free-guard-native-queue'
WAIT = P / 'paired-surface-successor-queue-v1-20261025'
PARENT = dict(path=str(P / 'paired-surface-authority-v1-20261025/authority.json'),
              pin='982fe10d9ae4c9a793e11211f41fdc56ca78b22e304fc83a20e5b8910c6dbc2c')
NEW = ('text/free_guard_native_plan.py', 'text/free_guard_native_measure.py', 'audit/free_guard_native_readback.py',
       'audit/free_guard_native_contract.py', 'audit/free_guard_native_queue.py')
STAGES = ('measure', 'cold', 'guard')
POLICY = dict(scope='one_cold_replayed_trajectory_mechanics_not_capability', inputs=15, occurrences=18,
    instruments=6, windows=127, reference_visible=False, device='canonical_cuda',
    wait='actual_paired_coordinator_exit_and_complete_report', automatic_retry=False,
    semantic_acceptance=False, retention_owner='ChrisRoyse/MNLM#29')


def owner_fd(fd):
    need(type(fd) is int and fd >= 0, 'FREE_GUARD_OWNER_FD')
    actual = os.fstat(fd); expected = LOCK.stat()
    need((actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino), 'FREE_GUARD_OWNER_IDENTITY')
    with LOCK.open('a+b') as other:
        try:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        raise ValueError('FREE_GUARD_OWNER_UNLOCKED')


def base(plan):
    parent, _, fresh = prior.authenticate(**PARENT); fresh()
    r = read(Path(plan['path']) / 'report.json', plan['pin'])
    need(r['complete'] and r['schema'] == 'free-guard-canonical-inputs-v1' and r['inputs'] == POLICY['inputs']
         and len(r['roster']) == POLICY['instruments'] and r['source_counts']['completed_atom_occurrences'] == POLICY['occurrences']
         and r['source_counts']['completed_answers'] == 0 and not r['serving'], 'FREE_GUARD_PLAN_SCOPE')
    source = read(Path(r['source']['path']) / 'report.json', r['source']['pin'])
    need(sum(v['total_windows'] for v in source['instrument_totals'].values()) == POLICY['windows'], 'FREE_GUARD_WINDOW_SCOPE')
    cold = read(Path(r['source_readback']['path']) / 'report.json', r['source_readback']['pin'])
    proof = read(Path(source['readback']['path']) / 'report.json', source['readback']['pin'])
    need(cold['complete'] and cold['input'] == r['source'] and cold['all_raw_spans_exact']
         and cold['all_native_joins_exact'] and cold['all_singleton_token_counts_exact'], 'FREE_GUARD_SOURCE_COLD')
    need(proof['complete'] and proof['all_free_logits_exact'] and proof['all_bytes_exact']
         and proof['all_terminals_recounted'], 'FREE_GUARD_FREE_COLD')
    codes = dict(parent['code'])
    for obj in (proof, cold, source, r):
        for name, h in obj['code'].items():
            need(name not in codes or codes[name] == h, 'FREE_GUARD_CODE_CONFLICT'); codes[name] = h
    for name in NEW:
        h = sha(REPO / 'scripts' / name)
        need(name not in codes or codes[name] == h, 'FREE_GUARD_NEW_CODE_CONFLICT'); codes[name] = h
    need(all(sha(REPO / 'scripts' / name) == h for name, h in codes.items()), 'FREE_GUARD_CODE_BINDING')
    files = {Path(plan['path']) / 'report.json': plan['pin'], Path(PARENT['path']): PARENT['pin']}
    for ref in (r['source'], r['source_readback'], source['readback']):
        files[Path(ref['path']) / 'report.json'] = ref['pin']
    for name, item in r['files'].items():
        files[Path(plan['path']) / name] = item['sha256']
    need(all(sha(path) == h for path, h in files.items()), 'FREE_GUARD_PHYSICAL_INPUT')
    return r, codes, files


def prepare(plan, output):
    _, codes, _ = base(plan); launch = WAIT / 'launch.json'; value = read(launch, sha(launch))
    need(value['authority'] == PARENT['pin'] and value['driver_pin'] == codes['audit/paired_surface_queue.py'], 'FREE_GUARD_PARENT_LAUNCH')
    wait = dict(path=str(WAIT), pid=value['pid'], starttime=str(value['starttime']), launch_pin=sha(launch))
    need(wait['pid'] == 3750983 and wait['starttime'] == '415225898', 'FREE_GUARD_PARENT_IDENTITY')
    check_owner(wait); need(identity(wait['pid']) == wait['starttime'], 'FREE_GUARD_PREPARE_LIVE')
    output.mkdir(exist_ok=False)
    write(output / 'authority.json', dict(schema='free-guard-native-successor-v1', plan=plan, parent=PARENT,
        wait=wait, code=codes, policy=POLICY, stages=list(STAGES)))
    return sha(output / 'authority.json')


def authenticate(path, pin):
    cfg = read(path, pin); plan, codes, files = base(cfg['plan'])
    need(set(cfg) == {'schema', 'plan', 'parent', 'wait', 'code', 'policy', 'stages'}
         and cfg['schema'] == 'free-guard-native-successor-v1' and cfg['parent'] == PARENT
         and cfg['code'] == codes and cfg['policy'] == POLICY and cfg['stages'] == list(STAGES), 'FREE_GUARD_AUTHORITY')
    wait = cfg['wait']
    need(wait['path'] == str(WAIT) and wait['pid'] == 3750983 and wait['starttime'] == '415225898', 'FREE_GUARD_WAIT')
    check_owner(wait); launch = read(WAIT / 'launch.json', wait['launch_pin'])
    need(launch['authority'] == PARENT['pin'] and launch['driver_pin'] == codes['audit/paired_surface_queue.py'], 'FREE_GUARD_PARENT_LAUNCH')
    files[Path(path)] = pin; files[WAIT / 'launch.json'] = wait['launch_pin']; stamps = {p: stamp(p) for p in files}
    def fresh():
        need(all(stamp(p) == s for p, s in stamps.items()), 'FREE_GUARD_SOURCE_CHANGED')
        need(all(sha(REPO / 'scripts' / n) == h for n, h in codes.items()), 'FREE_GUARD_CODE_CHANGED')
        check_owner(wait)
    fresh(); return cfg, plan, fresh


def complete(r, cfg):
    need(r['complete'] and r['authority'] == PARENT['pin']
         and r['driver_pin'] == cfg['code']['audit/paired_surface_queue.py'] and r['query_cases'] == 170
         and r['full_training'] and r['teacher_forced'] and not r['free_generation_tested']
         and [v['label'] for v in r['outputs']] == list(prior.STAGES), 'FREE_GUARD_PARENT_INCOMPLETE')


def predecessor(cfg):
    check_owner(cfg['wait']); need(identity(cfg['wait']['pid']) is None, 'FREE_GUARD_PREDECESSOR_LIVE')
    need(not (WAIT / 'failure.json').exists() and (WAIT / 'report.json').is_file(), 'FREE_GUARD_PARENT_FAILED')
    pin = sha(WAIT / 'report.json'); r = read(WAIT / 'report.json', pin); complete(r, cfg)
    parent = read(**PARENT); prior.terminal(parent)
    reports = {}
    for ref in r['outputs']:
        need(Path(ref['path']) == WAIT / ref['label'] / 'report.json', 'FREE_GUARD_PARENT_OUTPUT_PATH')
        reports[ref['label']] = read(ref['path'], ref['pin'])
    for scope in ('qualification', 'full'):
        first, cold = reports[scope], reports[scope + '-readback']
        ref = next(v for v in r['outputs'] if v['label'] == scope)
        need(cold['input_pin'] == ref['pin'], 'FREE_GUARD_PARENT_COLD_BINDING')
        prior.check_scores(first, cold, cuda=True, qualification=scope == 'qualification')
    return pin


def result_check(label, result, cfg, outputs):
    need(result['complete'] and result['plan'] == cfg['plan'], 'FREE_GUARD_CHILD_SCOPE')
    if label in ('measure', 'cold'):
        need(result['schema'] == 'free-guard-native-measurements-v1' and result['gpu_used']
             and not result['errors'] and result['inputs'] == POLICY['inputs'] and len(result['roster']) == POLICY['instruments'],
             'FREE_GUARD_CHILD_MEASUREMENT')
        if label == 'cold':
            first = outputs[0]
            need(result['cold_all_inputs_exact'] and result['cold_reference'] == dict(path=str(Path(first['path']).parent), pin=first['pin']),
                 'FREE_GUARD_CHILD_COLD')
    else:
        need(label == 'guard' and result['native_cold_all_exact'] and result['scored_atoms'] == POLICY['occurrences']
             and result['accepted_answers'] == 0 and not result['serving'] and not result['semantic_validity_established']
             and result['incomplete_answers_remain_refused'], 'FREE_GUARD_CHILD_SCORES')
        for key, ref in zip(('input', 'cold'), outputs[:2]):
            need(result[key] == dict(path=str(Path(ref['path']).parent), pin=ref['pin']), 'FREE_GUARD_CHILD_SCORE_BINDING')
