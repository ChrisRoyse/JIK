"""Actual bounded streaming consumer for ordinal history; no admission claim."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
from senses.contract import canonical
from production.input_plan import sha, code_hashes, environment, POLICY_SHA256
from production.stream import paths
from universal import specs


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    checks = []
    def check(name, ok):
        checks.append(dict(name=name, passed=bool(ok)))
        assert ok, name
    source, policies = a.input/'controls.sqlite', a.input/'policies.json'
    before = sha(source)
    bound = json.loads((a.input/'bindings.json').read_bytes())
    spec = next(s for s in specs() if s['channel'] == 'history')
    name, key = spec['name'], 'ordered-history'
    rows = a.output/'rows'
    rows.mkdir()
    rp = rows/(key+'.jsonl')
    rp.write_bytes((a.input/'target.json').read_bytes()+b'\n')
    plan = dict(schema='tier-input-plan-v1', purpose='engineering_boundary', storage='raw-f32',
        training_scope_sha256=POLICY_SHA256, generation=bound['second_generation'], policies_sha256=sha(policies),
        code_sha256=code_hashes(), environment=environment(), batch_rows=1, slot_budget_bytes=16 << 20,
        metadata_budget_bytes=1 << 20, blob_budget_bytes=256 << 20, work_budget_bytes=1 << 30,
        instruments={name:dict(spec=spec, weights_sha256=spec['weights_sha256'], tier='deterministic', slot_budget_bytes=1 << 20)},
        shards={key:dict(n=1, rows_sha256=sha(rp), rows_bytes=rp.stat().st_size, family='ordinal-boundary',
                        tier='deterministic', instruments=[name])})
    def cli(label, chosen, dest):
        pp, receipt = a.output/(label+'.plan.json'), a.output/(label+'.receipt.json')
        pp.write_bytes(canonical(chosen))
        r = subprocess.run([sys.executable, '-I', str(Path(__file__).resolve().parents[1]/'production/stream.py'),
            '--plan', str(pp), '--store', str(source), '--policies', str(policies), '--rows', str(rows),
            '--out', str(dest), '--receipt', str(receipt)], capture_output=True)
        (a.output/(label+'.stdout')).write_bytes(r.stdout)
        (a.output/(label+'.stderr')).write_bytes(r.stderr)
        return r, json.loads(receipt.read_bytes()) if receipt.exists() else None
    dest = a.output/'job'
    r, first = cli('first', plan, dest)
    check('actual_stream_completed', r.returncode == 0 and first is not None)
    check('engineering_is_not_stage4_acceptance', not first['stage4_accepted'] and not first['served'])
    path, cp = paths(dest, key, name)
    result = json.loads((a.input/'stage3-result.json').read_bytes())
    with np.load(result['families']['ordinal-boundary']['slot'], allow_pickle=False) as reference:
        with np.load(path, allow_pickle=False) as stored:
            check('native_rows_match_cold_checked_stage3', all(np.array_equal(stored[n], reference[n])
                for n in ('tokens', 'offsets', 'absent')))
    check('checkpoint_matches_physical_bytes', sha(path) == json.loads(cp.read_bytes())['slot']['sha256'])
    r, repeat = cli('restart', plan, dest)
    check('restart_reuses_without_loading_instruments', r.returncode == 0 and repeat['after'] == first['after']
          and repeat['checkpoints_reused'] == 1 and not repeat['loaded_instruments'])
    wrong = copy.deepcopy(plan)
    wrong['code_sha256']['senses/store.py'] = '0'*64
    blocked = a.output/'drift-job'
    r, _ = cli('resolver-drift', wrong, blocked)
    check('resolver_drift_refuses_before_output', r.returncode == 2 and b'MNLM_STREAM_CODE_DRIFT' in r.stdout
          and not blocked.exists())
    raw = policies.read_bytes()
    changed = json.loads(raw)
    changed['ordinal-boundary']['history_clock']['field'] = 'unapproved-order'
    policies.write_bytes(canonical(changed))
    try:
        r, _ = cli('policy-drift', plan, dest)
        check('changed_clock_policy_cannot_reuse_checkpoint', r.returncode == 2 and b'MNLM_STREAM_INPUT' in r.stdout)
    finally:
        policies.write_bytes(raw)
    production = copy.deepcopy(plan)
    production['purpose'] = 'production'
    blocked = a.output/'production-job'
    r, _ = cli('production-gate', production, blocked)
    check('production_admission_still_required', r.returncode == 2 and b'MNLM_STAGE4_ADMISSION_MISSING' in r.stdout
          and not blocked.exists())
    check('source_and_policy_restored_byte_exact', sha(source) == before and sha(policies) == plan['policies_sha256'])
    report = dict(acceptance='author-checked only', scientific_admission=False, checks=checks, passed=len(checks), failed=0)
    (a.output/'report.json').write_bytes(canonical(report))
    (a.output/'SHA256.json').write_bytes(canonical({p.relative_to(a.output).as_posix():sha(p)
        for p in a.output.rglob('*') if p.is_file() and not p.name.endswith(('-wal', '-shm'))}))
    print(json.dumps(dict(passed=len(checks), failed=0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args())
