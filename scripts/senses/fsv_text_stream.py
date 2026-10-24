"""Production consumer proof for retained typed text (#23); no admission claim."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
                str(Path(__file__).resolve().parents[1] / 'lens')]
from senses.contract import canonical
from senses.fsv_text_fields import real_reference
from senses.store import Store
from universal import specs
from production.input_plan import sha, code_hashes, environment, POLICY_SHA256
from production.stream import paths
from production.gate import require_admitted
from lenscommon import LensError


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    source = a.input / 'observations.sqlite'
    before = sha(source)
    reader = Store(source)
    generation = reader.head
    reader.close()
    with sqlite3.connect(source.resolve().as_uri()+'?mode=ro', uri=True) as db:
        records = [json.loads(x[0]) for x in db.execute('SELECT body FROM records ORDER BY atom')]
    expected = {r['atom_id']: np.asarray([v for item in r['channels']['behaviour']['items']
                for v in real_reference(item)], np.float32).reshape(-1, 128) for r in records}
    short = [r for r in records if expected[r['atom_id']].nbytes + 8192 < 1 << 20]
    long = max(records, key=lambda r: expected[r['atom_id']].nbytes)
    assert 1 < len(short) < len(records) and expected[long['atom_id']].nbytes > 1 << 20
    policies = a.input / 'policies.json'
    family = 'typed-text-boundary'
    spec = next(s for s in specs() if s['channel'] == 'behaviour')
    name = spec['name']
    plan = dict(schema='tier-input-plan-v1', purpose='engineering_boundary', storage='raw-f32',
                training_scope_sha256=POLICY_SHA256, generation=generation,
                policies_sha256=sha(policies), code_sha256=code_hashes(), environment=environment(),
                batch_rows=1, slot_budget_bytes=16 << 20, metadata_budget_bytes=1 << 20,
                blob_budget_bytes=256 << 20, work_budget_bytes=1 << 30,
                instruments={name: dict(spec=spec, weights_sha256=spec['weights_sha256'],
                                       tier='deterministic', slot_budget_bytes=1 << 20)}, shards={})
    rows_root = a.output / 'rows'
    rows_root.mkdir()

    def shard(key, record):
        row = {**record['locator'], 'atom': record['atom_id'], 'i': 0,
               'text': '\n'.join(x['text'] for x in record['channels']['surface']['items'])}
        path = rows_root / (key+'.jsonl')
        path.write_bytes(canonical(row)+b'\n')
        return dict(n=1, rows_sha256=sha(path), rows_bytes=path.stat().st_size,
                    family=family, tier='deterministic', instruments=[name])

    for i, record in enumerate(short):
        key = 'short-'+str(i)
        plan['shards'][key] = shard(key, record)
    too_long = copy.deepcopy(plan)
    too_long['shards'] = {'long': shard('long', long)}
    checks = []

    def check(label, ok, **details):
        checks.append(dict(name=label, passed=bool(ok), **details))

    def cli(label, chosen, output):
        planpath, receipt = a.output/(label+'.plan.json'), a.output/(label+'.receipt.json')
        planpath.write_bytes(canonical(chosen))
        result = subprocess.run([sys.executable, '-I', str(Path(__file__).resolve().parents[1]/'production/stream.py'),
                                 '--plan', str(planpath), '--store', str(source), '--policies', str(policies),
                                 '--rows', str(rows_root), '--out', str(output), '--receipt', str(receipt)],
                                capture_output=True)
        (a.output/(label+'.stdout')).write_bytes(result.stdout)
        (a.output/(label+'.stderr')).write_bytes(result.stderr)
        return result, json.loads(receipt.read_bytes()) if receipt.exists() else None

    output = a.output/'job'
    first, report = cli('first', plan, output)
    check('production_consumer_completed', first.returncode == 0 and report is not None,
          returncode=first.returncode)
    assert report is not None, first.stderr.decode(errors='replace')
    check('engineering_run_not_admission', not report['stage4_accepted'] and not report['served'])
    equal, total_rows = True, 0
    for i, record in enumerate(short):
        path, cp = paths(output, 'short-'+str(i), name)
        with np.load(path, allow_pickle=False) as z:
            target = expected[record['atom_id']]
            equal &= np.array_equal(z['tokens'], target) and z['offsets'].tolist() == [0, len(target)]
            equal &= z['absent'].tolist() == [not len(target)]
            total_rows += len(target)
        equal &= sha(path) == json.loads(cp.read_bytes())['slot']['sha256']
    check('persisted_rows_equal_independent_lexical_reference', equal, observations=len(short), rows=total_rows)
    second, repeated = cli('restart', plan, output)
    check('restart_reuses_exact_native_slots', second.returncode == 0 and repeated is not None
          and repeated['after'] == report['after'] and repeated['checkpoints_reused'] == len(short)
          and not repeated['loaded_instruments'])
    path, _ = paths(output, 'short-0', name)
    original = path.read_bytes()
    damaged = bytearray(original)
    damaged[-1] ^= 1
    path.write_bytes(damaged)
    bad, _ = cli('corrupt', plan, output)
    check('changed_checkpoint_refused', bad.returncode == 2 and b'MNLM_STREAM_CHECKPOINT' in bad.stdout)
    path.write_bytes(original)
    restored, reopened = cli('restore', plan, output)
    check('restored_checkpoint_reuses', restored.returncode == 0 and reopened['after'] == report['after'])
    long_output = a.output/'long-job'
    refused, _ = cli('over-budget', too_long, long_output)
    check('long_response_refuses_without_truncation', refused.returncode == 2 and b'MNLM_STREAM_SLOT_BUDGET' in refused.stdout,
          native_bytes=expected[long['atom_id']].nbytes, reservation_bytes=1 << 20)
    check('over_budget_publishes_no_native_slot', not list(long_output.rglob('*.npz'))
          and not list(long_output.rglob('*.checkpoint.json')) and not (long_output/'publication').exists())
    production = copy.deepcopy(plan)
    production['purpose'] = 'production'
    gated = a.output/'production-job'
    refused, _ = cli('admission-gate', production, gated)
    check('production_still_requires_admission', refused.returncode == 2
          and b'MNLM_STAGE4_ADMISSION_MISSING' in refused.stdout and not gated.exists())
    # Deliberately false certificates are negative controls, never source data.
    false_root = a.output/'false-certificate-control'
    false_root.mkdir()
    (false_root/'panel.json').write_bytes(canonical({'contract': 'universal-panel-v1', 'gates': {}}))
    code = None
    try:
        require_admitted(false_root, None)
    except LensError as exc:
        code = exc.code
    check('unadmitted_panel_refused_before_optional_import', code == 'MNLM_STAGE4_PANEL_UNADMITTED')
    if importlib.util.find_spec('joblib') is None:
        (false_root/'panel.json').write_bytes(canonical(dict(contract='universal-panel-v1',
            gates=dict(shannon_conditional_certified=True, families_served_ge3=True), lenses=[{}]*5)))
        code = None
        try:
            require_admitted(false_root, None, ["text"])
        except LensError as exc:
            code = exc.code
        check('missing_admission_runtime_has_typed_refusal', code == 'MNLM_STAGE4_RUNTIME_MISSING')
    check('original_observations_unchanged', sha(source) == before)
    result = dict(acceptance='author-checked only', scientific_admission=False,
                  passed=sum(c['passed'] for c in checks), failed=sum(not c['passed'] for c in checks),
                  checks=checks, source_sha256=before, selected_atoms=[r['atom_id'] for r in short])
    (a.output/'report.json').write_bytes(canonical(result))
    (a.output/'SHA256.json').write_bytes(canonical({p.relative_to(a.output).as_posix():sha(p)
        for p in a.output.rglob('*') if p.is_file() and not p.name.endswith(('-wal', '-shm'))}))
    print(json.dumps(result))
    return int(result['failed'] > 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    sys.exit(run(parser.parse_args()))
