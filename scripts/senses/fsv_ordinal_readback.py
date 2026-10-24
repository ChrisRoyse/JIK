"""Cold reference readback using stdlib/NumPy only; imports no MNLM code."""
import argparse
import collections
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import time

import numpy as np


def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def identity(r):
    return hashlib.sha256(canonical({k:v for k,v in r.items() if k != 'atom_id'})).hexdigest()


def reference_tokens(record):
    rows = []
    # Exact interpretation of the deliberately small known-truth control input.
    for item in record['channels']['surface']['items']:
        for word in re.findall(r'\w+|[^\w\s]', item['text']):
            rows.append([(word.casefold(), 1)])
    for item in record['channels']['fields']['items']:
        assert item['name'] == 'ordinal' and item['type'] == 'integer'
        key = '|'.join(str(item[k]) for k in ('name', 'type', 'unit', 'clock_role'))
        rows.append([(key, math.log(item['value']+1)), (key+'|observed', 1)])
    result = np.zeros((len(rows), 128), dtype=np.float32)
    for i, features in enumerate(rows):
        for name, value in features:
            h = hashlib.sha256(name.encode()).digest()
            pos = sum(h[j]*256**j for j in range(4)) % 128
            result[i, pos] += (1 if h[4] & 1 else -1)*value
    return result


def run(a):
    start = time.perf_counter()
    evidence = json.loads((a.run/'report.json').read_bytes())
    checks = []
    def check(name, ok, **details):
        checks.append(dict(name=name, passed=bool(ok), **details))
        assert ok, name
    inventory = json.loads((a.run/'SHA256.json').read_bytes())
    check('all_sealed_files_rehashed', all(sha(a.run/p) == h for p,h in inventory.items()), files=len(inventory))
    binding = json.loads((a.run/'bindings.json').read_bytes())
    p = json.loads((a.run/'policies.json').read_bytes())['ordinal-boundary']
    with sqlite3.connect((a.run/'controls.sqlite').resolve().as_uri()+'?mode=ro', uri=True) as db:
        stored = [(json.loads(body), seq) for body,seq in db.execute('SELECT body,seq FROM records')]
        previous = None
        for seq, h, body in db.execute('SELECT seq,hash,body FROM generations ORDER BY seq'):
            g = json.loads(body)
            assert hashlib.sha256(body).hexdigest() == h and g['parent'] == previous
            entries = sorted([r['atom_id'], {}] for r, n in stored if n == seq)
            assert hashlib.sha256(canonical(entries)).hexdigest() == g['entries_digest']
            assert len(entries) == g['added']
            previous = h
    check('physical_record_and_generation_identity', all(identity(r) == r['atom_id'] for r, _ in stored)
          and previous == binding['second_generation'] and len(stored) == 9)
    target = next(r for r, _ in stored if r['atom_id'] == binding['target'])
    eligible = []
    rank = lambda r: r['channels']['fields']['items'][0]['value']
    for r, _ in stored:
        if (r['unit'] != target['unit'] and r['unit'] not in p['blocked_units']
            and r['locator']['source'] == target['locator']['source'] and r['locator']['shard'] == target['locator']['shard']
            and {x['key'] for x in r['referents']} & {x['key'] for x in target['referents']} and rank(r) < rank(target)):
            eligible.append(copy.deepcopy(r))
    eligible.sort(key=rank)
    for r in eligible:
        for item in r['channels']['surface']['items']:
            item['text'] = item['text'].replace('hidden-label', '[MASKED]')
        r['atom_id'] = identity(r)
    check('independent_past_domain_unit_reconstruction', [r['locator']['row'] for r in eligible] == [0, 1]
          and [r['atom_id'] for r in eligible] == binding['after']['context_atoms']['history'])
    check('before_action_after_frozen_scope', binding['before']['context_atoms']['history'] == []
          and sum(seq == 1 for _, seq in stored) == 1 and sum(seq == 2 for _, seq in stored) == 8)
    expected = np.concatenate([reference_tokens(r) for r in eligible], axis=0)
    result = json.loads((a.run/'stage3-result.json').read_bytes())
    slot = Path(result['families'][p['question']]['slot'])
    with np.load(slot, allow_pickle=False) as z:
        check('independent_native_rows_match_stage3', np.array_equal(z['tokens'], expected)
              and z['offsets'].tolist() == [0, len(expected)] and z['absent'].tolist() == [False], rows=len(expected))
    check('recorded_negative_boundaries_passed', all(x['passed'] for x in evidence['checks'])
          and any(x['name'] == 'refuse_duplicate_past' for x in evidence['checks']))
    check('retained_controls_unchanged', sha(a.run/'controls.sqlite') == binding['source_sha256'])
    real = Path(evidence['real_source'])
    check('real_source_pin', sha(real) == evidence['real_source_sha256'])
    receipts = [json.loads(line) for line in (a.run/'real-receipts.jsonl').read_bytes().splitlines()]
    rows, units, referents = {}, set(), collections.defaultdict(set)
    with sqlite3.connect(real.resolve().as_uri()+'?mode=ro', uri=True) as db:
        for (body,) in db.execute("SELECT body FROM records WHERE source='swe-agent-trajectories'"):
            r = json.loads(body)
            assert r['atom_id'] == identity(r)
            rows[r['atom_id']] = r['unit']
            units.add(r['unit'])
            for ref in r['referents']:
                referents[ref['key']].add(r['unit'])
    check('all_real_exclusions_reconstructed_from_bytes', len(rows) == len(receipts) == 4142 and len(units) == 128
          and all(len(u) == 1 for u in referents.values()) and len(referents) == 128
          and {r['atom']:r['unit'] for r in receipts} == rows and all(r['history_atoms'] == [] for r in receipts))
    check('no_scientific_promotion', evidence['scientific_admission'] is False
          and evidence['control_inputs_are_training_data'] is False)
    out = dict(acceptance='author-checked separate reference reader; no independent-session acceptance',
        passed=len(checks), failed=0, checks=checks, seconds=time.perf_counter()-start,
        source_manifest_sha256=sha(a.run/'SHA256.json'), reader_sha256=sha(__file__))
    a.output.write_bytes(canonical(out))
    print(json.dumps(dict(passed=len(checks), failed=0, seconds=out['seconds'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args())
