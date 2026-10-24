"""Freeze label-blind closure fits on finance design rows, then publish all cases."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.calyx_resolution import SEEDS, sha, write, need
from audit.calyx_closure import case


def run(a):
    start = time.perf_counter()
    report = json.loads((a.profiles/'report.json').read_bytes())
    manifest = json.loads((a.profiles/'manifest.json').read_bytes())
    need(sha(a.profiles/'report.json') == manifest['report_sha256'] and
         sha(a.profiles/'intent.json') == manifest['intent_sha256'] and report['complete'], 'FINANCE_PROFILE_PUBLICATION')
    need(report['scientific_admission'] is False and report['served'] is False, 'FINANCE_PROFILE_CLAIM')
    a.output.mkdir(parents=True, exist_ok=False)
    write(a.output/'intent.json', dict(profile_report_sha256=manifest['report_sha256'],
        fixed_split='504 design / 1008 evaluation; original population roles', seeds=SEEDS,
        prefixes=report['prefixes'], comparisons=report['comparisons'],
        controls='one exact duplicate and one fixed-label permutation per retained comparison',
        scientific_admission=False, served=False))
    rows = []
    with threadpool_limits(limits=1):
        for q, record in sorted(report['files'].items()):
            path = a.profiles/record['file']; need(sha(path) == record['sha256'], 'FINANCE_PROFILE_ARRAY')
            with np.load(path, allow_pickle=False) as data:
                values = {key: data[key].copy() for key in data.files}
            y = values['labels']; fixed = values['design'], values['evaluation']
            for prefix in report['prefixes']:
                candidate = prefix[-1]; conditioning = prefix[:-1]
                name = q+'-'+hashlib.sha256('|'.join(prefix).encode()).hexdigest()[:16]
                source = dict(fixed_profile=str(path), sha256=record['sha256'], candidate=candidate, conditioning=conditioning)
                real = case(a.output, name, values[candidate+'__profile'], y,
                            [values[n+'__profile'] for n in conditioning], SEEDS[0], source=source, fixed_split=fixed)
                # The cold readback runs exact-duplicate and fixed-label-permutation
                # controls against these same frozen coefficients and distances.
                for seed in SEEDS:
                    entry = copy.deepcopy(real); entry['name'] = name+f'-{seed}'; entry['seed'] = seed
                    rows.append(entry)
    write(a.output/'report.json', dict(complete=True, rows=rows,
        profile_report_sha256=manifest['report_sha256'], scientific_admission=False,
        conditional_information_certified=False, served=False, wall_seconds=time.perf_counter()-start,
        code_sha256={str(p.relative_to(Path(__file__).parents[1])): sha(p) for p in
            (Path(__file__), Path(__file__).with_name('calyx_closure.py'), Path(__file__).parents[1]/'text/conditional_geometry.py')}))
    write(a.output/'manifest.json', dict(report_sha256=sha(a.output/'report.json'), intent_sha256=sha(a.output/'intent.json')))
    print(json.dumps(dict(comparisons=report['comparisons'], interval_cases=len(rows))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('profiles', 'output'): p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
