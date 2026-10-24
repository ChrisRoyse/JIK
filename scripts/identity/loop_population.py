"""Source-bound development inputs for the intrinsic loop (#10/#28; PC-07/13).

No fabricated episodes or behavioural labels. The source's interpretations keep
their tags. These exposed correction exemplars cannot become sealed evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.contract import require
from senses.trajectory_population import publish

PROFILE = '5f839acceb76bd4bd2db269af9651b0df9e072b4a001588068434391a42ea4f6'
RECORDS = '2f2149ca502f2db2713e1262e1c4eef3fa581e8b405e9c61cedf84c12d217a13'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source, output):
    started = time.perf_counter()
    require(sha(source/'profile.md') == PROFILE and sha(source/'records.jsonl') == RECORDS,
            'MNLM_LOOP_SOURCE_PIN', 'restore the exact operator profile and sealed identity records')
    records = [json.loads(line) for line in (source/'records.jsonl').read_bytes().splitlines()]
    require(len(records) == 195 and len({r['id'] for r in records}) == 195,
            'MNLM_LOOP_SOURCE_COUNT', 'identity source roster changed')
    rows, references = [], []
    for r in records:
        fields = ('text',) if r['type'] == 'standard' else ('echo', 'response', 'not_chosen') if r['type'] == 'correction' else ()
        for field in fields:
            text = r[field]
            require(isinstance(text, str) and bool(text.strip()), 'MNLM_LOOP_EMPTY_SOURCE', r['id']+':'+field)
            index = len(rows)
            rows.append(dict(row=index, chunk=0, entity=r['id']+':'+field, text=text,
                             start=0, end=len(text)))
            references.append(dict(unit=index, record=r['id'], field=field, source_sha256=RECORDS,
                                   text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                   refs=r['refs'], source_tag=r['tag'], exposes=r.get('exposes', []),
                                   role='standard' if r['type']=='standard' else field,
                                   epistemic_scope='operator_profile_interpretation',
                                   partition='development_exposed'))
    require(len(rows) == 76, 'MNLM_LOOP_FIELDS', 'expected all ten standards and all 22 three-field corrections')
    profile = (source/'profile.md').read_bytes().decode('utf-8')
    probe = next((p for p in re.split(r'\r?\n\r?\n', profile) if 1800 <= len(p) <= 3000), None)
    require(probe is not None, 'MNLM_LOOP_REAL_PROBE', 'no exact source block for the existing overflow check')
    start = profile.index(probe)
    rows.append(dict(row=76, chunk=0, entity='profile:overflow-probe', text=probe, start=start, end=start+len(probe)))
    references.append(dict(unit=76, record='profile', field=None, source_sha256=PROFILE,
                           text_sha256=hashlib.sha256(probe.encode()).hexdigest(),
                           source_char_span=[start, start+len(probe)], role='probe_only',
                           partition='development_exposed', epistemic_scope='operator_profile_interpretation'))
    parsed = time.perf_counter()
    require(not output.exists(), 'MNLM_LOOP_EXISTS', 'publication is immutable; choose a new version')
    (output/'identity/chunks').mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), output/'identity/chunks/atoms.parquet')
    with (output/'references.jsonl').open('x', encoding='utf-8', newline='\n') as f:
        for r in references:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n')
    publish(output/'identity/summary.json', dict(chunks=len(rows), chunks_sha256=sha(output/'identity/chunks/atoms.parquet')))
    files = {p: dict(sha256=sha(output/p), bytes=(output/p).stat().st_size)
             for p in ('identity/chunks/atoms.parquet', 'identity/summary.json', 'references.jsonl')}
    wall = time.perf_counter()-started
    publish(output/'report.json', dict(schema='identity-loop-population-v1', complete=True, errors=[],
            calibration_only=True, purpose='exposed_engineering_development_not_scientific_calibration',
            families={'identity': dict(ready=True, summary_sha256=sha(output/'identity/summary.json'))},
            source_pins=dict(profile=PROFILE, records=RECORDS), units=len(rows), files=files,
            identity_components=10, corrections=22, scientific_admission=False, meaning_asserted=False,
            behavioural_compliance_established=False, served=False,
            timing=dict(read_validate_seconds=parsed-started, write_hash_seconds=wall-(parsed-started), wall_seconds=wall)))
    return dict(report_sha256=sha(output/'report.json'), units=len(rows))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    print(json.dumps(build(a.source, a.output)))
