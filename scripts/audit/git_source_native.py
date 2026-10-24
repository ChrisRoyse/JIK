"""Independent source reconstruction, native conservation and refusal readback."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.git_source_population import SourcePopulation
from claims.episodic_memory import digest
from model.contracts import ModelConfig, validate_bank
from model.git_source_input import SourceNative
from oracle.git_source_slice import digest as sha, publish, require
from senses.contract import SenseError


def reconstruct(root, pin):
    raw = (root/'report.json').read_bytes()
    require(sha(raw) == pin, 'READBACK_PIN')
    report = json.loads(raw)
    source = Path(report['source']['root'])
    source_report = (source/'report.json').read_bytes()
    require(sha(source_report) == report['source']['pin'], 'READBACK_SOURCE_PIN')
    sr = json.loads(source_report)
    inventory = json.loads((source/'inventory.json').read_bytes())
    require(sha((source/'inventory.json').read_bytes()) == sr['inventory_sha256'], 'READBACK_INVENTORY_PIN')
    rows = pq.read_table(root/'repository/chunks/atoms.parquet').to_pylist()
    occurrences = json.loads((root/'occurrences.json').read_bytes())
    require(set(occurrences) == {v['path'] for v in inventory}, 'READBACK_FILE_SET')
    used, total, spans = set(), 0, 0
    for entry in inventory:
        name = entry['path']
        o = occurrences[name]
        actual = (source/'payload'/name).read_bytes()
        git_blob = hashlib.sha1(b'blob %d\0' % len(actual)+actual).hexdigest()
        require(sha(actual) == entry['sha256'] == o['sha256'] and
                len(actual) == entry['bytes'] == o['bytes'] and git_blob == entry['git_blob'] == o['git_blob'],
                'READBACK_RAW_SOURCE')
        rebuilt, cursor = bytearray(), 0
        for part in o['parts']:
            row = rows[part['unit']]
            piece = row['text'].encode('utf-8')
            require(row['row'] == part['unit'] and row['chunk'] == 0 and row['entity'] == sha(piece) == part['sha256'] and
                    part['start'] == cursor and part['end'] == cursor+len(piece) and len(piece) <= 2048,
                    'READBACK_CONTAINMENT')
            rebuilt.extend(piece)
            cursor += len(piece)
            used.add(part['unit'])
            spans += 1
        require(bytes(rebuilt) == actual and o['repository'] == sr['repository'] and o['commit'] == sr['commit'] and
                o['tree'] == sr['tree'] and o['split_group'] == sr['split_group'] and
                o['execution_outcome'] is None and not o['fragments_are_independent_atoms'], 'READBACK_OCCURRENCE')
        total += cursor
    require(used == set(range(len(rows))), 'READBACK_UNUSED_OR_MISSING_UNIT')
    counts = dict(files=len(inventory), source_bytes=total, fragments=spans, unique_fragments=len(rows),
                  unique_utf8_bytes=sum(len(v['text'].encode()) for v in rows),
                  suffixes=dict(Counter(Path(n).suffix for n in occurrences)), independent_lineages=1, executed_actions=0)
    require(counts == report['counts'], 'READBACK_COUNTS')
    return counts


def audit(spec, output):
    start = time.perf_counter()
    counts = reconstruct(Path(spec['population']), spec['population_pin'])
    native = SourceNative(**spec)
    bound = time.perf_counter()
    controls = []
    def refuses(name, code, fn):
        try:
            fn()
        except (ValueError, SenseError) as exc:
            require(code in str(exc), 'READBACK_NEGATIVE_CODE:'+name+':'+str(exc))
        else:
            raise AssertionError('READBACK_NEGATIVE_ACCEPTED:'+name)
        controls.append(dict(name=name, code=code))
    first = next(iter(native.population.occurrences))
    refuses('false_admission', 'NATIVE_UNADMITTED', lambda:native.materialize(first, scientific_admission=True))
    refuses('source_as_success', 'NO_EXECUTION_OUTCOME', lambda:native.materialize(first, require_execution=True))
    refuses('unknown_source', 'UNKNOWN_FILE', lambda:native.materialize('../absent.rs'))
    refuses('insufficient_native_lanes', 'MNLM_MODEL_MIN_INSTRUMENTS', lambda:validate_bank(native.specs[:4]))
    ep = native.population.episode(first)
    ep['content']['execution_outcome'] = 'passed'
    refuses('forged_episode_success', 'SOURCE_EPISODE', lambda:native.population.assess(ep))
    with tempfile.TemporaryDirectory(prefix='git-source-controls-', dir=output.parent) as temp:
        copied = Path(temp)/'population'
        shutil.copytree(native.population.root, copied)
        target = copied/'repository/chunks/atoms.parquet'
        original = target.read_bytes()
        target.write_bytes(original+b'corruption')
        refuses('changed_stored_surface', 'POPULATION_BYTES', lambda:SourcePopulation(copied, spec['population_pin']))
        target.write_bytes(original)
        original_report = json.loads((copied/'report.json').read_bytes())
        for name, mutate in (
            ('omitted_span', lambda o:o[first]['parts'].pop()),
            ('wrong_lineage', lambda o:o[first].update(split_group='independent-new-project'))):
            occurrences = copy.deepcopy(native.population.occurrences)
            mutate(occurrences)
            path = copied/'occurrences.json'
            path.write_text(json.dumps(occurrences), encoding='utf-8')
            r = copy.deepcopy(original_report)
            r['files']['occurrences.json'] = dict(bytes=path.stat().st_size, sha256=sha(path.read_bytes()))
            value = json.dumps(r).encode()
            (copied/'report.json').write_bytes(value)
            pin = sha(value)
            (copied/'COMMITTED.json').write_text(json.dumps(dict(report_sha256=pin)))
            refuses(name, 'POPULATION_SOURCE', lambda:SourcePopulation(copied, pin))
    # Read persisted native arrays afresh; compare ordered byte sequences with the typed consumer.
    stored = {}
    for spec_lane in native.specs:
        with np.load(native.root/spec_lane.name/'slots.npz', allow_pickle=False) as z:
            stored[spec_lane.name] = z['vectors'], z['offsets']
    receipts, windows = {}, Counter()
    for name, occurrence in native.population.occurrences.items():
        receipt = native.receipt(name, ModelConfig(), 'cpu', native.specs)
        for lane in native.specs:
            vectors, offsets = stored[lane.name]
            pieces = [vectors[offsets[p['unit']]:offsets[p['unit']+1]] for p in occurrence['parts']]
            x = np.concatenate(pieces) if pieces else np.empty((0, lane.dims), np.float32)
            expected = dict(instrument=lane.identity, shape=list(x.shape), sha256=sha(x.astype('<f4').tobytes()),
                            dtype='float32-le', absent=not len(x))
            require(receipt['slots'][lane.name] == expected, 'READBACK_TYPED_VALUES')
            windows[lane.name] += len(x)
        receipts[name] = receipt
    native.fresh()
    finish = time.perf_counter()
    report = dict(spec=spec, counts=counts, native_readback=native.readback, occurrence_windows=dict(windows),
        receipts_sha256=digest(receipts), controls=controls, acceptance='author-checked',
        scientific_admission=False, learned_skill=False, serving=False, world_outcomes_established=0,
        phases_seconds=dict(source_and_native_binding=bound-start, controls_and_typed_readback=finish-bound),
        wall_seconds=finish-start)
    publish(output, report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('controls','native_readback','spec')}, sort_keys=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    audit(json.loads(a.spec.read_bytes()), a.output)
