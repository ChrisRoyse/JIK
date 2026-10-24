"""Lossless source occurrences for the existing native-window reader.

Transport fragments retain file containment; they are not independent atoms,
task outcomes, admitted senses, or a measured kernel.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.episodic_memory import digest
from claims.repository_cohort_population import fragments, LINES, SPAN_BYTES
from oracle.git_source_slice import digest as sha, pinned, publish, require, validate_blob, validate_plan

FILES = ('repository/chunks/atoms.parquet', 'repository/summary.json', 'occurrences.json')


def code_pins():
    here = Path(__file__).parent
    return {n: sha((here/n).read_bytes()) for n in
            ('git_source_population.py', 'repository_cohort_population.py')}


class SourceSlice:
    def __init__(self, root, pin):
        self.root, self.pin = Path(root), pin
        r, _ = pinned(self.root/'report.json', pin)
        require(r['schema'] == 'git-source-slice-capture-v1' and r['complete'] and
                not r['errors'] and r['source_observations_only'] and
                not any(r[k] for k in ('repository_code_executed', 'training_ready',
                                      'scientific_admission', 'serving_changed')), 'SOURCE_SCOPE')
        require(json.loads((self.root/'COMMITTED.json').read_bytes()) ==
                dict(report_sha256=pin), 'SOURCE_MARKER')
        data = {n: pinned(self.root/(n+'.json'), p)[0] for n, p in r['inputs'].items()}
        rows = validate_plan(data['plan'], data['commit'], data['tree'])
        require(r['repository'] == data['plan']['repository'] and
                r['commit'] == data['plan']['commit'] and r['tree'] == data['plan']['tree'] and
                r['split_group'] == data['plan']['split_group'], 'SOURCE_LINEAGE')
        inventory, _ = pinned(self.root/'inventory.json', r['inventory_sha256'])
        self.raw, observed = {}, []
        for row in rows:
            path = self.root/'payload'/row['path']
            require(not path.is_symlink() and path.resolve().is_relative_to(
                    (self.root/'payload').resolve()), 'SOURCE_PATH')
            raw = path.read_bytes()
            observed.append(validate_blob(row, raw))
            raw.decode('utf-8')  # No replacement decoding or silent file exclusion.
            self.raw[row['path']] = raw
        require(observed == inventory and len(rows) == r['expected_files'] == r['acquired_files'] and
                sum(len(v) for v in self.raw.values()) == r['expected_bytes'] == r['acquired_bytes'],
                'SOURCE_INVENTORY')
        self.report, self.inventory = r, {v['path']: v for v in inventory}


def derive(source):
    rows, occurrences, by_content = [], {}, {}
    for name, raw in source.raw.items():
        parts, cursor = [], 0
        for piece in fragments(raw):
            pin = sha(piece)
            if pin not in by_content:
                by_content[pin] = len(rows)
                rows.append(dict(row=len(rows), chunk=0, entity=pin, text=piece.decode('utf-8'),
                                 start=0, end=len(piece.decode('utf-8'))))
            unit = by_content[pin]
            require(rows[unit]['text'].encode() == piece, 'SOURCE_CONTENT_COLLISION')
            parts.append(dict(unit=unit, start=cursor, end=cursor+len(piece), sha256=pin))
            cursor += len(piece)
        require(cursor == len(raw), 'SOURCE_CONSERVATION')
        occurrences[name] = dict(**source.inventory[name], parts=parts,
            repository=source.report['repository'], commit=source.report['commit'],
            tree=source.report['tree'], split_group=source.report['split_group'],
            partition='exposed_source_development', fragments_are_independent_atoms=False,
            claim_scope='pinned_artifact_contents_only', execution_outcome=None)
    return rows, occurrences


def build(root, source_root, source_pin):
    start = time.perf_counter()
    require(not root.exists() and root.resolve().is_relative_to(Path('/zfs/archive/mnlm')), 'POPULATION_DESTINATION')
    source = SourceSlice(source_root, source_pin)
    bound = time.perf_counter()
    rows, occurrences = derive(source)
    projected = time.perf_counter()
    (root/'repository/chunks').mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), root/FILES[0], compression='zstd')
    summary = publish(root/FILES[1], dict(chunks=len(rows), chunks_sha256=sha((root/FILES[0]).read_bytes())))
    publish(root/FILES[2], occurrences)
    files = {n: dict(bytes=(root/n).stat().st_size, sha256=sha((root/n).read_bytes())) for n in FILES}
    now = time.perf_counter()
    r = dict(schema='git-source-native-population-v1', source=dict(root=str(source_root), pin=source_pin),
        files=files, code=code_pins(), complete=True, errors=[], calibration_only=True,
        families=dict(repository=dict(ready=True, summary_sha256=summary)),
        counts=dict(files=len(occurrences), source_bytes=sum(len(v) for v in source.raw.values()),
                    fragments=sum(len(v['parts']) for v in occurrences.values()), unique_fragments=len(rows),
                    unique_utf8_bytes=sum(len(v['text'].encode()) for v in rows),
                    suffixes=dict(Counter(Path(n).suffix for n in occurrences)), independent_lineages=1,
                    executed_actions=0),
        fragment_policy=dict(lines=LINES, utf8_bytes=SPAN_BYTES), source_code_executed=False,
        fragments_are_independent_atoms=False, scientific_admission=False, serving=False,
        training_ready=False, kernel_recall_measured=False, acceptance='author-checked',
        phases_seconds=dict(binding=bound-start, projection=projected-bound, publication=now-projected),
        wall_seconds=now-start)
    pin = publish(root/'report.json', r)
    publish(root/'COMMITTED.json', dict(report_sha256=pin))
    print(json.dumps(dict(root=str(root), pin=pin, counts=r['counts'], wall_seconds=r['wall_seconds'])), flush=True)


class SourcePopulation:
    def __init__(self, root, pin):
        self.root, self.pin = Path(root), pin
        r, _ = pinned(self.root/'report.json', pin)
        require(r['schema'] == 'git-source-native-population-v1' and r['complete'] and not r['errors'] and
                r['code'] == code_pins() and not any(r[k] for k in
                ('source_code_executed', 'scientific_admission', 'serving', 'training_ready',
                 'kernel_recall_measured', 'fragments_are_independent_atoms')), 'POPULATION_SCOPE')
        require(json.loads((self.root/'COMMITTED.json').read_bytes()) == dict(report_sha256=pin), 'POPULATION_MARKER')
        require(set(r['files']) == set(FILES), 'POPULATION_ROSTER')
        for name, entry in r['files'].items():
            raw = (self.root/name).read_bytes()
            require(entry == dict(bytes=len(raw), sha256=sha(raw)), 'POPULATION_BYTES')
        self.source = SourceSlice(**r['source'])
        self.rows = pq.read_table(self.root/FILES[0]).to_pylist()
        self.occurrences = json.loads((self.root/FILES[2]).read_bytes())
        require((self.rows, self.occurrences) == derive(self.source), 'POPULATION_SOURCE')
        self.report = r

    def episode(self, name):
        require(name in self.occurrences, 'UNKNOWN_FILE')
        o = self.occurrences[name]
        body = dict(claim_kind='source_statement',
            content=dict(utf8_bytes=o['bytes'], text_sha256=o['sha256'], execution_outcome=None),
            source=dict(observation_set=self.pin, path=name, repository=o['repository'], commit=o['commit'],
                        git_blob=o['git_blob'], split_group=o['split_group']),
            evidence=[], event_time=None, scope='pinned_artifact_contents_only')
        return dict(id=digest(body), **body)

    def validate_episode(self, ep):
        require(ep == self.episode(ep.get('source', {}).get('path')), 'SOURCE_EPISODE')

    def assess(self, ep):
        self.validate_episode(ep)
        body = dict(checker='pinned-git-source-v1', episode=ep['id'], source=ep['source'],
                    claim_kind='source_statement', scope='pinned_artifact_contents_only', world_validity='unassessed')
        return dict(id=digest(body), **body)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--source-pin', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    build(a.output, a.source, a.source_pin)
