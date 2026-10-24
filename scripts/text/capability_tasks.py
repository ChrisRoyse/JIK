"""Source-addressed learning tasks: conditioning, private targets and consequences.

Transport fragments retain their parent field. Roles and relationships are metadata,
never new senses. Nothing in this module admits an instrument or a capability.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from claims.episodic_memory import canonical, digest
from claims.process_conditions import write_committed
from claims.repository_cohort_population import fragments
from production.publication import sync_dir
from text.evaluation_multiscale import file_sha


def check(ok, code):
    if not ok:
        raise ValueError('MNLM_CAPABILITY_TASK_' + code)


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class TaskBuilder:
    def __init__(self, family, source):
        check(family in ('coding', 'conversation', 'reasoning'), 'FAMILY')
        self.family, self.source = family, source
        self.fields, self.tasks, self.edges, self.gaps = {}, {}, {}, []

    def field(self, text, role, address, *, clock=None):
        check(isinstance(text, str) and bool(text.strip()), 'EMPTY_FIELD')
        check(isinstance(role, str) and isinstance(address, dict) and address, 'FIELD_SOURCE')
        body = dict(family=self.family, role=role, text=text, text_sha256=text_sha(text),
                    source=address, clock=clock)
        key = digest(body)
        self.fields[key] = dict(id=key, **body)
        return key

    def edge(self, origin, target, kind, evidence):
        check(origin in self.fields and target in self.fields and origin != target, 'EDGE_NODES')
        check(isinstance(evidence, dict) and evidence, 'EDGE_EVIDENCE')
        body = dict(origin=origin, target=target, kind=kind, evidence=evidence)
        self.edges[digest(body)] = body

    def task(self, *, context, targets, consequences, group, kind, provenance, checks=(),
             split='development_exposed', target_origin='attested_source'):
        check(bool(context) and bool(targets), 'MISSING_CONTEXT_OR_TARGET')
        selections = (context, targets, consequences)
        check(all(len(v) == len(set(v)) and all(x in self.fields for x in v)
                  for v in selections), 'TASK_FIELDS')
        check(not (set(context) & (set(targets) | set(consequences))), 'FUTURE_LEAKAGE')
        check(not set(targets) & set(consequences), 'TARGET_CONSEQUENCE_COLLISION')
        check(isinstance(group, str) and group and provenance, 'TASK_PROVENANCE')
        body = dict(family=self.family, kind=kind, context=context, targets=targets,
                    consequences=consequences, split_group=group, partition=split,
                    provenance=provenance, checks=list(checks), target_origin=target_origin,
                    scientific_admission=False, serving=False, sealed_evaluation=False)
        key = digest(body)
        check(key not in self.tasks, 'DUPLICATE_TASK')
        self.tasks[key] = dict(id=key, **body)
        return key

    def publish(self, root, *, scope, coverage):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        units, unit_index, fields = [], {}, []
        for key, field in self.fields.items():
            parts, cursor = [], 0
            for raw in fragments(field['text'].encode()):
                text = raw.decode()
                pin = text_sha(text)
                if pin not in unit_index:
                    unit_index[pin] = len(units)
                    units.append(dict(row=len(units), chunk=0, entity=pin, text=text,
                                      start=0, end=len(text)))
                index = unit_index[pin]
                check(units[index]['text'] == text, 'HASH_COLLISION')
                parts.append(dict(unit=index, start=cursor, end=cursor + len(raw), sha256=pin))
                cursor += len(raw)
            check(cursor == len(field['text'].encode()) and parts, 'FIELD_CONSERVATION')
            fields.append(dict(field, parts=parts, fragments_are_independent_atoms=False))
        family = root / self.family
        (family / 'chunks').mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(units), family / 'chunks/atoms.parquet')
        summary = dict(chunks=len(units), chunks_sha256=file_sha(family / 'chunks/atoms.parquet'))
        write_committed(family / 'summary.json', summary)
        for name, values in (('fields', fields), ('tasks', self.tasks.values()),
                             ('edges', self.edges.values()), ('gaps', self.gaps)):
            with (root / (name + '.jsonl')).open('xb') as stream:
                for value in values:
                    stream.write(canonical(value) + b'\n')
                stream.flush()
                os.fsync(stream.fileno())
        write_committed(root / 'source.json', self.source)
        files = [self.family + '/chunks/atoms.parquet', self.family + '/summary.json',
                 'fields.jsonl', 'tasks.jsonl', 'edges.jsonl', 'gaps.jsonl', 'source.json']
        counts = dict(tasks=len(self.tasks), fields=len(fields), edges=len(self.edges),
                      unique_fragments=len(units), unique_utf8_bytes=sum(len(x['text'].encode())
                                                                        for x in units),
                      role_occurrences=dict(Counter(f['role'] for f in fields)),
                      kinds=dict(Counter(t['kind'] for t in self.tasks.values())),
                      groups=len({t['split_group'] for t in self.tasks.values()}), gaps=len(self.gaps))
        report = dict(schema='connected-capability-tasks-v1', family=self.family,
            source=self.source, counts=counts, coverage=coverage, scope=scope,
            families={self.family: dict(ready=True, summary_sha256=file_sha(family / 'summary.json'))},
            files={n: dict(bytes=(root / n).stat().st_size, sha256=file_sha(root / n)) for n in files},
            complete=True, training_ready=False, scientific_admission=False, serving=False,
            sealed_evaluation=False, native_measurement_scope='all fields; task inputs remain masked',
            code_sha256=file_sha(Path(__file__)))
        pin = write_committed(root / 'report.json', report)
        for name in [*files, 'report.json']:
            with (root / name).open('rb') as stream:
                os.fsync(stream.fileno())
        sync_dir(family / 'chunks')
        sync_dir(family)
        # COMMITTED is published after all dependent files, following the existing population protocol.
        with (root / 'COMMITTED').open('x') as stream:
            stream.write(pin)
            stream.flush()
            os.fsync(stream.fileno())
        sync_dir(root)
        return dict(report_sha256=pin, **counts)
