"""Compare all declared whole-answer proposals using pinned, already-published scores.

No native vectors, predictors, reference targets or encoder weights are loaded.
Output is a new immutable publication. An interrupted publication is never accepted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# This process only mmaps metadata. BLAS pools have no work and compete with training.
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[name] = '1'
import numpy as np
from model.answer_architecture_comparison import Summary, compare, need

BASE = Path(__file__).resolve().parents[1]
CODE = ('model/answer_architecture_comparison.py', 'audit/legal_answer_architecture_comparison.py')
DTYPE = np.dtype([(n, '<i8') for n in
    ('document', 'side', 'component', 'collection', 'observed', 'donor', 'eligible', 'status', 'reversal_distinct')]
    + [('uids', '<i8', (2,)), ('donor_uids', '<i8', (2,)), ('layout', 'u1', (2, 6))])


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def stamp(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def memory(size):
    available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()
                     if s.startswith('MemAvailable:'))
    need(size*64+(1 << 20) <= available//4, 'PHYSICAL_MEMORY')


def child(root, name):
    need(isinstance(name, str) and bool(name) and Path(name).name == name and name not in ('.', '..'), 'PATH')
    root = Path(root).resolve(); path = root/name
    need(not path.is_symlink() and path.resolve().parent == root, 'PATH')
    return path


class Reads:
    def __init__(self):
        self.files = {}; self.bytes = 0; self.maximum_json_bytes = 0

    def load(self, path, pin):
        path = Path(path); before = stamp(path); memory(before[2]); raw = path.read_bytes()
        need(re.fullmatch('[0-9a-f]{64}', pin) and hashlib.sha256(raw).hexdigest() == pin, 'PIN')
        need(stamp(path) == before, 'INPUT_CHANGED')
        self.files[str(path)] = dict(pin=pin, stamp=before)
        self.bytes += len(raw); self.maximum_json_bytes = max(self.maximum_json_bytes, len(raw))
        return json.loads(raw)

    def array(self, path, reference):
        before = stamp(path)
        need(before[2] == reference['bytes'] and sha(path) == reference['sha256'], 'ROWS_PIN')
        need(stamp(path) == before, 'INPUT_CHANGED')
        self.files[str(path)] = dict(pin=reference['sha256'], stamp=before); self.bytes += before[2]
        return np.load(path, mmap_mode='r', allow_pickle=False)

    def fresh(self):
        need(all(stamp(Path(p)) == v['stamp'] for p,v in self.files.items()), 'INPUT_CHANGED')


def persist(path, value):
    raw = canonical(value)
    with Path(path).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


class Inputs:
    def __init__(self, score, pin, pack, scope):
        self.reads = Reads(); self.rows = None; self.score = Path(score); self.pack = Path(pack); self.models = None
        try:
            r = self.report = self.reads.load(score, pin)
            need(r['complete'] is True and r['scope'] == scope and scope in ('proof', 'all')
                 and r['all_declared_cases_accounted'] is True and r['architecture_winner'] is False
                 and r['semantic_validity_established'] is False, 'REPORT')
            self.root = child(self.score.parent, 'scores')
            b = self.binding = self.reads.load(child(self.root, 'binding.json'), r['binding'])
            need(b['schema'] == 'complete-answer-cohort-scores-v1' and b['scope'] == scope
                 and b['pack'] == r['pack'] and b['cohort'] == r['cohort'], 'BINDING')
            q = self.reads.load(child(self.pack, 'report.json'), r['pack'])
            need(q['complete'] is True and q['schema'] == 'legal-public-native-prefix-pack-v1'
                 and q['binding']['source'] == b['source'], 'PACK')
            self.roster = q['roster']; self.source = b['source']
            self.source_pin = hashlib.sha256(canonical(self.source)).hexdigest()
            need(len(self.roster) == len(set(self.roster)) == 6
                 and all(re.fullmatch('[a-z0-9-]+', n) for n in self.roster), 'ROSTER')
            self.rows = self.reads.array(child(self.pack, 'rows.npy'), q['files']['rows.npy'])
            need(self.rows.dtype == DTYPE and self.rows.ndim == 1
                 and len(self.rows) == q['summary']['cases'], 'ROWS_SCHEMA')
            ordinals = b['ordinals']
            need(ordinals and all(type(i) is int and 0 <= i < len(self.rows) for i in ordinals)
                 and ordinals == sorted(set(ordinals)), 'ORDINALS')
            need(len(r['cases']) == r['declared_cases'] == len(ordinals)
                 and [v['ordinal'] for v in r['cases']] == ordinals, 'CASE_COVERAGE')
            need(r['full_population'] == (ordinals == list(range(len(self.rows))))
                 and (scope != 'all' or r['full_population'] is True), 'FULL_COVERAGE')
            self.refs = {v['ordinal']:v for v in r['cases']}
            self.gaps = {v['ordinal']:v for v in r['control_unavailable']}
            need(len(self.gaps) == len(r['control_unavailable'])
                 and set(self.gaps) <= set(ordinals)
                 and r['scored_cases']+len(self.gaps) == r['declared_cases'], 'GAP_COVERAGE')
            for i in ordinals: self.meta(i)
        except BaseException:
            self.close(); raise

    def meta(self, ordinal):
        row = self.rows[ordinal]; side = int(row['side']); collection = int(row['collection'])
        need(side in (1, 2) and 0 <= collection < len(self.source['collections'])
             and int(row['status']) == 0 and int(row['reversal_distinct']) in (0, 1), 'ROW_IDENTITY')
        return dict(ordinal=ordinal, document=int(row['document']), component=int(row['component']),
            side=('train', 'development', 'query')[side], collection=self.source['collections'][collection],
            exposure=self.source['exposure'], reversal_distinct=bool(row['reversal_distinct']))

    def payload(self, ordinal):
        meta = self.meta(ordinal); ref = self.refs[ordinal]
        need(ref['path'] == f'case-{ordinal:06d}.json', 'CASE_PATH')
        c = self.reads.load(child(self.root, ref['path']), ref['pin'])
        need(c['binding'] == self.report['binding'] and c['ordinal'] == ordinal
             and c['document'] == meta['document'], 'COMMIT_IDENTITY')
        value = self.reads.load(child(self.root, c['payload']['path']), c['payload']['pin'])
        if ordinal in self.gaps:
            need(value == self.gaps[ordinal] == dict(status='CONTROL_UNAVAILABLE_NATIVE_IDENTICAL',
                 ordinal=ordinal, document=meta['document']), 'GAP_IDENTITY')
        else:
            need(value['identity']['source'] == self.source_pin, 'SOURCE')
            fixed = {a:{k:v for k,v in model.items() if k in
                ('model', 'target', 'roster', 'cache_bytes', 'serialized_event_bytes')}
                for a,model in value['models'].items()}
            need(set(fixed) == {'shared', 'own_slot', 'whole_proposals'}
                 and all(set(v) == {'model', 'target', 'roster', 'cache_bytes', 'serialized_event_bytes'}
                         and v['roster'] == self.roster for v in fixed.values()), 'MODELS')
            if self.models is None: self.models = fixed
            need(fixed == self.models, 'MODEL_CHANGED')
        return meta, c, value

    def case(self, ordinal):
        meta, c, value = self.payload(ordinal)
        ref = self.refs[ordinal]
        if ordinal in self.gaps:
            row = dict(meta=meta, source=self.source_pin, status=value['status'])
        else:
            row = compare(value, meta, self.roster)
        row['input_commit'] = ref['pin']; row['input_payload'] = c['payload']['pin']
        return row

    def close(self):
        if self.rows is not None: self.rows._mmap.close(); self.rows = None


def run(a):
    began = tick = time.perf_counter(); cpu = time.process_time()
    code = {name:sha(BASE/name) for name in CODE}
    data = Inputs(a.score, a.pin, a.pack, a.scope)
    try:
        phases = dict(authenticate=time.perf_counter()-tick); tick = time.perf_counter()
        summary = Summary(data.roster); refs = []
        ordinals = list(data.refs)
        if a.reverse: ordinals.reverse()
        with (a.output/'cases.jsonl').open('xb') as stream:
            for ordinal in ordinals:
                row = data.case(ordinal); summary.add(row)
                raw = canonical(row)+b'\n'; stream.write(raw)
                refs.append(dict(ordinal=ordinal, sha256=hashlib.sha256(raw).hexdigest()))
            stream.flush(); os.fsync(stream.fileno())
        phases['compare_and_persist'] = time.perf_counter()-tick; tick = time.perf_counter()
        result = summary.finish()
        need(result['cases'] == data.report['declared_cases'], 'FINAL_COVERAGE')
        summary_pin = persist(a.output/'summary.json', result)
        data.reads.fresh()
        need(all(sha(BASE/n) == p for n,p in code.items()), 'CODE_CHANGED')
        need('torch' not in sys.modules, 'UNEXPECTED_MODEL_RUNTIME')
        phases['summarize_and_authenticate'] = time.perf_counter()-tick
        seconds = time.perf_counter()-began
        need(.9 <= sum(phases.values())/seconds <= 1.001, 'TIMING_COVERAGE')
        report = dict(complete=True, schema='legal-answer-architecture-comparison-v1', code=code,
            input=dict(score=str(a.score), pin=a.pin, pack=str(a.pack), pack_pin=data.report['pack'], scope=a.scope),
            source=data.source_pin, exposure=data.source['exposure'], roster=data.roster,
            full_population=data.report['full_population'], declared_cases=result['cases'],
            scored_cases=data.report['scored_cases'], unavailable_cases=len(data.gaps),
            cases=sorted(refs, key=lambda r:r['ordinal']),
            files={n:dict(sha256=sha(a.output/n), bytes=(a.output/n).stat().st_size)
                   for n in ('cases.jsonl', 'summary.json')},
            summary_pin=summary_pin, read_inputs=data.reads.files, input_bytes=data.reads.bytes,
            models=data.models, timing_scope='through_summary_authentication_before_final_receipt',
            maximum_json_bytes=data.reads.maximum_json_bytes, phases_seconds=phases, seconds=seconds,
            cpu_seconds=time.process_time()-cpu, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            threads=int(next(s.split()[1] for s in Path('/proc/self/status').read_text().splitlines()
                             if s.startswith('Threads:'))),
            input_order='reverse' if a.reverse else 'forward', model_loads=0, encoder_calls=0,
            optimizer_updates=0, gpu_used=False, architecture_winner=False, semantic_validity_established=False)
        pin = persist(a.output/'report.json', report)
        fd = os.open(a.output, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
        return dict(complete=True, pin=pin, summary=summary_pin, cases=result['cases'], seconds=seconds)
    finally: data.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--score', type=Path, required=True); p.add_argument('--pin', required=True)
    p.add_argument('--pack', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scope', choices=('proof', 'all'), required=True); p.add_argument('--reverse', action='store_true')
    a = p.parse_args(); a.output.mkdir()
    try: print(json.dumps(run(a)), flush=True)
    except BaseException as exc:
        persist(a.output/'failure.json', dict(error=type(exc).__name__, detail=str(exc))); raise
