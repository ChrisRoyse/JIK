"""Pair saved whole-answer scores with authenticated exact-TRAIN baseline scores."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.legal_answer_architecture_comparison import (
    BASE, Inputs, canonical, child, memory, need, persist, sha, stamp)
from model.answer_baseline_comparison import BASELINE_POLICY, BaselineSummary, compare

CODE = ('model/answer_architecture_comparison.py', 'audit/legal_answer_architecture_comparison.py',
        'model/answer_baseline_comparison.py', 'audit/legal_answer_baseline_comparison.py')


class Baselines:
    """One sequential authentication scan, compact offsets, one selected JSON at a time."""
    def __init__(self, data, report, pin, readback, readback_pin):
        self.data = data; self.stream = None; self.index = []
        reads = data.reads
        try:
            r = self.report = reads.load(report, pin)
            need(r['complete'] is True and r['mode'] == 'full'
                 and r['source'] == data.source and r['roster'] == data.roster
                 and r['semantic_validity_established'] is False, 'BASELINE_REPORT')
            b = reads.load(child(Path(report).parent, 'binding.json'), r['binding'])
            need(b['source'] == data.source and b['mode'] == 'full' and b['policy'] == BASELINE_POLICY,
                 'BASELINE_POLICY')
            documents = [int(row['document']) for row in data.rows]
            need(r['documents'] == b['documents'] == documents, 'BASELINE_COVERAGE')
            q = reads.load(readback, readback_pin)
            need(q['complete'] is True and q['producer'] == pin and q['mode'] == 'full'
                 and q['source'] == data.source and q['documents'] == documents
                 and all(q[k] is True for k in ('all_native_exact', 'all_scores_exact', 'summary_exact')),
                 'BASELINE_READBACK')
            path = self.path = child(Path(report).parent, 'cases.jsonl'); before = stamp(path)
            self.stream = path.open('rb'); digest = hashlib.sha256(); offset = 0
            memory(len(documents)*128)
            # A malformed, unterminated line must not allocate the entire population.
            available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()
                             if s.startswith('MemAvailable:'))
            maximum = max(1, available//256)
            while raw := self.stream.readline(maximum+1):
                need(len(raw) <= maximum and raw.endswith(b'\n'), 'BASELINE_LINE_MEMORY')
                need(len(self.index) < len(documents), 'BASELINE_COVERAGE')
                digest.update(raw)
                self.index.append((offset, len(raw), hashlib.sha256(raw).hexdigest()))
                offset += len(raw)
            need(digest.hexdigest() == r['cases_pin'], 'BASELINE_CASES_PIN')
            need(len(self.index) == len(documents), 'BASELINE_COVERAGE')
            need(stamp(path) == before, 'INPUT_CHANGED')
            reads.files[str(path)] = dict(pin=r['cases_pin'], stamp=before); reads.bytes += offset
        except BaseException:
            self.close(); raise

    def case(self, ordinal):
        offset, length, pin = self.index[ordinal]; memory(length)
        self.stream.seek(offset); raw = self.stream.read(length)
        need(hashlib.sha256(raw).hexdigest() == pin, 'BASELINE_LINE_PIN')
        self.data.reads.bytes += len(raw)
        self.data.reads.maximum_json_bytes = max(self.data.reads.maximum_json_bytes, len(raw))
        return json.loads(raw), pin

    def close(self):
        if self.stream is not None: self.stream.close(); self.stream = None


def run(a):
    began = tick = time.perf_counter(); cpu = time.process_time(); code = {n:sha(BASE/n) for n in CODE}
    data = Inputs(a.score, a.pin, a.pack, a.scope); baseline = None
    try:
        baseline = Baselines(data, a.baseline, a.baseline_pin, a.baseline_readback, a.readback_pin)
        phases = dict(authenticate_and_index=time.perf_counter()-tick); tick = time.perf_counter()
        summary = BaselineSummary(data.roster); refs = []; ordinals = list(data.refs)
        if a.reverse: ordinals.reverse()
        with (a.output/'cases.jsonl').open('xb') as stream:
            for ordinal in ordinals:
                meta, commit, payload = data.payload(ordinal); b, pin = baseline.case(ordinal)
                row = compare(payload, b, meta, data.roster, data.source_pin)
                row.update(input_commit=data.refs[ordinal]['pin'], input_payload=commit['payload']['pin'],
                           baseline_line=pin)
                summary.add(row); raw = canonical(row)+b'\n'; stream.write(raw)
                refs.append(dict(ordinal=ordinal, sha256=hashlib.sha256(raw).hexdigest()))
            stream.flush(); os.fsync(stream.fileno())
        phases['compare_and_persist'] = time.perf_counter()-tick; tick = time.perf_counter()
        result = summary.finish()
        need(result['cases'] == data.report['declared_cases'], 'FINAL_COVERAGE')
        summary_pin = persist(a.output/'summary.json', result); data.reads.fresh()
        need(all(sha(BASE/n) == p for n,p in code.items()), 'CODE_CHANGED')
        need('torch' not in sys.modules, 'UNEXPECTED_MODEL_RUNTIME')
        phases['summarize_and_authenticate'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
        need(.9 <= sum(phases.values())/seconds <= 1.001, 'TIMING_COVERAGE')
        report = dict(complete=True, schema='legal-answer-baseline-comparison-v1', code=code,
            input=dict(score=str(a.score), pin=a.pin, pack=str(a.pack), pack_pin=data.report['pack'], scope=a.scope,
                baseline=str(a.baseline), baseline_pin=a.baseline_pin,
                baseline_readback=str(a.baseline_readback), readback_pin=a.readback_pin),
            source=data.source_pin, exposure=data.source['exposure'], roster=data.roster,
            full_population=data.report['full_population'], declared_cases=result['cases'],
            scored_cases=data.report['scored_cases'], unavailable_cases=len(data.gaps), models=data.models,
            stratum_cases=result['stratum_cases'], cases=sorted(refs, key=lambda r:r['ordinal']),
            files={n:dict(sha256=sha(a.output/n), bytes=(a.output/n).stat().st_size)
                   for n in ('cases.jsonl', 'summary.json')}, summary_pin=summary_pin,
            read_inputs=data.reads.files, input_bytes=data.reads.bytes,
            maximum_json_bytes=data.reads.maximum_json_bytes,
            baseline_index_rows=len(baseline.index), baseline_jsonl_bytes=baseline.path.stat().st_size,
            phases_seconds=phases, seconds=seconds, cpu_seconds=time.process_time()-cpu,
            timing_scope='through_summary_authentication_before_final_receipt',
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            threads=int(next(s.split()[1] for s in Path('/proc/self/status').read_text().splitlines()
                             if s.startswith('Threads:'))),
            input_order='reverse' if a.reverse else 'forward', model_loads=0, encoder_calls=0,
            optimizer_updates=0, gpu_used=False, architecture_winner=False, semantic_validity_established=False)
        pin = persist(a.output/'report.json', report)
        fd = os.open(a.output, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
        return dict(complete=True, pin=pin, summary=summary_pin, cases=result['cases'], seconds=seconds)
    finally:
        if baseline is not None: baseline.close()
        data.close()


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('score', 'pack', 'output', 'baseline', 'baseline-readback'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('pin', 'baseline-pin', 'readback-pin'): p.add_argument('--'+name, required=True)
    p.add_argument('--scope', choices=('proof', 'all'), required=True)
    p.add_argument('--reverse', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    a = arguments(); a.output.mkdir()
    try: print(json.dumps(run(a)), flush=True)
    except BaseException as exc:
        persist(a.output/'failure.json', dict(error=type(exc).__name__, detail=str(exc))); raise
