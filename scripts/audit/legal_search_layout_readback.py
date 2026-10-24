"""Real singleton/batch ANN result-layout proof without modifying live graph code."""
import argparse
import copy
import fcntl
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import usearch.index

from audit.native_index_reference import normalized, verify_candidates
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from kernel.certified_index_graph import CertifiedIndexLane, pinned
from kernel.native_graph_batch import query_batch
from kernel.native_search_layout import batch_layout
from text.constellation_corpus import digest, file_sha, need


class MatrixSearchView:
    """Proof-only view: use the candidate normalizer for matrix queries only."""
    def __init__(self, original):
        self.original = original
        self.receipts = []

    def __len__(self):
        return len(self.original)

    def __getattr__(self, name):
        return getattr(self.original, name)

    def search(self, values, *, count, threads):
        raw = self.original.search(values, count=count, threads=threads)
        if values.ndim == 1:
            return raw
        result = batch_layout(raw, len(values), count)
        self.receipts.append(dict(native_windows=len(values), raw_type=type(raw).__name__,
            raw_shape=list(raw.keys.shape), normalized_shape=list(result.keys.shape),
            raw_keys=digest(raw.keys.tobytes()), normalized_keys=digest(result.keys.tobytes()),
            raw_distances=digest(raw.distances.tobytes()), normalized_distances=digest(result.distances.tobytes()),
            raw_work=[raw.visited_members, raw.computed_distances],
            normalized_work=[result.visited_members, result.computed_distances]))
        return result


def code():
    scripts = Path(__file__).resolve().parents[1]
    names = ['audit/legal_search_layout_readback.py', 'kernel/native_search_layout.py',
             'kernel/native_graph_batch.py', 'kernel/certified_index_graph.py',
             'kernel/native_window_index.py', 'audit/native_index_reference.py']
    return {name: file_sha(scripts/name) for name in names}


def graph_digest(row):
    return digest(json.dumps({k: row[k] for k in ['uid', 'hits', 'candidate_uids']}, sort_keys=True).encode())


def run(a):
    began = tick = time.perf_counter(); cpu = time.process_time()
    lease_path = Path(__file__).resolve().parents[2]/'.lease/legal-search-layout'
    lease_path.parent.mkdir(exist_ok=True); lease = lease_path.open('a+')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    a.output.mkdir(parents=True, exist_ok=False)
    q = pinned(a.qualification/'report.json', a.qualification_pin)
    need(q['complete'], 'LAYOUT_NATIVE_QUALIFICATION')
    ref = q['binding']['certificate']; lane = CertifiedIndexLane(Path(ref['path']), ref['pin'])
    native_selection = pinned(a.qualification/'selection.json', q['selection_pin'])
    full_qualification = sorted(set(range(512)) | set(native_selection['uids']))
    failed = full_qualification[-1]
    need(len(full_qualification) == 769 and len(full_qualification) % 64 == 1
         and lane.index.lengths[failed] == 1, 'LAYOUT_FAILURE_POPULATION_CHANGED')
    single = next(u for u in native_selection['uids'] if lane.index.lengths[u] == 1 and u != failed)
    multi = next(u for u in native_selection['uids'] if lane.index.lengths[u] > 1)
    scope = list(dict.fromkeys([failed, single, multi]+native_selection['uids']))[:64]
    cases = dict(failing_singleton=[failed], other_singleton=[single], multiwindow_atom=[multi],
                 pair=[failed, single], mixed_pair=[failed, multi], full_tile=scope)
    selected = sorted({u for ids in cases.values() for u in ids})
    selection = dict(cases=cases, uids=selected, qualification_queries=len(full_qualification),
                     failed_uid=failed, source_selection_pin=q['selection_pin'])
    write_committed(a.output/'selection.json', selection)
    primary = '\n\n'.join(inspect.getsource(x) for x in
                          [usearch.index._search_in_compiled, usearch.index.Matches, usearch.index.BatchMatches])
    (a.output/'installed-search-source.txt').write_text(primary)
    binding = dict(native_qualification=dict(path=str(a.qualification), pin=a.qualification_pin),
                   lane=lane.binding, code=code(), usearch_version=importlib.metadata.version('usearch'),
                   installed_source=file_sha(Path(inspect.getfile(usearch.index))),
                   selection=digest(json.dumps(selection, sort_keys=True).encode()))
    previous = None
    if a.previous:
        previous = pinned(a.previous/'report.json', a.previous_pin)
        need(previous['complete'] and previous['binding'] == binding, 'LAYOUT_COLD_BINDING')
        for name, item in previous['files'].items():
            need(file_sha(a.previous/name) == item['sha256']
                 and (a.previous/name).stat().st_size == item['bytes'], 'LAYOUT_COLD_FILE')
    phases = dict(authentication=time.perf_counter()-tick); tick = time.perf_counter()
    original = lane.index.index
    values = lane([failed])[0]
    raw = original.search(values, count=128, threads=1)
    need(type(raw) is usearch.index.Matches and raw.keys.shape == (128,), 'LAYOUT_REPRODUCTION')
    failure = rejected(lambda: query_batch(lane, [failed], threads=1))
    need('GRAPH_BATCH_SEARCH_LAYOUT' in str(failure), 'LAYOUT_WRONG_ORIGINAL_FAILURE')
    np.savez(a.output/'original-singleton.npz', values=values, keys=raw.keys, distances=raw.distances)
    before = dict(uid=failed, input_shape=list(values.shape), raw_type=type(raw).__name__,
                  raw_shape=list(raw.keys.shape), has_counts=hasattr(raw, 'counts'), failure=failure,
                  work=[raw.visited_members, raw.computed_distances])
    write_committed(a.output/'before.json', before)
    refs = {}; native = a.output/'native'; native.mkdir()
    for uid in (list(reversed(selected)) if previous else selected):
        row = lane.query(uid)
        source = normalized(lane([uid]))[0]
        error = verify_candidates(row, source, lane)
        np.save(native/f'{uid}.npy', lane([uid])[0], allow_pickle=False)
        refs[uid] = dict(row=row, graph_digest=graph_digest(row), independent_error=error)
    write_committed(a.output/'scalar-reference.json', {str(k): refs[k] for k in sorted(refs)})
    phases['reproduce_and_reference'] = time.perf_counter()-tick; tick = time.perf_counter()
    view = MatrixSearchView(original); lane.index.index = view
    results = []; signatures = {}
    try:
        labels = list(cases)
        if previous: labels.reverse()
        for threads in [1, 4, 16, 32]:
            for label in labels:
                for reverse in [False, True]:
                    ids = cases[label][::-1] if reverse else cases[label]
                    before_count = len(view.receipts)
                    result = query_batch(lane, ids, threads=threads)
                    need(len(view.receipts) == before_count+1, 'LAYOUT_SEARCH_CALLS')
                    receipt = view.receipts[-1]
                    need(receipt['raw_keys'] == receipt['normalized_keys']
                         and receipt['raw_distances'] == receipt['normalized_distances']
                         and receipt['raw_work'] == receipt['normalized_work'], 'LAYOUT_CHANGED_SEARCH_RESULT')
                    for row in result['rows']:
                        need(graph_digest(row) == refs[row['uid']]['graph_digest'], 'LAYOUT_CHANGED_GRAPH_RESULT')
                    need(result['work']['computed_distances'] == receipt['raw_work'][1]
                         and result['work']['visited_members'] == receipt['raw_work'][0], 'LAYOUT_CHANGED_WORK')
                    key=f'{threads}-{label}-{int(reverse)}'
                    signatures[key] = dict(rows=[graph_digest(r) for r in result['rows']], search=receipt)
                    results.append(dict(key=key, result=result, search=receipt))
    finally:
        lane.index.index = original
    write_committed(a.output/'batches.json', results)
    phases['batch_action'] = time.perf_counter()-tick; tick = time.perf_counter()
    controls = {}
    controls['wrong_result_type'] = rejected(lambda: batch_layout({}, 1, 128))
    controls['wrong_singleton_cardinality'] = rejected(lambda: batch_layout(raw, 2, 128))
    controls['wrong_requested_count'] = rejected(lambda: batch_layout(raw, 1, 127))
    for kind in ['wrong_key_dtype', 'wrong_key_shape', 'nonfinite_distance', 'missing_distance', 'zero_work']:
        bad = copy.deepcopy(raw)
        if kind == 'wrong_key_dtype': bad.keys = bad.keys.astype(np.int64)
        if kind == 'wrong_key_shape': bad.keys = bad.keys[None, :]
        if kind == 'nonfinite_distance': bad.distances[0] = np.nan
        if kind == 'missing_distance': bad.distances = bad.distances[:-1]
        if kind == 'zero_work': bad.computed_distances = 0
        controls[kind] = rejected(lambda: batch_layout(bad, 1, 128))
    real_batch = original.search(np.concatenate(lane([failed, single])), count=128, threads=1)
    for kind in ['wrong_batch_count', 'wrong_batch_cardinality']:
        bad = copy.deepcopy(real_batch)
        if kind == 'wrong_batch_count': bad.counts[0] -= 1
        if kind == 'wrong_batch_cardinality': bad.keys = bad.keys[:1]
        controls[kind] = rejected(lambda: batch_layout(bad, 2, 128))
    for kind in ['unknown_owner', 'unknown_window']:
        bad = copy.deepcopy(raw)
        if kind == 'unknown_owner': bad.keys[0] = np.uint64(2**64-1)
        else: bad.keys[0] = (bad.keys[0] & np.uint64(0xffffffff00000000)) | np.uint64(2**32-1)
        class MutatedResultView(MatrixSearchView):
            def search(self, values, *, count, threads):
                return batch_layout(bad, len(values), count)
        lane.index.index = MutatedResultView(original)
        try: controls[kind] = rejected(lambda: query_batch(lane, [failed], threads=1))
        finally: lane.index.index = original
    if previous:
        need(previous['signatures'] == signatures and previous['before'] == before
             and previous['controls'] == controls, 'LAYOUT_COLD_RESULT')
        need(file_sha(a.output/'scalar-reference.json') == previous['files']['scalar-reference.json']['sha256'],
             'LAYOUT_COLD_SCALAR_REFERENCE')
    lane.fresh(); need(code() == binding['code'], 'LAYOUT_CODE_CHANGED')
    phases['controls_and_readback'] = time.perf_counter()-tick; tick = time.perf_counter()
    files = {str(f.relative_to(a.output)): dict(bytes=f.stat().st_size, sha256=file_sha(f))
             for f in sorted(a.output.rglob('*')) if f.is_file()}
    phases['publish'] = time.perf_counter()-tick
    seconds = time.perf_counter()-began
    need(sum(phases.values())/seconds >= .9, 'LAYOUT_PHASE_COVERAGE')
    report = dict(schema='native-search-layout-readback-v1', complete=True, binding=binding,
                  previous_pin=a.previous_pin, before=before, signatures=signatures, controls=controls,
                  files=files, queries=len(selected), batches=len(results), all_candidates_and_scores_exact=True,
                  all_search_keys_distances_work_unchanged=True, phases_seconds=phases, seconds=seconds,
                  cpu_seconds=time.process_time()-cpu, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  acceptance='author-checked', number_status='measured', model_updates=0, scientific_admission=False,
                  live_graph_code_modified=False)
    pin = write_committed(a.output/'report.json', report)
    write_committed(a.output/'COMMITTED', dict(report_sha256=pin))
    print(json.dumps(dict(complete=True, pin=pin, queries=len(selected), batches=len(results), seconds=seconds)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qualification', type=Path, required=True)
    p.add_argument('--qualification-pin', required=True)
    p.add_argument('--previous', type=Path); p.add_argument('--previous-pin')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        run(args)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write_committed(args.output/('failure-'+str(time.time_ns())+'.json'),
                        dict(complete=False, error=repr(error), code=code()))
        raise
