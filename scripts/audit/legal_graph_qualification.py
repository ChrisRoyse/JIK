"""Qualify certified-index graph access on actual full-corpus native readings."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import resource
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
from audit.legal_graph_readback import open_shard, replay, verify_scope
from audit.legal_stock_readback import rejected
from audit.native_index_reference import normalized, scores
from claims.process_conditions import write_committed
from kernel.certified_index_graph import CertifiedIndexLane, pinned, publish_shard
from kernel.legal_native_associations import BankLane
from text.constellation_corpus import digest, file_sha, need
from text.legal_native_batch import NativeBatchSession
from text.legal_reuse_input import ReuseInputs


def run(a):
    start = tick = time.perf_counter(); cpu = time.process_time()
    a.output.mkdir(parents=True, exist_ok=False)
    certificate = pinned(a.certificate, a.certificate_pin)
    lane = CertifiedIndexLane(a.certificate, a.certificate_pin)
    source = ReuseInputs(Path(lane.source['path']), lane.source['pin'])
    baseline = lane.baseline
    bank_ref = baseline['bank']; bank = NativeBatchSession(Path(bank_ref['path']), bank_ref['pin'], source)
    native = BankLane(bank, lane.name)
    ref = certificate['result']['source_receipt']; previous = pinned(Path(ref['path']), ref['pin'])
    known = {r['uid']: r for r in previous['assessment']['rows']}
    multi = lane.index.uids[lane.index.lengths > 1].tolist()
    spread = lane.index.uids[np.linspace(0, len(lane.index.uids)-1, 64, dtype=np.int64)].tolist()
    selected = sorted(set(known) | set(multi) | set(spread))
    selection = dict(uids=selected, saved_audit_queries=sorted(known), all_native_multiwindow=multi,
                     spread=spread, selection='source_address_before_scores')
    selection_pin = write_committed(a.output/'selection.json', selection)
    phases = dict(authenticate=time.perf_counter()-tick); tick = time.perf_counter(); windows = 0
    for lo in range(0, len(selected), 128):
        ids = selected[lo:lo+128]; direct = native(ids); cached = lane(ids)
        need(len(direct) == len(cached) and all(np.array_equal(x, y) for x, y in zip(direct, cached)),
             'GRAPH_QUALIFICATION_NATIVE_DIFFERENCE')
        windows += sum(map(len, cached))
    phases['ordinary_native_comparison'] = time.perf_counter()-tick; tick = time.perf_counter()
    pin = publish_shard(a.output/'graph', selected, lane)
    phases['graph_publish'] = time.perf_counter()-tick; tick = time.perf_counter()
    report, arrays = open_shard(a.output/'graph', pin, lane.binding)
    positions = {u: i for i, u in enumerate(selected)}
    for uid, old in known.items():
        i = positions[uid]
        need([[int(u), float(s)] for u, s in zip(arrays['targets'][i], arrays['scores'][i])] == old['hits'],
             'GRAPH_QUALIFICATION_SAVED_RANK')
        need(bytes(arrays['candidate_hashes'][i]).hex() == digest(np.asarray(old['candidate_uids'], dtype='<i8').tobytes()),
             'GRAPH_QUALIFICATION_SAVED_CANDIDATES')
    readback = replay(a.output/'graph', pin, lane, selected)
    readback_pin = write_committed(a.output/'readback.json', readback)
    phases['saved_results_and_independent_replay'] = time.perf_counter()-tick; tick = time.perf_counter()
    negatives = dict(wrong_certificate_pin=rejected(lambda: CertifiedIndexLane(a.certificate, '0'*64)),
                     wrong_lens=rejected(lambda: lane.index.query(selected[0], lane([selected[0]])[0], lane, lens_id='wrong')),
                     out_of_range=rejected(lambda: lane([len(lane.index.uids)])))
    need(bool(multi), 'GRAPH_QUALIFICATION_REQUIRES_MULTIWINDOW')
    uid = multi[0]; row = lane([uid])[0]
    negatives['dropped_native_window'] = rejected(lambda: lane.index.query(uid, row[:-1], lane, lens_id=lane.binding['lens_id']))
    for kind in ['self_edge', 'duplicate_edge', 'out_of_range_edge', 'nonfinite_score', 'wrong_order', 'missing_query']:
        changed = {k: v.copy() for k, v in arrays.items()}
        if kind == 'self_edge': changed['targets'][0, 0] = selected[0]
        if kind == 'duplicate_edge': changed['targets'][0, 0] = changed['targets'][0, 1]
        if kind == 'out_of_range_edge': changed['targets'][0, 0] = report['population']
        if kind == 'nonfinite_score': changed['scores'][0, 0] = np.nan
        if kind == 'wrong_order': changed['scores'][0, 0] = -1.
        if kind == 'missing_query': changed['queries'] = changed['queries'][:-1]
        negatives[kind] = rejected(lambda: verify_scope(report, changed, selected))
    negatives['wrong_graph_pin'] = rejected(lambda: open_shard(a.output/'graph', '0'*64, lane.binding))
    wrong = deepcopy(lane.binding); wrong['lens_id'] = 'wrong'
    negatives['wrong_graph_lens'] = rejected(lambda: open_shard(a.output/'graph', pin, wrong))
    changed = deepcopy(certificate); changed['complete'] = False
    invalid_path = a.output/'incomplete-certificate.json'; invalid_pin = write_committed(invalid_path, changed)
    negatives['incomplete_certificate'] = rejected(lambda: CertifiedIndexLane(invalid_path, invalid_pin))
    # Corrupt a disposable copy, never the real index or the committed graph.
    corrupt = a.output/'corrupt-copy'; corrupt.mkdir()
    for path in (a.output/'graph').iterdir():
        if path.is_file(): (corrupt/path.name).write_bytes(path.read_bytes())
    path = corrupt/'scores.npy'
    with path.open('r+b') as stream:
        stream.seek(-1, 2); old = stream.read(1); stream.seek(-1, 2); stream.write(bytes([old[0] ^ 1]))
    negatives['changed_file'] = rejected(lambda: open_shard(corrupt, pin, lane.binding))
    # A legitimate arbitrary pair outside the saved neighbour list remains readable.
    uid = selected[0]; neighbours = set(map(int, arrays['targets'][0]))
    target = next(u for u in selected[1:] if u not in neighbours)
    pair = lane.pair(uid, target); left, right = native([uid, target])
    pair_error = abs(pair['score']-float(scores(normalized([left]), normalized([right]))[0, 0]))
    need(pair_error <= 5e-13, 'GRAPH_QUALIFICATION_PAIR_SCORE')
    controls_pin = write_committed(a.output/'controls.json', dict(negative=negatives, arbitrary_pair=pair,
                                                                   independent_pair_error=pair_error))
    bank.fresh(); source.fresh(); lane.fresh(); phases['controls_and_freshness'] = time.perf_counter()-tick
    seconds = time.perf_counter()-start
    need(sum(phases.values())/seconds >= .9, 'GRAPH_QUALIFICATION_PHASE_ATTRIBUTION')
    result = dict(schema='legal-native-graph-qualification-v1', complete=True, binding=lane.binding,
                  selection_pin=selection_pin, graph_pin=pin, readback_pin=readback_pin, controls_pin=controls_pin,
                  queries=len(selected), windows_compared=windows, saved_queries_reproduced=len(known),
                  all_multiwindow_queries=len(multi), exact_native_reader_comparison=True,
                  phases_seconds=phases, seconds=seconds, cpu_seconds=time.process_time()-cpu,
                  graph_phases_seconds=report['phases_seconds'], graph_seconds=report['seconds'],
                  graph_cpu_seconds=report['cpu_seconds'], graph_native_read_seconds=report['native_read_seconds_in_queries'],
                  index_reader_stats=lane.stats, native_reader_stats=bank.stats,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  code={str(p.relative_to(Path(__file__).resolve().parents[1])): file_sha(p) for p in [
                      Path(__file__), Path(__file__).with_name('legal_graph_readback.py'),
                      Path(__file__).resolve().parents[1]/'kernel/certified_index_graph.py']},
                  source=lane.source, certificate_pin=a.certificate_pin, new_encoder_calls=0,
                  model_updates=0, scientific_admission=False, full_graph_complete=False,
                  acceptance='author-checked', number_status='measured')
    result_pin = write_committed(a.output/'report.json', result)
    print(json.dumps(dict(pin=result_pin, queries=len(selected), graph_seconds=report['seconds'], seconds=seconds)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--certificate', type=Path, required=True); p.add_argument('--certificate-pin', required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
