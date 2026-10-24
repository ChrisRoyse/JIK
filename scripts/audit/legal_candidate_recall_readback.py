"""Declared ANN candidate profiles against already independent full-gallery ranks.

No index rebuild, threshold change, graph promotion, or new sense admission.
The existing exact rankings are development evidence reused explicitly.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import resource
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1])]
import numpy as np

from audit.legal_stock_readback import rejected
from audit.native_index_reference import normalized, scores as independent_scores
from claims.process_conditions import write_committed
from kernel.certified_index_graph import CertifiedIndexLane, pinned
from kernel.native_window_index import score_table
from text.constellation_corpus import file_sha, need

PROFILES = ((128, 256), (256, 512), (512, 1024))
CERTIFICATE = ('legal-full-specter2-base-independent-v1-20261022/report.json',
               'beb9fbc9fc8f392ca7515cc5f913b6a435017897228d5ffe737729d54063c1c2')


def code():
    root = Path(__file__).resolve().parents[1]
    return {n: file_sha(root/n) for n in ('audit/legal_candidate_recall_readback.py',
        'audit/native_index_reference.py', 'kernel/certified_index_graph.py', 'kernel/native_window_index.py')}


def query(lane, uid, count):
    need(type(count) is int and count in [p[0] for p in PROFILES], 'CANDIDATE_PROFILE_COUNT')
    native = lane([uid])[0]; candidates = set(); work = 0; visited = 0
    for window in native:
        hit = lane.index.index.search(window, count=count, threads=1)
        keys = np.asarray(hit.keys, dtype=np.uint64); owners = (keys >> np.uint64(32)).astype('<i8')
        need(len(keys) == count and np.all((owners >= 0) & (owners < len(lane.index.uids)))
             and np.all((keys & np.uint64(2**32-1)) < lane.index.lengths[owners]), 'CANDIDATE_PROFILE_KEYS')
        candidates.update(map(int, owners)); work += int(hit.computed_distances); visited += int(hit.visited_members)
    candidates.discard(uid); ids = sorted(candidates); ranked = []; maximum_error = 0.
    for lo in range(0, len(ids), 512):
        part = ids[lo:lo+512]; values = lane(part)
        actual = score_table([native], values)[0]
        reference = independent_scores(normalized([native]), normalized(values))[0]
        maximum_error = max(maximum_error, float(np.max(np.abs(actual-reference))))
        need(maximum_error <= 5e-13, 'CANDIDATE_PROFILE_SCALAR_ERROR')
        ranked.extend(zip(part, actual.tolist()))
    hits = sorted(ranked, key=lambda pair: (-pair[1], pair[0]))[:10]
    need(len(hits) == 10, 'CANDIDATE_PROFILE_SCARCITY')
    return dict(uid=uid, hits=[[u,s] for u,s in hits], candidate_uids=ids,
                native_windows=len(native), computed_distances=work, visited_members=visited,
                maximum_scalar_error=maximum_error)


def recount(rows, exact, groups):
    need([r['uid'] for r in rows] == sorted(exact), 'CANDIDATE_RECOUNT_QUERY_SCOPE')
    results = {}; all_ids = list(exact)
    for name, ids in dict(all=all_ids, **groups).items():
        members = set(ids); selected = [r for r in rows if r['uid'] in members]
        need(len(selected) == len(members) > 0, 'CANDIDATE_RECOUNT_GROUP_SCOPE')
        recovered = 0; absent = 0
        for row in selected:
            need(len(row['hits']) == 10 and len({u for u,s in row['hits']}) == 10
                 and row['uid'] not in {u for u,s in row['hits']}, 'CANDIDATE_RECOUNT_HITS')
            expected = {u for u,s in exact[row['uid']]}; actual = {u for u,s in row['hits']}
            recovered += len(expected & actual); absent += len(expected-set(row['candidate_uids']))
        total = len(selected)*10
        results[name] = dict(queries=len(selected), recovered=recovered, requested=total,
                            recall=recovered/total, missing_from_candidates=absent, recall_gate_pass=recovered/total >= .95)
    return results


def run(a):
    began = tick = time.perf_counter(); cpu = time.process_time(); a.output.mkdir(parents=True, exist_ok=False)
    for name,pin in code().items():
        dest=a.output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes())
        need(file_sha(dest)==pin,'CANDIDATE_PROFILE_CODE_COPY')
    certificate = pinned(a.root/CERTIFICATE[0], CERTIFICATE[1]); lane = CertifiedIndexLane(a.root/CERTIFICATE[0],CERTIFICATE[1])
    ref = certificate['result']['source_receipt']; receipt = pinned(Path(ref['path']),ref['pin'])
    need(receipt['index'] == certificate['result']['index'], 'CANDIDATE_PROFILE_INDEX_BINDING')
    selection = pinned(Path(ref['path']).parent/'query-selection.json',lane.index.report['binding']['query_selection_pin'])
    old = {row['uid']:row for row in receipt['assessment']['rows']}
    need(list(old) == selection['union'], 'CANDIDATE_PROFILE_SELECTION')
    exact = {uid:row['exact_hits'] for uid,row in old.items()}
    groups = {name:selection[name] for name in ['canonical','nontrain']}
    groups['canonical_multiwindow'] = [u for u in groups['canonical'] if lane.index.lengths[u] > 1]
    plan = dict(code=code(),certificate=CERTIFICATE[1],index=receipt['index'],source=ref,profiles=PROFILES,
                groups=groups,query_uids=list(exact),same_exposed_development_queries=True,threads=1)
    write_committed(a.output/'plan.json',plan)
    phases = dict(authentication=time.perf_counter()-tick); tick = time.perf_counter(); results={}; profile_rows={}
    initial_expansion = lane.index.index.expansion_search
    need(initial_expansion == 256,'CANDIDATE_PROFILE_INITIAL_RUNTIME')
    try:
        for count,expansion in PROFILES:
            start = time.perf_counter(); cpu_start=time.process_time();lane.index.index.expansion_search=expansion
            need(lane.index.index.expansion_search==expansion,'CANDIDATE_PROFILE_RUNTIME')
            rows=[query(lane,uid,count) for uid in exact]
            if count==128:
                need(all(row['hits']==old[row['uid']]['hits'] and row['candidate_uids']==old[row['uid']]['candidate_uids']
                         for row in rows),'CANDIDATE_PROFILE_BASELINE_CHANGED')
            forward=time.perf_counter()-start
            reversed_rows=[query(lane,uid,count) for uid in reversed(exact)]
            need(rows==list(reversed(reversed_rows)),'CANDIDATE_PROFILE_REPEAT_CHANGED')
            groups_report=recount(rows,exact,groups);name=f'candidates{count}-expansion{expansion}'
            output=dict(profile=dict(candidates=count,expansion=expansion),rows=rows,groups=groups_report,
                        source=ref,query_scope=list(exact),reverse_repeat_exact=True)
            pin=write_committed(a.output/(name+'.json'),output);profile_rows[name]=rows
            result=dict(pin=pin,groups=groups_report,forward_seconds=forward,seconds=time.perf_counter()-start,
                        cpu_seconds=time.process_time()-cpu_start,
                        distance_work_per_pass=sum(r['computed_distances'] for r in rows),
                        candidate_owners_per_pass=sum(len(r['candidate_uids']) for r in rows),
                        maximum_scalar_error=max(r['maximum_scalar_error'] for r in rows))
            results[name]=result;print(json.dumps(dict(profile=name,groups=groups_report,seconds=result['seconds'])),flush=True)
    finally:lane.index.index.expansion_search=initial_expansion
    phases['profiles_and_independent_recount']=time.perf_counter()-tick;tick=time.perf_counter()
    bad=deepcopy(next(iter(profile_rows.values())));bad[0]['hits'][1]=list(bad[0]['hits'][0])
    negative=dict(invalid_profile=rejected(lambda:query(lane,next(iter(exact)),0)),
                  corrupt_hits=rejected(lambda:recount(bad,exact,groups)),
                  missing_query=rejected(lambda:recount(bad[1:],exact,groups)))
    lane.fresh();need(lane.index.index.expansion_search==initial_expansion,'CANDIDATE_PROFILE_NOT_RESTORED')
    phases['controls_and_freshness']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(sum(phases.values())/seconds>=.9,'CANDIDATE_PROFILE_PHASE_ATTRIBUTION')
    result=dict(complete=True,plan=plan,profiles=results,controls=negative,phases_seconds=phases,
        seconds=seconds,cpu_seconds=time.process_time()-cpu,reader_stats=lane.stats,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        files={str(f.relative_to(a.output)):file_sha(f) for f in a.output.rglob('*') if f.is_file()},
        promoted=False,scientific_admission=False,number_status='measured',acceptance='author-checked')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',result),seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except Exception as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error),code=code()))
        raise
