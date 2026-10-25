"""Publish and independently recount the complete legal answer evaluation plan."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import struct
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from claims.process_conditions import write_committed
from model import legal_answer_evaluation_plan as plan
from text.constellation_corpus import file_sha, need


def code():
    base = Path(__file__).resolve().parents[1]
    return {n: file_sha(base/n) for n in (
        'model/legal_answer_evaluation_plan.py', 'audit/legal_answer_evaluation_plan.py',
        'model/legal_complete_answer_plan.py', 'text/constellation_corpus.py',
        'text/constellation_input.py', 'claims/process_conditions.py')}


def independent_prefixes(source):
    """Read scalar source addresses/counts, without the producer's prefix table."""
    result = []; train = {}
    for number, bounds in enumerate(source.documents):
        start, stop = (int(x) for x in bounds)
        observed = 1 if stop-start == 2 else 2
        uids = tuple(int(source.mapping[start+i]) for i in range(observed))
        padded = uids + (-1,)*(2-observed)
        counts = tuple(int(source.counts[u, j]) for u in uids for j in range(6))
        counts += (0,)*(12-len(counts))
        collection = source.collections.index(source.metadata[number]['source_path'])
        entry = (collection, observed, padded, counts); result.append(entry)
        if int(source.sides[number]) == 0:
            train.setdefault((collection, observed, counts), {}).setdefault(padded, []).append(number)
    # Separate grouped-prefix reconstruction of the donor population and exclusion.
    pools = {}
    for group, members in train.items():
        order = []; positions = {}
        for uids in sorted(members):
            positions[uids] = (len(order), len(members[uids])); order.extend(sorted(members[uids]))
        pools[group] = order, positions
    return result, pools


def recount(source, rows, schedule, output):
    public, pools = independent_prefixes(source)
    expected_ids = [i for i in range(len(source.documents)) if int(source.sides[i]) in (1, 2)]
    need(rows['document'].tolist() == expected_ids, 'ANSWER_EVAL_RECOUNT_ALL_CASES')
    groups = {}; donors = Counter(); layout_gaps = []; address_gaps = []
    with (output/'public-inputs.jsonl').open('x') as stream:
        for ordinal, row in enumerate(rows):
            number = expected_ids[ordinal]; collection, observed, uids, counts = public[number]
            side = int(source.sides[number]); component = int(source.components[number])
            need((int(row['side']), int(row['component']), int(row['collection']), int(row['observed']))
                 == (side, component, collection, observed), 'ANSWER_EVAL_RECOUNT_SCOPE')
            need(row['uids'].tolist() == list(uids) and tuple(row['layout'].ravel().tolist()) == counts,
                 'ANSWER_EVAL_RECOUNT_PUBLIC_NATIVE')
            order, positions = pools.get((collection, observed, counts), ([], {}))
            offset, excluded = positions.get(uids, (0, 0)); eligible = len(order)-excluded
            if eligible:
                data = (b'legal-answer-evaluation-donor-v1\0' + source.binding['corpus']['pin'].encode('ascii')
                    + struct.pack('<Q', number) + struct.pack('<qqqq12B', collection, observed, *uids, *counts))
                rank = int(hashlib.sha256(data).hexdigest(), 16) % eligible
                selected = order[rank if rank < offset else rank+excluded]; status = 0
                dcollection, dn, duids, dcounts = public[selected]
                need(int(source.sides[selected]) == 0 and int(source.components[selected]) != component
                     and (dcollection, dn, dcounts) == (collection, observed, counts) and duids != uids,
                     'ANSWER_EVAL_RECOUNT_DONOR_SCOPE')
                donors[selected] += 1
            else:
                selected = -1; duids = (-1, -1); status = 1 if not order else 2
                (layout_gaps if status == 1 else address_gaps).append(number)
            need((int(row['donor']), int(row['eligible']), int(row['status'])) == (selected, eligible, status)
                 and tuple(row['donor_uids']) == duids, 'ANSWER_EVAL_RECOUNT_SELECTION')
            reversal = observed == 2 and uids[0] != uids[1]
            need(int(row['reversal_distinct']) == int(reversal), 'ANSWER_EVAL_RECOUNT_REVERSAL')
            request = schedule.request(ordinal)
            need(request == dict(operation='continue_observed_source', position_kind='attested_document_order',
                observed_uids=uids[:observed]), 'ANSWER_EVAL_RECOUNT_REQUEST')
            donor_request = schedule.donor_request(ordinal) if status == 0 else None
            if donor_request is not None:
                need(donor_request == dict(request, observed_uids=duids[:observed]), 'ANSWER_EVAL_RECOUNT_DONOR_REQUEST')
            stream.write(json.dumps(dict(ordinal=ordinal, document=number, request=request, donor_request=donor_request,
                status=plan.STATUS[status], reversal_changes_addresses=reversal), sort_keys=True)+'\n')
            for name in (plan.source_plan.SIDES[side], plan.source_plan.SIDES[side]+':'+source.collections[collection]):
                g = groups.setdefault(name, dict(documents=0, components=set(), ready=0, layout_gaps=0,
                    distinct_prefix_gaps=0, order_controls=0, identity_order_controls=0, public_native_windows=[0]*6))
                g['documents'] += 1; g['components'].add(component)
                g[['ready', 'layout_gaps', 'distinct_prefix_gaps'][status]] += 1
                g['order_controls' if reversal else 'identity_order_controls'] += 1
                for j in range(6):g['public_native_windows'][j] += counts[j]+counts[6+j]
    for side in ('development', 'query'):
        expected = source.binding['support'][side]; g = groups[side]
        need(g['documents'] == expected['documents'] and len(g['components']) == expected['components'],
             'ANSWER_EVAL_RECOUNT_DENOMINATOR')
    for g in groups.values():g['components'] = len(g['components'])
    return dict(cases=len(rows), groups=groups, unique_train_donors=len(donors),
        maximum_donor_reuse=max(donors.values(), default=0),
        control_gaps=dict(layout=layout_gaps, distinct_prefix=address_gaps),
        complete_grid_cases=len(rows)-len(layout_gaps)-len(address_gaps),
        nominal_trajectories=len(rows)*24, no_future_used_for_donor_selection=True,
        private_generation_length_available=False, native_vector_distinctness_not_yet_checked=True,
        outcomes_scored=False, models_loaded=0, encoder_calls=0, optimizer_updates=0)


def reject(call):
    try:call()
    except ValueError as exc:return str(exc)
    raise AssertionError('ANSWER_EVAL_NEGATIVE_ACCEPTED')


def controls(source, rows, schedule, output):
    result = {}
    result['missing_case'] = reject(lambda:plan.Schedule(source, rows[:-1], schedule.public, schedule.index))
    repeated = rows.copy(); repeated[1] = repeated[0]
    result['duplicate_case'] = reject(lambda:plan.Schedule(source, repeated, schedule.public, schedule.index))
    del repeated
    i = int(np.flatnonzero(rows['status'] == 0)[0]); original = rows[i]
    changes = dict(held_out_donor=('donor', int(original['document'])), wrong_side=('side', 0),
        false_gap=('status', 1), wrong_candidates=('eligible', int(original['eligible'])+1),
        wrong_component=('component', int(original['component'])+1), wrong_source=('collection', -1))
    for name, (field, value) in changes.items():
        row = original.copy(); row[field] = value
        result[name] = reject(lambda:schedule.validate(row))
    for field in ('uids', 'donor_uids', 'layout'):
        row = original.copy(); row[field].flat[0] += 1
        result['changed_'+field] = reject(lambda:schedule.validate(row))
    gaps = np.flatnonzero(rows['status'] != 0)
    for ordinal in gaps:
        need(reject(lambda:schedule.donor_request(int(ordinal))) == 'MNLM_TEXT_CORPUS_'+plan.STATUS[int(rows[ordinal]['status'])],
             'ANSWER_EVAL_GAP_REFUSAL')
    result['all_gap_requests_refused'] = len(gaps)
    # This corpus may have no natural gaps. Exercise both refusal branches by
    # restricting copies of the real TRAIN pool; authoritative source stays intact.
    number = int(original['document']); restricted = dict(schedule.index)
    del restricted[plan.key(schedule.public[number])]
    altered = rows.copy(); altered[i] = plan.expected(source, number, schedule.public, restricted)
    missing = plan.Schedule(source, altered, schedule.public, restricted)
    result['missing_layout_pool'] = reject(lambda:missing.donor_request(i))
    need(result['missing_layout_pool'] == 'MNLM_TEXT_CORPUS_CONTROL_UNAVAILABLE_LAYOUT', 'ANSWER_EVAL_GAP_CONTROL')
    train = int(np.flatnonzero(source.sides == 0)[0]); changed_public = schedule.public.copy()
    changed_public[number] = changed_public[train]
    restricted = {plan.key(changed_public[number]): [(tuple(map(int, changed_public[train]['uids'])), train)]}
    altered[i] = plan.expected(source, number, changed_public, restricted)
    identical = plan.Schedule(source, altered, changed_public, restricted)
    result['only_identical_prefix_pool'] = reject(lambda:identical.donor_request(i))
    need(result['only_identical_prefix_pool'] == 'MNLM_TEXT_CORPUS_CONTROL_UNAVAILABLE_DISTINCT_PREFIX',
         'ANSWER_EVAL_GAP_CONTROL')
    write_committed(output/'gap-controls.json', dict(real_held_out_document=number, real_train_document=train,
        synthetic_boundary_mutations_only=True, source_unchanged=True,
        missing_layout=result['missing_layout_pool'], identical_prefix=result['only_identical_prefix_pool']))
    del altered, changed_public, restricted, missing, identical
    path = output/'changed-source-control.json'; path.write_bytes(b'{"public":1}')
    pin = file_sha(path)
    result['wrong_receipt'] = reject(lambda:source.bind(path, '0'*64))
    source.bind(path, pin); path.write_bytes(b'{"public":2}')
    result['changed_persisted_source'] = reject(lambda:source.fresh(physical=True))
    del source.stamps[path]; del source.files[str(path)]
    return result


def run(a):
    start = tick = time.perf_counter(); cpu = time.process_time(); pins = code()
    a.output.mkdir(parents=True, exist_ok=False)
    for name, pin in pins.items():
        p = a.output/'source'/name; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes())
        need(file_sha(p) == pin, 'ANSWER_EVAL_CODE_COPY')
    source = plan.authenticate(a.root); phases = dict(authentication=time.perf_counter()-tick); tick = time.perf_counter()
    public = plan.prefixes(source); index = plan.buckets(source, public)
    phases['public_prefix_index'] = time.perf_counter()-tick; tick = time.perf_counter()
    if a.previous:
        need(a.previous_pin and file_sha(a.previous/'report.json') == a.previous_pin, 'ANSWER_EVAL_PREVIOUS_PIN')
        previous = json.loads((a.previous/'report.json').read_bytes())
        need(previous['complete'] and previous['mode'] == 'publish' and previous['binding'] == source.binding
             and previous['policy'] == plan.POLICY and previous['code'] == pins, 'ANSWER_EVAL_PREVIOUS_AUTHORITY')
        for name, pin in previous['files'].items():
            need(file_sha(a.previous/name) == pin, 'ANSWER_EVAL_PREVIOUS_FILE')
        path = a.previous/'evaluation-plan.npy'
    else:
        path = a.output/'evaluation-plan.npy'; plan.build(source, public, index, path)
    rows = np.load(path, mmap_mode='r', allow_pickle=False)
    schedule = plan.Schedule(source, rows, public, index)
    phases['publication_open'] = time.perf_counter()-tick; tick = time.perf_counter()
    summary = recount(source, rows, schedule, a.output)
    phases['independent_recount_and_public_consumption'] = time.perf_counter()-tick; tick = time.perf_counter()
    negatives = controls(source, rows, schedule, a.output)
    if a.previous:
        need(summary == previous['summary'] and negatives == previous['controls']
             and file_sha(a.output/'public-inputs.jsonl') == previous['files']['public-inputs.jsonl'],
             'ANSWER_EVAL_COLD_DIFFERENCE')
    refs = dict(source.files); source.close(); rows._mmap.close()
    need(code() == pins, 'ANSWER_EVAL_CODE_CHANGED')
    phases['controls_and_final_source_readback'] = time.perf_counter()-tick; tick = time.perf_counter()
    files = {str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['output_hashes'] = time.perf_counter()-tick; seconds = time.perf_counter()-start
    need(.9 <= sum(phases.values())/seconds <= 1.001, 'ANSWER_EVAL_PHASE_ATTRIBUTION')
    report = dict(complete=True, mode='cold' if a.previous else 'publish', previous_pin=a.previous_pin,
        policy=plan.POLICY, binding=source.binding, source_plan=plan.PLAN, source_plan_cold=plan.COLD,
        code=pins, source_files=refs, files=files, plan_sha256=file_sha(path), plan_bytes=path.stat().st_size,
        summary=summary, controls=negatives, seconds=seconds, cpu_seconds=time.process_time()-cpu,
        phases_seconds=phases, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        acceptance='author-checked', number_status='measured', capability_established=False)
    pin = write_committed(a.output/'report.json', report); (a.output/'COMMITTED').write_text(pin+'\n')
    print(json.dumps(dict(pin=pin, cases=summary['cases'], ready=summary['complete_grid_cases'],
        gaps={k:len(v) for k,v in summary['control_gaps'].items()}, seconds=seconds,
        peak_rss_bytes=report['peak_rss_bytes'], plan_bytes=report['plan_bytes'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--previous', type=Path); p.add_argument('--previous-pin')
    a = p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json', dict(error=repr(exc), code=code()))
        raise
