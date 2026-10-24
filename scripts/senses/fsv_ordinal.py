"""Persisted ordinal-history boundaries; controls are not training data (#24)."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
from senses.contract import CHANNELS, SenseError, absent, canonical, digest, field, present, seal_record
from senses.pipeline import file_hash
from senses.store import Store
from senses.views import view
from senses.instruments import measure, slot_digest
from universal import measure_universal, specs


def control(i, position, unit=None, source='ordinal-control', shard='one-sequence', ref='shared'):
    text = f'Known-truth control {i}: hidden-label; not training data.'
    r = dict(schema='ucr-v1', locator=dict(source=source, shard=shard, row=i, span=[0, len(text)],
        record_digest=digest([i, position]), manifest_digest=digest('explicit known-truth order')),
        unit=unit or f'unit-{i}', referents=[dict(key=ref, method='source_identifier', confidence=1,
        evidence='constructed boundary control only')], channels={c:absent('control_not_supplied') for c in CHANNELS})
    r['channels']['surface'] = present([dict(text=text)])
    r['channels']['fields'] = present([field('ordinal', position, 'integer', 'sequence_index')])
    return seal_record(r)


def policy(record, field_name='ordinal'):
    return dict(schema='ucr-view-v2', question='ordinal-boundary', allow_channels=list(CHANNELS),
        deny_fields=[], deny_relations=[], blocked_units=['blocked-unit'], label_aliases=['hidden-label'],
        rationale='Source-bound order control; same-unit exclusion remains mandatory; no sense admission.',
        history_clock=dict(schema='ucr-ordinal-clock-v1', field=field_name, unit='sequence_index',
            scope='source_shard', source_manifest_sha256=record['locator']['manifest_digest'],
            evidence='Retained sequence index, not an inferred wall-clock timestamp.'))


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    start, phases, checks = time.perf_counter(), {}, []
    def check(name, ok, **details):
        checks.append(dict(name=name, passed=bool(ok), **details))
        assert ok, name
    def refuses(name, fn, code):
        got = None
        try:
            fn()
        except SenseError as exc:
            got = exc.code
        check(name, got == code, code=got)
    def ingest(store, rows):
        return store.ingest(rows, dict(count=len(rows), input_sha256=digest(rows),
            kind='explicit known-truth controls, never training'), store.head)['generation']

    t = time.perf_counter()
    records = [control(0, 1), control(1, 2), control(2, 4), control(3, 5),
        control(4, 0, unit='unit-2'), control(5, 3, unit='blocked-unit'),
        control(6, 0, shard='another-clock'), control(7, 0, source='another-source'),
        control(8, 0, ref='unrelated')]
    target, p = records[2], policy(records[2])
    (a.output/'controls.jsonl').write_bytes(b''.join(canonical(r)+b'\n' for r in records))
    pp = a.output/'policies.json'
    pp.write_bytes(canonical({p['question']:p}))
    sp = a.output/'controls.sqlite'
    store = Store(sp, create=True, records_only=True)
    first = ingest(store, [target])
    before = view(store, target, p, first)[2]
    check('before_append_no_history', before['context_atoms']['history'] == [])
    second = ingest(store, records[:2]+records[3:])
    v, ctx, binding = view(store, target, p, second)
    check('strict_past_in_actual_sequence_order', [r['locator']['row'] for r in ctx['history']] == [0, 1])
    check('no_label_alias_in_any_visible_channel', b'hidden-label' not in canonical([v, ctx]))
    check('sibling_blocked_future_other_clock_excluded', len(ctx['history']) == 2)
    check('frozen_generation_replays_before_append', view(store, target, p, first)[2] == before)
    slot = measure(v, 'history', ctx['history'])
    check('history_retains_ordered_native_rows', slot['kind'] == 'Multi' and slot['dependencies'] ==
          [r['atom_id'] for r in ctx['history']])
    legacy = {k:x for k,x in p.items() if k != 'history_clock'}
    legacy['schema'] = 'ucr-view-v1'
    check('legacy_missing_wall_clock_still_absent', view(store, target, legacy, second)[1]['history'] == [])
    forged = copy.deepcopy(target)
    forged['channels']['fields']['items'][0]['value'] = 999
    refuses('direct_context_rejects_unretained_target', lambda: store.context(forged, 'history', second,
        history_clock=p['history_clock']), 'MNLM_UCR_VIEW_BINDING')
    refuses('withheld_clock_cannot_select_context', lambda: view(store, target,
        {**p, 'deny_fields':['ordinal']}, second), 'MNLM_UCR_ORDER_POLICY')
    refuses('clock_authority_must_match', lambda: view(store, target,
        {**p, 'history_clock':{**p['history_clock'], 'source_manifest_sha256':'0'*64}}, second),
        'MNLM_UCR_ORDER_BINDING')
    refuses('clock_cannot_change_coreference', lambda: store.context(target, 'co_reference', second,
        history_clock=p['history_clock']), 'MNLM_UCR_ORDER_POLICY')
    refuses('v1_cannot_silently_acquire_clock_semantics', lambda: view(store, target,
        {**p, 'schema':'ucr-view-v1'}, second), 'MNLM_UCR_POLICY')
    refuses('unchanged_context_budget_refuses', lambda: store.context(target, 'history', second,
        limit=1, history_clock=p['history_clock']), 'MNLM_UCR_CONTEXT_BUDGET')
    check('refusals_preserve_generation', store.head == second)
    store.close()
    original = file_hash(sp)
    cold = Store(sp)
    cold_view, cold_context, _ = view(cold, target, p, second)
    check('cold_native_history_reproduces', slot_digest(measure(cold_view, 'history', cold_context['history'])) == slot_digest(slot))
    cold.close()
    check('cold_queries_preserve_bytes', file_hash(sp) == original)
    (a.output/'bindings.json').write_bytes(canonical(dict(before=before, after=binding,
        first_generation=first, second_generation=second, target=target['atom_id'], source_sha256=original)))

    # Each malformed-source control is isolated and retained with its exact reason.
    for name, value, code in [('missing', None, 'MNLM_UCR_ORDER_VALUE'),
        ('wrong_unit', 'unit', 'MNLM_UCR_ORDER_VALUE'), ('negative', -1, 'MNLM_UCR_ORDER_VALUE'),
        ('duplicate_field', 'duplicate', 'MNLM_UCR_ORDER_VALUE'),
        ('equal_target', 4, 'MNLM_UCR_ORDER_AMBIGUOUS'), ('duplicate_past', 1, 'MNLM_UCR_ORDER_AMBIGUOUS'),
        ('foreign_manifest', 'manifest', 'MNLM_UCR_ORDER_BINDING')]:
        bad = control(9, 3)
        fields = bad['channels']['fields']['items']
        if value is None:
            bad['channels']['fields'] = absent('control_missing_clock')
        elif value == 'unit':
            fields[0]['unit'] = 'seconds'
        elif value == 'duplicate':
            fields.append(copy.deepcopy(fields[0]))
        elif value == 'manifest':
            bad['locator']['manifest_digest'] = digest('other authority')
        else:
            fields[0]['value'] = value
        seal_record(bad)
        bad_store = Store(a.output/(name+'.sqlite'), create=True, records_only=True)
        gen = ingest(bad_store, [records[0], target, bad])
        refuses('refuse_'+name, lambda: view(bad_store, target, p, gen), code)
        check('refusal_preserves_'+name, bad_store.head == gen)
        bad_store.close()
    phases['retention_views_and_negative_controls'] = time.perf_counter()-t

    t = time.perf_counter()
    row = {**target['locator'], 'atom':target['atom_id'], 'i':0,
           'text':target['channels']['surface']['items'][0]['text']}
    (a.output/'target.json').write_bytes(canonical(row))
    (a.output/'cal').mkdir()
    (a.output/'cal'/(p['question']+'.jsonl')).write_bytes(canonical(row)+b'\n')
    spec = next(s for s in specs() if s['channel'] == 'history')
    result = measure_universal(spec, {p['question']:[row]}, a.output/'stage3', {}, sp, pp)
    entry = result['families'][p['question']]
    with np.load(entry['slot'], allow_pickle=False) as saved:
        check('real_stage3_consumer_preserves_native_history', np.array_equal(saved['tokens'], slot['tokens'])
            and saved['offsets'].tolist() == [0, len(slot['tokens'])])
    check('stage3_contract_pins_context_resolver', result['contract']['context_code_sha256'] ==
          file_hash(Path(__file__).parent/'store.py'))
    (a.output/'stage3-result.json').write_bytes(canonical(result))
    phases['stage3_consumer'] = time.perf_counter()-t

    t = time.perf_counter()
    real_before, receipt, total, units = file_hash(a.real_store), [], 0, set()
    real = Store(a.real_store)
    queries = []
    real.db.set_trace_callback(lambda sql: queries.append(sql) if sql.startswith('SELECT DISTINCT r.atom,r.unit') else None)
    ids = [a[0] for a in real.db.execute("SELECT atom FROM records WHERE source='swe-agent-trajectories' ORDER BY atom")]
    for atom_id in ids:
        r = real.get(atom_id, real.head)
        q = policy(r, 'turn_index')
        history = real.context(r, 'history', real.head, history_clock=q['history_clock'])
        real.db.set_trace_callback(None)
        check_empty = history == []
        assert check_empty, 'real same-task label exclusion changed'
        receipt.append(dict(atom=atom_id, history_atoms=[], unit=r['unit']))
        total += 1
        units.add(r['unit'])
    query_plan = real.db.execute('EXPLAIN QUERY PLAN '+queries[0]).fetchall()
    check('real_query_visits_referent_index_before_record_primary_keys',
        'CROSS JOIN records' in queries[0] and 'SEARCH f USING COVERING INDEX' in query_plan[0][3]
        and 'SEARCH r USING INDEX sqlite_autoindex_records_1 (atom=?)' in query_plan[1][3])
    (a.output/'query-plan.json').write_bytes(canonical(dict(sql=queries[0], plan=query_plan)))
    real.close()
    check('all_real_trajectory_siblings_remain_excluded', total == 4142 and len(units) == 128,
          observations=total, units=len(units))
    check('real_store_unchanged', file_hash(a.real_store) == real_before)
    (a.output/'real-receipts.jsonl').write_bytes(b''.join(canonical(r)+b'\n' for r in receipt))
    phases['real_complete_trajectory_readback'] = time.perf_counter()-t
    report = dict(acceptance='author-checked only', scientific_admission=False,
        control_inputs_are_training_data=False, passed=sum(x['passed'] for x in checks), failed=0,
        checks=checks, phases=phases, seconds=time.perf_counter()-start,
        real_source=str(a.real_store), real_source_sha256=real_before,
        code={n:file_hash(Path(__file__).parent/n) for n in ('store.py', 'views.py', 'fsv_ordinal.py')})
    (a.output/'report.json').write_bytes(canonical(report))
    (a.output/'SHA256.json').write_bytes(canonical({p.relative_to(a.output).as_posix():file_hash(p)
        for p in a.output.rglob('*') if p.is_file() and not p.name.endswith(('-wal', '-shm'))}))
    print(json.dumps(dict(passed=report['passed'], failed=0, seconds=report['seconds'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-store', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
