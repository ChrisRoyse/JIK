"""Persist TRAIN-only complete-answer lexical and boundary controls on real tasks."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
from audit import complete_answer_baseline_source as source
from claims.process_conditions import write_committed
from model.complete_answer_baselines import CompleteAnswerBaselines, identity, validate
from text.constellation_corpus import canonical, digest, file_sha, need


def reject(fn, code):
    try: fn()
    except ValueError as e:
        need(code in str(e), 'COMPLETE_BASELINE_WRONG_REFUSAL'); return str(e)
    raise ValueError('COMPLETE_BASELINE_CONTROL_ACCEPTED_'+code)


def controls(state, model):
    row = state['bank'][0]; out = {}
    def altered(name, change, code):
        item = deepcopy(row); change(item); out[name] = reject(lambda:validate(item), code)
    altered('query_fit', lambda r:r.update(split='query'), 'COMPLETE_BASELINE_TRAIN_ONLY')
    altered('answer_text', lambda r:r['answer'].update(text=r['answer']['text']+'!'), 'COMPLETE_BASELINE_ANSWER_PIN')
    altered('atom_order', lambda r:r['answer']['atoms'][0].update(index=99), 'COMPLETE_BASELINE_ATOM_ORDER')
    altered('endpoint', lambda r:r['answer']['end'].update(byte_offset=-1), 'COMPLETE_BASELINE_END')
    altered('input_private_target', lambda r:r['public'].update(target='hidden'), 'COMPLETE_BASELINE_PUBLIC_SCHEMA')
    altered('input_text_type', lambda r:r['public']['fields'][0].update(text=None), 'COMPLETE_BASELINE_PUBLIC_SCHEMA')
    out['wrong_donor'] = reject(lambda:model.candidate('not-a-task','bm25'), 'COMPLETE_BASELINE_DONOR')
    out['query_target_fit'] = reject(lambda:state['view'].target(state['queries'][0]['task'], fitting=True), 'ANSWER_FIT_SPLIT')
    from model.capability_answer_targets import AnswerTargets
    out['source_pin'] = reject(lambda:AnswerTargets(state['view'].root, '0'*64), 'ANSWER_PUBLICATION_PIN')
    # Mutate a detached real row, never the publication. Restore and recheck the index seal.
    key = row['task']; original = model.bank[key]['answer']['text']
    model.bank[key]['answer']['text'] += '!'
    out['live_bank_change'] = reject(model.fresh, 'COMPLETE_BASELINE_INDEX_CHANGED')
    model.bank[key]['answer']['text'] = original; model.fresh()
    request = state['queries'][0]['public']; before = model.select(request)
    target = state['truth'].pop(state['queries'][0]['task'])
    try: need(model.select(request) == before, 'COMPLETE_BASELINE_PRIVATE_DEPENDENCY')
    finally: state['truth'][state['queries'][0]['task']] = target
    out['private_target_absent'] = True
    copied = model.candidate(key,'bm25'); copied['answer']['text'] += '!'; model.fresh()
    out['candidate_caller_ownership'] = True
    exact = model.select(row['public'])['exact']
    expected = {r['answer']['sha256'] for r in state['bank'] if identity(r['public']) == identity(row['public'])}
    need(exact['status'] == ('candidate' if len(expected)==1 else 'clarification_required'), 'COMPLETE_BASELINE_EXACT_CONTROL')
    out['real_exact_request'] = dict(status=exact['status'], distinct_answers=len(expected))
    ambiguous = [keys for index in model.indices.values() for keys in index['exact'].values()
                 if len({model.bank[k]['answer']['sha256'] for k in keys}) > 1]
    for keys in ambiguous:
        need(model.select(model.bank[keys[0]]['public'])['exact']['status'] == 'clarification_required',
             'COMPLETE_BASELINE_AMBIGUITY')
    need(ambiguous, 'COMPLETE_BASELINE_AMBIGUITY_CONTROL_MISSING')
    out['real_conflicting_requests'] = len(ambiguous)
    return out


def run(a):
    a.output.mkdir(exist_ok=False); began = tick = time.perf_counter(); phases = {}
    state = source.load(a.root)
    phases['source_authentication'] = time.perf_counter()-tick; tick = time.perf_counter()
    def save(name, obj): (a.output/name).write_bytes(canonical(obj))
    save('before.json', dict(code=state['code'], inputs=state['inputs'], train=319, query=170))
    save('training.json', state['bank']); save('requests.json', state['queries'])
    model = CompleteAnswerBaselines(state['bank']); model.fresh()
    phases['bank_index_publication'] = time.perf_counter()-tick; tick = time.perf_counter()
    predictions = [{n:r[n] for n in ('task','group','family')} |
                   dict(operation=r['public']['operation'], predictions=model.select(r['public'])) for r in state['queries']]
    # The physical predictions publication precedes all QUERY target scoring.
    save('predictions.json', predictions); prediction_pin = file_sha(a.output/'predictions.json')
    visits = model.visits; unique_fields = len(model._field_terms)
    phases['public_request_selection'] = time.perf_counter()-tick; tick = time.perf_counter()
    rows = [{n:r[n] for n in ('task','group','family','operation')} |
            dict(arms={name:dict(prediction=value, metrics=source.score(value,state['truth'][r['task']]))
                       for name,value in r['predictions'].items()}) for r in predictions]
    save('private-evaluation-targets.json', state['truth']); save('evaluation.json', rows)
    phases['private_target_scoring'] = time.perf_counter()-tick; tick = time.perf_counter()
    negative = controls(state,model); model.fresh(); state['fresh']()
    need(file_sha(a.output/'predictions.json') == prediction_pin, 'COMPLETE_BASELINE_PREDICTIONS_CHANGED')
    phases['controls_final_readback'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
    need(.9 <= sum(phases.values())/seconds <= 1.001, 'COMPLETE_BASELINE_PHASES')
    files = {p.name:file_sha(p) for p in a.output.iterdir()}
    value = dict(schema='complete-answer-baselines-v1', complete=True, source=dict(name=source.REF[0],pin=source.REF[1]),
        code=state['code'], inputs=state['inputs'], files=files, index_pin=model.seal, train=319, query=170,
        train_groups=len({r['group'] for r in state['bank']}), query_groups=len({r['group'] for r in state['queries']}),
        unique_training_fields=model.unique_training_fields, unique_fields_tokenized=unique_fields,
        train_term_occurrences=sum(sum(c.values()) for c in model.terms.values()),
        postings=sum(len(v) for index in model.indices.values() for v in index['postings'].values()),
        query_posting_visits=visits, summary=source.summary(rows), controls=negative,
        seconds=seconds, phases=phases, peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        gpu_used=False, native_readings_bound=False, correctness_established=False, architecture_winner=False,
        scientific_admission=False, acceptance='author-checked', exposure='previously_exposed_group_holdout')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',value), seconds=seconds, summary=value['summary'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=source.ROOT)
    p.add_argument('--output', type=Path, required=True); run(p.parse_args())
