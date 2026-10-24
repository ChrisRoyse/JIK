"""Independent flat-bank reconstruction of complete-answer baseline selections."""
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
from audit import complete_answer_baseline_source as source
from claims.process_conditions import write_committed
from text.constellation_corpus import canonical, digest, file_sha, need


def tokens(public):
    count = Counter()
    for field in public['fields']: count.update(re.findall(r'\w+|[^\w\s]',field['text'].casefold()))
    return count


def prepare(bank):
    result = {}
    for operation in {r['public']['operation'] for r in bank}:
        rows = sorted((r for r in bank if r['public']['operation'] == operation), key=lambda r:r['task'])
        counts = [tokens(r['public']) for r in rows]; lengths = [sum(c.values()) for c in counts]; df = Counter()
        for c in counts: df.update(c.keys())
        result[operation] = rows, counts, lengths, df
    return result


def reconstruct(prepared, request):
    # Deliberately no producer tokenizer, posting index, candidate constructor or selector.
    candidates, counts, document_lengths, frequencies = prepared[request['operation']]
    query = sorted(tokens(request)); average = sum(document_lengths)/len(counts); scores = []
    for count, length in zip(counts, document_lengths):
        score = []
        for term in query:
            if term in count:
                df = frequencies[term]; frequency = count[term]
                score.append(math.log1p((len(counts)-df+.5)/(df+.5))*frequency*2.2 /
                             (frequency+1.2*(.25+.75*length/average)))
        scores.append(math.fsum(score))
    chosen = min(range(len(candidates)), key=lambda i:(-scores[i], candidates[i]['task']))
    by_answer = Counter(r['answer']['sha256'] for r in candidates)
    most = min(by_answer, key=lambda h:(-by_answer[h],h))
    modal = next(r for r in candidates if r['answer']['sha256'] == most)
    lengths = Counter(len(r['answer']['atoms']) for r in candidates)
    def candidate(row, name, score=None):
        return dict(method=name,status='candidate',donor=row['task'],answer=deepcopy(row['answer']),score=score,
                    endpoint='TRAIN_donor_endpoint',validity='unassessed',donor_execution_transfers=False,native_readings_bound=False)
    out = dict(bm25=dict(candidate(candidates[chosen],'bm25',scores[chosen]), zero_overlap=scores[chosen]==0),
               modal=candidate(modal,'modal'), length_mode=dict(method='length_mode',status='length_only',
                   atoms=min(lengths,key=lambda n:(-lengths[n],n)),validity='unassessed',content_generated=False))
    matches = [r for r in candidates if r['public'] == request]; distinct = len({r['answer']['sha256'] for r in matches})
    if distinct == 1: out['exact'] = dict(candidate(matches[0],'exact'),matches=len(matches))
    else: out['exact'] = dict(method='exact',status='clarification_required' if distinct else 'unavailable',
        reason='CONFLICTING_TRAIN_ANSWERS' if distinct else 'NO_EXACT_TEXT_ROLE_MATCH', matches=len(matches),distinct_answers=distinct)
    return out


def metrics(proposal, truth):
    available = proposal['status'] == 'candidate'
    length = len(proposal['answer']['atoms']) if available else proposal.get('atoms')
    return dict(available=available,proposed_atoms=length,expected_atoms=len(truth['atoms']),
        exact_recorded_bytes=available and proposal['answer']['text'].encode() == truth['text'].encode(),
        exact_measurement_sequence=available and tuple(a['measurement'] for a in proposal['answer']['atoms']) ==
                                               tuple(a['measurement'] for a in truth['atoms']),
        length_match=length == len(truth['atoms']),correctness_established=False)


def same(observed, expected, code): need(canonical(observed) == canonical(expected), code)


def run(a):
    a.output.mkdir(exist_ok=False); began = tick = time.perf_counter(); phases = {}
    need(file_sha(a.input/'report.json') == a.pin, 'BASELINE_COLD_PIN')
    report = json.loads((a.input/'report.json').read_bytes())
    need(report['schema'] == 'complete-answer-baselines-v1' and report['complete'], 'BASELINE_COLD_SCOPE')
    for n,h in report['files'].items(): need(file_sha(a.input/n) == h, 'BASELINE_COLD_BYTES')
    state = source.load(a.root); same(state['code'],report['code'],'BASELINE_COLD_CODE')
    same(state['inputs'],report['inputs'],'BASELINE_COLD_INPUTS')
    def read(n): return json.loads((a.input/n).read_bytes())
    bank = read('training.json'); requests = read('requests.json'); truth = read('private-evaluation-targets.json')
    same(bank,state['bank'],'BASELINE_COLD_BANK'); same(requests,state['queries'],'BASELINE_COLD_REQUESTS')
    same(truth,state['truth'],'BASELINE_COLD_TRUTH')
    # Independently reassemble every source span and measurement from the original source bytes.
    atoms = 0
    for task, answer in [(r['task'],r['answer']) for r in bank]+list(truth.items()):
        pop,_,_ = state['view'].source.selected(task); raw = pop.fields[answer['source_field']]['text'].encode()
        lo,hi = answer['reference_span']; need(raw[lo:hi] == answer['text'].encode(),'BASELINE_COLD_SOURCE_SPAN')
        cursor = lo
        for i,atom in enumerate(answer['atoms']):
            left,right = atom['surface_span']
            need(left == cursor and atom['index'] == i and digest(raw[left:right]) == atom['surface_sha256'], 'BASELINE_COLD_ATOM')
            need(digest(b''.join(raw[x:y] for x,y in atom['measurement_spans'])) == atom['measurement'], 'BASELINE_COLD_MEASUREMENT')
            cursor = right; atoms += 1
        need(cursor == hi == answer['end']['byte_offset'], 'BASELINE_COLD_END')
    phases['authentication_source_spans'] = time.perf_counter()-tick; tick = time.perf_counter()
    observed = read('predictions.json'); evaluation = read('evaluation.json'); reconstructed = []; prepared = prepare(bank)
    need(len(observed) == len(evaluation) == len(requests) == 170, 'BASELINE_COLD_DENOMINATOR')
    for request, saved, scored in zip(requests,observed,evaluation):
        proposed = reconstruct(prepared,request['public'])
        expected = {n:request[n] for n in ('task','group','family')} | dict(operation=request['public']['operation'],predictions=proposed)
        same(saved,expected,'BASELINE_COLD_SELECTION')
        row = {n:expected[n] for n in ('task','group','family','operation')} | dict(arms={
            n:dict(prediction=v,metrics=metrics(v,truth[request['task']])) for n,v in proposed.items()})
        same(scored,row,'BASELINE_COLD_METRICS'); reconstructed.append(row)
    same(source.summary(reconstructed),report['summary'],'BASELINE_COLD_SUMMARY')
    phases['flat_bank_reconstruction'] = time.perf_counter()-tick; tick = time.perf_counter()
    controls = {}
    def corruption(name, obj, change, code):
        wrong = deepcopy(obj); change(wrong)
        try: same(wrong,obj,code)
        except ValueError as e:
            need(code in str(e),'BASELINE_COLD_WRONG_REFUSAL'); controls[name] = str(e); return
        raise ValueError('BASELINE_COLD_CONTROL_ACCEPTED')
    corruption('donor',observed[0],lambda r:r['predictions']['bm25'].update(donor='not-a-task'),'BASELINE_COLD_SELECTION')
    corruption('score',observed[0],lambda r:r['predictions']['bm25'].update(score=-1),'BASELINE_COLD_SELECTION')
    corruption('endpoint',observed[0],lambda r:r['predictions']['bm25']['answer']['end'].update(byte_offset=-1),'BASELINE_COLD_SELECTION')
    corruption('order',observed[0],lambda r:r['predictions']['bm25']['answer']['atoms'][0].update(index=99),'BASELINE_COLD_SELECTION')
    corruption('metric',evaluation[0],lambda r:r['arms']['bm25']['metrics'].update(correctness_established=True),'BASELINE_COLD_METRICS')
    corruption('code_pin',report['code'],lambda r:r.update({next(iter(r)):'0'*64}),'BASELINE_COLD_CODE')
    state['fresh'](); phases['controls_final_readback'] = time.perf_counter()-tick; seconds=time.perf_counter()-began
    need(.9 <= sum(phases.values())/seconds <= 1.001,'BASELINE_COLD_PHASES')
    value = dict(complete=True,source=dict(path=str(a.input),pin=a.pin),code=state['code'],tasks=170,methods=4,
        source_atoms=atoms,all_selections_exact=True,all_metrics_exact=True,controls=controls,summary=report['summary'],
        seconds=seconds,phases=phases,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        gpu_used=False,scientific_admission=False,correctness_established=False,acceptance='author-checked')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',value),seconds=seconds)),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=source.ROOT)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
