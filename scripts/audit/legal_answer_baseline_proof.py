"""Real saved-score integration, adversarial joins and independent readback."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.legal_answer_architecture_comparison import Inputs, canonical, persist, sha
from audit.legal_answer_architecture_proof import call, check
from audit.legal_answer_baseline_comparison import Baselines
from model.answer_baseline_comparison import compare, validate_baseline


def run(a):
    started = time.perf_counter(); a.output.mkdir(); out = a.output
    ap = json.loads((a.architecture_proof/'report.json').read_bytes())
    check(sha(a.architecture_proof/'report.json') == a.architecture_proof_pin
          and ap['complete'] is True and ap['source'] == dict(path=str(a.score),pin=a.pin), 'ARCHITECTURE_PROOF')
    before = {str(a.score):sha(a.score),str(a.baseline):sha(a.baseline),
              str(a.baseline.parent/'cases.jsonl'):sha(a.baseline.parent/'cases.jsonl')}

    def execute(name, score=a.score, pin=a.pin, reverse=False, error=None,
                baseline_pin=a.baseline_pin, baseline_path=a.baseline):
        dest = out/name
        args = ['--score',score,'--pin',pin,'--pack',a.pack,'--scope','proof','--output',dest,
                '--baseline',baseline_path,'--baseline-pin',baseline_pin,
                '--baseline-readback',a.baseline_readback,'--readback-pin',a.readback_pin]
        if reverse: args += ['--reverse']
        result = call('legal_answer_baseline_comparison.py', args, out/(name+'.log'), error is None, error)
        if error is None:
            report_pin = sha(dest/'report.json')
            call('legal_answer_baseline_readback.py', ['--root',dest,'--pin',report_pin,
                 '--output',out/(name+'-readback.json')],out/(name+'-readback.log'))
            result.update(report=dict(path=str(dest/'report.json'),pin=report_pin),summary=sha(dest/'summary.json'))
        else:
            check(not (dest/'report.json').exists() and (dest/'failure.json').exists(), 'REFUSAL')
            result['refusal'] = dict(path=str(dest/'failure.json'),pin=sha(dest/'failure.json'))
        return result

    runs = {n:execute(n,reverse=n=='reverse') for n in ('forward','cold','reverse')}
    check(len({v['summary'] for v in runs.values()}) == 1, 'ORDER_INVARIANCE')
    actual_strata = json.loads((out/'forward/summary.json').read_bytes())['stratum_cases']
    check(actual_strata == dict(no_exact_prefix=3,unique_answer=1,conflicting_answers=0), 'ACTUAL_STRATA')
    controls = [dict(name='wrong-baseline-pin', **execute('wrong-baseline-pin',baseline_pin='0'*64,
                     error='MNLM_ANSWER_PAIR_PIN'))]
    for name,error in (('baseline-source','BASELINE_REPORT'),('baseline-policy','BASELINE_POLICY'),
                       ('baseline-coverage','BASELINE_COVERAGE')):
        fixture = out/(name+'-input'); fixture.mkdir()
        report = json.loads(a.baseline.read_bytes())
        binding = json.loads((a.baseline.parent/'binding.json').read_bytes())
        if name == 'baseline-source':report['source']['exposure'] = 'different-source'
        elif name == 'baseline-policy':binding['policy']['reference_length_used_for_selection'] = True
        else:report['documents'].pop()
        report['binding'] = persist(fixture/'binding.json',binding)
        pin = persist(fixture/'report.json',report)
        controls.append(dict(name=name,**execute(name,baseline_path=fixture/'report.json',baseline_pin=pin,
                        error='MNLM_ANSWER_PAIR_'+error)))
    # Reuse already-persisted mutations; do not create another score corpus.
    errors = dict([('document-swap','CASE_IDENTITY'),('source-swap','SOURCE'),('component-swap','CASE_IDENTITY'),
        ('side-swap','CASE_IDENTITY'),('missing-arm','ARM_GRID'),('duplicate-arm','ARM_GRID'),
        ('wrong-arithmetic','ARITHMETIC'),('wrong-end','END'),('missing-refusal','REFUSAL'),
        ('missing-case','CASE_COVERAGE'),('false-full-population','FULL_COVERAGE'),('model-swap','MODEL_CHANGED'),
        ('posthoc-best-proposal','PROPOSALS')])
    for name,error in errors.items():
        score = a.architecture_proof/(name+'-input')/'result.json'
        controls.append(dict(name=name,**execute(name,score,sha(score),error='MNLM_ANSWER_PAIR_'+error)))
    boundaries = {}
    for name in ('one-gap','all-gap'):
        score = a.architecture_proof/(name+'-input')/'result.json'
        boundaries[name] = execute(name,score,sha(score))
        s = json.loads((out/name/'summary.json').read_bytes())
        check(s['cases'] == 4 and s['stratum_cases'] == actual_strata, 'GAP_DENOMINATOR')
        if name == 'all-gap':
            check(all(v['paired_mean'] is None and v['declared_mean_bounds'] == [-2.,2.]
                      for g in s['groups'] for v in g['slots'].values()), 'GAP_BOUNDS')

    # Real baseline samples exercise all three selection states independently.
    # They do not pretend the four model proof cases had available exact matches.
    data = Inputs(a.score,a.pin,a.pack,'proof'); baseline = None; samples = []
    try:
        baseline = Baselines(data,a.baseline,a.baseline_pin,a.baseline_readback,a.readback_pin)
        sample_docs = (7,1818,2923); selected = {}
        for i,row in enumerate(data.rows):
            if int(row['document']) in sample_docs: selected[int(row['document'])] = i
        for document in sample_docs:
            ordinal = selected[document]; v,pin = baseline.case(ordinal); meta = data.meta(ordinal)
            name = validate_baseline(v,meta,data.roster)
            path = out/f'baseline-{document}.json'; persist(path,dict(meta=meta,value=v,line_pin=pin,status=name))
            samples.append(dict(document=document,status=name,path=str(path),pin=sha(path)))
        check({x['status'] for x in samples} == {'no_exact_prefix','unique_answer','conflicting_answers'}, 'STATUSES')
        def reject(name, v, meta, mutate, expected):
            changed = copy.deepcopy(v); mutate(changed)
            path = out/(name+'.json'); persist(path,dict(meta=meta,value=changed,expected=expected))
            saved = json.loads(path.read_bytes()); error = None
            try: validate_baseline(saved['value'],saved['meta'],data.roster)
            except ValueError as exc: error = str(exc)
            check(error == 'MNLM_ANSWER_PAIR_'+expected, name)
            controls.append(dict(name=name,input=dict(path=str(path),pin=sha(path)),observed_error=error))
        for document in sample_docs:
            i = selected[document]; v,_ = baseline.case(i); meta = data.meta(i)
            reject(f'baseline-{document}-identity',v,meta,lambda x:x['identity'].__setitem__('component',-1),'BASELINE_IDENTITY')
        v,_ = baseline.case(selected[2923]); meta = data.meta(selected[2923])
        reject('conflict-suppressed',v,meta,lambda x:x['arms'].__setitem__('clarify_conflicts',copy.deepcopy(x['arms']['modal_control'])),
               'BASELINE_CONFLICT')
        reject('donor-end-changed',v,meta,lambda x:x['native'].__setitem__('donor_atoms',1),'BASELINE_DONOR_END')
        reject('selection-changed',v,meta,lambda x:x.__setitem__('selection_status',1),'BASELINE_UNIQUE')
        ordinal = next(iter(data.refs)); meta,c,payload = data.payload(ordinal); b,_ = baseline.case(ordinal)
        payload['identity']['reference_atoms'] += 1
        path = out/'wrong-reference-length.json'; persist(path,dict(meta=meta,payload=payload,baseline=b))
        value = json.loads(path.read_bytes()); error = None
        try: compare(value['payload'],value['baseline'],value['meta'],data.roster,data.source_pin)
        except ValueError as exc:error = str(exc)
        check(error == 'MNLM_ANSWER_PAIR_BASELINE_REFERENCE_LENGTH','REFERENCE_LENGTH')
        controls.append(dict(name='wrong-reference-length',input=dict(path=str(path),pin=sha(path)),observed_error=error))
        data.reads.fresh()
    finally:
        if baseline is not None:baseline.close()
        data.close()

    for name in ('bad-derived-delta','bad-aggregate','bad-stratum'):
        dest = out/name; dest.mkdir(); good = out/'forward'
        report = json.loads((good/'report.json').read_bytes())
        rows = [json.loads(z) for z in (good/'cases.jsonl').read_bytes().splitlines()]
        summary = json.loads((good/'summary.json').read_bytes())
        if name == 'bad-derived-delta':rows[0]['deltas'][0][report['roster'][0]] += .125
        elif name == 'bad-aggregate':summary['groups'][0]['slots'][report['roster'][0]]['positive'] += 1
        else:summary['stratum_cases']['unique_answer'] += 1
        refs = []
        with (dest/'cases.jsonl').open('xb') as f:
            for row in rows:
                raw = canonical(row)+b'\n'; f.write(raw)
                refs.append(dict(ordinal=row['meta']['ordinal'],sha256=hashlib.sha256(raw).hexdigest()))
        report['cases'] = sorted(refs,key=lambda v:v['ordinal'])
        report['summary_pin'] = persist(dest/'summary.json',summary)
        report['files'] = {n:dict(sha256=sha(dest/n),bytes=(dest/n).stat().st_size) for n in ('cases.jsonl','summary.json')}
        pin = persist(dest/'report.json',report); error = 'DELTAS' if name == 'bad-derived-delta' else 'AGGREGATES'
        result = call('legal_answer_baseline_readback.py',['--root',dest,'--pin',pin,'--output',out/(name+'-readback.json')],
                      out/(name+'-readback.log'),False,'MNLM_BASELINE_READBACK_'+error)
        check(not (out/(name+'-readback.json')).exists(),'READBACK_REFUSAL'); controls.append(dict(name=name,**result))
    check(all(sha(Path(p)) == pin for p,pin in before.items()), 'SOURCE_UNCHANGED')
    result = dict(complete=True,runs=runs,controls=controls,boundaries=boundaries,baseline_samples=samples,
        architecture_proof=dict(path=str(a.architecture_proof),pin=a.architecture_proof_pin),source_unchanged=before,
        source=dict(path=str(a.score),pin=a.pin),seconds=time.perf_counter()-started,
        capability_measurement=False,optimizer_updates=0,gpu_used=False,acceptance='author-checked only')
    pin = persist(out/'report.json',result)
    print(json.dumps(dict(complete=True,pin=pin,controls=len(controls),summary=runs['forward']['summary'],seconds=result['seconds'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('score','pack','baseline','baseline-readback','architecture-proof','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('pin','baseline-pin','readback-pin','architecture-proof-pin'):p.add_argument('--'+n,required=True)
    a = p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists() and not (a.output/'failure.json').exists():
            persist(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise
