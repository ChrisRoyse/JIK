"""Exercise the paired-answer CLI and independent reader on real saved scores.

Mutated publications are negative/boundary controls only, never capability data.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.legal_answer_architecture_comparison import BASE, canonical, persist, sha


def check(value, code):
    if not value: raise ValueError('MNLM_ANSWER_PAIR_PROOF_'+code)


def call(program, args, log, success=True, error=None):
    with log.open('xb') as stream:
        r = subprocess.run([sys.executable, '-I', str(BASE/'audit'/program), *map(str,args)],
                           stdout=stream, stderr=subprocess.STDOUT)
    raw = log.read_text()
    check((r.returncode == 0) == success and (error is None or error in raw), log.name)
    return dict(returncode=r.returncode,log=dict(path=str(log),sha256=sha(log)),expected_error=error)


def run(a):
    started = time.perf_counter(); a.output.mkdir(); output = a.output
    before = {str(a.score):sha(a.score)}
    r = json.loads(a.score.read_bytes()); check(before[str(a.score)] == a.pin and r['declared_cases'] == 4, 'SOURCE')
    sroot = a.score.parent/'scores'; b = json.loads((sroot/'binding.json').read_bytes())
    before[str(sroot/'binding.json')] = sha(sroot/'binding.json')
    originals = {}
    for ref in r['cases']:
        path = sroot/ref['path']; c = json.loads(path.read_bytes()); before[str(path)] = sha(path)
        p = sroot/c['payload']['path']; v = json.loads(p.read_bytes()); before[str(p)] = sha(p)
        check(before[str(path)] == ref['pin'] and before[str(p)] == c['payload']['pin'], 'SOURCE_PIN')
        originals[ref['ordinal']] = (c,v)
    check(sorted(v['identity']['reference_atoms'] for c,v in originals.values()) == [1,1,4,16], 'BOUNDARIES')

    def execute(name, score=a.score, pin=a.pin, reverse=False, success=True, error=None):
        dest = output/name
        args = ['--score', score, '--pin', pin, '--pack', a.pack, '--scope', 'proof', '--output', dest]
        if reverse: args += ['--reverse']
        result = call('legal_answer_architecture_comparison.py', args, output/(name+'.log'), success, error)
        if success:
            report = dest/'report.json'; report_pin = sha(report)
            call('legal_answer_architecture_readback.py', ['--root',dest,'--pin',report_pin,
                 '--output',output/(name+'-readback.json')],output/(name+'-readback.log'))
            check(json.loads((output/(name+'-readback.json')).read_bytes())['complete'], 'READBACK')
            result.update(report=dict(path=str(report),pin=report_pin),summary=sha(dest/'summary.json'))
        else:
            check(not (dest/'report.json').exists() and (dest/'failure.json').is_file(), 'REFUSAL_NOT_SUCCESS')
            result['refusal'] = dict(path=str(dest/'failure.json'),pin=sha(dest/'failure.json'))
        return result

    runs = {n:execute(n,reverse=n=='reverse') for n in ('forward','cold','reverse')}
    check(len({v['summary'] for v in runs.values()}) == 1, 'ORDER_INVARIANCE')

    def fixture(name, mutate):
        path = output/(name+'-input'); path.mkdir(); scores = path/'scores'; scores.mkdir()
        report = copy.deepcopy(r); cases = copy.deepcopy(originals)
        mutate(report,cases)
        binding_pin = persist(scores/'binding.json',b)
        for ref in report['cases']:
            ordinal = ref['ordinal']; c,v = cases[ordinal]; c['binding'] = binding_pin
            name = f'score-{ordinal:06d}.json'
            c['payload'] = dict(path=name,pin=persist(scores/name,v))
            ref['pin'] = persist(scores/ref['path'],c)
        report['binding'] = binding_pin
        return path/'result.json',persist(path/'result.json',report)

    first = next(iter(originals))
    controls = []
    controls.append(dict(name='wrong-report-pin',**execute('wrong-report-pin',pin='0'*64,
        success=False,error='MNLM_ANSWER_PAIR_PIN')))

    def payload_change(fn):
        return lambda report,cases:fn(cases[first][1])

    changes = [
        ('document-swap',payload_change(lambda v:v['identity'].__setitem__('document',-1)),'CASE_IDENTITY'),
        ('side-swap',payload_change(lambda v:v['identity'].__setitem__('side','query')),'CASE_IDENTITY'),
        ('component-swap',payload_change(lambda v:v['identity'].__setitem__('component',-1)),'CASE_IDENTITY'),
        ('source-swap',payload_change(lambda v:v['identity'].__setitem__('source','0'*64)),'SOURCE'),
        ('missing-arm',payload_change(lambda v:v['arms'].pop()),'ARM_GRID'),
        ('duplicate-arm',payload_change(lambda v:v['arms'].append(copy.deepcopy(v['arms'][0]))),'ARM_GRID'),
        ('wrong-arithmetic',payload_change(lambda v:v['arms'][0]['metrics']['full_output_mean'].__setitem__(v['roster'][0],0.)),'ARITHMETIC'),
        ('wrong-count',payload_change(lambda v:v['arms'][0]['metrics'].__setitem__('matched_positions',-1)),'COUNTS'),
        ('wrong-end',payload_change(lambda v:v['arms'][0]['metrics'].__setitem__('ended',not v['arms'][0]['metrics']['ended'])),'END'),
        ('missing-refusal',payload_change(lambda v:v['arms'][0]['metrics'].__setitem__('refusal',None)),'REFUSAL'),
        ('missing-case',lambda report,cases:report['cases'].pop(),'CASE_COVERAGE'),
        ('false-full-population',lambda report,cases:report.__setitem__('full_population',True),'FULL_COVERAGE'),
        ('model-swap',payload_change(lambda v:v['models']['shared'].__setitem__('model','0'*64)),'MODEL_CHANGED'),
        ('posthoc-best-proposal',payload_change(lambda v:v.__setitem__('proposal_selection','best')),'PROPOSALS'),
    ]
    for name,mutate,error in changes:
        score,pin = fixture(name,mutate)
        controls.append(dict(name=name,**execute(name,score,pin,success=False,error='MNLM_ANSWER_PAIR_'+error)))

    # Gap controls preserve the real metadata and explicitly replace score availability.
    # They are NOT observations that controls were absent in the four actual cases.
    def gap(ids):
        def mutate(report,cases):
            report['control_unavailable'] = []
            for ordinal in ids:
                c,v = cases[ordinal]; absent = dict(status='CONTROL_UNAVAILABLE_NATIVE_IDENTICAL',
                    ordinal=ordinal,document=c['document'])
                cases[ordinal] = c,absent; report['control_unavailable'].append(absent)
            report['scored_cases'] = len(cases)-len(ids)
        return mutate
    boundaries = {}
    for name,ids in [('one-gap',[first]),('all-gap',list(originals))]:
        score,pin = fixture(name,gap(ids)); boundaries[name] = execute(name,score,pin)
        summary = json.loads((output/name/'summary.json').read_bytes())
        check(summary['cases'] == 4, 'GAP_DENOMINATOR')
        if name == 'all-gap':
            check(all(s['paired_mean'] is None and s['declared_mean_bounds'] == [-2.,2.]
                      for g in summary['groups'] for s in g['slots'].values()), 'ALL_GAP_BOUNDS')

    # Corrupt already-persisted outputs, update their pins, and require the different
    # reader to detect the semantic accounting error against original source scores.
    for name in ('bad-derived-delta','bad-aggregate'):
        dest = output/name; dest.mkdir()
        good = output/'forward'; report = json.loads((good/'report.json').read_bytes())
        rows = [json.loads(line) for line in (good/'cases.jsonl').read_bytes().splitlines()]
        summary = json.loads((good/'summary.json').read_bytes())
        if name == 'bad-derived-delta': rows[0]['deltas'][0][report['roster'][0]] += .125
        else: summary['groups'][0]['slots'][report['roster'][0]]['positive'] += 1
        with (dest/'cases.jsonl').open('xb') as stream:
            import hashlib
            refs = []
            for row in rows:
                raw = canonical(row)+b'\n';stream.write(raw)
                refs.append(dict(ordinal=row['meta']['ordinal'],sha256=hashlib.sha256(raw).hexdigest()))
        report['cases'] = sorted(refs,key=lambda v:v['ordinal'])
        report['summary_pin'] = persist(dest/'summary.json',summary)
        report['files'] = {n:dict(sha256=sha(dest/n),bytes=(dest/n).stat().st_size)
                           for n in ('cases.jsonl','summary.json')}
        pin = persist(dest/'report.json',report)
        error = 'DELTAS' if name == 'bad-derived-delta' else 'AGGREGATES'
        control = call('legal_answer_architecture_readback.py', ['--root',dest,'--pin',pin,
            '--output',output/(name+'-readback.json')],output/(name+'-readback.log'),False,
            'MNLM_ANSWER_PAIR_READBACK_'+error)
        check(not (output/(name+'-readback.json')).exists(),'READER_REFUSAL')
        controls.append(dict(name=name,**control))
    check(all(sha(Path(p)) == pin for p,pin in before.items()), 'SOURCE_UNCHANGED')
    result = dict(complete=True,source=dict(path=str(a.score),pin=a.pin),runs=runs,controls=controls,
        boundaries=boundaries,source_unchanged=before,seconds=time.perf_counter()-started,
        native_inputs=False,model_loads=0,optimizer_updates=0,gpu_used=False,
        capability_measurement=False,acceptance='author-checked only')
    pin = persist(output/'report.json',result)
    print(json.dumps(dict(complete=True,pin=pin,controls=len(controls),summary=runs['forward']['summary'],
                         seconds=result['seconds'])),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--score',type=Path,required=True);parser.add_argument('--pin',required=True)
    parser.add_argument('--pack',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    try:run(args)
    except BaseException as exc:
        if args.output.exists() and not (args.output/'failure.json').exists():
            persist(args.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise
