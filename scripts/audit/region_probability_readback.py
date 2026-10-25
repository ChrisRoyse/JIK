"""Publish and independently recount proper region scores without a model forward."""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from model import region_count_probabilities as method
from audit import placement_branch_score_readback as source

CACHE = ('legal-target-cache-full-full-v2-20261023', '77319fe3eaac3ea9db2ede167519a418151133b5c4641ef19ce6c901d2d5fc72')
COLD = ('legal-target-cache-full-cold-v2-20261023', 'd4498a44902db8e642ab368e1e1ead435e7e87caed44cae274245215b2a9a77f')
PRIOR = ('shared-placement-diagnostic-v1-20261025', '43d135f1e777f269c03cbd8cde336ded48985237561ff38597fe7cbab22b05b3')
RECOUNT = ('shared-placement-score-cold-v1-20261025', '5b370d62934a041f66f6c71dae60aa9a8ddcc2fa24eeb443427325268c95027b')
NAMES = tuple(method.POLICY['baselines'])


def need(ok, code):
    if not ok: raise ValueError('REGION_SCORE_'+code)


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    with path.open('xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    return hashlib.sha256(raw).hexdigest()


def authenticate(root):
    files = {}
    def read(path, pin):
        need(path.is_file() and sha(path) == pin, 'INPUT_PIN'); files[str(path)] = pin
        return json.loads(path.read_bytes())
    reports = [read(root/name/'report.json', pin) for name,pin in (CACHE,COLD,PRIOR,RECOUNT)]
    cache,cold,prior,recount = reports; binding = prior['binding']['source']['cache']
    need(all(r['complete'] for r in reports) and binding['publication'] == CACHE[1] and binding['cold'] == COLD[1]
         and recount['input_pin'] == PRIOR[1] and recount['rows'] == prior['rows'] == 49644, 'AUTHORITY')
    need(cache['scope'] == cold['scope'] == 'full' and cache['mode'] == 'publish' and cold['mode'] == 'cold'
         and cold['previous_pin'] == CACHE[1] and cold['binding'] == cache['binding']
         and cache['binding']['source'] == binding['source'] and cache['binding']['target_checkpoint'] == binding['target']
         and cache['binding']['sizes'] == binding['sizes'] == [4,16,64]
         and cold['result']['all_rows_exact'] and cold['result']['all_exports_exact']
         and cold['result']['all_rows_canonical_singleton_exact'], 'COUNT_PUBLICATION')
    for name,pin in (CACHE,COLD):
        path=root/name/'COMMITTED'; need(path.read_text().strip() == pin, 'COUNT_COMMIT'); files[str(path)]=sha(path)
    counts = read(root/CACHE[0]/'baselines.json', cache['files']['baselines.json'])
    back = read(root/COLD[0]/'baselines.json', cold['files']['baselines.json'])
    need(counts == back and counts['rows'] == cold['result']['baseline_rows'], 'COUNT_COLD')
    base = Path(__file__).resolve().parents[1]
    code = {n:sha(base/n) for n in ('model/region_count_probabilities.py', 'audit/region_probability_readback.py', 'audit/placement_branch_score_readback.py')}
    inherited = dict(prior['code'], **cache['binding']['code'])
    need(all(sha(base/n) == h for n,h in inherited.items()), 'SOURCE_CODE')
    donor_path = root/'legal-answer-successor-v1-20261024/placement/full/donors.json'
    donor = read(donor_path, prior['binding']['donors'])
    model_rows = {}
    for stage in ('initial','final'):
        path=root/PRIOR[0]/prior['scores'][stage]['path']; pin=prior['scores'][stage]['pin']
        need(sha(path) == pin and recount['score_pins'][str(path)] == pin, 'MODEL_SCORE_PIN'); files[str(path)]=pin
        with path.open() as f: model_rows[stage]=[json.loads(line) for line in f]
        source.coverage(model_rows[stage], prior['binding']['source']['plan'], donor)
    need(len(model_rows['initial']) == len(model_rows['final']) == 24822, 'MODEL_ROWS')
    for a,b in zip(model_rows['initial'], model_rows['final']):
        need(all(a[k] == b[k] for k in ('document','offset','side','source','component','target','baselines')), 'MODEL_IDENTITY')
    return counts, binding['sizes'], model_rows, files, code, inherited


def reference(counts, sizes, target, current):
    """Independent list/integer count reductions, normalized with math.fsum."""
    a,b,_=target; h=counts['joint']; half=method.POLICY['half_count']
    total=counts['rows']; tails=sizes[1]*sizes[2]
    def normalize(xs):
        s=math.fsum(xs); need(s>0, 'REFERENCE_TOTAL'); return [x/s for x in xs]
    root=normalize([sum(sum(row) for row in block)+half*tails for block in h])
    middle=normalize([sum(row)+half*sizes[2] for row in h[a]])
    leaf=normalize([v+half for v in h[a][b]])
    marginal=[normalize([v+half for v in level]) for level in counts['level_histograms']]
    transition=normalize([v+half for v in counts['root_transition'][current[0]]])
    persistence=[]
    for j,ps in enumerate((root,middle,leaf)):
        weight=counts['copy_level'][j]/total
        persistence.append([(1-weight)*p+(weight if i==current[j] else 0) for i,p in enumerate(ps)])
    distributions=dict(uniform=[[1/n]*n for n in sizes], marginal=marginal, joint_prefix=[root,middle,leaf],
        root_transition=[transition,middle,leaf], persistence=persistence)
    out={}
    for name, levels in distributions.items():
        need(all(len(ps)==n and all(math.isfinite(p) and 0<p<=1 for p in ps) and abs(math.fsum(ps)-1)<=1e-12
                 for n,ps in zip(sizes,levels)), 'REFERENCE_PROBABILITY')
        out[name]=dict(ce=[-math.log(ps[y]) for ps,y in zip(levels,target)],
                      brier=[math.fsum((p-int(i==y))**2 for i,p in enumerate(ps)) for ps,y in zip(levels,target)])
    return out


def identities(r):
    return {k:r[k] for k in ('document','offset','side','source','component','target')}


def check_row(row, expected, counts, sizes):
    need(row['identity'] == identities(expected) and row['current'] == expected['baselines']['copy_current'], 'ROW_IDENTITY')
    ref=reference(counts,sizes,row['identity']['target'],row['current'])
    need(source.equal(row['baselines'],ref), 'ROW_SCORES')


def summary(rows, models, *, independent):
    mean = (lambda xs: math.fsum(xs)/len(xs)) if independent else (lambda xs: float(np.mean(xs)))
    grouped=defaultdict(list)
    for i,r in enumerate(rows): grouped[r['identity']['side']+'|'+r['identity']['source']].append(i)
    out={}
    for key,ids in grouped.items():
        docs=defaultdict(list)
        for i in ids:docs[rows[i]['identity']['document']].append(i)
        cell=dict(rows=len(ids),documents=len(docs),baselines={},models={})
        for name in NAMES:
            cell['baselines'][name]={score:[mean([rows[i]['baselines'][name][score][j] for i in ids]) for j in range(3)] for score in ('ce','brier')}
        for stage, ms in models.items():
            item={score:[mean([ms[i][score][j] for i in ids]) for j in range(3)] for score in ('ce','brier')}
            item['contrasts']={}
            for name in NAMES:
                contrasts={}
                for metric in ('ce_0','ce_1','ce_2','brier_0','brier_1','brier_2','joint_ce'):
                    def gap(i):
                        if metric=='joint_ce':return math.fsum(rows[i]['baselines'][name]['ce'])-math.fsum(ms[i]['ce'])
                        score,j=metric.split('_');return rows[i]['baselines'][name][score][int(j)]-ms[i][score][int(j)]
                    values=[mean([gap(i) for i in members]) for members in docs.values()]
                    contrasts[metric]=dict(atom_weighted_advantage=mean([gap(i) for i in ids]),
                        document_mean_advantage=mean(values),documents=len(docs),
                        win=sum(v>0 for v in values),tie=sum(v==0 for v in values),loss=sum(v<0 for v in values))
                item['contrasts'][name]=contrasts
            cell['models'][stage]=item
        out[key]=cell
    return out


def controls(counts,sizes,first,model_first,tables):
    out={}
    def reject(name, fn, code):
        try:fn()
        except ValueError as e:need(code in str(e), 'CONTROL_CODE:'+str(e));out[name]=str(e);return
        raise ValueError('REGION_SCORE_CORRUPTION_ACCEPTED:'+name)
    bad=deepcopy(counts);bad['fit_uses_held_out']=True
    reject('held_out_counts',lambda:method.Tables(bad,sizes), 'REGION_COUNT_FIT_SCOPE')
    bad=deepcopy(counts);bad['rows']+=1
    reject('wrong_total',lambda:method.Tables(bad,sizes), 'REGION_COUNT_TOTAL')
    bad=deepcopy(method.POLICY);bad['half_count']=1.
    reject('changed_prior',lambda:method.Tables(counts,sizes,bad), 'REGION_COUNT_POLICY')
    reject('invalid_target',lambda:tables.scores([sizes[0],0,0],[0,0,0]), 'REGION_COUNT_ADDRESS')
    ps=[p.copy() for p in tables.uniform];ps[0][0]=float('nan')
    reject('invalid_probability',lambda:method.probabilities_valid(ps,sizes), 'REGION_COUNT_PROBABILITY')
    bad=deepcopy(first);bad['identity']['target'][0]=(bad['identity']['target'][0]+1)%sizes[0]
    reject('wrong_target',lambda:check_row(bad,model_first,counts,sizes), 'REGION_SCORE_ROW_IDENTITY')
    bad=deepcopy(first);bad['baselines']['uniform']['ce'][0]+=.1
    reject('wrong_score',lambda:check_row(bad,model_first,counts,sizes), 'REGION_SCORE_ROW_SCORES')
    return out


def row_count(rows, expected):need(len(rows)==expected, 'ROW_COVERAGE')


def run(a):
    began=tick=time.perf_counter();a.output.mkdir(exist_ok=False)
    counts,sizes,models,files,code,inherited=authenticate(a.root)
    tables=method.Tables(counts,sizes);phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    binding=dict(inputs=files,code=code,inherited_code=inherited,policy=method.POLICY,sizes=sizes,training_rows=counts['rows'])
    previous=None
    if a.previous:
        need(a.previous_pin and sha(a.previous/'report.json')==a.previous_pin,'PREVIOUS_PIN')
        previous=json.loads((a.previous/'report.json').read_bytes())
        need(previous['complete'] and previous['binding']==binding and previous['mode']=='publish','PREVIOUS_BINDING')
        need(sha(a.previous/'scores.jsonl')==previous['scores_pin'],'PREVIOUS_SCORES')
        with (a.previous/'scores.jsonl').open() as f:rows=[json.loads(line) for line in f]
        row_count(rows,len(models['final']))
        for row,expected in zip(rows,models['final']):check_row(row,expected,counts,sizes)
        scores_pin=previous['scores_pin']
    else:
        rows=[]
        with (a.output/'scores.jsonl').open('x') as f:
            for expected in models['final']:
                target=expected['target'];current=expected['baselines']['copy_current']
                row=dict(identity=identities(expected),current=current,baselines=tables.scores(target,current))
                rows.append(row);f.write(json.dumps(row,sort_keys=True,separators=(',', ':'),allow_nan=False)+'\n')
            f.flush();os.fsync(f.fileno())
        scores_pin=sha(a.output/'scores.jsonl')
    phases['independent_recount' if previous else 'probability_scores']=time.perf_counter()-tick;tick=time.perf_counter()
    result=summary(rows,models,independent=bool(previous))
    if previous:need(source.equal(result,previous['summaries']),'AGGREGATE_RECOUNT')
    checks=controls(counts,sizes,rows[0],models['final'][0],tables)
    try:row_count(rows[:-1],len(models['final']))
    except ValueError as e:need('ROW_COVERAGE' in str(e),'CONTROL_CODE');checks['missing_row']=str(e)
    else:raise ValueError('REGION_SCORE_MISSING_ACCEPTED')
    if previous:
        bad=deepcopy(result);bad[next(iter(bad))]['models']['final']['ce'][0]+=.1
        need(not source.equal(bad,previous['summaries']),'AGGREGATE_CORRUPTION_ACCEPTED');checks['wrong_aggregate']='REGION_SCORE_AGGREGATE_RECOUNT'
        need(sha(a.previous/'scores.jsonl')==scores_pin and sha(a.previous/'report.json')==a.previous_pin,'PREVIOUS_CHANGED')
    phases['summaries_and_controls']=time.perf_counter()-tick;tick=time.perf_counter()
    base=Path(__file__).resolve().parents[1]
    need(all(sha(Path(p))==h for p,h in files.items()) and all(sha(base/n)==h for n,h in {**inherited,**code}.items()),'INPUT_CHANGED')
    need('torch' not in sys.modules,'MODEL_IMPORTED')
    phases['final_authentication']=time.perf_counter()-tick;wall=time.perf_counter()-began
    need(.9<=sum(phases.values())/wall<=1.000001,'PHASE_COVERAGE')
    out=dict(complete=True,mode='cold' if previous else 'publish',binding=binding,scores_pin=scores_pin,
        previous_pin=a.previous_pin,rows=len(rows),model_rows=2*len(rows),summaries=result,controls=checks,phases=phases,wall=wall,
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,model_forward=False,scientific_admission=False,model_promoted=False,
        scope='Existing source-split region probabilities only; teacher-forced within-code prefixes; not whole-answer, native distributional or grounded anchor capability.',acceptance='author-checked')
    print(json.dumps(dict(pin=save(a.output/'report.json',out),rows=len(rows),wall=wall,mode=out['mode'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','output','previous'):p.add_argument('--'+n,type=Path,required=n!='previous')
    p.add_argument('--previous-pin');a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.is_dir():save(a.output/'failure.json',dict(error=type(e).__name__,message=str(e)))
        raise
