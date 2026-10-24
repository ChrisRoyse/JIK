"""Pinned public QASC native observations; labels are a separate training boundary."""
import json
from pathlib import Path
import numpy as np
from audit.connected_population_readback import readback as read_population
from audit.window_readback import readback as read_native
from text.evaluation_multiscale import file_sha
from text.connected_training_census import norm

POPULATION='55583a92affe1d330c67bda07c6a3976ae50427aa23404332085fd2461e63ee3'
NATIVE='efa6540a740e65e12ff084c6cfab023eb1e895fc668fdf36eab82fae4a1787ea'
POLICIES=('qasc-composition','qasc-fact-disjoint')
LANES=('all-mpnet-base-v2','codebert-mean','e5-base-v2','finbert-mean','pubmedbert-embeddings','specter2-base')

def check(ok,code):
    if not ok:raise ValueError('MNLM_QASC_'+code)

def rows(path):return [json.loads(line) for line in path.read_bytes().splitlines()]

def load(population,native,raw):
    check(file_sha(population/'report.json')==POPULATION,'POPULATION_PIN')
    check(file_sha(native/'report.json')==NATIVE,'NATIVE_PIN')
    source=read_population(population,raw/'multirc-training-v1',raw/'qasc-hf-training-v1',raw/'multirc-v1')
    readings=read_native(native,population)
    check(tuple(sorted(readings['lenses']))==LANES,'NATIVE_ROSTER')
    cases=[r for r in rows(population/'cases.jsonl') if r['family']=='qasc']
    refs={r['id']:r for r in rows(population/'references.jsonl')}
    targets={r['case']:r for r in rows(population/'private/targets.jsonl') if r['family']=='qasc'}
    check(len(cases)==8134 and all(len(c['facts'])==2 and len(c['options'])==8 for c in cases),'CASE_SHAPE')
    original=np.array([[refs[r]['unit'] for r in [c['question'],*c['facts'],*c['options']]] for c in cases],np.int64)
    units=np.unique(original);check(len(units)==27956,'UNITS')
    maps=np.searchsorted(units,original);check(np.array_equal(units[maps],original),'GATHER')
    values=[];lane_pins={}
    for name in LANES:
        root=native/name;rep=json.loads((root/'report.json').read_bytes())
        with np.load(root/'slots.npz',allow_pickle=False) as z:
            ptr=z['offsets'];check(np.all(np.diff(ptr)[units]==1),'QASC_WINDOW_OVERFLOW')
            values.append(z['vectors'][ptr[units]].copy())
        lane_pins[name]=dict(report_sha256=file_sha(root/'report.json'),lens_id=rep['lens_id'])
    labels=np.array([targets[c['id']]['answer_index'] for c in cases],np.int64)
    check(labels.shape==(8134,) and np.all((labels>=0)&(labels<8)),'LABELS')
    # Normalized lexical identities are used for strata only, never as model features.
    import pyarrow.parquet as pq
    surfaces=pq.read_table(population/'connected/chunks/atoms.parquet',columns=['text'])['text'].to_pylist()
    facts=[[norm(surfaces[u]) for u in row[1:3]] for row in original]
    assignment={(r['policy'],r['case']):r for r in rows(population/'assignments.jsonl') if r['policy'] in POLICIES}
    policies={}
    for policy in POLICIES:
        rr=[dict(id=c['id'],**{k:v for k,v in assignment[policy,c['id']].items() if k not in ('case','policy')}) for c in cases]
        check(all(r['eligible'] for r in rr),'INELIGIBLE')
        trainfacts={f for i,r in enumerate(rr) if r['partition']=='train' for f in facts[i]}
        for i,r in enumerate(rr):r['seen_training_facts']=sum(f in trainfacts for f in facts[i])
        policies[policy]=rr
    before={str(population/'report.json'):POPULATION,str(native/'report.json'):NATIVE}
    for root in (population,native):
        report=json.loads((root/'report.json').read_bytes())
        for n in report.get('files',{}):before[str(root/n)]=file_sha(root/n)
    for name in LANES:
        root=native/name;rep=json.loads((root/'report.json').read_bytes());before[str(root/'report.json')]=file_sha(root/'report.json')
        for n in rep['files']:before[str(root/n)]=rep['files'][n]['sha256']
    return dict(native=values,maps=maps,units=units,original=original,labels=labels,policies=policies,before=before,
                source=source,readings=readings,lanes=lane_pins)
