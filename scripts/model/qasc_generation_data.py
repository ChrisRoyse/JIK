"""Private supervision and public three-atom inputs, with separate native addresses."""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from model.qasc_context import POPULATION,NATIVE,POLICIES,LANES,check,rows
from audit.connected_population_readback import readback as read_population
from audit.qasc_composition_target_readback import read as read_targets
from audit.window_readback import readback as read_native
from text.connected_training_census import norm
from text.evaluation_multiscale import file_sha

TARGETS='2fcc98ff39f60a1271ef9389234e6554c9808cbd7f9ab34d3e49d10d56fe9bd9'
TARGET_NATIVE='a608bcca4324f9cb4b3d0ff0a9a3048e777eaa17ffbb4572c5c7e77e996caf8b'

def load(population,native,targets,target_native,raw,proof):
    roots=[(population,POPULATION),(native,NATIVE),(targets,TARGETS),(target_native,TARGET_NATIVE)]
    for root,pin in roots:check(file_sha(root/'report.json')==pin,'GENERATION_PARENT_PIN')
    fsv=json.loads((proof/'native-fsv-linux-v2/fsv.json').read_bytes())
    replay=json.loads((proof/'native-replay-linux-v1/report.json').read_bytes())
    check(fsv['passed']==fsv['total']==21 and fsv['cold_readback']['report_sha256']==TARGET_NATIVE,'GENERATION_NATIVE_FSV')
    check(replay['publication_sha256']==TARGET_NATIVE and replay['source_and_instruments_unchanged']
          and all(v['all_replayed_vectors_bitwise_equal'] for v in replay['lanes'].values()),'GENERATION_NATIVE_REPLAY')
    cold=dict(population=read_population(population,raw/'multirc-training-v1',raw/'qasc-hf-training-v1',raw/'multirc-v1'),
              targets=read_targets(targets,population,native,raw),native=read_native(native,population),
              target_native=read_native(target_native,targets))
    cases=rows(targets/'cases.jsonl');private=rows(targets/'targets.jsonl')
    original=np.array([[r['question_unit'],*r['fact_units']] for r in cases],np.int64)
    units=np.unique(original);maps=np.searchsorted(units,original)
    check(original.shape==(8134,3) and len(units)==14628 and np.array_equal(units[maps],original),'GENERATION_INPUT_MAP')
    target_index=np.array([r['target'] for r in cases],np.int64);check(len(private)==7883,'GENERATION_TARGETS')
    inputs=[];truth=[];lanes={}
    fresh=[(i,r['native']['row']+1) for i,r in enumerate(private) if r['native']['route']=='fresh']
    cached=[(i,r['native']['row']) for i,r in enumerate(private) if r['native']['route']=='public']
    check(len(fresh)==7878 and len(cached)==5,'GENERATION_CACHE')
    for name in LANES:
        a=json.loads((native/name/'report.json').read_bytes());b=json.loads((target_native/name/'report.json').read_bytes())
        check(a['lens_id']==b['lens_id'],'GENERATION_INSTRUMENT_IDENTITY');y=np.empty((7883,768),np.float32)
        with np.load(native/name/'slots.npz',allow_pickle=False) as z:
            check(np.all(np.diff(z['offsets'])[units]==1),'GENERATION_INPUT_MULTI_WINDOW')
            inputs.append(z['vectors'][z['offsets'][units]].copy())
            for i,u in cached:
                check(z['offsets'][u+1]-z['offsets'][u]==1,'GENERATION_CACHED_MULTI_WINDOW');y[i]=z['vectors'][z['offsets'][u]]
        with np.load(target_native/name/'slots.npz',allow_pickle=False) as z:
            ii,uu=np.asarray(fresh).T
            check(np.all(uu>0) and np.all(np.diff(z['offsets'])[uu]==1),'GENERATION_PROBE_OR_MULTI_WINDOW')
            y[ii]=z['vectors'][z['offsets'][uu]]
        check(np.isfinite(y).all() and np.all(abs(np.linalg.norm(y,axis=1)-1)<2e-5),'GENERATION_TARGET_VECTOR')
        truth.append(y);lanes[name]=dict(lens_id=a['lens_id'],input_report=file_sha(native/name/'report.json'),target_report=file_sha(target_native/name/'report.json'))
    surfaces=pq.read_table(population/'connected/chunks/atoms.parquet',columns=['text'])['text'].to_pylist()
    facts=[[norm(surfaces[u]) for u in row[1:]] for row in original]
    policies=json.loads((targets/'policies.json').read_bytes())
    check(set(policies)==set(POLICIES),'GENERATION_POLICIES')
    for rr in policies.values():
        seen={f for i,r in enumerate(rr) if r['partition']=='train' for f in facts[i]}
        for i,r in enumerate(rr):r['seen_training_facts']=sum(f in seen for f in facts[i])
    before={}
    for root,pin in roots:
        before[str(root/'report.json')]=pin;report=json.loads((root/'report.json').read_bytes())
        for n,v in report.get('files',{}).items():before[str(root/n)]=v['sha256']
    for root in (native,target_native):
        for name in LANES:
            folder=root/name;report=json.loads((folder/'report.json').read_bytes());before[str(folder/'report.json')]=file_sha(folder/'report.json')
            for n,v in report['files'].items():before[str(folder/n)]=v['sha256']
    before[str(raw/'qasc-hf-training-v1/train.parquet')]=file_sha(raw/'qasc-hf-training-v1/train.parquet')
    return dict(inputs=inputs,truth=truth,maps=maps,units=units,original=original,target_index=target_index,
                cases=cases,texts=[r['text'] for r in private],policies=policies,before=before,cold=cold,lanes=lanes)
