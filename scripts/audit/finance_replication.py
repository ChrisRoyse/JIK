"""Read new native observations through the frozen successful calibration method."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.calyx_closure_readback import (sha, write, require, prediction, standardize,
    distance, scan, bits, intervals)
from audit.finance_profiles import native_sets, similarity
from audit.finance_profiles_readback import sim
from audit.window_readback import readback
from text.calyx_bits import unit
from text.calyx_indexed import indexed_counts
from text.conditional_geometry import predict
from text.evaluation_overlap import WORD

NAMES=['all-mpnet-base-v2','finbert-mean','specter2-base']


def population_readback(population,census):
    manifest=json.loads((population/'manifest.json').read_bytes())
    report=json.loads((population/'report.json').read_bytes())
    require(sha(population/'report.json')==manifest['report_sha256'] and report['complete'],'REPLICATION_POPULATION')
    for name,pin in report['files'].items():require(sha(population/name)==pin,'REPLICATION_FILE')
    intent=json.loads((population/'intent.json').read_bytes())
    prior={}
    for name,pin in intent['prior_sources'].items():
        require(sha(name)==pin,'REPLICATION_PRIOR_PIN')
        for line in Path(name).read_bytes().splitlines():
            row=json.loads(line);prior[row['entity']]=row['text']
    proof=population/'overlap'
    documents=[json.loads(line) for line in (proof/'documents.jsonl').read_bytes().splitlines()]
    words=json.loads((proof/'words.json').read_bytes()); word_ids={w:i for i,w in enumerate(words)}
    tuples=np.load(proof/'shingles.npy',allow_pickle=False)
    gram_ids={tuple(map(int,row)):i for i,row in enumerate(tuples)}
    ptr=np.load(proof/'indptr.npy',allow_pickle=False);flat=np.load(proof/'members.npy',allow_pickle=False)
    require(len(ptr)==len(documents)+1 and ptr[0]==0 and ptr[-1]==len(flat) and np.all(np.diff(ptr)>=0),'REPLICATION_FEATURE_OFFSETS')
    sets=[set(map(int,flat[ptr[i]:ptr[i+1]])) for i in range(len(documents))]
    positions={r['id']:i for i,r in enumerate(documents)}
    selected=[json.loads(line) for line in (population/'finance/rows.jsonl').read_bytes().splitlines()]
    anchors=json.loads((population/'finance/anchors.json').read_bytes())['P159']
    original=json.loads((census/'report.json').read_bytes())
    require(sha(census/'report.json')==report['census_report_sha256'] and
            sha(census/'private/eligible.jsonl')==original['eligible_sha256'],'REPLICATION_ORACLE_PIN')
    oracle={r['entity']:r for r in map(json.loads,(census/'private/eligible.jsonl').read_bytes().splitlines())}
    complete=dict(prior);complete.update({r['entity']:r['text'] for r in selected})
    for entity,text in complete.items():
        row=documents[positions[entity]];pin=hashlib.sha256(text.encode()).hexdigest()
        require(row['fields'][0]['sha256']==pin,'REPLICATION_TEXT_HASH')
        ids=[word_ids[w] for w in WORD.findall(text.lower())]
        expected={gram_ids[tuple(ids[i:i+5])] for i in range(len(ids)-4)}
        require(expected==sets[positions[entity]],'REPLICATION_EXACT_SHINGLES')
    require(len(selected)==600 and len({r['entity'] for r in selected})==600 and not set(prior)&{r['entity'] for r in selected},'REPLICATION_NEW_ENTITIES')
    for i,row in enumerate(selected):
        source=oracle[row['entity']];label=anchors['labels'][i]
        require(anchors['rows'][i]==i and row['row']==i and source['values']['P159']==[anchors['classes'][label]] and
                row['text_source']==source['text_source'] and hashlib.sha256(row['text'].encode()).hexdigest()==source['text_source']['sha256'],'REPLICATION_LABEL_SOURCE')
    require(list(Counter(anchors['labels']).values())==[100]*6,'REPLICATION_BALANCE')
    # Independent full static prefix index. No streaming window or producer join.
    chosen=[positions[r['entity']] for r in selected];background=[positions[e] for e in prior]+chosen
    frequency=Counter(v for i in background for v in sets[i]);index=defaultdict(list);prefix={}
    for i in background:
        s=sets[i];length=len(s)-(3*len(s)+4)//5+1 if s else 0
        prefix[i]=sorted(s,key=lambda v:(frequency[v],v))[:length]
        for value in prefix[i]:index[value].append(i)
    compared=0
    for i in chosen:
        require(bool(sets[i]),'REPLICATION_SHORT_TEXT')
        candidates=set(j for value in prefix[i] for j in index[value]);candidates.discard(i)
        for j in candidates:
            if 5*min(len(sets[i]),len(sets[j]))<3*max(len(sets[i]),len(sets[j])):continue
            compared+=1;intersection=len(sets[i]&sets[j]);union=len(sets[i]|sets[j])
            require(5*intersection<3*union,'REPLICATION_OVERLAP')
    return dict(source_report_sha256=manifest['report_sha256'],source_feature_sets_rebuilt=len(complete),
                exact_prefix_candidate_comparisons=compared,entities=len(selected),lexical_overlap_found=False)


def run(a):
    start=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    with threadpool_limits(limits=1):
        population=population_readback(a.population,a.census)
        cold=readback(a.readings,a.population)
        fsv=json.loads((a.fsv/'fsv.json').read_bytes())
        require(fsv['passed']==fsv['total'] and fsv['cold_readback']==cold,'REPLICATION_NATIVE_FSV')
        original=json.loads((a.original/'result-v1.json').read_bytes())
        require(sha(a.original/'result-v1.json')=='d1a4b4e27761b94ba7603f87b23aa99f057b0d393dfc8e8200e4c2132ee5e12f','REPLICATION_ORIGINAL_RESULT')
        fit_report=json.loads((a.original/'closure-v1/report.json').read_bytes())
        require(sha(a.original/'closure-v1/report.json')==original['reports']['closure-v1/report.json'],'REPLICATION_FIT_REPORT')
        fits=[r for r in fit_report['rows'] if r['source']['candidate']==NAMES[-1] and r['source']['conditioning']==NAMES[:-1] and
              Path(r['source']['fixed_profile']).name=='P159.npz' and r['seed']==7]
        require(len(fits)==1,'REPLICATION_FIXED_CASE');frozen=fits[0]
        model_file=a.original/'closure-v1'/frozen['model_file'];require(sha(model_file)==frozen['model_sha256'],'REPLICATION_MODEL_PIN')
        model=json.loads(model_file.read_bytes());profile_path=Path(frozen['source']['fixed_profile'])
        require(sha(profile_path)==frozen['source']['sha256'],'REPLICATION_LANDMARK_PIN')
        with np.load(profile_path,allow_pickle=False) as z: old={k:z[k] for k in z.files}
        original_pop=Path(json.loads((a.original/'profiles-v1/report.json').read_bytes())['sources']['population']['file']).parent
        # Full native binding of the unchanged landmark sources, not just cached profiles.
        original_cold=readback(a.original/'public-v1',original_pop)
        require(original_cold==json.loads((a.original/'public-fsv-v1/fsv.json').read_bytes())['cold_readback'],'REPLICATION_ORIGINAL_NATIVE')
        old_chunks=pq.read_table(original_pop/'finance/chunks/atoms.parquet').to_pylist()
        new_chunks=pq.read_table(a.population/'finance/chunks/atoms.parquet').to_pylist()
        profile={};readings_root=a.readings.parent
        require(a.readings.name=='readings-v1','REPLICATION_PATH')
        for name in NAMES:
            old_sets=native_sets(a.original,name,old_chunks,4343)
            with np.load(a.readings/name/'slots.npz',allow_pickle=False) as z:vectors,offsets=z['vectors'],z['offsets']
            grouped=defaultdict(list)
            for i,row in enumerate(new_chunks):grouped[row['row']].append(vectors[offsets[i]:offsets[i+1]])
            require(set(grouped)==set(range(600)),'REPLICATION_WINDOW_COVERAGE')
            sets=[unit(np.concatenate(grouped[i]).astype(float)) for i in range(600)]
            landmarks=[old_sets[old['rows'][i]] for i in old['landmarks']]
            prof=unit(np.array([[similarity(v,b) for b in landmarks] for v in sets]))
            independent=unit(np.array([[sim(v,b) for b in landmarks] for v in sets]))
            require(np.allclose(prof,independent,rtol=0,atol=1e-12),'REPLICATION_MAXSIM')
            profile[name]=prof
        labels=np.array(json.loads((a.population/'finance/anchors.json').read_bytes())['P159']['labels'])
        design=old['design'];x=profile[NAMES[-1]];blocks=[profile[n] for n in NAMES[:-1]]
        panel=np.concatenate(blocks,axis=1);pred=predict(model,panel)
        require(np.allclose(pred,prediction(model,panel),rtol=0,atol=1e-10),'REPLICATION_FROZEN_PREDICTION')
        xd=old[NAMES[-1]+'__profile'][design]
        sx=standardize(xd,x);sp=standardize(xd,pred)
        sz=[standardize(old[n+'__profile'][design],profile[n]) for n in NAMES[:-1]]
        variants=dict(panel=blocks,joint=blocks+[x],closed_panel=blocks+[pred],closed_joint=blocks+[pred,x],
            scaled_panel=sz,scaled_joint=sz+[sx],scaled_closed_panel=sz+[sp],scaled_closed_joint=sz+[sp,sx])
        arrays=dict(labels=labels,prediction=pred,**{n+'__profile':p for n,p in profile.items()});distances={}
        for name,values in variants.items():
            count,radius=indexed_counts(values,labels);actual,r=scan(values,labels,np.arange(len(labels)))
            require(np.array_equal(count,actual) and np.array_equal(radius,r),'REPLICATION_COUNTS')
            arrays[name+'_counts']=count;arrays[name+'_radii']=radius;distances[name]=distance(values)
        require(np.array_equal(distance(blocks+[blocks[-1]]),distances['panel']),'REPLICATION_DUPLICATE')
        with (a.output/'native-profiles-counts.npz').open('xb') as f:np.savez(f,**arrays)
        (a.output/'frozen-model.json').write_bytes(model_file.read_bytes())
        definitions=dict(raw=('joint','panel'),closed=('closed_joint','closed_panel'),scaled=('scaled_joint','scaled_panel'),scaled_closed=('scaled_closed_joint','scaled_closed_panel'))
        rows=[];permuted=np.random.default_rng(20261020).permutation(labels)
        for seed in (7,17,27,37,47):
            statistics,saved=intervals(labels,distances,definitions,seed)
            controls,negative=intervals(permuted,distances,definitions,seed)
            with (a.output/f'resamples-{seed}.npz').open('xb') as f:np.savez(f,**saved,**{'permuted_'+k:v for k,v in negative.items()})
            rows.append(dict(seed=seed,statistics=statistics,controls=controls))
        summary={method:dict(point=rows[0]['statistics'][method]['point'],outer_low=min(r['statistics'][method]['low'] for r in rows),
            outer_high=max(r['statistics'][method]['high'] for r in rows),permuted_point=rows[0]['controls'][method]['point'],
            permuted_outer_low=min(r['controls'][method]['low'] for r in rows)) for method in definitions}
        require(sha(model_file)==frozen['model_sha256'] and sha(profile_path)==frozen['source']['sha256'],'REPLICATION_FROZEN_AFTER')
        result=dict(complete=True,acceptance='author-checked',calibration_replication=True,scientific_admission=False,
            conditional_information_certified=False,served=False,independently_accepted=False,population=population,
            frozen_calibration_case=frozen['name'],frozen_model_sha256=frozen['model_sha256'],frozen_landmark_profile_sha256=frozen['source']['sha256'],
            native_fsv_sha256=sha(a.fsv/'fsv.json'),rows=rows,summary=summary,exact_duplicate_gain=0.,
            files={p.name:sha(p) for p in a.output.iterdir() if p.is_file()},wall_seconds=time.perf_counter()-start,
            code_sha256=sha(Path(__file__)),limitations=['same underlying Wikidata source collection','no pretraining-independence certificate','limited reconstruction family'])
        write(a.output/'report.json',result);write(a.output/'manifest.json',dict(report_sha256=sha(a.output/'report.json')))
        print(json.dumps(dict(summary=summary,population=population,wall_seconds=result['wall_seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('population','census','readings','fsv','original','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
