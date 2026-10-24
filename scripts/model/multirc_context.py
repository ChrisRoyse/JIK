"""Source-bound question/answer/context maps; labels never enter native batches.

One cold source/native read per process. Compact arrays retain each selected
native window. Preparation is linear in source references and selected windows;
batch consumers gather fixed maps, never reopen documents. See #25 preflight.
"""
import json
from pathlib import Path
import numpy as np
from model.qasc_context import POPULATION,NATIVE,LANES,rows
from audit.connected_population_readback import readback as read_population
from audit.window_readback import readback as read_native
from text.multirc_anchor_quality import audit_raw
from text.evaluation_multiscale import file_sha

QUALITY={'conflicts.json':'fc5e626d7b620bad99c1a531d03a25d128b973a000b778b5c0bde8e0076c650d',
         'excluded-cases.json':'41aba273dd3015b33548c5711406efb7d05b39f41b5412a6ffe1ea5c6238f30a',
         'contrasts.json':'650d1cc2974c2a4f2559b006308772ec3d6ead78213edc19f7a9da5447739719'}

def check(ok,code):
    if not ok:raise ValueError('MNLM_MULTIRC_BINDING_'+code)

def load(population,native,quality,raw):
    check(file_sha(population/'report.json')==POPULATION and file_sha(native/'report.json')==NATIVE,'PARENT_PINS')
    cold=dict(population=read_population(population,raw/'multirc-training-v1',raw/'qasc-hf-training-v1',raw/'multirc-v1'),native=read_native(native,population))
    for name,pin in QUALITY.items():check(file_sha(quality/name)==pin,'QUALITY_PIN')
    quality_values=[json.loads((quality/name).read_bytes()) for name in QUALITY]
    cold['quality']=audit_raw(population,raw/'multirc-training-v1/train.json',*quality_values)
    excluded={r['case'] for r in quality_values[1]}
    cases=[c for c in rows(population/'cases.jsonl') if c['family']=='multirc'];refs={r['id']:r for r in rows(population/'references.jsonl')}
    targets={r['case']:r for r in rows(population/'private/targets.jsonl') if r['family']=='multirc'}
    assignment={r['case']:r for r in rows(population/'assignments.jsonl') if r['policy']=='multirc'}
    documents=rows(population/'documents.jsonl');docmap={d['id']:i for i,d in enumerate(documents)}
    check(len(cases)==5131 and len(documents)==456,'SOURCE_COUNTS')
    option_rows=[];labels=[];wanted=set();view_refs=[]
    for d in documents:
        rr=[refs[r] for r in [*d['sentences'],*d['pairs'],d['paragraph']]];view_refs.append(rr)
        check(all(r['owner']==d['id'] and r['unit'] is not None for r in rr),'CONTEXT_SOURCE')
        wanted.update(r['unit'] for r in rr)
    for case in cases:
        target=targets[case['id']];a=assignment[case['id']];eligible=a['eligible'];clean=case['id'] not in excluded
        check(len(case['options'])==len(target['labels']) and target['family']=='multirc','SOURCE_LABELS')
        for oi,rid in enumerate(case['options']):
            q=refs[case['question']];answer=refs[rid]
            if eligible:check(q['unit'] is not None and answer['unit'] is not None,'ELIGIBLE_ABSENCE');wanted.update((q['unit'],answer['unit']))
            option_rows.append(dict(index=len(option_rows),case=case['id'],document=docmap[case['document']],source_row=case['source_row'],
                source_question=case['source_question'],option=oi,partition=a['partition'],group=a['group'],available=eligible,quality_clean=clean,
                question_reference=case['question'],answer_reference=rid,question_unit=q['unit'],answer_unit=answer['unit']))
            labels.append(int(target['labels'][oi]))
    check(len(option_rows)==27243 and sum(r['available'] for r in option_rows)==27228 and len(wanted)==35701,'OPTION_COUNTS')
    train=np.asarray([i for i,r in enumerate(option_rows) if r['partition']=='train' and r['available'] and r['quality_clean']],np.int64)
    check(len(train)==17971,'TRAIN_SCOPE')
    native_arrays=[];maps=[];means=[];lane_pins={};unique_units=np.asarray(sorted(wanted),np.int64)
    for name in LANES:
        root=native/name;report=json.loads((root/'report.json').read_bytes());lane_pins[name]=dict(report_sha256=file_sha(root/'report.json'),lens_id=report['lens_id'])
        with np.load(root/'slots.npz',allow_pickle=False) as z:
            ptr=z['offsets'];all_ids=np.concatenate([np.arange(ptr[u],ptr[u+1],dtype=np.int64) for u in unique_units]);values=z['vectors'][all_ids].copy()
        def address(unit):
            check(unit is not None and ptr[unit+1]-ptr[unit]==1,'QUERY_OR_ANSWER_MULTIPLE_WINDOWS')
            j=int(np.searchsorted(all_ids,ptr[unit]));check(j<len(all_ids) and all_ids[j]==ptr[unit],'QUERY_OR_ANSWER_ADDRESS');return j
        q=np.full(len(option_rows),-1,np.int64);a=q.copy();contexts=[]
        for i,r in enumerate(option_rows):
            if r['available']:q[i]=address(r['question_unit']);a[i]=address(r['answer_unit'])
        for doc,rr in zip(documents,view_refs):
            seq=[];ns=len(doc['sentences'])
            for ref in rr:
                u=ref['unit'];lo,hi=ref['source_sentence_range'];count=int(ptr[u+1]-ptr[u]);level={'sentence':1,'sentence_pair':2,'paragraph':3}[ref['role']]
                for j,old in enumerate(range(ptr[u],ptr[u+1])):
                    ix=int(np.searchsorted(all_ids,old));check(ix<len(all_ids) and all_ids[ix]==old,'CONTEXT_ADDRESS')
                    seq.append((ix,level,(lo/ns,hi/ns,j/max(count-1,1))))
            contexts.append(seq)
        width=max(map(len,contexts));expected=64 if name=='all-mpnet-base-v2' else 63;check(width==expected,'CONTEXT_COVERAGE')
        index=np.zeros((len(documents),width),np.int64);mask=np.zeros_like(index,dtype=bool);levels=np.zeros_like(index);geometry=np.zeros((*index.shape,3),np.float32)
        for d,seq in enumerate(contexts):
            for j,(ix,level,geo) in enumerate(seq):index[d,j]=ix;mask[d,j]=True;levels[d,j]=level;geometry[d,j]=geo
        used=set(q[train].tolist())|set(a[train].tolist())
        for d in {option_rows[i]['document'] for i in train}:used.update(index[d,mask[d]].tolist())
        training_native=np.asarray(sorted(used),np.int64);mean=values[training_native].mean(0,dtype=np.float64).astype(np.float32)
        native_arrays.append(values);means.append(mean)
        maps.append(dict(question=q,answer=a,indices=index,mask=mask,levels=levels,geometry=geometry,source_native_rows=all_ids,mean_training_native_rows=training_native))
    check(sum(len(v) for v in native_arrays)==214626,'WINDOW_COUNTS')
    before={}
    for root in (population,native):
        before[str(root/'report.json')]=file_sha(root/'report.json');report=json.loads((root/'report.json').read_bytes())
        for name,v in report.get('files',{}).items():before[str(root/name)]=v['sha256']
    for name in LANES:
        root=native/name;before[str(root/'report.json')]=file_sha(root/'report.json')
        for fname,v in json.loads((root/'report.json').read_bytes())['files'].items():before[str(root/fname)]=v['sha256']
    for name,pin in QUALITY.items():before[str(quality/name)]=pin
    before[str(raw/'multirc-training-v1/train.json')]=file_sha(raw/'multirc-training-v1/train.json')
    return dict(native=native_arrays,maps=maps,means=means,rows=option_rows,labels=np.asarray(labels,np.int64),train=train,
                documents=documents,refs=refs,cases=cases,targets=targets,contrasts=quality_values[2],before=before,cold=cold,lanes=lane_pins)
