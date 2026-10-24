"""Cold source maps, raw annotation boundaries and diagnostic result arithmetic."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from model.multirc_context import check,LANES,QUALITY,POPULATION,NATIVE
from text.evaluation_multiscale import file_sha

def read_json(p):return json.loads(p.read_bytes())

def files(root):
    return {p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in root.rglob('*') if p.is_file()}

def manifest(root,report):
    for name,v in report['files'].items():
        p=root/name;check(p.resolve().is_relative_to(root.resolve()) and p.is_file(),'MANIFEST_PATH')
        check(p.stat().st_size==v['bytes'] and file_sha(p)==v['sha256'],'FILE_HASH')

def preparation(root,ctx,native,raw):
    report=read_json(root/'report.json');manifest(root,report)
    check(report['population_sha256']==POPULATION and report['native_sha256']==NATIVE and report['quality_payloads']==QUALITY,'PARENT_PINS')
    check(read_json(root/'rows.json')==ctx['rows'] and read_json(root/'lanes.json')==ctx['lanes'],'SOURCE_SCOPE')
    rr=ctx['rows'];original=read_json(raw/'multirc-training-v1/train.json')['data']
    labels=np.load(root/'labels.npy',allow_pickle=False)
    expected=np.array([original[r['source_row']]['paragraph']['questions'][r['source_question']]['answers'][r['option']]['isAnswer'] for r in rr],np.int64)
    check(np.array_equal(labels,expected) and np.array_equal(labels,ctx['labels']),'RAW_LABELS')
    train=np.load(root/'train.npy',allow_pickle=False)
    expected_train=np.array([i for i,r in enumerate(rr) if r['partition']=='train' and r['available'] and r['quality_clean']],np.int64)
    check(np.array_equal(train,expected_train) and np.array_equal(train,ctx['train']),'TRAIN_SCOPE')
    for k,name in enumerate(LANES):
        with np.load(root/(name+'.npz'),allow_pickle=False) as z:m={n:z[n] for n in z.files}
        check(set(m)==set(ctx['maps'][k])|{'mean'},'MAP_FIELDS')
        for key in ctx['maps'][k]:check(np.array_equal(m[key],ctx['maps'][k][key]),'MAP_'+key.upper())
        with np.load(native/name/'slots.npz',allow_pickle=False) as z:
            offsets=z['offsets'];source=z['vectors'];addresses=m['source_native_rows']
            check(np.array_equal(source[addresses],ctx['native'][k]),'NATIVE_VALUES')
            for i,r in enumerate(rr):
                if not r['available']:
                    check(m['question'][i]==m['answer'][i]==-1,'QUARANTINE');continue
                for role in ('question','answer'):
                    unit=r[role+'_unit'];index=m[role][i]
                    check(offsets[unit+1]-offsets[unit]==1 and addresses[index]==offsets[unit],'QUERY_SOURCE_ADDRESS')
            for d,doc in enumerate(ctx['documents']):
                position=0;ns=len(doc['sentences'])
                for rid in [*doc['sentences'],*doc['pairs'],doc['paragraph']]:
                    ref=ctx['refs'][rid];unit=ref['unit'];count=offsets[unit+1]-offsets[unit]
                    for window,source_row in enumerate(range(offsets[unit],offsets[unit+1])):
                        check(m['mask'][d,position] and addresses[m['indices'][d,position]]==source_row,'CONTEXT_SOURCE_ADDRESS')
                        check(m['levels'][d,position]=={'sentence':1,'sentence_pair':2,'paragraph':3}[ref['role']],'LEVEL')
                        expected_geo=np.array([ref['source_sentence_range'][0]/ns,ref['source_sentence_range'][1]/ns,window/max(count-1,1)],np.float32)
                        check(np.array_equal(m['geometry'][d,position],expected_geo),'GEOMETRY');position+=1
                check(m['mask'][d].sum()==position and not m['mask'][d,position:].any(),'WINDOW_COVERAGE')
            needed=set()
            for i in train:needed.update((int(m['question'][i]),int(m['answer'][i])))
            for d in {rr[i]['document'] for i in train}:needed.update(m['indices'][d,m['mask'][d]].tolist())
            needed=np.array(sorted(needed),np.int64)
            check(np.array_equal(needed,m['mean_training_native_rows']),'MEAN_SCOPE')
            mean=source[addresses[needed]].mean(0,dtype=np.float64).astype(np.float32)
            check(np.array_equal(mean,m['mean']) and np.array_equal(mean,ctx['means'][k]),'MEAN_TRAIN_ONLY')
    check(report['world_truth_assessed'] is False and report['scientific_admission'] is False and report['served'] is False,'CLAIMS')
    return dict(report_sha256=file_sha(root/'report.json'),raw_labels_equal=True,all_native_addresses_and_windows_equal=True,
                all_train_only_means_equal=True,options=len(rr),native_windows=sum(len(v) for v in ctx['native']))

def independent_metrics(logits,ids,ctx):
    """Separate scalar/count reconstruction; does not call producer scorer."""
    n=len(ctx['rows']);target=ctx['labels'];prediction=np.full(n,-1,np.int64);loss=np.zeros(n);error=np.zeros(n)
    check(logits.shape==(len(ids),2) and np.isfinite(logits).all(),'OUTPUT_VALUES')
    x=np.asarray(logits,np.float64);x-=x.max(1,keepdims=True);e=np.exp(x);p=e/e.sum(1,keepdims=True)
    prediction[ids]=p.argmax(1);loss[ids]=-np.log(np.maximum(p[np.arange(len(ids)),target[ids]],1e-300));error[ids]=(p[:,1]-target[ids])**2
    available=set(ids.tolist());question=defaultdict(list)
    for i,r in enumerate(ctx['rows']):question[r['case']].append(i)
    result={}
    for split in ('train','calibration','evaluation'):
        result[split]={}
        for cohort in ('original','eligible','clean'):
            selected=[i for i,r in enumerate(ctx['rows']) if r['partition']==split and (cohort=='original' or r['available']) and (cohort!='clean' or r['quality_clean'])]
            chosen=set(selected);observed=[i for i in selected if i in available];qq=[indices for indices in question.values() if indices[0] in chosen]
            exact=f1=paircount=both=0
            for q in qq:
                positives=[i for i in q if target[i]==1];negatives=[i for i in q if target[i]==0]
                right=sum(prediction[i]==target[i] for i in q);exact+=int(right==len(q))
                tp=sum(prediction[i]==1 for i in positives);den=sum(prediction[i]==1 for i in q)+len(positives)
                if all(i in available for i in q):f1+=2*tp/den if den else 1.
                paircount+=len(positives)*len(negatives);both+=tp*sum(prediction[i]==0 for i in negatives)
            correct=sum(prediction[i]==target[i] for i in selected)
            result[split][cohort]=dict(options=len(selected),questions=len(qq),components=len({ctx['rows'][i]['group'] for i in selected}),
                scored_options=len(observed),unavailable_options=len(selected)-len(observed),correct_options=int(correct),accuracy=correct/len(selected),
                nll_scored=float(loss[observed].mean()),brier_scored=float(error[observed].mean()),exact_questions=exact,question_exact=exact/len(qq),
                question_macro_f1=f1/len(qq),opposite_label_pairs=paircount,pairs_both_correct=int(both),pair_accuracy=both/paircount if paircount else None)
        lookup={(r['case'],r['option']):i for i,r in enumerate(ctx['rows'])};groups=pairs=both=0
        for c in ctx['contrasts']:
            if c['members'][0]['partition']!=split:continue
            pp=[lookup[m['case'],m['option']] for m in c['members'] if m['label']==1]
            nn=[lookup[m['case'],m['option']] for m in c['members'] if m['label']==0]
            groups+=1;pairs+=len(pp)*len(nn);both+=sum(prediction[i]==1 for i in pp)*sum(prediction[i]==0 for i in nn)
        result[split]['same_answer_contrasts']=dict(groups=groups,pairs=pairs,both_correct=int(both),pair_accuracy=both/pairs)
    return result

def close(actual,expected,code):
    if isinstance(expected,dict):
        check(isinstance(actual,dict) and set(actual)==set(expected),code)
        for k in expected:close(actual[k],expected[k],code)
    elif isinstance(expected,list):
        check(isinstance(actual,list) and len(actual)==len(expected),code)
        for a,b in zip(actual,expected):close(a,b,code)
    elif isinstance(expected,(float,np.floating)):check(np.isclose(actual,expected,rtol=1e-12,atol=1e-12),code)
    else:check(actual==expected,code)
