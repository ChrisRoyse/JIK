"""Description controls for the fixed source-readout experiment; never senses."""
import argparse
import json
from pathlib import Path
import re
import sys
import time
import unicodedata

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.anchor_readout_readback import read as read_native, metrics
from audit.anchor_population_readback import lines
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha

SCHEMA="source-description-controls-v1"
CONFIG=dict(k=5,prior_weight=.1,ranking_decimals=12,word_pattern=r"(?u)\b\w\w+\b",
            char_width=3,seeds=[7,17,27,37,47],resamples=999,
            scalars=["characters","utf8_bytes","words","digits","punctuation"],
            kinds=["word","char3","structure"])


def require(ok,code):
    if not ok:raise ValueError("MNLM_DESCRIPTION_"+code)


def char3(text):
    s=text.lower()
    return [s[i:i+3] for i in range(len(s)-2)]


def source_texts(population,rows):
    parents={f:lines(population/"parents/lineage/parents"/(f+"-rows.jsonl")) for f in ("biomedical","science")}
    import hashlib
    texts=[]
    for r in rows:
        source=parents[r["family"]][r["record_row"]];s=source["text"]
        require(source["entity"]==r["entity"] and hashlib.sha256(s.encode()).hexdigest()==r["text_sha256"],"SOURCE")
        texts.append(s)
    return texts


def descriptors(texts):
    return np.asarray([[len(s),len(s.encode()),len(re.findall(CONFIG["word_pattern"],s.lower())),
                        sum(c.isdigit() for c in s),sum(unicodedata.category(c).startswith('P') for c in s)] for s in texts],np.float64)


def fit(kind,texts,train,root,ids):
    state=dict(kind=kind,input_rows=ids.tolist(),training_rows=ids[train].tolist())
    if kind=="structure":
        raw=descriptors(texts);x=np.log1p(raw);mean=x[train].mean(0);scale=x[train].std(0);active=scale>0
        value=np.zeros_like(x);value[:,active]=(x[:,active]-mean[active])/scale[active]
        state.update(mean=mean.tolist(),scale=scale.tolist(),active=active.tolist())
        np.save(root/"raw.npy",raw);np.save(root/"features.npy",value)
    else:
        vectorizer=TfidfVectorizer(analyzer=char3 if kind=="char3" else "word",lowercase=kind=="word",
            token_pattern=CONFIG["word_pattern"] if kind=="word" else None,dtype=np.float64)
        vectorizer.fit([texts[i] for i in train]);value=vectorizer.transform(texts)
        state.update(vocabulary=vectorizer.get_feature_names_out().tolist(),idf=vectorizer.idf_.tolist())
        sparse.save_npz(root/"features.npz",value)
    publish(root/"state.json",state)
    return value


def donors(features,ids,train,groups,kind):
    chosen=np.full((len(ids),5),-1,np.int64);scores=np.zeros((len(ids),5),np.float64)
    for i in range(len(ids)):
        if kind=="structure":sim=1/(1+np.abs(features[train]-features[i]).mean(1))
        else:sim=(features[train]@features[i].T).toarray().ravel()
        allowed=[j for j in range(len(train)) if groups[train[j]]!=groups[i] and sim[j]>0]
        best=sorted(allowed,key=lambda j:(-round(float(sim[j]),12),int(ids[train[j]])))[:5]
        chosen[i,:len(best)]=ids[train[best]];scores[i,:len(best)]=sim[best]
    return chosen,scores


def distributions(chosen,rows):
    target=np.zeros((len(rows),6),np.float64)
    for i,r in enumerate(rows):target[i,r["positive_classes"]]=1/len(r["positive_classes"])
    priors={q:target[[i for i,r in enumerate(rows) if r["question"]==q and r["split"]=="train"]].mean(0) for q in range(3)}
    result=np.empty_like(target)
    for i,r in enumerate(rows):
        ix=chosen[i][chosen[i]>=0];base=priors[r["question"]]
        result[i]=.9*target[ix].mean(0)+.1*base if len(ix) else base
    return result


def difference(prob,reference,rows,ids):
    def values(p):
        p=np.asarray(p,dtype=np.float64)
        correct=np.asarray([int(p[i].argmax()) in rows[i]["positive_classes"] for i in ids],float)
        loss=np.asarray([-np.log(p[i,rows[i]["positive_classes"]].sum()) for i in ids])
        return correct,loss
    bc,bl=values(prob);rc,rl=values(reference)
    return np.stack([rc-bc,bl-rl],axis=1)


def bounds(samples,point):
    per={str(s):dict(low=np.quantile(samples[i],.025,axis=0).tolist(),high=np.quantile(samples[i],.975,axis=0).tolist()) for i,s in enumerate(CONFIG["seeds"])}
    return dict(point=point.tolist(),per_seed=per,low=np.min([v["low"] for v in per.values()],axis=0).tolist(),
                high=np.max([v["high"] for v in per.values()],axis=0).tolist())


def compare(root,result,rows,probabilities):
    arms={}
    for seed in (25,26,27):
        for name in ("public-six","all-ten"):
            arms[str(seed)+"/"+name]=np.load(result/("seed-"+str(seed))/name/"probabilities.npy",allow_pickle=False)
    findings={};all_samples={};plans={}
    for q in range(3):
        ids=np.asarray([i for i,r in enumerate(rows) if r["question"]==q and r["split"]=="evaluation"])
        groups=sorted({rows[i]["group"] for i in ids});lookup={g:j for j,g in enumerate(groups)}
        membership=np.asarray([lookup[rows[i]["group"]] for i in ids]);counts=np.bincount(membership,minlength=len(groups))
        draw=np.asarray([np.random.default_rng(s).integers(0,len(groups),size=(999,len(groups))) for s in CONFIG["seeds"]],np.int32)
        np.savez(root/("bootstrap-"+str(q)+".npz"),draw=draw,rows=ids,membership=membership,counts=counts)
        plans[str(q)]=dict(groups=groups,rows=len(ids));denom=counts[draw].sum(2)
        for kind,prob in probabilities.items():
            for name,reference in arms.items():
                delta=difference(prob,reference,rows,ids);sums=np.zeros((len(groups),2))
                np.add.at(sums,membership,delta)
                samples=sums[draw].sum(2)/denom[:,:,None]
                key=kind+"|"+name+"|"+str(q);all_samples[key]=samples;findings[key]=bounds(samples,delta.mean(0))
    np.savez(root/"paired-samples.npz",**all_samples)
    publish(root/"paired-comparisons.json",dict(plans=plans,metrics=["native_minus_baseline_accuracy","baseline_minus_native_nll_nats"],
            scope="evaluation overlap-component descriptive bootstrap; no IID or CMI certificate",findings=findings))


def run(a):
    require(not a.output.exists(),"EXISTS");start=time.perf_counter();phases={}
    cold=read_native(a.population,a.result,a.result_sha256,audit_inputs=False)
    rows=lines(a.population/"rows.jsonl");texts=source_texts(a.population,rows)
    a.output.mkdir(parents=True);publish(a.output/"config.json",CONFIG)
    publish(a.output/"native-reference-metrics.json",cold["arms"])
    publish(a.output/"input-rows.json",[dict(case=r["case"],text_sha256=r["text_sha256"],group=r["group"],split=r["split"],question=r["question"]) for r in rows])
    phases["cold_source_and_native_result"]=time.perf_counter()-start;tick=time.perf_counter();probabilities={};counts={}
    for kind in CONFIG["kinds"]:
        folder=a.output/kind;folder.mkdir();chosen=np.full((len(rows),5),-1,np.int64);scores=np.zeros((len(rows),5),np.float64)
        for q in range(3):
            ids=np.asarray([i for i,r in enumerate(rows) if r["question"]==q]);train=np.asarray([j for j,i in enumerate(ids) if rows[i]["split"]=="train"])
            part=folder/str(q);part.mkdir();features=fit(kind,[texts[i] for i in ids],train,part,ids)
            phases[kind+str(q)+"_features"]=time.perf_counter()-tick;tick=time.perf_counter()
            chosen[ids],scores[ids]=donors(features,ids,train,[rows[i]["group"] for i in ids],kind)
            phases[kind+str(q)+"_donors"]=time.perf_counter()-tick;tick=time.perf_counter()
        p=distributions(chosen,rows);np.save(folder/"donors.npy",chosen);np.save(folder/"similarities.npy",scores);np.save(folder/"probabilities.npy",p)
        publish(folder/"metrics.json",metrics(p,rows));probabilities[kind]=p
        counts[kind]=dict(no_positive_description_match=int(np.sum(chosen[:,0]<0)),production_assertions=0)
        phases[kind+"_probabilities_metrics"]=time.perf_counter()-tick;tick=time.perf_counter()
    compare(a.output,a.result,rows,probabilities)
    phases["paired_group_bootstrap"]=time.perf_counter()-tick;tick=time.perf_counter()
    require(cold==read_native(a.population,a.result,a.result_sha256,audit_inputs=False),"SOURCE_DRIFT")
    files={p.relative_to(a.output).as_posix():dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in a.output.rglob('*') if p.is_file()}
    phases["source_drift_and_output_hashes"]=time.perf_counter()-tick;wall=time.perf_counter()-start
    report=dict(schema=SCHEMA,config=CONFIG,rows=len(rows),population_sha256=cold["population_sha256"],result_sha256=a.result_sha256,
        files=files,scientific_admission=False,production_assertions=False,shannon_cmi_claim=False,baseline_is_sense=False,source_truth_only=True,
        refusal="MNLM_DESCRIPTION_DIAGNOSTIC_ONLY",counts=counts,sklearn_version=sklearn.__version__,code_sha256=file_sha(Path(__file__)),
        phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(a.output/"report.json",report)
    print(json.dumps(dict(report_sha256=file_sha(a.output/"report.json"),counts=counts,wall_seconds=wall,phase_coverage=report["phase_coverage"])))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("population","result","output"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--result-sha256",required=True);run(p.parse_args())
