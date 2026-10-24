"""Independent text/count/donor/cluster readback; no producer or sklearn imports."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.anchor_readout_readback import read as native_read,metrics,same_metrics
from audit.anchor_population_readback import lines
from text.evaluation_multiscale import file_sha

CONFIG=dict(k=5,prior_weight=.1,ranking_decimals=12,word_pattern=r"(?u)\b\w\w+\b",
            char_width=3,seeds=[7,17,27,37,47],resamples=999,
            scalars=["characters","utf8_bytes","words","digits","punctuation"],kinds=["word","char3","structure"])


def check(ok,code):
    if not ok:raise ValueError("MNLM_DESCRIPTION_READ_"+code)


def equal(a,b,code,tol=1e-12):
    a=np.asarray(a);b=np.asarray(b)
    check(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,rtol=0,atol=tol),code)


def reconstruct(population,rows):
    chunks=[]
    for family in ("biomedical","science"):
        chunks.extend(pq.read_table(population/"parents/calibration"/family/"chunks/atoms.parquet").to_pylist())
    texts=[]
    for row in rows:
        parts=[chunks[j] for j in row["chunks"]];text=''.join(p["text"] for p in parts);end=0
        for part in parts:
            check(part["start"]==end and part["entity"]==row["entity"] and text[part["start"]:part["end"]]==part["text"],"CHUNK_JOIN")
            end=part["end"]
        check(end==len(text) and hashlib.sha256(text.encode()).hexdigest()==row["text_sha256"],"SOURCE")
        texts.append(text)
    return texts


def counts(text,kind):
    text=text.lower()
    if kind=="word":return Counter(m.group() for m in re.finditer(r"(?u)\b\w\w+\b",text))
    return Counter(text[j:j+3] for j in range(max(0,len(text)-2)))


def feature_read(root,kind,texts,ids,train):
    state=json.loads((root/"state.json").read_bytes())
    extra={"mean","scale","active"} if kind=="structure" else {"vocabulary","idf"}
    check(set(state)=={"kind","input_rows","training_rows"}|extra and state["kind"]==kind,"FEATURE_STATE")
    check(state["input_rows"]==ids.tolist() and state["training_rows"]==ids[train].tolist(),"TRAINING_ROWS")
    if kind=="structure":
        raw=np.asarray([[len(t),len(t.encode('utf-8')),sum(1 for _ in re.finditer(r"(?u)\b\w\w+\b",t.lower())),
            sum(1 for c in t if c.isdigit()),sum(1 for c in t if unicodedata.category(c)[0]=='P')] for t in texts],np.float64)
        equal(np.load(root/"raw.npy",allow_pickle=False),raw,"SCALARS",0)
        x=np.log1p(raw);mean=np.sum(x[train],axis=0)/len(train);scale=np.sqrt(np.sum((x[train]-mean)**2,axis=0)/len(train));active=scale>0
        equal(state["mean"],mean,"NORMALIZATION");equal(state["scale"],scale,"NORMALIZATION");equal(state["active"],active,"NORMALIZATION",0)
        features=np.zeros_like(x)
        for j in range(5):
            if active[j]:features[:,j]=(x[:,j]-mean[j])/scale[j]
        equal(np.load(root/"features.npy",allow_pickle=False),features,"FEATURES")
        return features
    bags=[counts(t,kind) for t in texts];vocabulary=sorted(set().union(*(set(bags[j]) for j in train)))
    check(state["vocabulary"]==vocabulary,"TRAINING_VOCABULARY")
    df=Counter()
    for j in train:df.update(bags[j].keys())
    idf=np.asarray([math.log((1+len(train))/(1+df[w]))+1 for w in vocabulary]);equal(state["idf"],idf,"IDF")
    lookup={w:j for j,w in enumerate(vocabulary)};ptr=[0];indices=[];data=[]
    for bag in bags:
        pairs=sorted((lookup[w],n) for w,n in bag.items() if w in lookup)
        values=np.asarray([n*idf[j] for j,n in pairs]);norm=math.sqrt(float(values@values))
        if norm:values/=norm
        indices.extend(j for j,n in pairs);data.extend(values);ptr.append(len(indices))
    features=sparse.csr_matrix((np.asarray(data),np.asarray(indices),np.asarray(ptr)),shape=(len(texts),len(vocabulary)))
    saved=sparse.load_npz(root/"features.npz");saved.sort_indices()
    check(saved.shape==features.shape,"FEATURES");equal(saved.indptr,features.indptr,"FEATURES",0);equal(saved.indices,features.indices,"FEATURES",0)
    equal(saved.data,features.data,"FEATURES")
    return features


def donor_read(root,kind,features,ids,train,rows,chosen,similarities):
    for j,i in enumerate(ids):
        if kind=="structure":values=1/(1+np.sum(np.abs(features[train]-features[j]),axis=1)/5)
        else:values=(features[j]@features[train].T).toarray()[0]
        ranked=sorted(((round(float(values[k]),12),int(ids[t]),float(values[k])) for k,t in enumerate(train)
            if rows[ids[t]]["group"]!=rows[i]["group"] and values[k]>0),key=lambda x:(-x[0],x[1]))[:5]
        expected=[v[1] for v in ranked]+[-1]*(5-len(ranked));scores=[v[2] for v in ranked]+[0]*(5-len(ranked))
        equal(chosen[i],expected,"DONORS",0);equal(similarities[i],scores,"SIMILARITIES")


def predict(chosen,rows):
    priors={};target={}
    for i,r in enumerate(rows):
        if r["split"]=="train":
            vector=np.zeros(6)
            for j in r["positive_classes"]:vector[j]=1/len(r["positive_classes"])
            target[i]=vector
    for q in range(3):priors[q]=sum(target[i] for i in target if rows[i]["question"]==q)/sum(rows[i]["question"]==q for i in target)
    output=[]
    for i,r in enumerate(rows):
        donors=[int(j) for j in chosen[i] if j>=0]
        check(all(j in target and rows[j]["question"]==r["question"] and rows[j]["group"]!=r["group"] for j in donors),"DONOR_SCOPE")
        output.append(.9*sum(target[j] for j in donors)/len(donors)+.1*priors[r["question"]] if donors else priors[r["question"]])
    return np.asarray(output)


def percentile(x,p):
    ordered=np.sort(x,axis=0);at=p*(len(x)-1);lo=int(at);hi=min(len(x)-1,lo+1)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(at-lo)


def pair_read(root,result,rows,probabilities):
    report=json.loads((root/"paired-comparisons.json").read_bytes());check(set(report)=={"plans","metrics","scope","findings"},"COMPARISONS")
    check(report["metrics"]==["native_minus_baseline_accuracy","baseline_minus_native_nll_nats"] and
          report["scope"]=="evaluation overlap-component descriptive bootstrap; no IID or CMI certificate","COMPARISONS")
    expected={};plans={}
    with np.load(root/"paired-samples.npz",allow_pickle=False) as saved:
        for q in range(3):
            ids=np.asarray([i for i,r in enumerate(rows) if r["question"]==q and r["split"]=="evaluation"])
            groups=sorted({rows[i]["group"] for i in ids});membership=np.asarray([groups.index(rows[i]["group"]) for i in ids]);count=np.bincount(membership)
            draw=np.stack([np.random.default_rng(seed).integers(len(groups),size=(999,len(groups))) for seed in (7,17,27,37,47)])
            with np.load(root/("bootstrap-"+str(q)+".npz"),allow_pickle=False) as z:
                check(set(z.files)=={"draw","rows","membership","counts"},"BOOTSTRAP_PLAN")
                for k,v in dict(draw=draw,rows=ids,membership=membership,counts=count).items():equal(z[k],v,"BOOTSTRAP_PLAN",0)
            weights=np.asarray([np.bincount(sample,minlength=len(groups)) for sample in draw.reshape(-1,len(groups))]);denom=weights@count
            plans[str(q)]=dict(groups=groups,rows=len(ids))
            for kind,base in probabilities.items():
                for seed in (25,26,27):
                    for name in ("public-six","all-ten"):
                        native=np.load(result/("seed-"+str(seed))/name/"probabilities.npy",allow_pickle=False);d=[]
                        for i in ids:
                            positive=rows[i]["positive_classes"]
                            d.append([int(int(native[i].argmax()) in positive)-int(int(base[i].argmax()) in positive),
                                      math.log(sum(float(native[i,j]) for j in positive))-math.log(sum(float(base[i,j]) for j in positive))])
                        delta=np.asarray(d);grouped=np.stack([delta[membership==g].sum(0) for g in range(len(groups))])
                        sample=((weights@grouped)/denom[:,None]).reshape(5,999,2);key=kind+"|"+str(seed)+"/"+name+"|"+str(q)
                        equal(saved[key],sample,"BOOTSTRAP_SAMPLES")
                        per={str(s):dict(low=percentile(sample[j],.025).tolist(),high=percentile(sample[j],.975).tolist()) for j,s in enumerate((7,17,27,37,47))}
                        value=report["findings"][key];point=delta.mean(0);low=np.min([v["low"] for v in per.values()],axis=0);high=np.max([v["high"] for v in per.values()],axis=0)
                        check(set(value)=={"point","per_seed","low","high"} and set(value["per_seed"])==set(per),"COMPARISONS")
                        for k,v in dict(point=point,low=low,high=high).items():equal(value[k],v,"COMPARISONS")
                        for s,v in per.items():
                            for k,x in v.items():equal(value["per_seed"][s][k],x,"COMPARISONS")
                        expected[key]=value
        check(set(saved.files)==set(expected)==set(report["findings"]) and plans==report["plans"],"COMPARISON_SET")
    return expected


def read(population,result,root,pin,audit_features=True):
    check(file_sha(root/"report.json")==pin,"REPORT_PIN");r=json.loads((root/"report.json").read_bytes())
    check(r["schema"]=="source-description-controls-v1" and r["config"]==CONFIG and r["rows"]==1512,"CONTRACT")
    check(all(r[k] is False for k in ("scientific_admission","production_assertions","shannon_cmi_claim","baseline_is_sense"))
          and r["source_truth_only"] is True and r["refusal"]=="MNLM_DESCRIPTION_DIAGNOSTIC_ONLY","CLAIM")
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()};check(actual==set(r["files"])|{"report.json"},"FILE_SET")
    for name,f in r["files"].items():check((root/name).stat().st_size==f["bytes"] and file_sha(root/name)==f["sha256"],"FILE_PIN")
    check(json.loads((root/"config.json").read_bytes())==CONFIG,"CONTRACT")
    native=native_read(population,result,r["result_sha256"],audit_inputs=False);check(native["population_sha256"]==r["population_sha256"],"POPULATION_PIN")
    reference=json.loads((root/"native-reference-metrics.json").read_bytes());check(set(reference)==set(native["arms"]),"NATIVE_REFERENCE")
    for arm,values in native["arms"].items():same_metrics(reference[arm],values)
    rows=lines(population/"rows.jsonl");texts=reconstruct(population,rows)
    expected=[dict(case=x["case"],text_sha256=x["text_sha256"],group=x["group"],split=x["split"],question=x["question"]) for x in rows]
    check(json.loads((root/"input-rows.json").read_bytes())==expected,"INPUT_ROWS")
    probabilities={};summary={};counts={}
    for kind in CONFIG["kinds"]:
        folder=root/kind;chosen=np.load(folder/"donors.npy",allow_pickle=False);scores=np.load(folder/"similarities.npy",allow_pickle=False)
        check(chosen.shape==scores.shape==(1512,5) and chosen.dtype==np.int64 and np.isfinite(scores).all(),"DONOR_ARRAYS")
        if audit_features:
            for q in range(3):
                ids=np.asarray([i for i,x in enumerate(rows) if x["question"]==q]);train=np.asarray([j for j,i in enumerate(ids) if rows[i]["split"]=="train"])
                features=feature_read(folder/str(q),kind,[texts[i] for i in ids],ids,train)
                donor_read(folder,kind,features,ids,train,rows,chosen,scores)
        prob=np.load(folder/"probabilities.npy",allow_pickle=False);equal(prob,predict(chosen,rows),"PROBABILITIES")
        check((prob>0).all() and np.allclose(prob.sum(1),1,rtol=0,atol=1e-12),"PROBABILITIES")
        values=metrics(prob,rows);same_metrics(json.loads((folder/"metrics.json").read_bytes()),values)
        summary[kind]=values;probabilities[kind]=prob;counts[kind]=dict(no_positive_description_match=int(np.sum(chosen[:,0]<0)),production_assertions=0)
    check(r["counts"]==counts,"COUNTS")
    pairs=pair_read(root,result,rows,probabilities)
    return dict(author_checked=True,report_sha256=pin,population_sha256=r["population_sha256"],result_sha256=r["result_sha256"],
        baselines=summary,native_reference=native["arms"],paired_comparisons=pairs,counts=counts,rows=1512,
        production_refusals=4536,scientific_admission=False,source_truth_only=True)
