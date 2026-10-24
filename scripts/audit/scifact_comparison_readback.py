"""Independent source/count/donor/bootstrap reconstruction for relation controls."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.description_readback import feature_read,donor_read,percentile
from audit.scifact_population_readback import reference as population_reference
from audit.scifact_result_readback import read_replayed,score,equal
from text.evaluation_multiscale import file_sha

CONFIG=dict(kinds=["word","char3"],scopes=["claim","evidence","pair"],k=5,prior_weight=.1,
            seeds=[7,17,27,37,47],resamples=999,source_only=True)


def check(ok,code):
    if not ok:raise ValueError("MNLM_RELATION_COMPARISON_"+code)


def close(a,b,code,tol=1e-12):
    a=np.asarray(a);b=np.asarray(b);check(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        and np.allclose(a,b,rtol=0,atol=tol),code)


def reference(root,population,inventory,result,replay):
    ctx=population_reference(inventory);r=json.loads((root/"report.json").read_bytes())
    cold=read_replayed(result,r["result_sha256"],population,r["population_sha256"],inventory,replay,r["replay_sha256"],ctx)
    inv=json.loads(inventory.read_bytes());source=Path(inv["publication"]["path"])
    lines=lambda p:[json.loads(v) for v in p.read_bytes().splitlines()]
    claims={x["id"]:x for x in lines(source/"claims.jsonl")};docs={x["id"]:x for x in lines(source/"documents.jsonl")}
    rows=ctx["rows"];texts={s:[] for s in CONFIG["scopes"]}
    for row in rows:
        c=claims[row["claim"]]["text"];d=docs[row["document"]];e='\n'.join([d["title"]]+d["views"]["source_sentences"])
        texts["claim"].append(c);texts["evidence"].append(e);texts["pair"].append(c+'\n'+e)
    binding=[dict(case=r["id"],group=r["group"],partition=r["partition"],text_sha256={k:hashlib.sha256(v[i].encode()).hexdigest()
        for k,v in texts.items()}) for i,r in enumerate(rows)]
    ids=np.arange(len(rows));train=np.array([i for i,r in enumerate(rows) if r["partition"]=="train"])
    labels=np.array([{"NOT_ENOUGH_EVIDENCE":0,"CONTRADICT":1,"SUPPORT":2}[t["label"]] for t in ctx["targets"]])
    prior=np.array([np.sum(labels[train]==k)/len(train) for k in range(3)]);probabilities={"prior":np.tile(prior,(len(rows),1))};controls={}
    for kind in CONFIG["kinds"]:
        for scope in CONFIG["scopes"]:
            name=kind+'-'+scope;folder=root/name;features=feature_read(folder,kind,texts[scope],ids,train)
            chosen=np.load(folder/"donors.npy",allow_pickle=False);similarities=np.load(folder/"similarities.npy",allow_pickle=False)
            check(chosen.dtype==np.int64 and chosen.shape==similarities.shape==(1258,5),"DONOR_SHAPE")
            donor_read(folder,kind,features,ids,train,rows,chosen,similarities)
            pred=[]
            for i,donor in enumerate(chosen):
                selected=donor[donor>=0];check(all(j in train and rows[j]["group"]!=rows[i]["group"] for j in selected),"DONOR_SCOPE")
                empirical=np.array([np.sum(labels[selected]==k)/len(selected) for k in range(3)]) if len(selected) else prior
                pred.append(.9*empirical+.1*prior if len(selected) else prior)
            prob=np.array(pred);probabilities[name]=prob
            controls[name]=dict(features=features,state=json.loads((folder/"state.json").read_bytes()),donors=chosen,
                similarities=similarities,probabilities=prob,metrics=score(prob,rows,labels))
    for key in cold["arms"]:
        seed,name=key.split('/');probabilities[key]=np.load(result/("seed-"+seed)/name/"probabilities.npy",allow_pickle=False)
    evaluation=np.array([i for i,r in enumerate(rows) if r["partition"]=="evaluation"]);groups=sorted({rows[i]["group"] for i in evaluation})
    member=np.array([groups.index(rows[i]["group"]) for i in evaluation]);count=np.bincount(member)
    draw=np.stack([np.random.default_rng(s).integers(len(groups),size=(999,len(groups))) for s in CONFIG["seeds"]])
    weights=np.stack([np.bincount(v,minlength=len(groups)) for v in draw.reshape(-1,len(groups))]);denom=weights@count
    pairs=[(n,b) for n in probabilities if '/' in n for b in probabilities if '/' not in b]
    pairs += [(str(s)+"/joint-all",str(s)+"/"+n) for s in (25,26,27) for n in ("joint-public","claim-only","evidence-only","sentences-only")]
    samples={};findings={}
    for name,control in pairs:
        p=probabilities[name];b=probabilities[control]
        delta=np.array([[int(p[i].argmax()==labels[i])-int(b[i].argmax()==labels[i]),
            np.log(float(p[i,labels[i]]))-np.log(float(b[i,labels[i]]))] for i in evaluation])
        grouped=np.stack([delta[member==g].sum(0) for g in range(len(groups))]);sample=((weights@grouped)/denom[:,None]).reshape(5,999,2)
        key=name+'|'+control;samples[key]=sample
        per={str(s):dict(low=percentile(sample[j],.025).tolist(),high=percentile(sample[j],.975).tolist()) for j,s in enumerate(CONFIG["seeds"])}
        findings[key]=dict(point=delta.mean(0).tolist(),per_seed=per,low=np.min([v["low"] for v in per.values()],axis=0).tolist(),
            high=np.max([v["high"] for v in per.values()],axis=0).tolist())
    contrasts={}
    for split in ("train","calibration","evaluation"):
        docs=defaultdict(list)
        for i,row in enumerate(rows):
            if row["partition"]==split:docs[row["document"]].append(i)
        documents={d:ix for d,ix in docs.items() if any(labels[i]==1 for i in ix) and any(labels[i]==2 for i in ix)}
        pair=[(i,j) for ix in documents.values() for i in ix for j in ix if labels[i]==2 and labels[j]==1];arms={}
        for name,prob in probabilities.items():
            both=sum(int(int(prob[i].argmax())==2 and int(prob[j].argmax())==1) for i,j in pair)
            arms[name]=dict(both_correct=both,predicted_labels_differ=sum(int(prob[i].argmax()!=prob[j].argmax()) for i,j in pair),
                both_probabilities_ordered=sum(int(prob[i,2]>prob[j,2] and prob[j,1]>prob[i,1]) for i,j in pair),
                both_correct_rate=both/len(pair) if pair else None)
        contrasts[split]=dict(documents=len(documents),pairs=[list(v) for v in pair],arms=arms)
    return dict(sources=ctx["sources"],native=cold,bindings=binding,controls=controls,contrasts=contrasts,
        plan=dict(rows=evaluation,membership=member,counts=count,draw=draw),samples=samples,comparisons=dict(groups=groups,
            metrics=["head_minus_control_accuracy","control_minus_head_nll_nats"],findings=findings,
            scope="fixed evaluation overlap-component descriptive bootstrap; no IID, population-risk or Shannon CMI certificate"))


def read(root,pin,ctx):
    check(file_sha(root/"report.json")==pin,"PIN");r=json.loads((root/"report.json").read_bytes())
    check(r["schema"]=="scifact-comparisons-v1" and r["assertion_verdict"]=="REFUSE_UNADMITTED_PANEL"
        and all(r[k] is False for k in ("scientific_admission","baseline_is_sense","shannon_cmi_claim","served")),"CLAIM")
    check((r["population_sha256"],r["result_sha256"],r["replay_sha256"])==(ctx["native"]["population_sha256"],ctx["native"]["result_sha256"],ctx["native"]["fresh_replay_sha256"]),"PARENTS")
    names={"config.json","native-readback.json","input-rows.json","bootstrap-plan.npz","bootstrap-samples.npz","paired-comparisons.json","same-document-contrasts.json"}
    for name in ctx["controls"]:names|={name+'/'+v for v in ("state.json","features.npz","donors.npy","similarities.npy","probabilities.npy","metrics.json")}
    check(set(r["files"])==names and {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==names|{"report.json"},"FILES")
    for name,v in r["files"].items():check(file_sha(root/name)==v["sha256"] and (root/name).stat().st_size==v["bytes"],"FILE_HASH")
    load=lambda n:json.loads((root/n).read_bytes())
    check(load("config.json")==CONFIG and equal(load("native-readback.json"),ctx["native"]),"CONFIG_NATIVE")
    check(load("input-rows.json")==ctx["bindings"],"SOURCE_BINDINGS")
    for name,expected in ctx["controls"].items():
        check(load(name+"/state.json")==expected["state"],"TRAINING_FEATURE_STATE")
        got=sparse.load_npz(root/name/"features.npz");got.sort_indices();want=expected["features"]
        check(got.shape==want.shape,"FEATURES");close(got.indices,want.indices,"FEATURES",0);close(got.indptr,want.indptr,"FEATURES",0);close(got.data,want.data,"FEATURES")
        for field in ("donors","similarities","probabilities"):close(np.load(root/name/(field+".npy"),allow_pickle=False),expected[field],field.upper(),0 if field=="donors" else 1e-12)
        check(equal(load(name+"/metrics.json"),expected["metrics"]),"METRICS")
    for file,key in (("bootstrap-plan.npz","plan"),("bootstrap-samples.npz","samples")):
        with np.load(root/file,allow_pickle=False) as z:
            check(set(z.files)==set(ctx[key]),"BOOTSTRAP_KEYS")
            for name,value in ctx[key].items():close(z[name],value,"BOOTSTRAP_"+key.upper(),0 if key=="plan" else 1e-12)
    check(equal(load("paired-comparisons.json"),ctx["comparisons"]),"INTERVALS")
    check(equal(load("same-document-contrasts.json"),ctx["contrasts"]),"CONTRASTS")
    check(r["phase_coverage"]>=.9 and abs(sum(r["phases_seconds"].values())/r["wall_seconds"]-r["phase_coverage"])<1e-12,"PHASES")
    return dict(report_sha256=pin,result_sha256=r["result_sha256"],controls={n:v["metrics"] for n,v in ctx["controls"].items()},
        paired_comparisons=ctx["comparisons"],same_document_contrasts=ctx["contrasts"],scientific_admission=False,
        all_source_features_donors_predictions_metrics_intervals_reconstructed=True)
