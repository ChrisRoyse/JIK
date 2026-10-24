"""Independent cold objectives, split boundaries, scores and bootstrap readback."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from model.qasc_context import check,POPULATION,NATIVE,POLICIES
from model.qasc_combination import CONFIG,ARMS
from text.evaluation_multiscale import file_sha

def equal(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return isinstance(b,(float,int)) and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9)
    return a==b

def score(p,rows,y):
    check(p.dtype==np.float32 and p.shape==(8134,8) and np.isfinite(p).all() and (p>0).all() and np.allclose(p.sum(1),1,rtol=0,atol=2e-6),'COLD_PROBABILITY')
    subsets={s:[i for i,r in enumerate(rows) if r['partition']==s] for s in ('train','calibration','evaluation')};out={}
    for s in ('calibration','evaluation'):
        for n in (0,1,2):subsets[f'{s}/facts-seen-{n}']=[i for i in subsets[s] if rows[i]['seen_training_facts']==n]
    for key,ids in subsets.items():
        if not ids:out[key]=dict(cases=0);continue
        a=p[ids].astype(float);truth=y[ids];pred=np.argmax(a,1);confidence=np.max(a,1);curves={}
        for t in (.5,.75,.9,.95,.99):
            accepted=[j for j,c in enumerate(confidence) if c>=t];wrong=sum(pred[j]!=truth[j] for j in accepted);n=len(accepted)
            curves[str(t)]=dict(accepted=n,wrong=int(wrong),coverage=n/len(ids),error_rate=float(wrong/n) if n else None,production_assertions=0)
        nll=math.fsum(-math.log(a[j,v]) for j,v in enumerate(truth))/len(ids)
        brier=math.fsum(math.fsum((v-int(k==truth[j]))**2 for k,v in enumerate(row)) for j,row in enumerate(a))/len(ids)
        out[key]=dict(cases=len(ids),groups=len({rows[i]['group'] for i in ids}),accuracy=float(np.count_nonzero(pred==truth)/len(ids)),nll_nats=nll,brier=brier,confidence=curves)
    return out,subsets

def bootstrap(root,outputs,rows,labels,seed,subsets):
    saved=json.loads((root/'comparisons.json').read_bytes());summary={};rng=np.random.default_rng(seed)
    with np.load(root/'bootstrap-plan.npz') as plan,np.load(root/'bootstrap-samples.npz') as samples:
        expected_keys=[]
        for subset,ids in subsets.items():
            if not subset.startswith('evaluation') or not ids:continue
            ids=np.asarray(ids);groups=sorted({rows[i]['group'] for i in ids});members=[[j for j,i in enumerate(ids) if rows[i]['group']==g] for g in groups]
            expected=rng.integers(0,len(groups),size=(500,len(groups)));check(np.array_equal(plan[subset],expected),'COLD_BOOTSTRAP_PLAN')
            base=outputs['both-facts']['probabilities'][ids].argmax(1)==labels[ids]
            alternatives={n:o['probabilities'] for n,o in outputs.items() if n!='both-facts'}
            alternatives.update({'intervention-'+n:p for n,p in outputs['both-facts'].items() if n!='probabilities'})
            for name,p in alternatives.items():
                key=subset+'|'+name;delta=base.astype(float)-(p[ids].argmax(1)==labels[ids]);sums=np.array([delta[ms].sum() for ms in members]);counts=np.array([len(ms) for ms in members])
                values=sums[expected].sum(1)/counts[expected].sum(1);check(np.array_equal(samples[key],values),'COLD_BOOTSTRAP_SAMPLES')
                summary[key]=dict(cases=len(ids),groups=len(groups),paired_accuracy_difference=float(delta.mean()),percentile_95=np.quantile(values,[.025,.975]).tolist(),draws=500)
                expected_keys.append(key)
        check(set(samples.files)==set(expected_keys),'COLD_BOOTSTRAP_KEYS')
    check(equal(summary,saved),'COLD_COMPARISONS');return summary

def read(root,ctx,rehash=True):
    j=lambda n:json.loads((root/n).read_bytes());r=j('report.json')
    check(r['schema']=='qasc-combination-results-v1' and r['complete'] is True and r['errors']==[],'COLD_COMPLETE')
    check(r['population_sha256']==POPULATION and r['native_sha256']==NATIVE,'COLD_PARENTS')
    check(all(r[k] is False for k in ('scientific_admission','served','sealed_evaluation','free_text_composition_proven'))
          and r['production_assertions']==0 and r['assertion_verdict']=='REFUSE_UNADMITTED_PANEL' and r['source_supplied_evidence'] is True,'COLD_CLAIMS')
    if rehash:
        for n,v in r['files'].items():
            p=(root/n).resolve();check(p.is_relative_to(root.resolve()),'COLD_PATH');check(p.stat().st_size==v['bytes'] and file_sha(p)==v['sha256'],'COLD_FILE_HASH')
    check({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==set(r['files'])|{'report.json'},'COLD_FILE_SET')
    check(j('config.json')==CONFIG and j('before.json')==j('after.json')==ctx['before'],'COLD_BOUNDARY')
    for n,pin in j('runtime.json')['code'].items():check(file_sha(Path(__file__).parents[1]/'model'/n)==pin,'COLD_CODE')
    check(j('policies.json')==ctx['policies'] and j('lanes.json')==ctx['lanes'],'COLD_POLICIES')
    with np.load(root/'gather.npz') as z:check(set(z.files)=={'maps','units','original'} and all(np.array_equal(z[k],ctx[k]) for k in z.files),'COLD_GATHER')
    labels=np.load(root/'labels.npy');check(np.array_equal(labels,ctx['labels']),'COLD_LABELS')
    all_outputs={};all_metrics={};comparisons={};initials={};expected_arms=[]
    for policy in POLICIES:
        rows=ctx['policies'][policy];train=np.array([i for i,v in enumerate(rows) if v['partition']=='train']);folder=root/policy
        donor=np.load(folder/'donors.npy');check(donor.shape==(8134,) and np.all((donor>=0)&(donor<8134)),'COLD_DONOR_SHAPE')
        check(all(rows[i]['partition']==rows[d]['partition'] and rows[i]['group']!=rows[d]['group'] for i,d in enumerate(donor)),'COLD_DONOR_CROSSING')
        for seed in (25,26,27):
            seedroot=folder/f'seed-{seed}';steps=np.load(seedroot/'training-rows.npy');rng=np.random.default_rng(seed)
            expected=np.array([rng.choice(train,32,replace=False) for _ in range(400)]);check(np.array_equal(steps,expected),'COLD_TRAINING_SPLIT')
            initial=torch.load(seedroot/'initial.pt',map_location='cpu',weights_only=True)
            if seed in initials:check(all(torch.equal(v,initials[seed][k]) for k,v in initial.items()),'COLD_MATCHED_INITIAL')
            else:initials[seed]=initial
            outputs={}
            for name in [*ARMS,*(['shuffled'] if seed==25 else [])]:
                arm=seedroot/name;key=f'{policy}/seed-{seed}/{name}';expected_arms.append(key);ar=json.loads((arm/'report.json').read_bytes())
                check(file_sha(arm/'report.json')==r['arms'][key] and ar['arm']==name and ar['seed']==seed and ar['policy']==policy,'COLD_ARM_BINDING')
                check(ar['initial_sha256']==file_sha(seedroot/'initial.pt') and ar['model_sha256']==file_sha(arm/'model.pt'),'COLD_MODEL_PIN')
                targets=labels[train] if name!='shuffled' else np.random.default_rng(seed).permutation(labels[train])
                check(np.array_equal(np.load(arm/'training-labels.npy'),targets),'COLD_TRAINING_LABELS')
                with np.load(arm/'objective.npz') as z:
                    check(np.array_equal(z['rows'],steps[0]) and np.array_equal(z['labels'],targets[np.searchsorted(train,steps[0])]),'COLD_OBJECTIVE_TARGET')
                    losses=[float(v.max())+math.log(math.fsum(math.exp(float(t)-float(v.max())) for t in v))-float(v[y]) for v,y in zip(z['logits'],z['labels'])];loss=math.fsum(losses)/32
                obj=json.loads((arm/'objective.json').read_bytes());check(abs(obj['loss']-loss)<1e-5,'COLD_OBJECTIVE_LOSS')
                state=torch.load(arm/'model.pt',map_location='cpu',weights_only=True)
                check(set(state)==set(initial)==set(obj['gradients']) and any(not torch.equal(v,initial[k]) for k,v in state.items()),'COLD_LEARNING')
                check(all(v.shape==initial[k].shape and v.dtype==torch.float32 and torch.isfinite(v).all() for k,v in state.items()) and ar['parameters']==sum(v.numel() for v in state.values()),'COLD_MODEL_SHAPE')
                losses=np.load(arm/'losses.npy');check(losses.shape==(400,) and np.isfinite(losses).all() and abs(losses[0]-loss)<1e-5,'COLD_LOSS_TRACE')
                with np.load(arm/'outputs.npz') as z:o={n:z[n].copy() for n in z.files}
                check(set(o)==({'probabilities','fact1-only','fact2-only','question-only','other-group-facts'} if name=='both-facts' else {'probabilities'}),'COLD_OUTPUT_KEYS')
                for p in o.values():metric,subsets=score(p,rows,labels)
                metric,subsets=score(o['probabilities'],rows,labels);check(equal(metric,json.loads((arm/'metrics.json').read_bytes())),'COLD_METRICS')
                control=json.loads((arm/'controls.json').read_bytes());check(control['private_target_refused'] and control['removed_inputs_bitwise_irrelevant'] and max(control['choice_permutation_max_error'],control['fact_swap_max_error'])<2e-5,'COLD_CONTROLS')
                timing=json.loads((arm/'training-time.json').read_bytes());check(timing['updates']==400 and timing['case_updates']==12800 and timing['seconds']>0 and timing['peak_reserved_bytes']>=timing['peak_allocated_bytes']>0,'COLD_TRAINING_TIME')
                check(.9<=ar['phase_coverage']<=1 and math.isclose(sum(ar['phases_seconds'].values())/ar['wall_seconds'],ar['phase_coverage'],abs_tol=1e-12),'COLD_ARM_PHASES')
                all_metrics[key]=dict(metrics=metric,training=timing,model_sha256=ar['model_sha256'],parameters=ar['parameters']);outputs[name]=o;all_outputs[key]=o
            comparisons[f'{policy}/seed-{seed}']=bootstrap(seedroot,outputs,rows,labels,seed,subsets)
    check(set(expected_arms)==set(r['arms']),'COLD_ARM_SET')
    check(.9<=r['phase_coverage']<=1 and math.isclose(sum(r['phases_seconds'].values())/r['wall_seconds'],r['phase_coverage'],abs_tol=1e-12),'COLD_PHASES')
    return dict(cold=dict(report_sha256=file_sha(root/'report.json'),arms=all_metrics,comparisons=comparisons,scientific_admission=False,served=False,production_assertions=0),outputs=all_outputs)
