"""Reconstruct complete binding-run scope, training boundaries, and all metrics."""
import numpy as np
import torch
from model.multirc_context import check
from model.multirc_binding import CONFIG,ARMS,schedule,donors
from audit.multirc_binding_readback import read_json,manifest,preparation,independent_metrics,close
from text.evaluation_multiscale import file_sha

def component_scores(logits,ids,ctx):
    rr=ctx['rows'];y=ctx['labels'];pred=np.full(len(rr),-1,np.int64);loss=np.zeros(len(rr));err=np.zeros(len(rr))
    z=np.asarray(logits,np.float64);z=z-z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True)
    pred[ids]=p.argmax(1);loss[ids]=-np.log(np.maximum(p[np.arange(len(ids)),y[ids]],1e-300));err[ids]=(p[:,1]-y[ids])**2
    groups=sorted({r['group'] for r in rr if r['partition']=='evaluation' and r['available'] and r['quality_clean']})
    lookup={v:i for i,v in enumerate(groups)};a=np.zeros((len(groups),7));qs={}
    for i,r in enumerate(rr):
        if r['partition']=='evaluation' and r['available'] and r['quality_clean']:
            qs.setdefault(r['case'],[]).append(i);a[lookup[r['group']],:4]+=[1,int(pred[i]==y[i]),loss[i],err[i]]
    for ii in qs.values():
        j=lookup[rr[ii[0]]['group']];tp=int(((pred[ii]==1)&(y[ii]==1)).sum());den=int((pred[ii]==1).sum()+(y[ii]==1).sum())
        a[j,4:]+=[1,int(np.array_equal(pred[ii],y[ii])),2*tp/den if den else 1.]
    return a

def bootstrap(a,b):
    def rate(x):return x[..., [1,2,3,5,6]]/x[..., [0,0,0,4,4]]
    intervals=[]
    for seed in range(5):
        samples=np.random.default_rng(seed).integers(0,len(a),size=(999,len(a)))
        difference=rate(a[samples].sum(axis=1))-rate(b[samples].sum(axis=1))
        intervals.append(np.quantile(difference,[.025,.975],axis=0))
    values=np.array(intervals)
    return dict(metrics=['accuracy','nll','brier','question_exact','question_macro_f1'],difference=(rate(a.sum(0))-rate(b.sum(0))).tolist(),
                seed_intervals=values.tolist(),descriptive_envelope=[values[:,0].min(0).tolist(),values[:,1].max(0).tolist()],
                groups=len(a),draws_per_seed=999,estimator_seeds=5,sealed_confirmation=False)

def pair_components(logits,ids,ctx):
    """Exact grouped pair counts; never materialize an all-pairs matrix."""
    rr=ctx['rows'];y=ctx['labels'];pred=np.full(len(rr),-1,np.int64);pred[ids]=logits.argmax(1)
    groups=sorted({r['group'] for r in rr if r['partition']=='evaluation' and r['available'] and r['quality_clean']})
    lookup={g:i for i,g in enumerate(groups)};rows={};out=np.zeros((len(groups),4),np.int64)
    for i,r in enumerate(rr):
        if r['partition']=='evaluation' and r['available'] and r['quality_clean']:rows.setdefault(r['case'],[]).append(i)
    def count(ii):
        ii=np.array(ii,np.int64);pos=ii[y[ii]==1];neg=ii[y[ii]==0]
        return [len(pos)*len(neg),int((pred[pos]==1).sum())*int((pred[neg]==0).sum())]
    for ii in rows.values():out[lookup[rr[ii[0]]['group']],:2]+=count(ii)
    option={(r['case'],r['option']):i for i,r in enumerate(rr)}
    for g in ctx['contrasts']:
        if g['members'][0]['partition']=='evaluation':
            ii=[option[m['case'],m['option']] for m in g['members']];out[lookup[rr[ii[0]]['group']],2:]+=count(ii)
    check(out[:,0].sum()==8326 and out[:,2].sum()==457,'PAIR_DENOMINATORS')
    return out

def pair_bootstrap(a,b):
    check(np.array_equal(a[:,[0,2]],b[:,[0,2]]),'PAIR_COMPONENTS')
    rate=lambda x:x[..., [1,3]]/x[..., [0,2]]
    intervals=[]
    for seed in range(5):
        ix=np.random.default_rng(seed).integers(len(a),size=(999,len(a)));aa=a[ix].sum(1);bb=b[ix].sum(1)
        check((aa[:,[0,2]]>0).all(),'PAIR_BOOTSTRAP_EMPTY')
        intervals.append(np.quantile(rate(aa)-rate(bb),[.025,.975],axis=0))
    interval=np.array(intervals)
    return dict(metrics=['within_question_pair_accuracy','same_answer_cross_question_pair_accuracy'],
                difference=(rate(a.sum(0))-rate(b.sum(0))).tolist(),seed_intervals=interval.tolist(),
                descriptive_envelope=[interval[:,0].min(0).tolist(),interval[:,1].max(0).tolist()],
                components=len(a),draws_per_seed=999,estimator_seeds=5,sealed_confirmation=False)

def read(root,ctx,native,raw):
    report=read_json(root/'report.json');manifest(root,report)
    check(report['complete'] and report['errors']==[],'INCOMPLETE')
    expected={f'seed-{s}/{a}' for s in (25,26,27) for a in (*ARMS,*(['shuffled'] if s==25 else []))}
    check(set(report['results'])==expected,'FIT_COVERAGE')
    check(all(report[k] is False for k in ('scientific_admission','served','epistemic_understanding_proven','world_truth_assessed','sealed_evaluation'))
          and report['production_assertions']==0 and report['assertion_verdict']=='REFUSE_UNADMITTED_PANEL'
          and report['anchor_scope']=='source_relative_answer_correctness','CLAIMS')
    check(read_json(root/'config.json')==CONFIG,'CONFIG')
    prep=preparation(root/'preparation',ctx,native,raw);check(read_json(root/'preparation-readback.json')==prep,'PREPARATION_READBACK')
    archived_before=read_json(root/'preparation/before.json')
    # Host-local paths differ on import. The cold loader pins each named parent;
    # the complete hash multiset binds its bytes to the original before/after log.
    check(read_json(root/'after.json')==archived_before and sorted(archived_before.values())==sorted(ctx['before'].values()),'PARENT_AFTER')
    ids=np.array([i for i,r in enumerate(ctx['rows']) if r['available']],np.int64)
    heldout=np.array([i for i in ids if ctx['rows'][i]['partition']!='train'],np.int64);donor=donors(ctx['rows'])
    with np.load(root/'evaluation-scope.npz',allow_pickle=False) as z:
        check(np.array_equal(z['ids'],ids) and np.array_equal(z['heldout'],heldout) and np.array_equal(z['donors'],donor),'EVALUATION_SCOPE')
    prior=float(ctx['labels'][ctx['train']].mean());base=np.broadcast_to(np.log([1-prior,prior]),(len(ids),2))
    check(np.array_equal(np.load(root/'prior-logits.npy',allow_pickle=False),base),'TRAIN_PRIOR')
    close(read_json(root/'prior-metrics.json'),independent_metrics(base.copy(),ids,ctx),'PRIOR_METRICS')
    sums={'prior':component_scores(base,ids,ctx)};pairs={'prior':pair_components(base,ids,ctx)};fits={};duplicates={}
    for i,r in enumerate(ctx['rows']):
        if r['available']:duplicates.setdefault((r['document'],r['question_unit'],r['answer_unit']),[]).append(i)
    duplicates=[np.array(v,np.int64) for v in duplicates.values() if len(v)>1]
    for seed in (25,26,27):
        parent=root/f'seed-{seed}';steps=np.load(parent/'training-rows.npy',allow_pickle=False)
        check(np.array_equal(steps,schedule(ctx['train'],seed)),'TRAIN_ORDER')
        initial=torch.load(parent/'initial.pt',map_location='cpu',weights_only=True)
        check(all(bool(torch.isfinite(v).all()) for v in initial.values()),'INITIAL_WEIGHTS')
        for arm in (*ARMS,*(['shuffled'] if seed==25 else [])):
            key=f'seed-{seed}/{arm}';folder=root/key;r=read_json(folder/'report.json');manifest(folder,r)
            check(file_sha(folder/'report.json')==report['results'][key],'FIT_HASH')
            check(r['arm']==arm and r['seed']==seed and r['scope']==('joint' if arm=='shuffled' else arm),'FIT_SCOPE')
            check(r['initial_sha256']==file_sha(parent/'initial.pt'),'INITIAL_PIN')
            check(all(r[k] is False for k in ('world_truth_assessed','scientific_admission','served','epistemic_understanding_proven')),'FIT_CLAIMS')
            labels=ctx['labels'].copy()
            if arm=='shuffled':labels[ctx['train']]=np.random.default_rng(seed).permutation(labels[ctx['train']])
            check(np.array_equal(np.load(folder/'training-labels.npy',allow_pickle=False),labels[ctx['train']]),'TRAIN_LABELS')
            objective=read_json(folder/'objective.json')
            with np.load(folder/'objective.npz',allow_pickle=False) as z:
                check(np.array_equal(z['rows'],steps[0]) and np.array_equal(z['labels'],labels[steps[0]]),'OBJECTIVE_SCOPE')
                x=z['logits'].astype(np.float64);check(x.shape==(64,2) and np.isfinite(x).all(),'OBJECTIVE_LOGITS')
                shift=x.max(1);ce=np.mean(np.log(np.exp(x-shift[:,None]).sum(1))+shift-x[np.arange(64),z['labels']])
                check(np.isclose(ce,objective['loss'],rtol=1e-6,atol=1e-7),'OBJECTIVE_LOSS')
            grads=objective['gradients'];check(any(v is not None and v>0 for v in grads.values()) and all(v is None or np.isfinite(v) for v in grads.values()),'GRADIENTS')
            losses=np.load(folder/'losses.npy',allow_pickle=False);check(losses.shape==(2000,) and np.isfinite(losses).all() and (losses>=0).all(),'LOSSES')
            weights=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
            check(set(weights)==set(initial) and all(v.shape==initial[n].shape and bool(torch.isfinite(v).all()) for n,v in weights.items()),'FINAL_WEIGHTS')
            check(any(not torch.equal(v,initial[n]) for n,v in weights.items()),'NO_TRAINING')
            timing=read_json(folder/'training-time.json');check(timing['updates']==2000 and timing['case_updates']==128000,'UPDATE_COUNT')
            logits=np.load(folder/'logits.npy',allow_pickle=False);metrics=independent_metrics(logits,ids,ctx)
            close(read_json(folder/'metrics.json'),metrics,'METRICS');sums[key]=component_scores(logits,ids,ctx);pairs[key]=pair_components(logits,ids,ctx)
            totals=pairs[key].sum(0)
            check(totals[1]==metrics['evaluation']['clean']['pairs_both_correct'] and totals[3]==metrics['evaluation']['same_answer_contrasts']['both_correct'],'PAIR_COUNTS')
            ctl=read_json(folder/'controls.json');check(all(ctl[k] is True for k in ('private_fields_refused','absent_inputs_bitwise_irrelevant','padding_bitwise_irrelevant'))
                and ctl['source_correctness_is_world_truth'] is False and all(v>0 and np.isfinite(v) for v in ctl['input_changes'].values()),'FORWARD_CONTROLS')
            check(.9<=r['phase_coverage']<=1.001 and np.isclose(r['phase_coverage'],sum(r['phases_seconds'].values())/r['wall_seconds']),'PHASE_COVERAGE')
            if arm=='joint':
                summaries=read_json(folder/'interventions.json');base_logits=logits[np.searchsorted(ids,heldout)].astype(np.float64)
                base_probs=np.exp(base_logits-base_logits.max(1,keepdims=True));base_probs/=base_probs.sum(1,keepdims=True)
                with np.load(folder/'interventions.npz',allow_pickle=False) as z:
                    check(set(z.files)=={'remove-context','question-swap','passage-swap'},'INTERVENTION_COVERAGE')
                    for kind in z.files:
                        value=z[kind].astype(np.float64);check(value.shape==base_logits.shape and np.isfinite(value).all(),'INTERVENTION_VALUES')
                        q=np.exp(value-value.max(1,keepdims=True));q/=q.sum(1,keepdims=True)
                        expected_change=dict(rows=len(q),argmax_changed=int((q.argmax(1)!=base_probs.argmax(1)).sum()),
                            mean_absolute_probability_change=float(np.abs(base_probs-q).mean()),
                            mean_kl=float((base_probs*(np.log(np.maximum(base_probs,1e-300))-np.log(np.maximum(q,1e-300)))).sum(1).mean()),
                            semantic_labels_for_swaps=None,world_truth_assessed=False)
                        close(summaries[kind],expected_change,'INTERVENTION_METRICS')
            spread=[float(np.ptp(logits[np.searchsorted(ids,v)],axis=0).max()) for v in duplicates]
            fits[key]=dict(options=len(ids),raw_labels_and_all_metrics_reconstructed=True,checkpoint_sha256=file_sha(folder/'model.pt'),
                           exact_same_question_answer_passage_groups=len(duplicates),maximum_duplicate_logit_spread=max(spread,default=0.),
                           exact_input_conflicting_label_groups=sum(len(set(ctx['labels'][v].tolist()))>1 for v in duplicates))
    comparisons={}
    for seed in (25,26,27):
        key=f'seed-{seed}/joint'
        for arm in ARMS[1:]:comparisons[key+'/'+arm]=bootstrap(sums[key],sums[f'seed-{seed}/{arm}'])
        comparisons[key+'/train-prior']=bootstrap(sums[key],sums['prior'])
    close(read_json(root/'comparisons.json'),comparisons,'COMPONENT_BOOTSTRAP')
    pair_comparisons={}
    for seed in (25,26,27):
        key=f'seed-{seed}/joint'
        for arm in ARMS[1:]:pair_comparisons[key+'/'+arm]=pair_bootstrap(pairs[key],pairs[f'seed-{seed}/{arm}'])
        pair_comparisons[key+'/train-prior']=pair_bootstrap(pairs[key],pairs['prior'])
    pair_comparisons['seed-25/joint/shuffled']=pair_bootstrap(pairs['seed-25/joint'],pairs['seed-25/shuffled'])
    check(.9<=report['phase_coverage']<=1.001 and np.isclose(report['phase_coverage'],sum(report['phases_seconds'].values())/report['wall_seconds']),'PHASE_COVERAGE')
    return dict(report_sha256=file_sha(root/'report.json'),fits=fits,pair_component_comparisons=pair_comparisons,all_metrics_and_component_intervals_reconstructed=True,
                all_source_maps_and_private_raw_labels_reconstructed=True,scientific_admission=False,served=False,acceptance='author-checked')
