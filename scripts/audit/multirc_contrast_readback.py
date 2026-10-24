"""Independent paired-objective arithmetic, source scope and result readback."""
from pathlib import Path
import numpy as np
import torch
from model.multirc_context import check
from model.multirc_binding import donors
from model.multirc_contrast import BASELINE,CONFIG,ARMS,scope,schedule
from audit.multirc_binding_readback import read_json,manifest,preparation,independent_metrics,close
from audit.multirc_contrast_scope import validate
from audit.multirc_result_readback import component_scores,bootstrap,pair_components,pair_bootstrap
from text.evaluation_multiscale import file_sha

def loss_terms(logits,labels,pairs,ranked):
    x=np.array(logits,np.float64,copy=True);shift=x.max(1)
    ce=np.mean(np.log(np.exp(x-shift[:,None]).sum(1))+shift-x[np.arange(64),labels])
    score=x[:,1]-x[:,0];terms=[ce];counts=[]
    for part in (pairs[:12],pairs[12:]):
        sign=labels[part[:,0]]-labels[part[:,1]];valid=sign!=0;counts.append(int(valid.sum()))
        terms.append(float(np.logaddexp(0,-sign[valid]*(score[part[valid,0]]-score[part[valid,1]])).mean()) if valid.any() else 0.)
    return terms[0]+(.5*(terms[1]+terms[2]) if ranked else 0),terms,counts

def read(root,ctx,native,raw,baseline):
    report=read_json(root/'report.json');manifest(root,report)
    check(file_sha(baseline/'report.json')==BASELINE==report['baseline_report_sha256'],'CONTRAST_BASELINE_PIN')
    base=read_json(baseline/'report.json');manifest(baseline,base)
    check(report['complete'] and not report['errors'],'CONTRAST_INCOMPLETE')
    expected={f'seed-{s}/{a}' for s in (25,26,27) for a in (*ARMS,*(['shuffled-rank'] if s==25 else []))}
    check(set(report['results'])==expected,'CONTRAST_FIT_COVERAGE')
    check(all(report[k] is False for k in ('scientific_admission','served','world_truth_assessed','epistemic_understanding_proven','sealed_evaluation'))
        and report['production_assertions']==0 and report['assertion_verdict']=='REFUSE_UNADMITTED_PANEL' and report['anchor_scope']=='source_relative_answer_correctness','CONTRAST_CLAIMS')
    check(read_json(root/'config.json')==CONFIG,'CONTRAST_CONFIG')
    prep=preparation(root/'preparation',ctx,native,raw);check(prep==read_json(root/'preparation-readback.json'),'CONTRAST_PREPARATION')
    check(prep['report_sha256']==file_sha(baseline/'preparation/report.json'),'CONTRAST_PREPARATION_REUSE')
    before={**ctx['before'],str(baseline/'report.json'):BASELINE}
    for name,v in base['files'].items():before[str(baseline/name)]=v['sha256']
    archived=read_json(root/'source-before.json');check(read_json(root/'after.json')==archived and sorted(archived.values())==sorted(before.values()),'CONTRAST_PARENT_AFTER')
    ids=np.array([i for i,r in enumerate(ctx['rows']) if r['available']],np.int64)
    heldout=np.array([i for i in ids if ctx['rows'][i]['partition']!='train'],np.int64);donor=donors(ctx['rows'])
    with np.load(root/'evaluation-scope.npz',allow_pickle=False) as z:
        check(np.array_equal(z['ids'],ids) and np.array_equal(z['heldout'],heldout) and np.array_equal(z['donors'],donor),'CONTRAST_EVALUATION_SCOPE')
    prior=float(ctx['labels'][ctx['train']].mean());prior_logits=np.broadcast_to(np.log([1-prior,prior]),(len(ids),2)).copy()
    check(np.array_equal(np.load(root/'prior-logits.npy',allow_pickle=False),prior_logits),'CONTRAST_PRIOR')
    close(read_json(root/'prior-metrics.json'),independent_metrics(prior_logits.copy(),ids,ctx),'CONTRAST_PRIOR_METRICS')
    sums={'prior':component_scores(prior_logits,ids,ctx)};pairsums={'prior':pair_components(prior_logits,ids,ctx)}
    grouped=read_json(root/'training-groups.json');fits={};schedules={}
    for seed in (25,26,27):
        parent=root/f'seed-{seed}'
        with np.load(parent/'schedule.npz',allow_pickle=False) as z:selected={n:z[n] for n in z.files}
        proof=validate(grouped,selected,ctx);check(proof==read_json(parent/'schedule-readback.json'),'CONTRAST_SCHEDULE_READBACK')
        regenerated=schedule(ctx['train'],grouped,seed)
        check(all(np.array_equal(selected[n],regenerated[n]) for n in selected),'CONTRAST_SEEDED_SCHEDULE');schedules[str(seed)]=proof
        steps=selected['rows'];positions=selected['pairs']
        check(file_sha(parent/'initial.pt')==file_sha(baseline/f'seed-{seed}/initial.pt'),'CONTRAST_INITIAL_REUSE')
        initial=torch.load(parent/'initial.pt',map_location='cpu',weights_only=True)
        old=np.load(baseline/f'seed-{seed}/joint/logits.npy',allow_pickle=False)
        sums[f'seed-{seed}/uniform-ce']=component_scores(old,ids,ctx);pairsums[f'seed-{seed}/uniform-ce']=pair_components(old,ids,ctx)
        for arm in (*ARMS,*(['shuffled-rank'] if seed==25 else [])):
            key=f'seed-{seed}/{arm}';out=root/key;r=read_json(out/'report.json');manifest(out,r);ranked=arm!='joint-ce'
            check(file_sha(out/'report.json')==report['results'][key],'CONTRAST_FIT_HASH')
            check(r['arm']==arm and r['seed']==seed and r['scope']==scope(arm) and r['ranked']==ranked and r['parameters']==1391330,'CONTRAST_FIT_SCOPE')
            check(r['initial_sha256']==file_sha(parent/'initial.pt'),'CONTRAST_INITIAL_PIN')
            check(all(r[k] is False for k in ('world_truth_assessed','scientific_admission','served','epistemic_understanding_proven')),'CONTRAST_FIT_CLAIMS')
            labels=ctx['labels'].copy()
            if arm=='shuffled-rank':labels[ctx['train']]=np.random.default_rng(seed).permutation(labels[ctx['train']])
            check(np.array_equal(np.load(out/'training-labels.npy',allow_pickle=False),labels[ctx['train']]),'CONTRAST_TRAIN_LABELS')
            obj=read_json(out/'objective.json');check(obj['ranked']==ranked,'CONTRAST_OBJECTIVE_SCOPE')
            with np.load(out/'objective.npz',allow_pickle=False) as z:
                check(np.array_equal(z['rows'],steps[0]) and np.array_equal(z['labels'],labels[steps[0]]) and np.array_equal(z['pairs'],positions[0]),'CONTRAST_OBJECTIVE_SCOPE')
                check(z['logits'].shape==(64,2) and np.isfinite(z['logits']).all(),'CONTRAST_OBJECTIVE_LOGITS')
                loss,terms,counts=loss_terms(z['logits'],z['labels'],z['pairs'],ranked)
                check(np.isclose(loss,obj['loss'],rtol=1e-6,atol=1e-7) and np.allclose(terms,obj['terms'],rtol=1e-6,atol=1e-7) and counts==obj['counts'],'CONTRAST_OBJECTIVE_ARITHMETIC')
            gradients=obj['gradients'];check(any(v is not None and v>0 for v in gradients.values()) and all(v is None or np.isfinite(v) for v in gradients.values()),'CONTRAST_GRADIENTS')
            oc=read_json(out/'objective-controls.json');check(oc==dict(pair_endpoint_swap_bitwise_invariant=True,invalid_pair_address_refused=True,equal_labels_add_zero_ranking=True,objective_uses_fit_labels_only=True,world_truth_assessed=False),'CONTRAST_OBJECTIVE_CONTROLS')
            losses=np.load(out/'losses.npy',allow_pickle=False);check(losses.shape==(2000,6) and np.isfinite(losses).all() and (losses>=0).all(),'CONTRAST_LOSSES')
            lhs=np.take_along_axis(labels[steps],positions[:,:,0],axis=1);rhs=np.take_along_axis(labels[steps],positions[:,:,1],axis=1)
            expected_counts=np.stack(((lhs[:,:12]!=rhs[:,:12]).sum(1),(lhs[:,12:]!=rhs[:,12:]).sum(1)),axis=1)
            check(np.array_equal(losses[:,4:],expected_counts),'CONTRAST_ALL_OBJECTIVE_LABEL_COUNTS')
            expected_loss=losses[:,1]+(.5*(losses[:,2]+losses[:,3]) if ranked else 0)
            check(np.allclose(losses[:,0],expected_loss,rtol=1e-6,atol=1e-7),'CONTRAST_LOSS_COMPOSITION')
            timing=read_json(out/'training-time.json');check(timing['updates']==2000 and timing['case_updates']==128000,'CONTRAST_UPDATES')
            state=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
            check(set(state)==set(initial) and all(v.shape==initial[n].shape and bool(torch.isfinite(v).all()) for n,v in state.items()),'CONTRAST_WEIGHTS')
            check(any(not torch.equal(v,initial[n]) for n,v in state.items()),'CONTRAST_NO_TRAINING')
            logits=np.load(out/'logits.npy',allow_pickle=False);metrics=independent_metrics(logits,ids,ctx);close(read_json(out/'metrics.json'),metrics,'CONTRAST_METRICS')
            sums[key]=component_scores(logits,ids,ctx);pairsums[key]=pair_components(logits,ids,ctx)
            totals=pairsums[key].sum(0);check(totals[1]==metrics['evaluation']['clean']['pairs_both_correct'] and totals[3]==metrics['evaluation']['same_answer_contrasts']['both_correct'],'CONTRAST_PAIR_COUNTS')
            control=read_json(out/'controls.json');check(all(control[k] is True for k in ('private_fields_refused','absent_inputs_bitwise_irrelevant','padding_bitwise_irrelevant'))
                and control['source_correctness_is_world_truth'] is False and all(v>0 and np.isfinite(v) for v in control['input_changes'].values()),'CONTRAST_FORWARD_CONTROLS')
            if arm=='joint-rank':
                summaries=read_json(out/'interventions.json');x=logits[np.searchsorted(ids,heldout)].astype(np.float64);pp=np.exp(x-x.max(1,keepdims=True));pp/=pp.sum(1,keepdims=True)
                with np.load(out/'interventions.npz',allow_pickle=False) as z:
                    check(set(z.files)=={'remove-context','question-swap','passage-swap'},'CONTRAST_INTERVENTION_COVERAGE')
                    for kind in z.files:
                        v=z[kind].astype(np.float64);check(v.shape==x.shape and np.isfinite(v).all(),'CONTRAST_INTERVENTION_VALUES')
                        qq=np.exp(v-v.max(1,keepdims=True));qq/=qq.sum(1,keepdims=True)
                        expected_change=dict(rows=len(qq),argmax_changed=int((qq.argmax(1)!=pp.argmax(1)).sum()),mean_absolute_probability_change=float(np.abs(pp-qq).mean()),
                            mean_kl=float((pp*(np.log(np.maximum(pp,1e-300))-np.log(np.maximum(qq,1e-300)))).sum(1).mean()),semantic_labels_for_swaps=None,world_truth_assessed=False)
                        close(summaries[kind],expected_change,'CONTRAST_INTERVENTION_METRICS')
            check(.9<=r['phase_coverage']<=1.001 and np.isclose(r['phase_coverage'],sum(r['phases_seconds'].values())/r['wall_seconds']),'CONTRAST_PHASE_COVERAGE')
            fits[key]=dict(options=len(ids),all_metrics_reconstructed=True,checkpoint_sha256=file_sha(out/'model.pt'))
    cc={};pc={};tests={}
    for seed in (25,26,27):
        left=f'seed-{seed}/joint-rank'
        for other in ('joint-ce','no-context-rank','no-question-rank','no-answer-rank','uniform-ce',*(['shuffled-rank'] if seed==25 else [])):
            right=f'seed-{seed}/{other}';cc[left+'/'+other]=bootstrap(sums[left],sums[right]);pc[left+'/'+other]=pair_bootstrap(pairsums[left],pairsums[right])
        cc[left+'/train-prior']=bootstrap(sums[left],sums['prior']);pc[left+'/train-prior']=pair_bootstrap(pairsums[left],pairsums['prior'])
        for other,axis in (('joint-ce',1),('no-context-rank',1),('no-question-rank',1),('no-answer-rank',0)):
            key=left+'/'+other;tests[key]=pc[key]['descriptive_envelope'][0][axis]>0
    close(read_json(root/'comparisons.json'),cc,'CONTRAST_COMPONENT_INTERVALS');close(read_json(root/'pair-comparisons.json'),pc,'CONTRAST_PAIR_INTERVALS')
    decision=dict(all_predeclared_research_checks_pass=all(tests.values()),checks=tests,world_truth_assessed=False,epistemic_understanding_proven=False,
                  scientific_admission=False,episodic_promotion_authorized=False,sealed_confirmation=False)
    check(read_json(root/'research-advancement.json')==decision,'CONTRAST_ADVANCEMENT')
    check(.9<=report['phase_coverage']<=1.001 and np.isclose(report['phase_coverage'],sum(report['phases_seconds'].values())/report['wall_seconds']),'CONTRAST_PHASE_COVERAGE')
    return dict(report_sha256=file_sha(root/'report.json'),fits=fits,schedules=schedules,all_objectives_metrics_and_intervals_reconstructed=True,
                research_advancement=decision,scientific_admission=False,served=False,acceptance='author-checked')
