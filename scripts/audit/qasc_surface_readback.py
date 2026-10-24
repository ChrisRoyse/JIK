"""Independent complete-byte and source-condition audit of persisted renderer results."""
from collections import Counter
import json
import math
import numpy as np
import torch
from model.qasc_context import POLICIES,LANES
from model.qasc_generation_data import TARGETS,TARGET_NATIVE
from model.scifact_readout import array_sha
from audit.qasc_region_readback import check,read_json,time_check,same,unit
from text.evaluation_multiscale import file_sha

def records(path):return [json.loads(r) for r in path.read_bytes().splitlines()]

def decode(tokens):
    check(type(tokens)==list and all(type(v)==int for v in tokens) and 0<len(tokens)<=513,'SURFACE_TOKENS')
    if 257 not in tokens:return 'MNLM_SURFACE_NO_EOS',None
    content=tokens[:tokens.index(257)]
    if len(content)>512:return 'MNLM_SURFACE_OUTPUT_CAPACITY',None
    if any(x<0 or x>=256 for x in content):return 'MNLM_SURFACE_INVALID_SYMBOL',None
    try:text=bytes(content).decode('utf-8',errors='strict')
    except UnicodeDecodeError:return 'MNLM_SURFACE_INVALID_UTF8',None
    return ('complete_unasserted',text) if text.strip() else ('MNLM_SURFACE_EMPTY_OUTPUT',None)

def read(root,ctx,region):
    r=read_json(root/'report.json');check(r['schema']=='qasc-complete-surfaces-v1' and r['complete'] and r['errors']==[],'SURFACE_STATE')
    check(r['region_report_sha256']==file_sha(region/'report.json') and r['targets_sha256']==TARGETS and r['target_native_sha256']==TARGET_NATIVE,'SURFACE_PINS')
    check(all(r[k] is False for k in ('scientific_admission','served','sealed_evaluation','free_text_composition_proven','guard_evaluated'))
          and r['production_assertions']==0 and r['assertion_verdict']=='REFUSE_UNADMITTED_PANEL','SURFACE_CLAIMS');time_check(r)
    check(set(r['results'])=={p+'/'+a for p in POLICIES for a in ('true','gaussian','predicted')},'SURFACE_ARMS')
    check({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==set(r['files'])|{'report.json'},'SURFACE_FILES')
    for name,pin in r['files'].items():
        path=(root/name).resolve();check(path.is_relative_to(root.resolve()) and path.stat().st_size==pin['bytes'] and file_sha(path)==pin['sha256'],'SURFACE_FILE')
    before=read_json(root/'before.json');expected={**ctx['before'],str(region/'report.json'):file_sha(region/'report.json')}
    expected.update({str(region/n):v['sha256'] for n,v in read_json(region/'report.json')['files'].items()})
    check(before==read_json(root/'after.json') and Counter(before.values())==Counter(expected.values()),'SURFACE_PARENT_DRIFT')
    check(read_json(root/'policies.json')==ctx['policies'],'SURFACE_POLICIES')
    config=dict(width=256,layers=4,heads=8,steps=600,batch=32,seed=25,lr=.0003,weight_decay=.05,betas=[.9,.95],
                clip_gradient_norm=1.,capacity=512,gaussian_norm_sigma=.3,precision='bf16-fp32-loss')
    check(read_json(root/'config.json')==config,'SURFACE_CONFIG')
    texts=[ctx['texts'][i] for i in ctx['target_index']];lengths=np.array([len(t.encode())+2 for t in texts])
    with np.load(root/'complete-target-bytes.npz',allow_pickle=False) as z:
        check(set(z.files)=={'ids','pad','lengths'} and np.array_equal(z['lengths'],lengths) and z['ids'].shape==z['pad'].shape==(8134,152),'SURFACE_CODEC_SHAPE')
        for i,t in enumerate(texts):
            target=np.array([256,*t.encode('utf-8'),257]);n=len(target)
            check(np.array_equal(z['ids'][i,:n],target) and not z['pad'][i,:n].any() and z['pad'][i,n:].all()
                  and np.all(z['ids'][i,n:]==257),'SURFACE_COMPLETE_BYTES')
    summaries={};total_renders=0
    for policy in POLICIES:
        folder=root/policy;rr=ctx['policies'][policy];train=np.array([i for i,x in enumerate(rr) if x['partition']=='train']);heldout=np.array([i for i,x in enumerate(rr) if x['partition']!='train'])
        order=np.random.default_rng(25).choice(train,(600,32),replace=True);selected={s:np.array([i for i,x in enumerate(rr) if x['partition']==s][:128]) for s in ('calibration','evaluation')}
        check(np.array_equal(np.load(folder/'training-rows.npy',allow_pickle=False),order) and np.array_equal(np.load(folder/'heldout.npy',allow_pickle=False),heldout),'SURFACE_TRAIN_SCOPE')
        with np.load(folder/'selected.npz',allow_pickle=False) as z:check(set(z.files)==set(selected) and all(np.array_equal(z[s],ii) for s,ii in selected.items()),'SURFACE_SELECTION')
        donor=np.load(folder/'donors.npy',allow_pickle=False);check(np.array_equal(donor,np.load(region/policy/'donors.npy',allow_pickle=False)),'SURFACE_DONOR_PIN')
        check(all(rr[i]['partition']==rr[j]['partition'] and rr[i]['group']!=rr[j]['group'] for i,j in enumerate(donor)),'SURFACE_DONOR_GROUP')
        condition=read_json(folder/'conditioning.json')
        check(condition['codebook_sha256']==file_sha(region/policy/'codebook.npz') and condition['predictor_outputs_sha256']==file_sha(region/policy/'seed-25/region/outputs.npz')
              and condition['actual_predictor_seed']==25 and condition['train_predictions_in_sample'] and condition['predictions_out_of_fold'] is False
              and condition['heldout_predictions_target_free'] and condition['unsnapped'],'SURFACE_CONDITION_SOURCE')
        with np.load(region/policy/'seed-25/region/outputs.npz',allow_pickle=False) as z:
            check(condition['hashes']['predicted']==[array_sha(z[f'slots_{k}']) for k in range(6)],'SURFACE_ACTUAL_PREDICTIONS')
        with np.load(folder/'gaussian.npz',allow_pickle=False) as z:
            check(set(z.files)=={f'slots_{k}' for k in range(6)} and condition['hashes']['gaussian']==[array_sha(z[f'slots_{k}']) for k in range(6)],'SURFACE_NOISE_HASHES')
            check(all(z[n].shape==(8134,768) and z[n].dtype==np.float32 and np.isfinite(z[n]).all() and np.all(abs(np.linalg.norm(z[n],axis=1)-1)<2e-5) for n in z.files),'SURFACE_NOISE_VALUES')
        initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=True)
        for arm in ('true','gaussian','predicted'):
            path=policy+'/'+arm;f=root/path;rep=read_json(f/'report.json');time_check(rep)
            check(file_sha(f/'report.json')==r['results'][path] and rep['initial_sha256']==file_sha(folder/'initial.pt'),'SURFACE_FIT_PINS')
            check(rep['policy']==policy and rep['arm']==arm and all(rep[k] is False for k in ('scientific_admission','served','guard_evaluated','free_text_composition_proven')),'SURFACE_FIT_SCOPE')
            for name,pin in rep['files'].items():check(pin==r['files'][path+'/'+name],'SURFACE_FIT_FILES')
            objective=read_json(f/'objective.json');check(all(v is not None and np.isfinite(v) for v in objective['gradients'].values()) and any(v>0 for v in objective['gradients'].values()),'SURFACE_GRADIENTS')
            with np.load(f/'objective.npz',allow_pickle=False) as z:
                check(np.array_equal(z['rows'],order[0]) and np.array_equal(z['bytes'],lengths[order[0]]-2),'SURFACE_OBJECTIVE_BYTES')
                loss=(z['byte_nll_bits'].astype(float).sum()+z['eos_nll_bits'].astype(float).sum())*math.log(2)/(z['bytes'].sum()+32)
                check(abs(loss-objective['loss'])<2e-5,'SURFACE_OBJECTIVE')
            losses=np.load(f/'losses.npy',allow_pickle=False);check(losses.shape==(600,) and np.isfinite(losses).all() and losses[0]==objective['loss'],'SURFACE_LOSSES')
            checkpoint=torch.load(f/'renderer.pt',map_location='cpu',weights_only=True)
            check(set(checkpoint)==set(initial) and all(checkpoint[k].shape==initial[k].shape and torch.isfinite(checkpoint[k]).all() for k in initial)
                  and any(not torch.equal(checkpoint[k],initial[k]) for k in initial) and sum(v.numel() for v in checkpoint.values())==rep['parameters'],'SURFACE_CHECKPOINT')
            scores=records(f/'scores.jsonl');expected_rows=[(route,int(i)) for route in ('true','predicted','other-true','other-predicted') for i in heldout]
            check([(x['route'],x['row']) for x in scores]==expected_rows,'SURFACE_SCORE_COVERAGE')
            for x in scores:
                i=x['row'];check(x['partition']==rr[i]['partition'] and x['conditioning_row']==(int(donor[i]) if x['route'].startswith('other-') else i)
                    and x['bytes']==lengths[i]-2 and np.isfinite(x['byte_nll_bits']) and np.isfinite(x['eos_nll_bits'])
                    and x['byte_nll_bits']>=0 and x['eos_nll_bits']>=0,'SURFACE_SCORE')
            metrics={}
            for split in selected:
                metrics[split]={}
                for route in ('true','predicted','other-true','other-predicted'):
                    part=[x for x in scores if x['partition']==split and x['route']==route]
                    metrics[split][route]=dict(rows=len(part),bytes=sum(x['bytes'] for x in part),bits_per_byte=sum(x['byte_nll_bits'] for x in part)/sum(x['bytes'] for x in part),mean_eos_bits=sum(x['eos_nll_bits'] for x in part)/len(part))
            check(same(metrics,read_json(f/'metrics.json')),'SURFACE_METRICS')
            generated=records(f/'renders.jsonl');expected_rows=[(s,route,int(i)) for s,ii in selected.items() for route in ('true','predicted') for i in ii]
            check([(x['partition'],x['route'],x['row']) for x in generated]==expected_rows,'SURFACE_RENDER_COVERAGE')
            for x in generated:
                i=x['row'];status,text=decode(x['tokens'])
                check(x['case']==rr[i]['case'] and x['group']==rr[i]['group'] and x['policy']==policy and x['arm']==arm,'SURFACE_RENDER_SOURCE')
                check(x['asserted'] is False and x['status']==status and x['text']==text and x['exact_provider_target']==(text==texts[i]),'SURFACE_RENDER_CODEC')
            check(dict(Counter(x['status'] for x in generated))==read_json(f/'surface-statuses.json'),'SURFACE_STATUS_COUNTS')
            total_renders+=len(generated);summaries[path]=dict(evaluation=metrics['evaluation'],renders=len(generated),exact_provider_targets=sum(x['exact_provider_target'] for x in generated))
    check(total_renders==3072,'SURFACE_TOTAL_RENDERS')
    return dict(report_sha256=file_sha(root/'report.json'),fits=6,renders=total_renders,complete_targets_preserved=True,
            actual_predictor_conditions_pinned=True,all_likelihood_metrics_reconstructed=True,all_render_bytes_decoded=True,
            summary=summaries,scientific_admission=False,served=False,free_text_composition_proven=False,acceptance='author-checked')
