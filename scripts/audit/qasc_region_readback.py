"""Cold reconstruction of target boundaries, objectives and typed region metrics."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from model.qasc_context import LANES,POLICIES,POPULATION,NATIVE
from model.qasc_generation_data import TARGETS,TARGET_NATIVE
from text.evaluation_multiscale import file_sha

def check(ok,code):
    if not ok:raise ValueError('MNLM_REGION_READBACK_'+code)

def read_json(p):return json.loads(p.read_bytes())

def unit(x):return x/np.linalg.norm(x,axis=-1,keepdims=True)

def same(a,b,tol=2e-6):
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(same(a[k],b[k],tol) for k in a)
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(same(x,y,tol) for x,y in zip(a,b))
    if isinstance(a,(float,np.floating)):return isinstance(b,(int,float)) and np.isfinite(b) and abs(a-b)<=tol
    return a==b

def stats(cos,correct,rows):
    out={}
    for split in ('train','calibration','evaluation','familiar-facts-unseen-pair'):
        ii=[i for i,r in enumerate(rows) if (r['partition']=='evaluation' and r['seen_training_facts']==2)] if split=='familiar-facts-unseen-pair' else [i for i,r in enumerate(rows) if r['partition']==split]
        out[split]=dict(cases=len(ii),groups=len({rows[i]['group'] for i in ii}),
                       centered_cosine_by_sense=np.mean(cos[ii],axis=0,dtype=np.float64).tolist() if ii else None,
                       region_accuracy_by_sense=np.mean(correct[ii],axis=0,dtype=np.float64).tolist() if correct is not None and ii else None)
    return out

def cross_entropy(logits,labels):
    x=logits.astype(np.float64);m=np.max(x,axis=1)
    return np.mean(m+np.log(np.exp(x-m[:,None]).sum(1))-x[np.arange(len(x)),labels])

def time_check(r):
    check(r['wall_seconds']>0 and .9<=r['phase_coverage']<=1
          and abs(sum(r['phases_seconds'].values())/r['wall_seconds']-r['phase_coverage'])<1e-10,'COST')

def read(root,ctx):
    r=read_json(root/'report.json');check(r['schema']=='qasc-typed-regions-v1' and r['complete'] and r['errors']==[],'STATE')
    check(r['targets_sha256']==TARGETS and r['target_native_sha256']==TARGET_NATIVE,'TARGET_PINS')
    check(r['population_sha256']==POPULATION and r['native_sha256']==NATIVE,'INPUT_PINS')
    check(all(r[k] is False for k in ('scientific_admission','served','sealed_evaluation','free_text_composition_proven','teacher_forced_regions'))
          and r['assertion_verdict']=='REFUSE_UNADMITTED_PANEL' and r['production_assertions']==0,'CLAIMS')
    time_check(r)
    expected={f'{policy}/seed-{seed}/{arm}' for policy in POLICIES for seed in (25,26,27)
              for arm in (('region','direct','shuffled') if seed==25 else ('region','direct'))}
    check(set(r['results'])==expected,'FIT_COVERAGE')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    check(actual==set(r['files'])|{'report.json'},'FILE_COVERAGE')
    for n,m in r['files'].items():
        p=(root/n).resolve();check(p.is_relative_to(root.resolve()) and p.stat().st_size==m['bytes'] and file_sha(p)==m['sha256'],'FILE')
    before=read_json(root/'before.json');check(before==read_json(root/'after.json') and Counter(before.values())==Counter(ctx['before'].values()),'PARENT_DRIFT')
    check(read_json(root/'policies.json')==ctx['policies'] and read_json(root/'lanes.json')==ctx['lanes'],'SOURCE_SCOPE')
    with np.load(root/'gather.npz',allow_pickle=False) as z:
        check(set(z.files)=={'maps','units','original','target_index'} and all(np.array_equal(z[n],ctx[n]) for n in z.files),'INPUT_TARGET_MAP')
    expected_cfg=dict(width=96,layers=2,heads=4,regions=64,lloyd_updates=25,steps=600,batch=32,seeds=[25,26,27],learning_rate=.001,weight_decay=.01,temperature=.1,precision='fp32')
    check(read_json(root/'config.json')==expected_cfg,'CONFIG');count=0;summary={}
    for policy in POLICIES:
        folder=root/policy;rr=ctx['policies'][policy];train=np.array([i for i,x in enumerate(rr) if x['partition']=='train']);unique=np.unique(ctx['target_index'][train])
        with np.load(folder/'codebook.npz',allow_pickle=False) as z:book={n:z[n].copy() for n in z.files}
        check(np.array_equal(book['train_rows'],train) and np.array_equal(book['unique_training_targets'],unique),'BOOK_TRAIN_SCOPE')
        check(np.array_equal(book['initial_positions'],np.random.default_rng(25).choice(len(unique),64,replace=False)),'BOOK_INITIAL')
        truth=[];inputs=[];code=[];centers=[]
        for k,name in enumerate(LANES):
            mean=book[name+'_mean'];c=book[name+'_centers'];counts=book[name+'_counts']
            check(mean.shape==(768,) and np.allclose(mean,ctx['truth'][k][unique].mean(0),rtol=0,atol=2e-6),'BOOK_MEAN_TRAIN_ONLY')
            check(c.shape==(64,768) and c.dtype==np.float32 and np.isfinite(c).all() and np.all(abs(np.linalg.norm(c,axis=1)-1)<2e-5),'BOOK_CENTERS')
            check(counts.shape==(25,64) and np.all(counts>=0) and np.all(counts.sum(1)==len(unique)),'BOOK_COUNTS')
            y=unit(ctx['truth'][k]-mean);x=unit(ctx['inputs'][k]-mean);truth.append(y);inputs.append(x);centers.append(c)
            code.append((y@c.T).argmax(1))
        code=np.stack(code,1);check(np.array_equal(code,book['codes']),'BOOK_CODES')
        baseline=read_json(folder/'baselines.json')
        with np.load(folder/'baselines.npz',allow_pickle=False) as bz:
            for name in ('copy-fact1','copy-fact2','training-modal-region'):
                cosine=[]
                for k in range(6):
                    value=inputs[k][ctx['maps'][:,1 if name=='copy-fact1' else 2]] if name!='training-modal-region' else np.broadcast_to(centers[k][np.bincount(code[unique,k],minlength=64).argmax()],(8134,768))
                    cosine.append(np.sum(value.astype(np.float64)*truth[k][ctx['target_index']].astype(np.float64),axis=1))
                cosine=np.stack(cosine,1);check(np.allclose(cosine,bz[name],rtol=0,atol=2e-6) and same(stats(cosine,None,rr),baseline[name]),'BASELINE')
        for seed in (25,26,27):
            sf=folder/f'seed-{seed}';initial=torch.load(sf/'initial.pt',map_location='cpu',weights_only=True)
            steps=np.load(sf/'training-rows.npy',allow_pickle=False)
            check(np.array_equal(steps,np.random.default_rng(seed).choice(train,(600,32),replace=True)),'TRAIN_ORDER')
            for arm in (('region','direct','shuffled') if seed==25 else ('region','direct')):
                path=f'{policy}/seed-{seed}/{arm}';f=root/path;rep=read_json(f/'report.json');time_check(rep)
                check(file_sha(f/'report.json')==r['results'][path] and rep['initial_sha256']==file_sha(sf/'initial.pt')
                      and rep['codebook_sha256']==file_sha(folder/'codebook.npz'),'FIT_PINS')
                check(rep['policy']==policy and rep['seed']==seed and rep['arm']==arm and rep['placement']==('region' if arm=='shuffled' else arm),'FIT_ID')
                check(all(rep[k] is False for k in ('scientific_admission','served','free_text_composition_proven')),'FIT_CLAIMS')
                for n,m in rep['files'].items():check(r['files'][path+'/'+n]==m,'FIT_FILES')
                target=ctx['target_index'].copy()
                if arm=='shuffled':target[train]=np.random.default_rng(seed).permutation(target[train])
                check(np.array_equal(np.load(f/'training-targets.npy',allow_pickle=False),target[train]),'TRAIN_TARGETS')
                objective=read_json(f/'objective.json')
                check(all(v is not None and np.isfinite(v) for v in objective['gradients'].values()) and any(v>0 for v in objective['gradients'].values()),'GRADIENTS')
                with np.load(f/'objective.npz',allow_pickle=False) as z:
                    check(np.array_equal(z['rows'],steps[0]) and np.array_equal(z['targets'],target[steps[0]]) and np.array_equal(z['codes'],code[z['targets']]),'OBJECTIVE_TARGETS')
                    ce=[];cos=[];nce=[];tid=z['targets'];mask=tid[:,None]==tid[None,:];np.fill_diagonal(mask,False)
                    for k in range(6):
                        x=z[f'slots_{k}'].astype(np.float64);y=truth[k][tid].astype(np.float64)
                        ce.append(cross_entropy(z[f'logits_{k}'],code[tid,k]));cos.append(np.mean(1-(x*y).sum(1)))
                        scores=x@y.T/.1;scores[mask]=-np.inf;nce.append(cross_entropy(scores,np.arange(32)))
                    terms=[np.mean(v) for v in (ce,cos,nce)]
                    check(np.allclose(terms,objective['terms'],rtol=0,atol=2e-5) and abs(sum(terms)-objective['loss'])<3e-5,'OBJECTIVE')
                losses=np.load(f/'losses.npy',allow_pickle=False);check(losses.shape==(600,3) and np.isfinite(losses).all() and np.allclose(losses[0],objective['terms'],rtol=0,atol=1e-8),'LOSSES')
                final=torch.load(f/'model.pt',map_location='cpu',weights_only=True)
                check(set(final)==set(initial) and all(final[k].shape==initial[k].shape and torch.isfinite(final[k]).all() for k in final),'CHECKPOINT')
                check(any(not torch.equal(final[k],initial[k]) for k in final if not k.startswith('book_')),'NO_LEARNING')
                check(all(np.array_equal(final['book_'+str(k)].numpy(),centers[k]) and torch.equal(final['book_'+str(k)],initial['book_'+str(k)]) for k in range(6)),'FROZEN_BOOK')
                check(sum(v.numel() for k,v in final.items() if not k.startswith('book_'))==rep['parameters'],'PARAMETERS')
                with np.load(f/'outputs.npz',allow_pickle=False) as z:
                    check(set(z.files)=={f'{kind}_{k}' for kind in ('slots','logits') for k in range(6)},'OUTPUT_KEYS')
                    cosine=[];correct=[]
                    for k in range(6):
                        v=z[f'slots_{k}'];l=z[f'logits_{k}']
                        check(v.shape==(8134,768) and l.shape==(8134,64) and v.dtype==l.dtype==np.float32
                              and np.isfinite(v).all() and np.isfinite(l).all() and np.all(abs(np.linalg.norm(v,axis=1)-1)<2e-5),'OUTPUT_VALUES')
                        cosine.append(np.sum(v.astype(np.float64)*truth[k][ctx['target_index']].astype(np.float64),axis=1));correct.append(l.argmax(1)==code[ctx['target_index'],k])
                cosine=np.stack(cosine,1);correct=np.stack(correct,1);metrics=read_json(f/'metrics.json')
                with np.load(f/'scores.npz',allow_pickle=False) as z:check(np.allclose(cosine,z['cosine'],rtol=0,atol=2e-6) and np.array_equal(correct,z['correct']),'SCORES')
                check(same(stats(cosine,correct,rr),metrics['primary']),'METRICS')
                if arm=='region':
                    with np.load(f/'interventions.npz',allow_pickle=False) as z:
                        for scope in ('fact1-only','fact2-only','question-only','facts-only','other-group-facts'):
                            c=z[scope+'_cosine'];ok=z[scope+'_correct'];check(c.shape==ok.shape==(8134,6) and np.isfinite(c).all() and same(stats(c,ok,rr),metrics[scope]),'INTERVENTIONS')
                control=read_json(f/'controls.json');check(control['private_field_refused'] and control['teacher_forced_region'] is False and control['unsnapped']
                      and set(control['input_changes'])=={'question','fact1','fact2'} and all(v>0 for v in control['input_changes'].values()),'CONTROLS')
                count+=1;summary[path]=metrics['primary']['evaluation']
    return dict(report_sha256=file_sha(root/'report.json'),fits=count,all_source_maps_exact=True,train_only_codebooks=True,
                all_initial_objectives_reconstructed=True,all_scores_reconstructed=True,summary=summary,
                scientific_admission=False,served=False,free_text_composition_proven=False,acceptance='author-checked')
