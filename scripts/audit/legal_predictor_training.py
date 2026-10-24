"""Small real CUDA document-training qualification and complete checkpoint replay."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch

from audit import legal_region_predictor_readback as qualification
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from lenscommon import take_lease,release_lease
from model.constellation import ordered_distance
from model import native_document_training as steps
from model.native_population_training import cpu,fingerprint
from model.native_region_predictor import NativeRegionPredictor,target_labels
from senses.contract import canonical,digest
from text.constellation_corpus import file_sha,need

QUAL=('legal-region-predictor-qualification-v2-20261023','d5d4477fb98face26b4556d7101351a5ea4cdfd5271af912e35a0df776a54570')
COLD=('legal-region-predictor-cold-v2-20261023','a9aab63aa684030d7445446df93e814fd803615346442453b1c2045ea0d511d2')
POLICY=dict(device='cuda',threads=1,seed=20261025,tile=8,epochs=2,learning_rate=1e-4,weight_decay=.01,
            foreach=False,fused=False,precision='fp32_tf32_disabled_fp64_categorical_reduction',
            scope='source_document_training_path_qualification',promotion=False)


def code():
    root=Path(__file__).resolve().parents[1]
    return dict(qualification.code(),**{n:file_sha(root/n) for n in ('audit/legal_predictor_training.py',
        'model/native_document_training.py','model/constellation.py')})


def make(target,pin,state=None):
    model=NativeRegionPredictor(target,target_pin=pin,seed=POLICY['seed'])
    if state is not None:model.load_state_dict(state,strict=True)
    model=model.to('cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=POLICY['learning_rate'],weight_decay=POLICY['weight_decay'],
                               foreach=False,fused=False)
    return model,optimizer


def distance(left,right):
    a=left.detach().cpu().double().numpy();b=right.detach().cpu().double().numpy()
    an=np.linalg.norm(a,axis=1,keepdims=True);bn=np.linalg.norm(b,axis=1,keepdims=True)
    need(np.isfinite(a).all() and np.isfinite(b).all() and bool((an>0).all()) and bool((bn>0).all()),
         'PREDICTOR_EVALUATION_NATIVE')
    return ordered_distance(np.clip((a/an)@(b/bn).T,-1,1))


@torch.no_grad()
def evaluate(model,target,ref,items,stats,output):
    output.mkdir();clone=NativeRegionPredictor(target,target_pin=ref['pin'],seed=POLICY['seed'])
    clone.load_state_dict(cpu(model.state_dict()),strict=True);results=[]
    train_codes=torch.cat([i['codes'] for i in items if i['side']=='train'])
    majority=torch.tensor([torch.bincount(train_codes[:,l],minlength=size).argmax() for l,size in enumerate(clone.sizes)])
    for item in items:
        frames=item['frames'];hidden=clone.encoder.encode(frames,clone.projectors)[:-1]
        predicted=clone.forecast(hidden);native=clone.render(predicted,target)
        forced=clone.loss(hidden,item['codes'],item['counts']);copy_codes,_=target_labels(target,frames[:-1])
        ce=[];brier=[]
        for l,score in enumerate(forced['logits']):
            value=score.double().numpy();high=value.max(1,keepdims=True)
            probability=np.exp(value-high);probability/=probability.sum(1,keepdims=True)
            label=item['codes'][:,l].numpy();n=np.arange(len(label))
            ce.append(-np.log(probability[n,label]));expected=np.eye(clone.sizes[l])[label]
            brier.append(np.square(probability-expected).sum(1))
            need(abs(float(ce[-1].mean())-float(forced['region'][l]))<=5e-7,'PREDICTOR_EVALUATION_CE')
        rows=[]
        for j in range(len(item['codes'])):
            lanes={}
            for name in clone.roster:
                lanes[name]=dict(target_count=int(item['counts'][name][j]),predicted_count=int(predicted['counts'][name][j]),
                    ordered_prediction=distance(frames[j+1][name],native[j][name]),
                    ordered_copy_current=distance(frames[j+1][name],frames[j][name]),
                    ordered_train_mean=distance(frames[j+1][name],stats['centers'][name].float()[None]))
            rows.append(dict(target_codes=item['codes'][j].tolist(),predicted_codes=predicted['codes'][j].tolist(),
                copy_current_codes=copy_codes[j].tolist(),majority_codes=majority.tolist(),
                conditional_code_ce=[float(x[j]) for x in ce],conditional_code_brier=[float(x[j]) for x in brier],lanes=lanes))
        results.append(dict(document=item['document'],side=item['side'],rows=rows))
    summary={}
    for side in ['train','development','query']:
        rows=[row for d in results if d['side']==side for row in d['rows']]
        summary[side]=dict(rows=len(rows),conditional_code_ce=np.mean([r['conditional_code_ce'] for r in rows],axis=0).tolist(),
            conditional_code_brier=np.mean([r['conditional_code_brier'] for r in rows],axis=0).tolist(),
            accuracy={key:np.mean([[int(x==y) for x,y in zip(r[key],r['target_codes'])] for r in rows],axis=0).tolist()
                      for key in ['predicted_codes','copy_current_codes','majority_codes']},
            per_lane={n:dict(predicted_count_accuracy=sum(r['lanes'][n]['target_count']==r['lanes'][n]['predicted_count'] for r in rows)/len(rows),
                **{k:sum(r['lanes'][n][k] for r in rows)/len(rows) for k in ['ordered_prediction','ordered_copy_current','ordered_train_mean']})
                for n in clone.roster})
    payload=dict(results=results,summary=summary,majority_scope='66_TRAIN_transitions_of_qualification_only',
                 metric='per_instrument_ordered_cosine_edit_distance_v1',autonomous_counts=True,skill_claim=False)
    pin=write_committed(output/'report.json',payload)
    need(json.loads((output/'report.json').read_bytes())==payload,'PREDICTOR_EVALUATION_READBACK')
    return dict(pin=pin,summary=summary)


def load_items(root,q,target):
    plan=qualification.training.pinned(root/QUAL[0]/'selection.json',q['binding']['selection']);items=[]
    for row in plan['documents']:
        path=root/QUAL[0]/f"document-{row['document']}"/'outputs.pt'
        need(file_sha(path)==q['files'][str(path.relative_to(root/QUAL[0]))],'PREDICTOR_NATIVE_PROOF_PIN')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        need(fingerprint(saved['frames'])==saved['native_pin'],'PREDICTOR_NATIVE_PROOF_VALUES')
        codes,counts=target_labels(target,saved['frames'][1:])
        need(torch.equal(codes,saved['codes']) and fingerprint(counts)==fingerprint(saved['counts']),
             'PREDICTOR_FROZEN_TARGET_LABELS')
        items.append(dict(document=row['document'],side=row['side'],frames=saved['frames'],codes=codes,counts=counts,
                          native_pin=saved['native_pin'],label_pin=fingerprint((codes,counts))))
    return items,plan


def run(a):
    began=tick=time.perf_counter();cpu_start=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    for name,pin in code().items():
        saved=a.output/'source'/name;saved.parent.mkdir(parents=True,exist_ok=True)
        saved.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes());need(file_sha(saved)==pin,'PREDICTOR_SOURCE_COPY')
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
    need(not active.strip() and torch.cuda.is_available(),'PREDICTOR_GPU_BUSY')
    take_lease('29','ordered-native-predictor-training-qualification')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    q=qualification.training.pinned(a.root/QUAL[0]/'report.json',QUAL[1])
    qc=qualification.training.pinned(a.root/COLD[0]/'report.json',COLD[1])
    need(q['complete'] and qc['complete'] and qc['previous_pin']==QUAL[1] and q['binding']==qc['binding']
         and q['binding']['code']==qualification.code(),'PREDICTOR_BOUNDARY_QUALIFICATION')
    _,_,data,c,_,_,_,_=qualification.training.authenticate(a.root)
    path=a.root/qualification.initialization.STATS[0];st=qualification.training.pinned(path/'report.json',qualification.initialization.STATS[1])
    need(file_sha(path/'statistics.pt')==st['files']['statistics.pt'],'PREDICTOR_STATISTICS_PIN')
    stats=torch.load(path/'statistics.pt',weights_only=True,map_location='cpu')
    target,fit,ref=qualification.load_target(a.root,stats,{n:len(stats['centers'][n]) for n in data.roster})
    tc=qualification.training.pinned(a.target_cold/'report.json',a.target_cold_pin)
    need(tc['complete'] and tc['mode']=='cold' and tc['scope']=='full' and tc['previous_pin']==qualification.FIT[1]
         and all(tc[k]==fit[k] for k in ['binding','rows','updates','ledger_head','native_windows','evaluations']),
         'PREDICTOR_FULL_TARGET_COLD_REQUIRED')
    need(q['binding']['source']==data.binding==fit['binding']['source'],'PREDICTOR_SOURCE_BINDING')
    items,plan=load_items(a.root,q,target);target_before=fingerprint(target.state_dict())
    binding=dict(code=code(),policy=POLICY,qualification=QUAL[1],qualification_cold=COLD[1],
                 target=ref,target_cold=a.target_cold_pin,source=data.binding,torch_version=str(torch.__version__),
                 device=torch.cuda.get_device_name(0),selection=q['binding']['selection'])
    previous=qualification.training.pinned(a.previous/'report.json',a.previous_pin) if a.previous else None
    if previous:need(previous['complete'] and previous['binding']==binding,'PREDICTOR_COLD_BINDING')
    model,opt=make(target,ref['pin']);chain=digest(binding);rows=0;step=0
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    evaluations=dict(initial=evaluate(model,target,ref,items,stats,a.output/'initial-evaluation'))
    checkpoints={'0':steps.checkpoint(a.output/'checkpoint-0.pt',model,opt,binding,0,0,chain)}
    if previous:
        saved=steps.load(a.previous/'checkpoint-0.pt',previous['checkpoints']['0']['pin'],model,opt,binding,0,0,POLICY)
        need(saved['ledger']==chain,'PREDICTOR_COLD_INITIAL_LEDGER')
        need(file_sha(a.previous/'steps.jsonl')==previous['files']['steps.jsonl'],'PREDICTOR_COLD_LEDGER_PIN')
    train=[x for x in items if x['side']=='train'];old=(a.previous/'steps.jsonl').open() if previous else None
    phases['initial_evaluation_and_checkpoint']=time.perf_counter()-tick;tick=time.perf_counter()
    with (a.output/'steps.jsonl').open('x') as ledger:
        for epoch in range(POLICY['epochs']):
            for item in train:
                torch.cuda.synchronize();update=steps.update(model,opt,item,step,POLICY);torch.cuda.synchronize()
                rows+=update['targets'];step+=1;record=dict(parent=chain,epoch=epoch,update=update)
                chain=digest(record);record['hash']=chain
                if old:need(record==json.loads(old.readline()),'PREDICTOR_COLD_UPDATE')
                ledger.write(canonical(record).decode()+'\n');ledger.flush();os.fsync(ledger.fileno())
                if step in [1,len(train)*POLICY['epochs']]:
                    cp=a.output/f'checkpoint-{step}.pt';checkpoints[str(step)]=steps.checkpoint(cp,model,opt,binding,step,rows,chain)
                    if previous:
                        actual=checkpoints[str(step)];expected=previous['checkpoints'][str(step)]
                        need(actual['state']==expected['state'] and actual['rows']==expected['rows'],'PREDICTOR_COLD_CHECKPOINT')
                        model,opt=make(target,ref['pin'])
                        saved=steps.load(a.previous/cp.name,expected['pin'],model,opt,binding,step,rows,POLICY)
                        need(saved['ledger']==chain,'PREDICTOR_COLD_CHECKPOINT_LEDGER')
    if old:need(not old.read(),'PREDICTOR_COLD_EXTRA_UPDATE');old.close()
    phases['document_updates_and_checkpoint']=time.perf_counter()-tick;tick=time.perf_counter()
    evaluations['final']=evaluate(model,target,ref,items,stats,a.output/'final-evaluation')
    held=next(x for x in items if x['side']=='query');changed=dict(train[0],label_pin='0'*64)
    cp=a.output/f'checkpoint-{step}.pt';pin=checkpoints[str(step)]['pin'];state_before=fingerprint(model.state_dict())
    negative=dict(nontrain=rejected(lambda:steps.update(model,opt,held,step,POLICY)),
        changed_target=rejected(lambda:steps.update(model,opt,changed,step,POLICY)),
        changed_checkpoint=rejected(lambda:steps.load(cp,'0'*64,model,opt,binding,step,rows,POLICY)),
        changed_cursor=rejected(lambda:steps.load(cp,pin,model,opt,binding,step+1,rows,POLICY)))
    need(state_before==fingerprint(model.state_dict()) and target_before==fingerprint(target.state_dict()),'PREDICTOR_CONTROL_CHANGED_STATE')
    data.fresh();c.fresh()
    need(all(fingerprint(i['frames'])==i['native_pin'] for i in items),'PREDICTOR_NATIVE_CHANGED')
    if previous:need(evaluations==previous['evaluations'],'PREDICTOR_COLD_EVALUATION')
    phases['final_evaluation_controls']=time.perf_counter()-tick;tick=time.perf_counter()
    files={str(f.relative_to(a.output)):file_sha(f) for f in a.output.rglob('*') if f.is_file()}
    phases['final_hashes']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(sum(phases.values())/seconds>=.9,'PREDICTOR_PHASE_ATTRIBUTION')
    result=dict(complete=True,binding=binding,mode='cold' if previous else 'qualify',previous_pin=a.previous_pin,
        files=files,evaluations=evaluations,checkpoints=checkpoints,ledger_head=chain,updates=step,target_presentations=rows,
        train_documents=len(train),controls=negative,phases_seconds=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu_start,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        target_state=target_before,model_promotion=False,scientific_admission=False,number_status='measured',acceptance='author-checked')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',result),updates=step,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--target-cold',type=Path,required=True);p.add_argument('--target-cold-pin',required=True)
    p.add_argument('--previous',type=Path);p.add_argument('--previous-pin');a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error),code=code()))
        raise
    finally:release_lease()
