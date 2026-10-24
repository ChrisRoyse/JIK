"""Qualify complete real TRAIN documents through the longest source history."""
import argparse
import hashlib
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

from audit import legal_predictor_training as small
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from lenscommon import take_lease,release_lease
from model.legal_native_training_stream import CertifiedNativeFrames
from model.native_population_training import fingerprint
from model.native_region_predictor import target_labels
from senses.contract import canonical,digest
from text.constellation_corpus import file_sha,need

QUAL=('legal-predictor-training-qualification-v1-20261023','2a63bc79f538a294b824239b07849eff8f5b53fd6758fc88925de729fd995aed')
COLD=('legal-predictor-training-cold-v1-20261023','76b0d6e48a3970564b1e94b3293bdd2ed7825e0920bb9544762e20da2b5a9ee2')
POLICY=dict(small.POLICY,tile=32,epochs=1,scope='complete_long_TRAIN_document_path_qualification')
q=small.qualification


def code():
    return dict(small.code(),**{'audit/legal_long_document_training.py':file_sha(Path(__file__))})


def selection(data,root):
    lengths=np.asarray(data.documents[:,1]-data.documents[:,0]);train=np.flatnonzero(data.doc_sides==0)
    selected={}
    for target in [15,198,681,3210]:
        distances=np.abs(lengths[train]-target);candidates=train[distances==distances.min()]
        i=min(candidates.tolist(),key=lambda i:hashlib.sha256(f'long-document-20261023:{i}'.encode()).digest())
        selected.setdefault(i,[]).append('length_nearest_'+str(target))
    old=q.training.pinned(root/small.QUAL[0]/'selection.json',
                         q.training.pinned(root/small.QUAL[0]/'report.json',small.QUAL[1])['binding']['selection'])
    row=next(x for x in old['documents'] if x['side']=='train' and 'native_multiwindow' in x['reasons'])
    selected.setdefault(row['document'],[]).append('shortest_real_multiwindow_proof')
    rows=[dict(document=i,side='train',start=int(data.documents[i,0]),stop=int(data.documents[i,1]),
               reasons=selected[i]) for i in sorted(selected,key=lambda i:(lengths[i],i))]
    return dict(documents=rows,score_blind=True,source=data.binding,
                population=dict(documents=len(train),atoms=int(lengths[train].sum()),
                    successors=int((lengths[train]-1).sum()),max_document_atoms=int(lengths[train].max()),
                    sum_document_lengths_squared=int(np.square(lengths[train]).sum())))


def read_document(data,reader,target,row):
    indices=data.indices(list(range(row['start'],row['stop'])),'train',fit=True)
    need(np.array_equal(data.documents[row['document']],[row['start'],row['stop']]),'LONG_TRAIN_DOCUMENT_BOUNDARY')
    uids=data.mapping[indices].tolist();frames=[]
    began=time.perf_counter()
    for lo in range(0,len(uids),128):frames.extend(reader.frames(uids[lo:lo+128]))
    phases=dict(native_read_seconds=time.perf_counter()-began);began=time.perf_counter()
    codes,counts=target_labels(target,frames);native=fingerprint(tuple(frames))
    phases['target_label_seconds']=time.perf_counter()-began
    payload=dict(document=row['document'],uids=uids,codes=codes,counts=counts,native_pin=native,
                 source_start=row['start'],source_stop=row['stop'])
    item=dict(document=row['document'],side='train',frames=tuple(frames),codes=codes[1:],
              counts={n:x[1:] for n,x in counts.items()},native_pin=native)
    item['label_pin']=fingerprint((item['codes'],item['counts']))
    windows={n:sum(len(f[n]) for f in frames) for n in data.roster}
    phases.update(atoms=len(frames),targets=len(frames)-1,native_windows=windows,
                  native_bytes=sum(x.numel()*x.element_size() for f in frames for x in f.values()),
                  total_attention_positions=sum(windows.values())+len(frames)*len(data.roster))
    return item,payload,phases


def run(a):
    began=tick=time.perf_counter();cpu_start=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    for name,pin in code().items():
        dest=a.output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes());need(file_sha(dest)==pin,'LONG_TRAIN_SOURCE')
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
    need(not active.strip() and torch.cuda.is_available(),'LONG_TRAIN_GPU_BUSY');take_lease('29','long-native-document-qualification')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    proof=q.training.pinned(a.root/QUAL[0]/'report.json',QUAL[1]);cold=q.training.pinned(a.root/COLD[0]/'report.json',COLD[1])
    need(proof['complete'] and cold['complete'] and proof['binding']==cold['binding']
         and cold['previous_pin']==QUAL[1] and proof['binding']['code']==small.code(),'LONG_TRAIN_SMALL_AUTHORITY')
    _,_,data,c,_,_,_,_=q.training.authenticate(a.root)
    path=a.root/q.initialization.STATS[0];sr=q.training.pinned(path/'report.json',q.initialization.STATS[1])
    need(file_sha(path/'statistics.pt')==sr['files']['statistics.pt'],'LONG_TRAIN_STATISTICS')
    stats=torch.load(path/'statistics.pt',map_location='cpu',weights_only=True)
    target,fit,ref=q.load_target(a.root,stats,{n:len(stats['centers'][n]) for n in data.roster})
    need(proof['binding']['target']==ref and proof['binding']['source']==data.binding,'LONG_TRAIN_TARGET_BINDING')
    plan=selection(data,a.root);plan_pin=write_committed(a.output/'selection.json',plan)
    binding=dict(code=code(),policy=POLICY,source=data.binding,target=ref,qualification=QUAL[1],cold=COLD[1],
                 selection=plan_pin,torch_version=str(torch.__version__),device=torch.cuda.get_device_name(0))
    previous=q.training.pinned(a.previous/'report.json',a.previous_pin) if a.previous else None
    if previous:need(previous['complete'] and previous['binding']==binding,'LONG_TRAIN_COLD_BINDING')
    reader=CertifiedNativeFrames(data,a.root/q.training.DEFAULTS['audit'][0],q.training.DEFAULTS['audit'][1])
    model,opt=small.make(target,ref['pin']);target_state=fingerprint(target.state_dict());chain=digest(binding)
    checkpoints={'0':small.steps.checkpoint(a.output/'checkpoint-0.pt',model,opt,binding,0,0,chain)}
    if previous:
        saved=small.steps.load(a.previous/'checkpoint-0.pt',previous['checkpoints']['0']['pin'],model,opt,binding,0,0,POLICY)
        need(saved['ledger']==chain,'LONG_TRAIN_COLD_INITIAL')
        need(file_sha(a.previous/'steps.jsonl')==previous['files']['steps.jsonl'],'LONG_TRAIN_COLD_LEDGER')
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter();profiles=[];rows=0
    old=(a.previous/'steps.jsonl').open() if previous else None
    with (a.output/'steps.jsonl').open('x') as ledger:
        for step,row in enumerate(plan['documents']):
            item,payload,profile=read_document(data,reader,target,row);label=a.output/f"labels-{row['document']}.pt"
            torch.save(payload,label)
            need(fingerprint(torch.load(label,map_location='cpu',weights_only=True))==fingerprint(payload),'LONG_TRAIN_LABEL_READBACK')
            if previous:
                need(file_sha(a.previous/label.name)==previous['files'][label.name],'LONG_TRAIN_PREVIOUS_LABEL_PIN')
                need(fingerprint(torch.load(a.previous/label.name,map_location='cpu',weights_only=True))==fingerprint(payload),
                     'LONG_TRAIN_COLD_NATIVE_AND_LABEL')
            torch.cuda.synchronize();start=time.perf_counter();cpu0=time.process_time();torch.cuda.reset_peak_memory_stats()
            update=small.steps.update(model,opt,item,step,POLICY);torch.cuda.synchronize()
            profile.update(update_seconds=time.perf_counter()-start,update_cpu_seconds=time.process_time()-cpu0,
                           peak_gpu_bytes=torch.cuda.max_memory_allocated(),document=row['document'])
            rows+=update['targets'];record=dict(parent=chain,update=update);chain=digest(record);record['hash']=chain
            if old:need(record==json.loads(old.readline()),'LONG_TRAIN_COLD_UPDATE')
            ledger.write(canonical(record).decode()+'\n');ledger.flush();os.fsync(ledger.fileno())
            path=a.output/f'checkpoint-{step+1}.pt';checkpoints[str(step+1)]=small.steps.checkpoint(path,model,opt,binding,step+1,rows,chain)
            if previous:
                expected=previous['checkpoints'][str(step+1)]
                need(checkpoints[str(step+1)]['state']==expected['state'],'LONG_TRAIN_COLD_CHECKPOINT')
                model,opt=small.make(target,ref['pin'])
                saved=small.steps.load(a.previous/path.name,expected['pin'],model,opt,binding,step+1,rows,POLICY)
                need(saved['ledger']==chain,'LONG_TRAIN_RELOADED_LEDGER')
            profiles.append(profile);write_committed(a.output/f"profile-{row['document']}.json",profile)
            print(json.dumps(dict(document=row['document'],atoms=profile['atoms'],update_seconds=profile['update_seconds'],
                                  peak_gpu_bytes=profile['peak_gpu_bytes'])),flush=True)
    if old:need(not old.read(),'LONG_TRAIN_COLD_EXTRA_STEP');old.close()
    phases['document_read_label_update_checkpoint']=time.perf_counter()-tick;tick=time.perf_counter()
    last=plan['documents'][-1];wrong=dict(last,stop=last['stop']-1)
    held=int(np.flatnonzero(data.doc_sides==2)[0]);lo,hi=map(int,data.documents[held])
    negative=dict(changed_document_boundary=rejected(lambda:read_document(data,reader,target,wrong)),
        held_fit=rejected(lambda:read_document(data,reader,target,dict(document=held,start=lo,stop=hi))),
        stale_cursor=rejected(lambda:small.steps.load(path,checkpoints[str(len(profiles))]['pin'],model,opt,binding,
                                                        len(profiles)+1,rows,POLICY)))
    reader.close();c.fresh();need(fingerprint(target.state_dict())==target_state,'LONG_TRAIN_TARGET_CHANGED')
    phases['controls_final_authentication']=time.perf_counter()-tick;tick=time.perf_counter()
    files={str(f.relative_to(a.output)):file_sha(f) for f in a.output.rglob('*') if f.is_file()}
    phases['hashes']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(sum(phases.values())/seconds>=.9,'LONG_TRAIN_PHASE_ATTRIBUTION')
    result=dict(complete=True,binding=binding,mode='cold' if previous else 'qualify',previous_pin=a.previous_pin,
                profiles=profiles,updates=len(profiles),targets=rows,ledger_head=chain,checkpoints=checkpoints,
                files=files,controls=negative,phases_seconds=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu_start,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,reader=reader.stats,
                requires_full_canonical_label_cache_comparison=True,skill_claim=False,model_promotion=False,
                number_status='measured',acceptance='author-checked')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',result),seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path);p.add_argument('--previous-pin');a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error),code=code()))
        raise
    finally:release_lease()
