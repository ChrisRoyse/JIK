"""One complete source-bound training pass, with durable moments and cold replay.

This experiment learns a derived target codebook. It does not forecast the next
atom, admit a sense, or establish that a source's assertion is true.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch

from audit import legal_streaming_rq_readback as initialization, legal_training_statistics as training
from audit import legal_native_training_probe as probe, legal_rq_growth as growth
from audit.legal_rq_growth_readback import recount
from audit.legal_rq_intervention_steps import loss_reference
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from lenscommon import take_lease,release_lease
from model.legal_native_training_stream import CertifiedNativeFrames
from model.native_rq_packed import PackedNativeRQ
from model import native_population_training as steps
from senses.contract import canonical,digest
from text.constellation_corpus import file_sha,need

POLICY=dict(device='cuda',batch=128,passes=1,learning_rate=1e-4,weight_decay=.01,seed=20261024,
            foreach=False,fused=False,precision='fp32_tf32_disabled',held='phase45_source_separated_with_all_multiwindow',
            cadence_rows=[13184,131072,1048576],promotion=False)


def code():
    root=Path(__file__).resolve().parents[1]
    return dict(initialization.code(),**{n:file_sha(root/n) for n in ['audit/legal_population_training.py',
        'model/native_population_training.py','audit/legal_rq_growth.py','audit/legal_rq_growth_readback.py',
        'audit/legal_rq_intervention_steps.py']})


def make(stats,dimensions,state):
    model=probe.construct(PackedNativeRQ,stats,dimensions,state).to(POLICY['device'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=POLICY['learning_rate'],weight_decay=POLICY['weight_decay'],
                               foreach=False,fused=False)
    return model,optimizer


def checkpoint(path,model,optimizer,binding,step,rows,chain):
    pin=steps.save(path,model,optimizer,binding,step,rows,chain)
    saved=torch.load(path,map_location='cpu',weights_only=True)
    # Pickle memoization can differ after loading identical AdamW state. Bind
    # both physical bytes and the canonical complete tensor/configuration state.
    return dict(pin=pin,state=steps.fingerprint(saved),rows=rows)


def evaluate(model,reader,data,ids,held,stats,dimensions,root):
    root.mkdir();clone=probe.construct(PackedNativeRQ,stats,dimensions,steps.cpu(model.state_dict()));results={}
    populations=dict(train_probe=(ids,0),development=(held['development'],1),query=(held['query'],2))
    for name,(uids,side) in populations.items():
        cases=growth.components_for(data,uids,side);result=growth.evaluate(clone,reader,uids,cases,stats,root/name)
        arrays=[np.load(root/name/f,allow_pickle=False) for f in ['codes.npy','errors.npy','eligible.npy','windows.npy']]
        recount(result,*arrays,reader.roster);results[name]=result
    return results


def controls(model,opt,reader,uids,held,c,binding,checkpoint,pin,step,rows):
    frames=reader.frames([int(uids[0])]);broken={n:x.clone() for n,x in frames[0].items()}
    broken[reader.roster[0]][0,0]=float('nan');broken={n:x.to(POLICY['device']) for n,x in broken.items()}
    return dict(nontrain=rejected(lambda:training.training_only(np.asarray(held['query'][:1],dtype='<i8'),c.presence)),
        changed_pin=rejected(lambda:steps.load(checkpoint,'0'*64,model,opt,binding,step,rows,POLICY)),
        changed_cursor=rejected(lambda:steps.load(checkpoint,pin,model,opt,binding,step+1,rows,POLICY)),
        nonfinite=rejected(lambda:model.latent([broken])))


def run(a):
    start=tick=time.perf_counter();cpu_start=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
    need(not active.strip() and torch.cuda.is_available(),'POPULATION_GPU_BUSY')
    take_lease('29','complete-native-population-learning')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    _,_,data,c,selected,held,_,_=training.authenticate(a.root)
    source=training.pinned(a.initialization/'report.json',a.initialization_pin)
    cold=training.pinned(a.initialization_cold/'report.json',a.initialization_cold_pin)
    need(source['complete'] and cold['complete'] and cold['mode']=='cold' and cold['previous_pin']==a.initialization_pin
         and source['binding']==cold['binding'] and source['binding']['code']==initialization.code()
         and source['binding']['source']==data.binding,'POPULATION_INITIALIZATION_AUTHORITY')
    for filename in ['initialized.pt','uids.npy']:
        need(file_sha(a.initialization/filename)==source['files'][filename],'POPULATION_INITIALIZATION_FILE')
    previous=None
    if a.mode=='cold':
        previous=training.pinned(a.previous/'report.json',a.previous_pin)
        need(previous['complete'] and previous['mode'] in ['qualify','fit'],'POPULATION_COLD_SOURCE')
        scope=previous['scope']
    else:scope='qualify' if a.mode=='qualify' else 'full'
    need(source['scope']==scope,'POPULATION_INITIALIZATION_SCOPE')
    uids=np.load(a.initialization/'uids.npy',mmap_mode='r',allow_pickle=False)
    for lo in range(0,len(uids),16384):training.training_only(uids[lo:lo+16384],c.presence)
    statsroot=a.root/initialization.STATS[0];sr=training.pinned(statsroot/'report.json',initialization.STATS[1])
    need(file_sha(statsroot/'statistics.pt')==sr['files']['statistics.pt'],'POPULATION_STATISTICS_PIN')
    stats=torch.load(statsroot/'statistics.pt',map_location='cpu',weights_only=True)
    state=torch.load(a.initialization/'initialized.pt',map_location='cpu',weights_only=True)
    dimensions={n:len(stats['centers'][n]) for n in data.roster}
    binding=dict(code=code(),policy=POLICY,source=data.binding,initialization=a.initialization_pin,
        initialization_cold=a.initialization_cold_pin,uids=source['files']['uids.npy'],statistics=initialization.STATS[1],
        torch_version=str(torch.__version__),device_name=torch.cuda.get_device_name(0))
    if previous:need(previous['binding']==binding,'POPULATION_COLD_BINDING')
    if a.mode=='fit':
        q=training.pinned(a.qualification/'report.json',a.qualification_pin)
        qc=training.pinned(a.qualification_cold/'report.json',a.qualification_cold_pin)
        need(q['complete'] and q['scope']=='qualify' and qc['complete'] and qc['mode']=='cold'
             and qc['previous_pin']==a.qualification_pin and q['binding']==qc['binding']
             and all(q['binding'][k]==binding[k] for k in ['code','policy','source','statistics','torch_version','device_name']),
             'POPULATION_QUALIFICATION')
    model,opt=make(stats,dimensions,state);reader=CertifiedNativeFrames(data,a.root/training.DEFAULTS['audit'][0],training.DEFAULTS['audit'][1])
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter();proof_ids=selected.tolist()
    result_evaluations={'initial':evaluate(model,reader,data,proof_ids,held,stats,dimensions,a.output/'initial-evaluation')}
    clone=probe.construct(PackedNativeRQ,stats,dimensions,steps.cpu(model.state_dict()))
    raw_loss=loss_reference(clone,reader,proof_ids,growth.components_for(data,proof_ids,0),a.output)
    phases['initial_evaluation']=time.perf_counter()-tick;tick=time.perf_counter()
    chain=digest(binding);last_after=steps.fingerprint(model.state_dict());checkpoints={}
    checkpoints['0']=checkpoint(a.output/'checkpoint-0.pt',model,opt,binding,0,0,chain)
    if previous:
        steps.load(a.previous/'checkpoint-0.pt',previous['checkpoints']['0']['pin'],model,opt,binding,0,0,POLICY)
        need(file_sha(a.previous/'steps.jsonl')==previous['files']['steps.jsonl'],'POPULATION_COLD_LEDGER_PIN')
    count=(len(uids)+127)//128;rows=0;windows={n:0 for n in data.roster};eligible_rows=0
    points={1,count-1,count}|{min(count,(n+127)//128) for n in POLICY['cadence_rows']};points.discard(0)
    timings=dict(reading=0.,update_and_hashes=0.,ledger=0.,checkpoint=0.,evaluation=0.)
    old=(a.previous/'steps.jsonl').open() if previous else None
    with (a.output/'steps.jsonl').open('x') as ledger, (a.output/'timings.jsonl').open('x') as cost:
        for number in range(count):
            began=time.perf_counter();lo=number*128;ids=[int(x) for x in uids[lo:lo+128]]
            frames=reader.frames(ids);cases=growth.components_for(data,ids,0);read_seconds=time.perf_counter()-began
            for frame in frames:
                for n in data.roster:windows[n]+=len(frame[n])
            torch.cuda.synchronize();began=time.perf_counter();row=steps.step(model,opt,frames,ids,cases,number,POLICY)
            torch.cuda.synchronize();update_seconds=time.perf_counter()-began
            need(row['before']==last_after,'POPULATION_STEP_CHAIN');last_after=row['after'];rows+=len(ids)
            eligible_rows+=sum(row['eligible']);record=dict(parent=chain,update=row);chain=digest(record);record['hash']=chain
            if old:
                expected=json.loads(old.readline());need(record==expected,'POPULATION_COLD_STEP')
            began=time.perf_counter();ledger.write(canonical(record).decode()+'\n');ledger.flush();os.fsync(ledger.fileno())
            ledger_seconds=time.perf_counter()-began
            for name,value in [('reading',read_seconds),('update_and_hashes',update_seconds),('ledger',ledger_seconds)]:timings[name]+=value
            cost.write(json.dumps(dict(step=number+1,reading=read_seconds,update_and_hashes=update_seconds,ledger=ledger_seconds))+'\n')
            if number+1 in points:
                began=time.perf_counter();path=a.output/f'checkpoint-{number+1}.pt'
                checkpoints[str(number+1)]=checkpoint(path,model,opt,binding,number+1,rows,chain)
                if previous:
                    item=previous['checkpoints'][str(number+1)]
                    need(all(item[k]==checkpoints[str(number+1)][k] for k in ['rows','state']),'POPULATION_COLD_CHECKPOINT')
                    model,opt=make(stats,dimensions,state)
                    saved=steps.load(a.previous/path.name,item['pin'],model,opt,binding,number+1,rows,POLICY)
                    need(saved['ledger']==chain and steps.fingerprint(saved)==item['state'],'POPULATION_COLD_CHECKPOINT_CHAIN')
                timings['checkpoint']+=time.perf_counter()-began;began=time.perf_counter()
                if number+1==count:
                    result_evaluations['final']=evaluate(model,reader,data,proof_ids,held,stats,dimensions,a.output/'final-evaluation')
                timings['evaluation']+=time.perf_counter()-began
                print(json.dumps(dict(stage='checkpoint',step=number+1,rows=rows,total=len(uids),loss=row['loss'])),flush=True)
            if (number+1)%128==0:cost.flush();print(json.dumps(dict(stage='training',step=number+1,rows=rows,total=len(uids))),flush=True)
        cost.flush();os.fsync(cost.fileno())
    if old:need(not old.read(),'POPULATION_COLD_LEDGER_EXTRA');old.close()
    need(rows==len(uids),'POPULATION_FULL_COVERAGE')
    if scope=='full':need(windows=={n:sr['lanes'][n]['windows'] for n in data.roster},'POPULATION_WINDOW_COVERAGE')
    phases['training']=time.perf_counter()-tick;tick=time.perf_counter()
    if previous:need(result_evaluations==previous['evaluations'],'POPULATION_COLD_EVALUATIONS')
    final=a.output/f'checkpoint-{count}.pt'
    negatives=controls(model,opt,reader,uids,held,c,binding,final,checkpoints[str(count)]['pin'],count,rows)
    data.fresh();c.fresh();reader.close();uids._mmap.close()
    phases['controls']=time.perf_counter()-tick;tick=time.perf_counter()
    files={str(f.relative_to(a.output)):file_sha(f) for f in a.output.rglob('*') if f.is_file()}
    phases['final_hashes']=time.perf_counter()-tick;seconds=time.perf_counter()-start
    need(sum(phases.values())/seconds>=.9,'POPULATION_TIMING_COVERAGE')
    result=dict(complete=True,mode=a.mode,scope=scope,binding=binding,previous_pin=a.previous_pin,files=files,
        checkpoints=checkpoints,ledger_head=chain,rows=rows,updates=count,native_windows=windows,
        contrast_eligible_rows=eligible_rows,evaluations=result_evaluations,independent_initial_loss=raw_loss,
        controls=negatives,reader_stats=reader.stats,update_phases_seconds=timings,phases_seconds=phases,
        seconds=seconds,cpu_seconds=time.process_time()-cpu_start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        model_promotion=False,scientific_admission=False,number_status='measured',acceptance='author-checked')
    print(json.dumps(dict(complete=True,pin=write_committed(a.output/'report.json',result),seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--mode',choices=['qualify','fit','cold'],required=True);p.add_argument('--output',type=Path,required=True)
    for name in ['initialization','initialization-cold','previous','qualification','qualification-cold']:
        p.add_argument('--'+name,type=Path,required=name.startswith('initialization'));p.add_argument('--'+name+'-pin',required=name.startswith('initialization'))
    a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.is_dir():write_committed(a.output/'failure.json',dict(error=type(exc).__name__+':'+str(exc)))
        raise
    finally:release_lease()
