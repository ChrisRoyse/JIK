"""Matched real-window objective, gradient, update and cost readback for packing."""
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
import torch
from audit import legal_native_training_probe as probe, legal_training_statistics as training
from audit.legal_rq_batch_readback import difference
from audit.legal_rq_growth import components_for
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from lenscommon import take_lease,release_lease
from model.native_rq_batch import BatchedInterventionNativeRQ
from model.native_rq_data import matched_donors
from model.native_rq_packed import PackedNativeRQ
from model.loop_training_checkpoint import tensor_fingerprint
from senses.contract import digest
from text.constellation_corpus import file_sha,need


def code():
    root=Path(__file__).resolve().parents[1]
    return dict(probe.pins(),**{n:file_sha(root/n) for n in ['audit/legal_packed_training_readback.py','model/native_rq_packed.py']})


def compare_state(actual,expected):
    need(actual.keys()==expected.keys(),'PACKED_READBACK_STATE_KEYS');errors={}
    for name,x in actual.items():
        y=expected[name]
        if x.dtype in [torch.bool,torch.int64]:need(torch.equal(x,y),'PACKED_READBACK_STATE_INTEGER')
        else:errors[name]=difference({name:y},{name:x})[name]
    return errors


def objective(model,frames,ids,cases,device):
    model=model.to(device);native=tuple({n:x.to(device) for n,x in f.items()} for f in frames)
    donors,eligible=matched_donors(frames,ids,cases,model.roster)
    with torch.no_grad():out=model.training_loss(native,donors.to(device),eligible.to(device))
    return {n:out[n].cpu() for n in ['loss','reconstruction','contrast','quantizer_loss']}


def run(a):
    start=tick=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    resident=subprocess.check_output(['nvidia-smi'],text=True);(a.output/'nvidia-before.txt').write_text(resident)
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
    need(not active.strip() and torch.cuda.is_available(),'PACKED_READBACK_GPU_BUSY')
    take_lease('29','packed-native-student-readback')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=training.pinned(a.source/'report.json',a.source_pin)
    cold=training.pinned(a.cold/'report.json',a.cold_pin)
    need(source['complete'] and cold['complete'] and cold['previous_pin']==a.source_pin
         and source['code']==cold['code']==probe.pins(),'PACKED_READBACK_SOURCE_BINDING')
    path=a.source/'state.pt';need(file_sha(path)==source['files']['state.pt'],'PACKED_READBACK_SOURCE_STATE')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    _,_,data,c,selected,_,_,_=training.authenticate(a.root)
    need(payload['uids'][:len(selected)]==selected.tolist() and data.binding==source['source_binding'],'PACKED_READBACK_TRAIN_SCOPE')
    frames=payload['frames'];train=frames[:len(selected)];cases=components_for(data,selected.tolist(),0)
    dimensions={n:frames[0][n].shape[1] for n in data.roster};stats=payload['statistics']
    native_pin=digest(tensor_fingerprint(frames));sig=[tuple(len(f[n]) for n in data.roster) for f in train]
    workloads={label:[i for i,s in enumerate(sig) if (max(s)>1)==multi][:128]
               for label,multi in [('single_window',False),('multiwindow',True)]}
    phases=dict(source_authentication=time.perf_counter()-tick);tick=time.perf_counter()
    results={};states={};outputs={};objectives={};comparisons={}
    for state_name,state in [('initial',payload['initial']),('trained',payload['steps']['multiwindow:cpu']['model'])]:
        for device in ['cpu','cuda']:
            models={kind:probe.construct(cls,stats,dimensions,state).to(device)
                    for kind,cls in [('reference',BatchedInterventionNativeRQ),('packed',PackedNativeRQ)]}
            native=tuple({n:x.to(device) for n,x in f.items()} for f in frames)
            measured={kind:probe.evaluate(model,native) for kind,model in models.items()}
            comparisons[state_name+':'+device+':outputs']=difference(measured['reference'],measured['packed'])
            outputs.update({state_name+':'+device+':'+kind:values for kind,values in measured.items()})
            del models,native
            for label,indices in workloads.items():
                batch=[train[i] for i in indices];ids=[int(selected[i]) for i in indices];components=[cases[i] for i in indices]
                key=state_name+':'+device+':'+label
                for kind,cls in [('reference',BatchedInterventionNativeRQ),('packed',PackedNativeRQ)]:
                    model=probe.construct(cls,stats,dimensions,state)
                    objectives[key+':'+kind]=objective(model,batch,ids,components,device)
                    measured=probe.run_steps(device,probe.construct(cls,stats,dimensions,state),batch,ids,components)
                    states[key+':'+kind]={n:measured[n] for n in ['model','optimizer','gradients']}
                    results[key+':'+kind]={n:measured[n] for n in ['transfer_seconds','steps','eligible']}
                comparisons[key+':objective']=difference(objectives[key+':reference'],objectives[key+':packed'])
                comparisons[key+':gradients']=difference(states[key+':reference']['gradients'],states[key+':packed']['gradients'])
                comparisons[key+':weights']=compare_state(states[key+':packed']['model'],states[key+':reference']['model'])
            print(json.dumps(dict(stage='matched_runtime',state=state_name,device=device)),flush=True)
    phases['objectives_gradients_updates_and_timing']=time.perf_counter()-tick;tick=time.perf_counter()
    model=probe.construct(PackedNativeRQ,stats,dimensions,payload['initial'])
    damaged={n:x.clone() for n,x in train[0].items()};damaged[data.roster[0]][0,0]=float('nan')
    controls=dict(nonfinite=rejected(lambda:model.latent([damaged])),
        missing_slot=rejected(lambda:model.latent([{n:x for n,x in train[0].items() if n!=data.roster[0]}])),
        physical_memory=rejected(lambda:model.groups([{n:1<<20 for n in data.roster}])),
        negative_step=rejected(lambda:model.set_step(-1)))
    model.set_step(100000);need(model.stage()=='quantized_joint' and not model.frozen_names(),'PACKED_READBACK_CURSOR')
    need(digest(tensor_fingerprint(frames))==native_pin,'PACKED_READBACK_NATIVE_CHANGED')
    all_states=dict(states=states,outputs=outputs,objectives=objectives)
    torch.save(all_states,a.output/'state.pt')
    need(tensor_fingerprint(torch.load(a.output/'state.pt',map_location='cpu',weights_only=True))==tensor_fingerprint(all_states),
         'PACKED_READBACK_PERSISTED_STATE')
    if a.previous:
        previous=training.pinned(a.previous/'report.json',a.previous_pin)
        need(previous['complete'] and previous['code']==code() and previous['source_pin']==a.source_pin,'PACKED_READBACK_COLD_BINDING')
        old=a.previous/'state.pt';need(file_sha(old)==previous['files']['state.pt'],'PACKED_READBACK_COLD_PIN')
        need(tensor_fingerprint(torch.load(old,map_location='cpu',weights_only=True))==tensor_fingerprint(all_states),
             'PACKED_READBACK_COLD_STATE')
    data.fresh();c.fresh();phases['persistence_and_controls']=time.perf_counter()-tick;seconds=time.perf_counter()-start
    need(sum(phases.values())/seconds>=.9,'PACKED_READBACK_PHASE_COVERAGE')
    report=dict(complete=True,code=code(),source_pin=a.source_pin,cold_source_pin=a.cold_pin,
        previous_pin=a.previous_pin,native_pin=native_pin,inputs=len(frames),train=len(train),workloads={n:len(v) for n,v in workloads.items()},
        results=results,comparisons=comparisons,controls=controls,files={'state.pt':file_sha(a.output/'state.pt')},
        phases_seconds=phases,seconds=seconds,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),model_updates=48,model_promotion=False,scientific_admission=False,
        number_status='measured',acceptance='author-checked')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    for name in ['source','cold','output']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['source-pin','cold-pin']:p.add_argument('--'+name,required=True)
    p.add_argument('--previous',type=Path);p.add_argument('--previous-pin');a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.is_dir():write_committed(a.output/'failure.json',dict(error=type(exc).__name__+':'+str(exc)))
        raise
    finally:release_lease()
