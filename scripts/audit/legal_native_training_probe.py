"""Byte-exact certified input readback and measured CPU/GPU student costs."""
import argparse
import copy
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
from audit import legal_training_statistics as training
from audit.legal_rq_batch_readback import difference, flat
from audit.legal_rq_growth import components_for
from audit.legal_residual_codebook_readback import CONFIG, POLICY as RQ_POLICY
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from lenscommon import take_lease, release_lease
from model.legal_native_training_stream import CertifiedNativeFrames
from model.loop_training_checkpoint import tensor_fingerprint
from model.native_rq_batch import BatchedInterventionNativeRQ
from model.native_rq_data import NativeRQData, fingerprint_frames, matched_donors
from model.native_rq_interventions import InterventionNativeRQ
from senses.contract import digest
from text.constellation_corpus import file_sha, need


def pins():
    root=Path(__file__).resolve().parents[1]
    names=['audit/legal_native_training_probe.py','model/legal_native_training_stream.py',
           'model/native_rq_batch.py','model/native_rq_interventions.py','model/attention.py']
    return dict(training.code_pins(),**{n:file_sha(root/n) for n in names})


def construct(cls, stats, dimensions, state=None):
    torch.manual_seed(RQ_POLICY['seed'])
    model=cls(dimensions,CONFIG,RQ_POLICY['sizes'],'no_encoder_quantizer',stats)
    if state is not None:model.load_state_dict(state,strict=True)
    return model


def evaluate(model,frames):
    outputs=[]
    with torch.no_grad():
        for lo in range(0,len(frames),128):
            outputs.append(flat(model(frames[lo:lo+128])))
    return {k:torch.cat([row[k].detach().cpu() for row in outputs]) for k in outputs[0]}


def cpu_state(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_state(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(cpu_state(v) for v in value)
    return value


def run_steps(device,model,frames,uids,cases):
    sync=(lambda:torch.cuda.synchronize()) if device=='cuda' else (lambda:None)
    sync();start=time.perf_counter();model=model.to(device)
    native=tuple({n:x.to(device) for n,x in f.items()} for f in frames)
    donors,eligible=matched_donors(frames,uids,cases,model.roster)
    donors=donors.to(device);eligible=eligible.to(device);sync();transfer=time.perf_counter()-start
    opt=torch.optim.AdamW(model.parameters(),lr=RQ_POLICY['learning_rate'],weight_decay=RQ_POLICY['weight_decay'],foreach=False,fused=False)
    records=[];first_gradient=None
    for step in range(3):
        model.set_step(step);model.check_statistics();opt.zero_grad(set_to_none=True);sync();tick=time.perf_counter()
        out=model.training_loss(native,donors,eligible);sync();forward=time.perf_counter()-tick;tick=time.perf_counter()
        out['loss'].backward();sync();backward=time.perf_counter()-tick;tick=time.perf_counter()
        gradients={n:p.grad for n,p in model.named_parameters()}
        need(all(g is not None and torch.isfinite(g).all() for g in gradients.values()),'TRAIN_PROBE_GRADIENT')
        if first_gradient is None:first_gradient=cpu_state(gradients)
        opt.step();sync();update=time.perf_counter()-tick
        records.append(dict(step=step+1,loss=float(out['loss'].detach()),forward=forward,backward=backward,update_and_checks=update))
    model.check_statistics()
    return dict(transfer_seconds=transfer,steps=records,eligible=int(eligible.sum()),
        gradients=first_gradient,model=cpu_state(model.state_dict()),optimizer=cpu_state(opt.state_dict()))


def run(a):
    start=tick=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    # Even CPU AdamW can initialize a CUDA context in this installed Torch build.
    # Acquire the host lease before optimizer/runtime setup, not after CPU trials.
    resident=subprocess.check_output(['nvidia-smi'],text=True)
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
    (a.output/'nvidia-before.txt').write_text(resident)
    need(not active.strip() and torch.cuda.is_available(),'TRAIN_PROBE_GPU_BUSY_OR_MISSING')
    take_lease('29','legal-native-student-runtime-probe')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    records,bank,data,c,selected,held,scans,binding=training.authenticate(a.root)
    report=training.pinned(a.statistics/'report.json',a.statistics_pin)
    need(report['complete'] and report['binding']==binding,'TRAIN_PROBE_STATISTICS_BINDING')
    stat_file=a.statistics/'statistics.pt'
    need(file_sha(stat_file)==report['files']['statistics.pt'],'TRAIN_PROBE_STATISTICS_PIN')
    stats=torch.load(stat_file,map_location='cpu',weights_only=True)
    uids=selected.tolist()+held['development']+held['query']
    need(len(set(uids))==len(uids),'TRAIN_PROBE_COHORT_DISJOINT')
    certified=CertifiedNativeFrames(data,a.root/training.DEFAULTS['audit'][0],training.DEFAULTS['audit'][1])
    base=records['baseline'];ordinary=NativeRQData(data,Path(base['proof']['path']),base['proof']['pin'])
    phases=dict(bindings=time.perf_counter()-tick);tick=time.perf_counter();frames=[]
    before=digest(tensor_fingerprint(stats));native_hash=hashlib.sha256();reader_timings={}
    # Authenticate/open the real input bank in the new path, then independently
    # bind every selected row through the original source/trace reader.
    for label,reader in [('certified',certified),('ordinary',ordinary)]:
        began=time.perf_counter();cursor=0
        for ids,chunk in reader.tiles(uids):
            if label=='certified':frames.extend(chunk);fingerprint_frames(native_hash,ids,chunk,data.roster)
            else:
                need(all(torch.equal(f[n],frames[cursor+i][n]) for i,f in enumerate(chunk) for n in data.roster),
                     'TRAIN_PROBE_NATIVE_BYTES')
            cursor+=len(ids)
        reader_timings[label]=time.perf_counter()-began
    payload=sum(x.numel()*x.element_size() for frame in frames for x in frame.values())
    need(payload<=64<<20,'TRAIN_PROBE_PHYSICAL_MEMORY')
    # A second certified read measures reuse separately from first authentication.
    began=time.perf_counter()
    for ids,chunk in certified.tiles(uids):pass
    reader_timings['certified_warm']=time.perf_counter()-began
    phases['input_bytes_readback']=time.perf_counter()-tick;tick=time.perf_counter()
    print(json.dumps(dict(stage='native_readback',inputs=len(uids),seconds=phases['input_bytes_readback'])),flush=True)
    dimensions={n:frames[0][n].shape[1] for n in data.roster}
    model=construct(BatchedInterventionNativeRQ,stats,dimensions)
    train=frames[:len(selected)]
    with torch.no_grad():
        latents=torch.cat([model.latent(train[i:i+128]) for i in range(0,len(train),128)])
        model.initialize_latents(latents)
    initial=cpu_state(model.state_dict());reference=evaluate(construct(InterventionNativeRQ,stats,dimensions,initial),frames)
    batched=evaluate(model,frames);output_errors=difference(reference,batched)
    cases=components_for(data,selected.tolist(),0)
    signatures=[tuple(len(f[n]) for n in data.roster) for f in train]
    workloads={label:[i for i,s in enumerate(signatures) if (max(s)>1)==multi][:128]
               for label,multi in [('single_window',False),('multiwindow',True)]}
    need(all(len(v)>0 for v in workloads.values()),'TRAIN_PROBE_WORKLOAD')
    phases['initialization_and_cpu_outputs']=time.perf_counter()-tick;tick=time.perf_counter();steps={};gradient_errors={}
    for label,indices in workloads.items():
        steps[label+':cpu']=run_steps('cpu',construct(BatchedInterventionNativeRQ,stats,dimensions,initial),
            [train[i] for i in indices],[int(selected[i]) for i in indices],[cases[i] for i in indices])
    phases['cpu_steps']=time.perf_counter()-tick;tick=time.perf_counter()
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
    need(all(int(line.split(',')[0])==os.getpid() for line in active.splitlines() if line.strip()),
         'TRAIN_PROBE_GPU_OTHER_OWNER')
    try:
        gpu=construct(BatchedInterventionNativeRQ,stats,dimensions,initial).cuda()
        gpu_frames=tuple({n:x.cuda() for n,x in frame.items()} for frame in frames)
        gpu_output=evaluate(gpu,gpu_frames);gpu_errors=difference(reference,gpu_output)
        del gpu,gpu_frames;torch.cuda.empty_cache()
        for label,indices in workloads.items():
            steps[label+':cuda']=run_steps('cuda',construct(BatchedInterventionNativeRQ,stats,dimensions,initial),
                [train[i] for i in indices],[int(selected[i]) for i in indices],[cases[i] for i in indices])
            gradient_errors[label]=difference(steps[label+':cpu']['gradients'],steps[label+':cuda']['gradients'])
        peak_gpu=torch.cuda.max_memory_allocated()
    finally:release_lease()
    phases['gpu_outputs_steps']=time.perf_counter()-tick;tick=time.perf_counter()
    name=data.roster[0];key=next(iter(certified.refs[name]));old=certified.refs[name][key]
    certified.refs[name][key]=(old[0],'0'*64)
    controls=dict(wrong_shard=rejected(lambda:certified.frames([key])),
                  outside_source=rejected(lambda:certified.frames([len(c.presence)])),
                  oversized_batch=rejected(lambda:certified.frames([0]*129)),
                  implicit_admission=rejected(certified.training_authority))
    certified.refs[name][key]=old;certified.finish();c.fresh()
    need(before==digest(tensor_fingerprint(stats)),'TRAIN_PROBE_STATISTICS_MUTATION')
    torch.save(dict(uids=uids,frames=frames,statistics=stats,initial=initial,reference=reference,batched=batched,
                    gpu=gpu_output,steps=steps),a.output/'state.pt')
    loaded=torch.load(a.output/'state.pt',map_location='cpu',weights_only=True)
    need(tensor_fingerprint(loaded['steps'])==tensor_fingerprint(steps),'TRAIN_PROBE_PERSISTED_UPDATE')
    if a.previous:
        previous=training.pinned(a.previous/'report.json',a.previous_pin)
        need(previous['complete'] and previous['code']==pins() and previous['statistics_pin']==a.statistics_pin,
             'TRAIN_PROBE_COLD_BINDING')
        need(file_sha(a.previous/'state.pt')==previous['files']['state.pt'],'TRAIN_PROBE_COLD_STATE_PIN')
        old=torch.load(a.previous/'state.pt',map_location='cpu',weights_only=True)
        for key in ['uids','frames','statistics','initial','reference','batched','gpu']:
            need(tensor_fingerprint(old[key])==tensor_fingerprint(loaded[key]),'TRAIN_PROBE_COLD_OUTPUT')
        for key in steps:
            for field in ['model','optimizer','gradients']:
                need(tensor_fingerprint(old['steps'][key][field])==tensor_fingerprint(steps[key][field]),'TRAIN_PROBE_COLD_UPDATE')
    phases['persistence_controls_cold']=time.perf_counter()-tick;seconds=time.perf_counter()-start
    need(sum(phases.values())/seconds>=.9,'TRAIN_PROBE_PHASE_COVERAGE')
    result=dict(complete=True,code=pins(),statistics_pin=a.statistics_pin,source_binding=data.binding,
        previous_pin=a.previous_pin,inputs=len(uids),train=len(selected),held={k:len(v) for k,v in held.items()},
        native_pin=native_hash.hexdigest(),native_bytes=payload,reader_timings=reader_timings,
        reader_stats=certified.stats,authentication=data.native.stats,workload_rows={k:len(v) for k,v in workloads.items()},
        step_timings={k:{field:v[field] for field in ['transfer_seconds','steps','eligible']} for k,v in steps.items()},
        output_errors=output_errors,gpu_output_errors=gpu_errors,gradient_errors=gradient_errors,
        controls=controls,files={'state.pt':file_sha(a.output/'state.pt'),'nvidia-before.txt':file_sha(a.output/'nvidia-before.txt')},
        phases_seconds=phases,seconds=seconds,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        peak_gpu_bytes=peak_gpu,model_updates=12,model_promotion=False,scientific_admission=False,
        number_status='measured',acceptance='author-checked')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',result),seconds=seconds)),flush=True)
    certified.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--statistics',type=Path,required=True);p.add_argument('--statistics-pin',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path);p.add_argument('--previous-pin')
    a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.is_dir():write_committed(a.output/'failure.json',dict(error=type(exc).__name__+':'+str(exc)))
        raise
    finally:release_lease()
