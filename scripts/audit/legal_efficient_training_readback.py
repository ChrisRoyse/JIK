"""Compare every real optimizer update across original and batched-transport paths."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if '--cpu-proof' in sys.argv: os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch

from audit import legal_cached_predictor_training as original
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from lenscommon import take_lease, release_lease
from model import native_document_efficient as efficient
from model.native_device_transport import transfer, EfficientNativePrefixStream, validate_optimizer
from model.native_population_training import cpu, fingerprint
from model.native_region_predictor import NativeRegionPredictor
from text.constellation_corpus import file_sha, need


def code():
    root=Path(__file__).resolve().parents[1]
    return dict(original.code(), **{n:file_sha(root/n) for n in ['model/native_device_transport.py',
        'model/native_document_efficient.py','audit/legal_efficient_training_readback.py']})


def make(target, ref, device):
    m=NativeRegionPredictor(target,target_pin=ref['pin'],seed=original.POLICY['seed']).to(device)
    o=torch.optim.AdamW(m.parameters(),lr=original.POLICY['learning_rate'],weight_decay=original.POLICY['weight_decay'],foreach=False,fused=False)
    return m,o


def synchronize(device):
    if device=='cuda': torch.cuda.synchronize()


def run(a):
    began=tick=time.perf_counter(); cpu0=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    device='cpu' if a.cpu_proof else 'cuda'
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    if device=='cuda':
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
        need(not active.strip() and torch.cuda.is_available(),'EFFICIENT_TRAIN_GPU_BUSY')
        take_lease('29','exact-native-transfer-qualification')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for name,pin in code().items():
        path=a.output/'source'/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes())
        need(file_sha(path)==pin,'EFFICIENT_TRAIN_SOURCE_COPY')
    io=original.pinned(a.io,a.io_pin)
    need(io['complete'] and io['all_native_bytes_exact'] and io['all_handles_closed'],'EFFICIENT_TRAIN_NATIVE_AUTHORITY')
    original.physical(a.io,io)
    q=original.small.qualification;statsroot=a.root/q.initialization.STATS[0]
    sr=original.pinned(statsroot,q.initialization.STATS[1])
    need(file_sha(statsroot/'statistics.pt')==sr['files']['statistics.pt'],'EFFICIENT_TRAIN_STATISTICS')
    stats=torch.load(statsroot/'statistics.pt',map_location='cpu',weights_only=True)
    target,_,ref=q.load_target(a.root,stats,{n:len(value) for n,value in stats['centers'].items()})
    binding=dict(code=code(),native=a.io_pin,policy=original.POLICY,device=device,target=ref,
                 arithmetic='unchanged_GEMM_shapes_addition_order_losses_and_AdamW',torch_version=str(torch.__version__))
    models={n:make(target,ref,device) for n in ['original','efficient']}
    initial={n:fingerprint((m.state_dict(),o.state_dict())) for n,(m,o) in models.items()}
    need(initial['original']==initial['efficient'],'EFFICIENT_TRAIN_INITIAL')
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    steps=[]; timings=[];checkpoints=[];target_pin=fingerprint(target.state_dict())
    prior=original.pinned(a.root/original.LONG[0],original.LONG[1])
    old_steps=[json.loads(line)['update'] for line in (a.root/original.LONG[0]/'steps.jsonl').read_text().splitlines()]
    need(file_sha(a.root/original.LONG[0]/'steps.jsonl')==prior['files']['steps.jsonl'],'EFFICIENT_TRAIN_PRIOR_LEDGER')
    total=0
    for step,row in enumerate(io['plan']):
        p=torch.load(a.io/f"document-{row['document']}.pt",map_location='cpu',weights_only=True)
        item=dict(document=row['document'],side='train',frames=p['frames'],codes=p['codes'][1:],
                  counts={n:x[1:] for n,x in p['counts'].items()},native_pin=p['native_pin'])
        item['label_pin']=fingerprint((item['codes'],item['counts']))
        outputs={};cp={}
        for name in (['original','efficient'] if step%2==0 else ['efficient','original']):
            m,o=models[name];synchronize(device);start=time.perf_counter()
            fn=original.small.steps.update if name=='original' else efficient.update
            outputs[name]=fn(m,o,item,step,original.POLICY);synchronize(device)
            timings.append(dict(step=step+1,document=row['document'],arm=name,seconds=time.perf_counter()-start))
            path=a.output/f'{name}-checkpoint-{step+1}.pt'
            cp[name]=original.small.steps.checkpoint(path,m,o,binding,step+1,total+len(item['codes']),fingerprint(outputs[name]))
        need(outputs['original']==outputs['efficient'],'EFFICIENT_TRAIN_UPDATE_NOT_EXACT')
        need(cp['original']['state']==cp['efficient']['state'],'EFFICIENT_TRAIN_CHECKPOINT_NOT_EXACT')
        if device=='cuda':need(outputs['original']==old_steps[step],'EFFICIENT_TRAIN_PRIOR_GPU_NOT_EXACT')
        steps.append(outputs['efficient']);checkpoints.append(cp);total+=len(item['codes'])
        # Resume after each update, checking full moments and cursor.
        models['efficient']=make(target,ref,device);m,o=models['efficient']
        original.small.steps.load(a.output/f'efficient-checkpoint-{step+1}.pt',cp['efficient']['pin'],m,o,binding,step+1,total,original.POLICY)
        print(json.dumps(dict(step=step+1,exact=True,timings=timings[-2:])),flush=True)
    phases['five_paired_complete_document_updates']=time.perf_counter()-tick;tick=time.perf_counter()
    m,o=models['efficient'];before=fingerprint((m.state_dict(),o.state_dict()))
    controls=dict(nontrain=rejected(lambda:efficient.update(m,o,dict(item,side='query'),5,original.POLICY)),
                  native_pin=rejected(lambda:efficient.update(m,o,dict(item,native_pin='0'*64),5,original.POLICY)),
                  label_pin=rejected(lambda:efficient.update(m,o,dict(item,label_pin='0'*64),5,original.POLICY)),
                  cursor=rejected(lambda:validate_optimizer(m,o,original.POLICY,4)))
    frames=transfer(item['frames'][:2],device)
    need(fingerprint(transfer(frames,'cpu'))==fingerprint(item['frames'][:2]),'EFFICIENT_TRANSFER_ROUNDTRIP')
    broken=tuple({n:x.clone() for n,x in f.items()} for f in frames);broken[0][m.roster[0]][0,0]=float('nan')
    stream=EfficientNativePrefixStream(m.encoder,m.projectors,cache_bytes=m.cache_bytes)
    controls['nonfinite']=rejected(lambda:stream.step(broken,context='negative',offset=0,detach_history=True))
    state=next(iter(o.state.values()));old=state['exp_avg'].clone();state['exp_avg'].view(-1)[0]=float('nan')
    try:controls['nonfinite_moment']=rejected(lambda:validate_optimizer(m,o,original.POLICY,5))
    finally:state['exp_avg'].copy_(old)
    need(fingerprint((m.state_dict(),o.state_dict()))==before and fingerprint(target.state_dict())==target_pin,'EFFICIENT_TRAIN_CONTROL_MUTATION')
    if a.previous:
        old=original.pinned(a.previous,a.previous_pin);original.physical(a.previous,old)
        need(old['complete'] and old['binding']==binding and old['steps']==steps and old['controls']==controls,
             'EFFICIENT_TRAIN_COLD_NOT_EXACT')
        need([c['efficient']['state'] for c in old['checkpoints']]==[c['efficient']['state'] for c in checkpoints],'EFFICIENT_TRAIN_COLD_CHECKPOINT')
    need(device!='cpu' or not torch.cuda.is_initialized(),'EFFICIENT_TRAIN_UNEXPECTED_GPU')
    phases['controls_and_cold_readback']=time.perf_counter()-tick;tick=time.perf_counter()
    files={str(f.relative_to(a.output)):file_sha(f) for f in a.output.rglob('*') if f.is_file()}
    phases['hashes']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(sum(phases.values())/seconds>=.9,'EFFICIENT_TRAIN_PHASE_ATTRIBUTION')
    result=dict(complete=True,binding=binding,steps=steps,timings=timings,checkpoints=checkpoints,controls=controls,
                previous_pin=a.previous_pin,files=files,all_updates_exact=True,prior_gpu_exact=device=='cuda',
                phases_seconds=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu0,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                peak_gpu_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
                acceptance='author-checked',number_status='measured',scientific_admission=False)
    print(json.dumps(dict(complete=True,pin=write_committed(a.output/'report.json',result))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--io',type=Path,required=True);p.add_argument('--io-pin',required=True)
    p.add_argument('--cpu-proof',action='store_true');p.add_argument('--previous',type=Path);p.add_argument('--previous-pin')
    a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error)))
        raise
    finally:release_lease()
