"""Matched updates, cold restart and corruption controls on an explicit CPU/CUDA runtime."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
if '--cpu-proof' in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import legal_grouped_placement_training as source
from audit import legal_placement_runtime_readback as runtime
from claims.process_conditions import write_committed
from lenscommon import take_lease,release_lease
from model import native_batched_validation as validation
from model import native_placement_batched_checks as candidate
from model import native_placement_efficient as original
from model.native_population_training import cpu,fingerprint
from model.native_rq_packed import PackedNativeRQ
from senses.contract import digest
from text.constellation_corpus import file_sha,need


def refuse(call,code):
    try:call()
    except (ValueError,RuntimeError) as exc:
        need(code in str(exc),'VALIDATION_PROOF_WRONG_REFUSAL:'+str(exc));return str(exc)
    raise ValueError('VALIDATION_PROOF_ACCEPTED:'+code)


def controls(model,opt,policy,step):
    before=fingerprint((model.state_dict(),opt.state_dict()));out={}
    check=lambda:validation.validate_optimizer(model,opt,policy,step)
    parameter=next(model.parameters());saved=opt.state.pop(parameter)
    try:out['missing_state']=refuse(check,'INCOMPLETE_STATE')
    finally:opt.state[parameter]=saved
    for key,value,code in [('step',saved['step']+1,'OPTIMIZER_CURSOR'),
                           ('exp_avg',saved['exp_avg'].reshape(-1)[:1],'OPTIMIZER_STATE'),
                           ('exp_avg',saved['exp_avg'].double(),'OPTIMIZER_STATE')]:
        old=saved[key];saved[key]=value
        try:out[key+'_'+code+'_'+str(value.dtype)]=refuse(check,code)
        finally:saved[key]=old
    last=next(reversed(opt.state.values()));old=last['exp_avg_sq']
    for label,number in [('nan',float('nan')),('inf',float('inf')),('negative_inf',-float('inf'))]:
        bad=old.clone();bad.view(-1)[-1]=number;last['exp_avg_sq']=bad
        try:out['late_'+label]=refuse(check,'OPTIMIZER_STATE')
        finally:last['exp_avg_sq']=old
    group=opt.param_groups[0];old_lr=group['lr'];group['lr']=old_lr*2
    try:out['recipe']=refuse(check,'OPTIMIZER_RECIPE')
    finally:group['lr']=old_lr
    values=[v for state in opt.state.values() for v in state.values()]
    out['all_real_values']=validation.finite_tensors(values)==original.finite_tensors(values)==True
    real=next(v for v in values if v.ndim==2 and min(v.shape)>1).clone()
    for label,value in [('contiguous',real),('noncontiguous',real.T)]:
        need(validation.finite_tensors([value],tile_bytes=64),'VALIDATION_PROOF_FINITE')
        for position in (0,value.numel()//2,value.numel()-1):
            bad=real.clone();bad.view(-1)[position]=float('nan')
            view=bad if label=='contiguous' else bad.T
            need(not validation.finite_tensors([view],tile_bytes=64),'VALIDATION_PROOF_NONFINITE')
        out[label+'_first_middle_last']=True
    out['tile_policy']=refuse(lambda:validation.finite_tensors(values,tile_bytes=0),'VALIDATION_TILE_BYTES')
    check();need(fingerprint((model.state_dict(),opt.state_dict()))==before,'VALIDATION_PROOF_CONTROL_MUTATION')
    out['state_unchanged']=True
    return out


def snapshot(root,name,model,opt,step,binding):
    payload=dict(model=cpu(model.state_dict()),optimizer=cpu(opt.state_dict()),step=step,binding=binding)
    path=root/(name+'.pt');ref=runtime.persist(path,payload);return dict(path=str(path),**ref)


def run(a):
    began=time.perf_counter();cpu_start=time.process_time();a.output.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cpu' if a.cpu_proof else 'cuda'
    if device=='cuda':
        need(a.qualification and a.qualification_pin,'VALIDATION_PROOF_CPU_QUALIFICATION_REQUIRED')
        need(file_sha(a.qualification/'report.json')==a.qualification_pin,'VALIDATION_PROOF_QUALIFICATION_PIN')
        q=json.loads((a.qualification/'report.json').read_bytes())
        need(q['complete'] and q['cold_readback_exact'] and q['all_updates_exact'] and q['midpoint_resume_exact']
             and q['binding']['device']=='cpu','VALIDATION_PROOF_CPU_QUALIFICATION')
        (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
        take_lease('29','batched-validation-qualification')
        need(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
             and torch.cuda.is_available(),'VALIDATION_PROOF_GPU_BUSY')
    src,bundle=source.source_bundle(a.root);policy=dict(bundle['policy'],device=device)
    items=[i for i in bundle['items'] if i['side']=='train']
    need(len(items)==5 and sum(len(i['codes']) for i in items)==66,'VALIDATION_PROOF_REAL_COHORT')
    target=PackedNativeRQ(bundle['dimensions'],bundle['config'],bundle['sizes'],'no_encoder_quantizer',bundle['statistics'])
    target.load_state_dict(bundle['target'],strict=True);target.to(device).eval().requires_grad_(False)
    base=Path(__file__).resolve().parents[1];code=dict(src['code'])
    for n in ('audit/placement_validation_proof.py','model/native_batched_validation.py','model/native_placement_batched_checks.py',
              'model/native_placement_efficient.py','model/native_device_transport.py','model/native_placement_checked.py',
              'audit/legal_grouped_placement_training.py','model/legal_placement_comparison.py',
              'claims/codata_source.py','claims/process_conditions.py','claims/episodic_memory.py',
              'senses/contract.py','text/constellation_corpus.py','lens/lenscommon.py','canon/common.py'):
        code[n]=file_sha(base/n)
    need(all(file_sha(base/n)==h for n,h in code.items()),'VALIDATION_PROOF_CODE')
    binding=dict(source=source.SOURCE[1],code=code,policy=policy,device=device,torch=str(torch.__version__),
                 target=fingerprint(target.state_dict()),documents=[i['document'] for i in items])
    if device=='cuda':
        need(q['binding']['code']==code and q['binding']['source']==binding['source']
             and q['binding']['target']==binding['target'],'VALIDATION_PROOF_QUALIFICATION_CODE')
    def sync():
        if device=='cuda':torch.cuda.synchronize()
    previous=None
    if a.previous:
        need(file_sha(a.previous/'report.json')==a.previous_pin,'VALIDATION_PROOF_PREVIOUS')
        previous=json.loads((a.previous/'report.json').read_bytes())
        need(previous['complete'] and previous['binding']==binding,'VALIDATION_PROOF_PREVIOUS_BINDING')
    timing={};results={};files={};resumed={};cases={};census={}
    for name in runtime.original.ARCHITECTURES:
        runs=[];expected=None
        for repeat in range(1 if previous else 3):
            model,opt=runtime.make(bundle,target,name,device);other,other_opt=runtime.make(bundle,target,name,device)
            need(fingerprint((model.state_dict(),opt.state_dict()))==fingerprint((other.state_dict(),other_opt.state_dict())),
                 'VALIDATION_PROOF_INITIAL_STATE')
            durations=dict(original=0.,candidate=0.);records=[]
            for step,item in enumerate(items):
                outputs={}
                order=('original','candidate') if repeat%2==0 else ('candidate','original')
                for label in order:
                    m,o=(model,opt) if label=='original' else (other,other_opt)
                    fn=original.update if label=='original' else candidate.update
                    sync();tick=time.perf_counter();outputs[label]=fn(m,o,item,step,policy,target=target);sync()
                    durations[label]+=time.perf_counter()-tick
                old,new=outputs['original'],outputs['candidate']
                need(old[0]==new[0],'VALIDATION_PROOF_UPDATE_CHANGED')
                if device=='cpu':need(new[0]==bundle['cpu_reference'][name]['updates'][step]['record'],
                                      'VALIDATION_PROOF_PRIOR_CPU_CHANGED')
                need(fingerprint(old[1:])==fingerprint(new[1:]),'VALIDATION_PROOF_GRADIENT_NOISE_CHANGED')
                need(fingerprint((model.state_dict(),opt.state_dict()))==fingerprint((other.state_dict(),other_opt.state_dict())),
                     'VALIDATION_PROOF_STATE_CHANGED')
                records.append(new[0])
                if repeat==0 and step in (1,4):files[name+'-'+str(step+1)]=snapshot(a.output,name+'-'+str(step+1),other,other_opt,step+1,binding)
            if expected is None:expected=records
            else:need(records==expected,'VALIDATION_PROOF_REPEAT_CHANGED')
            runs.append(durations)
        results[name]=expected;timing[name]=runs
        cases[name]=controls(other,other_opt,policy,5)
        census[name]=dict(parameters=sum(p.numel() for p in other.parameters()),parameter_tensors=len(list(other.parameters())),
                          optimizer_tensors=sum(len(s) for s in other_opt.state.values()),
                          optimizer_bytes=sum(v.numel()*v.element_size() for s in other_opt.state.values() for v in s.values()))
        # Cold replay consumes the prior process's physical midpoint, not this run's copy.
        ref=(previous['files'] if previous else files)[name+'-2'];path=Path(ref['path'])
        need(file_sha(path)==ref['sha256'],'VALIDATION_PROOF_CHECKPOINT_PIN')
        state=torch.load(path,map_location='cpu',weights_only=True)
        need(fingerprint(state)==ref['state'] and state['binding']==binding and state['step']==2,'VALIDATION_PROOF_CHECKPOINT')
        restored,optimizer=runtime.make(bundle,target,name,device);restored.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer']);validation.validate_optimizer(restored,optimizer,policy,2)
        for step in range(2,5):
            row,_,_=candidate.update(restored,optimizer,items[step],step,policy,target=target)
            need(row==expected[step],'VALIDATION_PROOF_RESUME_CHANGED')
        resumed[name]=fingerprint((restored.state_dict(),optimizer.state_dict()))
        need(resumed[name]==fingerprint((other.state_dict(),other_opt.state_dict())),'VALIDATION_PROOF_RESUME_STATE')
    need((device!='cpu' or not torch.cuda.is_initialized()) and fingerprint(target.state_dict())==binding['target'],
         'VALIDATION_PROOF_FROZEN_RUNTIME')
    need(all(file_sha(base/n)==h for n,h in code.items()),'VALIDATION_PROOF_CODE_CHANGED')
    stable=dict(binding=binding,records=results,controls=cases,census=census,resumed=resumed)
    result_pin=digest(stable)
    if previous:need(result_pin==previous['result_pin'],'VALIDATION_PROOF_COLD_CHANGED')
    result=dict(complete=True,**stable,result_pin=result_pin,timing=timing,files=files,previous=a.previous_pin,
                cold_readback_exact=previous is not None,all_updates_exact=True,midpoint_resume_exact=True,
                wall=time.perf_counter()-began,cpu=time.process_time()-cpu_start,
                rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,gpu_qualified=device=='cuda',
                qualification=a.qualification_pin,promoted=False,acceptance='author-checked')
    pin=write_committed(a.output/'report.json',result)
    print(json.dumps(dict(pin=pin,result_pin=result_pin,wall=result['wall'],timing=timing,census=census,cold=result['cold_readback_exact'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--previous',type=Path);p.add_argument('--previous-pin');p.add_argument('--cpu-proof',action='store_true')
    p.add_argument('--qualification',type=Path);p.add_argument('--qualification-pin');a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.is_dir():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,message=str(exc)))
        raise
    finally:release_lease()
