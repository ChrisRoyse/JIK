"""Physical corruption and native-input refusals for the prefix intervention."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import sys

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import decoder_prefix_training as trial
from audit.decoder_prefix_cohort import code
from claims.process_conditions import write_committed
from model import decoder_prefix_mask as masking
from model.native_population_training import fingerprint
from model.surface_device_runtime import batches
from text.constellation_corpus import file_sha,need


def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    s=trial.load(a.root);model,opt=trial.new(s);model.train();controls={}
    initial=fingerprint(dict(model=model.state_dict(),optimizer=opt.state_dict()))
    batch=next(batches(s['train'][:128],model.config,model.specs,'gathered',128));mask=masking.mask_for(batch,51150,.4)
    name=next(iter(batch.native));values=batch.native[name].values.clone();values[0,0,0]=float('nan')
    native=dict(batch.native);native[name]=replace(native[name],values=values)
    controls['mutated_nonfinite_reading']=trial.rejection(lambda:masking.forward(model,replace(batch,native=native),mask),'MNLM_MODEL_VALUES')
    bad=mask.clone();padding=[]
    for i,r in enumerate(batch.records):
        if r['prefix_length']<batch.prefix.shape[1]:padding.append((i,r['prefix_length']))
    need(padding,'PREFIX_CONTROL_PADDING_REQUIRED');i,j=padding[0];bad[i,j]=True
    controls['padding_mask']=trial.rejection(lambda:masking.forward(model,batch,bad),'PREFIX_MASK_NONBYTE_OR_PADDING')
    with torch.no_grad():
        controls['no_grad_train']=trial.rejection(lambda:masking.forward(model,batch,mask),'PREFIX_MASK_TRAIN_ONLY')
    # Corrupt a copied, real artifact. The authoritative source is never edited.
    testroot=a.output/'corruption';copy=testroot/trial.COHORT;copy.mkdir(parents=True)
    for name in ('report.json','observations.pt','initial.pt'):shutil.copyfile(a.root/trial.COHORT/name,copy/name)
    for name in ('observations.pt','initial.pt'):
        path=copy/name;original=file_sha(path)
        with path.open('ab') as f:f.write(b'physical-negative-control');f.flush();os.fsync(f.fileno())
        controls[name]=dict(original=original,corrupt=file_sha(path),refusal=trial.rejection(lambda:trial.load(testroot),'PREFIX_INPUT_PIN'))
        # Restore only this explicit disposable control copy for the next probe.
        shutil.copyfile(a.root/trial.COHORT/name,path)
        need(file_sha(path)==original,'PREFIX_CONTROL_RESTORE')
    path=copy/'report.json'
    with path.open('ab') as f:f.write(b'physical-negative-control');f.flush();os.fsync(f.fileno())
    controls['report_pin']=trial.rejection(lambda:trial.load(testroot),'PREFIX_COHORT_PIN')
    need(initial==fingerprint(dict(model=model.state_dict(),optimizer=opt.state_dict()))
         and not model.tokens._forward_hooks and all(p.grad is None for p in model.parameters()),'PREFIX_CONTROL_STATE_CHANGED')
    s['fresh']();need(not torch.cuda.is_initialized(),'PREFIX_CONTROL_CPU_ONLY')
    value=dict(complete=True,cohort=trial.PIN,controls=controls,unchanged_state=initial,code=code(),
               no_hook_or_gradient_leak=True,acceptance='author-checked')
    pin=write_committed(a.output/'report.json',value);print(json.dumps(dict(pin=pin,controls=len(controls))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
