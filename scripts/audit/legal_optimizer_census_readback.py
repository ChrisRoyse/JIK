"""Qualify complete optimizer continuation against real sealed placement updates."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch

from audit import legal_placement_training_readback as prior
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model import native_placement_checked as checked
from model.native_population_training import cpu,fingerprint
from text.constellation_corpus import file_sha,need

QUAL=('legal-placement-training-qualification-v1-20261023','7de6be5f715b95ac5fc50b8043960b8feab9d80776489bbea50b51b0567a238c')
COLD=('legal-placement-training-cold-v1-20261023','96f32f4b320b0d988eb57dcf56a9b583add3e245a69f313f36166084810b3e4e')


def code():
    root=Path(__file__).resolve().parents[1]
    return dict(prior.code(),**{n:file_sha(root/n) for n in
        ('model/native_placement_checked.py','audit/legal_optimizer_census_readback.py')})


def load(root,report,name):
    f=root/name;need(file_sha(f)==report['files'][name],'OPTIMIZER_CENSUS_SOURCE_BYTES')
    return torch.load(f,map_location='cpu',weights_only=True)


def numeric(value):return {k:v for k,v in value.items() if k!='binding'}


def controls(model,optimizer,item,target):
    policy=prior.POLICY;names=dict(model.named_parameters());parameters=list(names.values());original=optimizer.state
    frozen=fingerprint((model.state_dict(),optimizer.state_dict()));result={};calls=[]
    real_stream=model.predictor.stream
    def observed_stream():
        calls.append('forward_stream');return real_stream()
    model.predictor.stream=observed_stream
    try:
        # Demonstrate the actual old acceptance, without permitting an old update.
        optimizer.state=dict(original);optimizer.state.pop(parameters[0])
        checked.validate_existing(model,optimizer,policy,2)
        result['legacy_accepts_missing_state']=True
        missing={}
        for name,p in names.items():
            optimizer.state=dict(original);optimizer.state.pop(p)
            missing[name]=rejected(lambda:checked.validate_optimizer(model,optimizer,policy,2))
            need('INCOMPLETE_STATE' in missing[name],'OPTIMIZER_CENSUS_MISSING_REASON')
        result['missing_parameter_states']=missing
        actual={}
        for index in sorted({0,len(parameters)//2,len(parameters)-1}):
            optimizer.state=dict(original);optimizer.state.pop(parameters[index])
            before=fingerprint((model.state_dict(),optimizer.state_dict()))
            actual[list(names)[index]]=rejected(lambda:checked.update(model,optimizer,item,2,policy,target=target))
            need(before==fingerprint((model.state_dict(),optimizer.state_dict())),'OPTIMIZER_CENSUS_REJECT_MUTATION')
        result['real_update_missing_state']=actual
        optimizer.state=dict(original);foreign=torch.nn.Parameter(parameters[0].detach().clone())
        optimizer.state[foreign]=original[parameters[0]]
        result['foreign_parameter']=rejected(lambda:checked.validate_optimizer(model,optimizer,policy,2))
        optimizer.state=dict(original);optimizer.state[parameters[0]]=dict(original[parameters[0]])
        del optimizer.state[parameters[0]]['exp_avg']
        result['missing_moment_buffer']=rejected(lambda:checked.validate_optimizer(model,optimizer,policy,2))
        optimizer.state=dict(original);optimizer.state[parameters[0]]=dict(original[parameters[0]])
        optimizer.state[parameters[0]]['exp_avg']=original[parameters[0]]['exp_avg'].double()
        result['wrong_moment_dtype']=rejected(lambda:checked.validate_optimizer(model,optimizer,policy,2))
        optimizer.state=dict(original);optimizer.state[parameters[0]]=dict(original[parameters[0]])
        bad=original[parameters[0]]['exp_avg'].clone();bad.reshape(-1)[0]=float('nan');optimizer.state[parameters[0]]['exp_avg']=bad
        result['nonfinite_moment']=rejected(lambda:checked.validate_optimizer(model,optimizer,policy,2))
        optimizer.state={}
        result['weights_only_restart']=rejected(lambda:checked.validate_optimizer(model,optimizer,policy,2))
        optimizer.state=original
        result['stale_cursor']=rejected(lambda:checked.validate_optimizer(model,optimizer,policy,3))
        result['populated_step_zero']=rejected(lambda:checked.validate_optimizer(model,optimizer,policy,0))
        group=optimizer.param_groups[0];lr=group['lr'];group['lr']=lr*2
        try:result['wrong_recipe']=rejected(lambda:checked.validate_optimizer(model,optimizer,policy,2))
        finally:group['lr']=lr
        order=group['params'];group['params']=list(reversed(order))
        try:result['wrong_parameter_order']=rejected(lambda:checked.validate_optimizer(model,optimizer,policy,2))
        finally:group['params']=order
    finally:optimizer.state=original;model.predictor.stream=real_stream
    need(not calls and all(p.grad is None for p in model.parameters()),'OPTIMIZER_CENSUS_FORWARD_ON_REJECTION')
    need(frozen==fingerprint((model.state_dict(),optimizer.state_dict())),'OPTIMIZER_CENSUS_CONTROL_MUTATION')
    return dict(**result,forward_stream_calls=len(calls),unchanged_state=frozen)


def timings(model,optimizer):
    # Interleave identical validators on a fixed state; report overhead, no speed claim.
    rows=[]
    for i in range(10):
        for name,fn in [('existing',checked.validate_existing),('checked',checked.validate_optimizer)]:
            tick=time.perf_counter();fn(model,optimizer,prior.POLICY,2)
            rows.append(dict(repetition=i,kind=name,seconds=time.perf_counter()-tick))
    return rows


def run(a):
    started=tick=time.perf_counter();process=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    pins=code()
    for name,pin in pins.items():
        dest=a.output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes());need(file_sha(dest)==pin,'OPTIMIZER_CENSUS_CODE_COPY')
    qroot,croot=a.root/QUAL[0],a.root/COLD[0]
    q=prior.prior.branch.proof.pinned(qroot,QUAL[1]);c=prior.prior.branch.proof.pinned(croot,COLD[1])
    need(q['complete'] and c['complete'] and c['previous_pin']==QUAL[1] and q['binding']==c['binding']
         and q['results_pin']==c['results_pin'] and q['binding']['code']==prior.code(),'OPTIMIZER_CENSUS_PREREQUISITE')
    target,ref,labels,items,source=prior.prior.source(a.root);frozen=fingerprint(target.state_dict())
    need(source==q['binding']['source'] and ref==q['binding']['target'],'OPTIMIZER_CENSUS_SOURCE')
    fitting=[i for i in items if i['side']=='train'];item=fitting[2]
    item['label_pin']=fingerprint((item['codes'],item['counts']))
    binding=dict(code=pins,source=source,target=ref,qualification=QUAL[1],cold=COLD[1],policy=prior.POLICY,
                 document=item['document'],targets=len(item['codes']))
    previous=prior.prior.branch.proof.pinned(a.previous,a.previous_pin) if a.previous else None
    if previous:
        need(previous['complete'] and previous['binding']==binding,'OPTIMIZER_CENSUS_COLD_BINDING')
        for name,pin in previous['files'].items():need(file_sha(a.previous/name)==pin,'OPTIMIZER_CENSUS_COLD_BYTES')
    phases=dict(authentication=time.perf_counter()-tick);results={};measurements={}
    for arch in checked.ARCHITECTURES:
        tick=time.perf_counter();out=a.output/arch;out.mkdir()
        cp=load(qroot,q,arch+'/checkpoint-2.pt');expected=load(qroot,q,arch+'/checkpoint-3.pt')
        need(fingerprint(cp)==fingerprint(load(croot,c,arch+'/checkpoint-2.pt'))
             and fingerprint(expected)==fingerprint(load(croot,c,arch+'/checkpoint-3.pt')),'OPTIMIZER_CENSUS_SOURCE_COLD')
        model=prior.make(target,ref,arch);optimizer=checked.optimizer_for(model,prior.POLICY)
        checked.validate_optimizer(model,optimizer,prior.POLICY,0)
        model.load_state_dict(cp['model'],strict=True);optimizer.load_state_dict(cp['optimizer'])
        need(cp['step']==2 and cp['noise_cursor']==prior.POLICY['noise_seed']+2,'OPTIMIZER_CENSUS_CURSOR')
        checked.validate_optimizer(model,optimizer,prior.POLICY,2)
        negatives=controls(model,optimizer,item,target);measurements[arch]=timings(model,optimizer)
        record,grads,noise=checked.update(model,optimizer,item,2,prior.POLICY,target=target)
        value=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),gradients=grads,noise=noise,
                   step=3,noise_cursor=prior.POLICY['noise_seed']+3,record=record)
        need(fingerprint(value)==fingerprint(numeric(expected)),'OPTIMIZER_CENSUS_CHANGED_UPDATE')
        pin=prior.prior.branch.persist(out/'continued.pt',dict(binding=binding,numeric=value,reference_checkpoint=q['files'][arch+'/checkpoint-3.pt']))
        need(fingerprint(target.state_dict())==frozen,'OPTIMIZER_CENSUS_FROZEN_CHANGED')
        results[arch]=dict(complete=True,negative=negatives,parameters=sum(p.numel() for p in model.parameters()),
            parameter_tensors=len(list(model.parameters())),state_tensors=len(optimizer.state),numeric_pin=fingerprint(value),
            saved_pin=pin,reference_checkpoint=q['files'][arch+'/checkpoint-3.pt'],targets=len(item['codes']))
        phases[arch]=time.perf_counter()-tick;print(json.dumps(dict(architecture=arch,parameter_tensors=results[arch]['parameter_tensors'],seconds=phases[arch])),flush=True)
    tick=time.perf_counter();need(not torch.cuda.is_initialized(),'OPTIMIZER_CENSUS_UNEXPECTED_CUDA')
    results_pin=write_committed(a.output/'results.json',results);write_committed(a.output/'timings.json',measurements)
    if previous:need(previous['results_pin']==results_pin,'OPTIMIZER_CENSUS_COLD_EXACT')
    files={str(f.relative_to(a.output)):file_sha(f) for f in a.output.rglob('*') if f.is_file()}
    phases['readback']=time.perf_counter()-tick;seconds=time.perf_counter()-started
    need(sum(phases.values())/seconds>=.9,'OPTIMIZER_CENSUS_PHASE_ACCOUNTING')
    r=dict(complete=True,binding=binding,files=files,results_pin=results_pin,previous_pin=a.previous_pin,
           phase_seconds=phases,seconds=seconds,cpu_seconds=time.process_time()-process,
           peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
           new_training_population=False,gpu_qualified=False,scientific_admission=False,acceptance='author-checked',number_status='measured')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',r),seconds=seconds)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--previous',type=Path);parser.add_argument('--previous-pin')
    a=parser.parse_args();need(bool(a.previous)==bool(a.previous_pin),'OPTIMIZER_CENSUS_PREVIOUS_ARGUMENTS');run(a)
