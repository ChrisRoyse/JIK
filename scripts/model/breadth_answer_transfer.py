"""Named broad-predictor transfer with preserved Adam ages and complete answers."""
from copy import deepcopy
import time

import torch
from model import answer_full_history as history
from model.native_answer_sequence import versions
from model.native_conditioned_answer import check
from model.native_population_training import cpu,fingerprint
from model.native_placement_checked import validate_optimizer as source_optimizer
from model.saved_gradient_storage import SavedStorage
from text.constellation_corpus import need

PREFIX='forecaster.predictor.'
RETAIN=('encoder.','projectors.','region.')


def validate(model,opt,policy,binding,cursor):
    need(type(cursor) is int and cursor>=0 and type(opt) is torch.optim.AdamW
         and len(opt.param_groups)==1,'TRANSFER_OPTIMIZER_RECIPE')
    params=dict(model.named_parameters());group=opt.param_groups[0]
    need(list(params)==binding['names'] and set(params)==set(binding['ages']) and len(group['params'])==len(params)
         and all(a is b for a,b in zip(group['params'],params.values())), 'TRANSFER_OPTIMIZER_OWNER')
    recipe=dict(lr=policy['learning_rate'],weight_decay=policy['weight_decay'],betas=(.9,.999),eps=1e-8,
                amsgrad=False,maximize=False,foreach=False,capturable=False,differentiable=False,fused=False)
    need(all(group[k]==v for k,v in recipe.items()),'TRANSFER_OPTIMIZER_RECIPE')
    need(binding['pin']==fingerprint({k:v for k,v in binding.items() if k!='pin'}),'TRANSFER_BINDING')
    expected={p for n,p in params.items() if binding['ages'][n]+cursor>0}
    need(set(opt.state)==expected,'TRANSFER_MOMENT_CENSUS')
    for n,p in params.items():
        base=binding['ages'][n];need(type(base) is int and base>=0,'TRANSFER_AGE')
        if p not in expected:continue
        state=opt.state[p]
        need(set(state)=={'step','exp_avg','exp_avg_sq'} and state['step'].numel()==1
             and float(state['step'])==base+cursor,'TRANSFER_AGE')
        need(all(state[k].shape==p.shape and state[k].dtype==p.dtype and state[k].device==p.device
                 for k in ('exp_avg','exp_avg_sq'))
             and all(bool(v.isfinite().all()) for v in state.values()),'TRANSFER_MOMENT_VALUES')
        if cursor==0:need(fingerprint(state)==binding['moment_origins'][n],'TRANSFER_MOMENT_ORIGIN')


def install(model,source,saved,policy,*,source_pin):
    need(model.forecaster.architecture=='shared','TRANSFER_ARCHITECTURE')
    child=model.forecaster.predictor
    need(child._target==source._target and child.roster==source.roster and child.sizes==source.sizes
         and child.config==source.config,'TRANSFER_TARGET')
    need(fingerprint(saved)==source_pin and fingerprint(source.state_dict())==fingerprint(saved['model']),
         'TRANSFER_SOURCE_PIN')
    source_opt=torch.optim.AdamW(source.parameters(),lr=policy['learning_rate'],weight_decay=policy['weight_decay'],
                                foreach=False,fused=False)
    source_opt.load_state_dict(deepcopy(saved['optimizer']))
    source_optimizer(source,source_opt,policy,saved['step'])
    source_names=dict(source.named_parameters());destination=dict(model.named_parameters())
    mapping={n:PREFIX+n for n in source_names if n.startswith(RETAIN)}
    need(len(source_names)==59 and len(mapping)==47 and all(n.startswith('counts.') for n in source_names if n not in mapping),
         'TRANSFER_PARAMETER_MAPPING')
    original=cpu(model.state_dict());proposed=deepcopy(original)
    kept={n:v for n,v in saved['model'].items() if n.startswith(RETAIN)}
    need(set(kept)=={n[len(PREFIX):] for n in original if n.startswith(tuple(PREFIX+x for x in RETAIN))},
         'TRANSFER_BUFFER_MAPPING')
    for n,value in kept.items():
        name=PREFIX+n
        need(proposed[name].shape==value.shape and proposed[name].dtype==value.dtype,'TRANSFER_SHAPE')
        proposed[name]=value.clone()
    ages={n:(saved['step'] if n in mapping.values() else 0) for n in destination}
    binding=dict(schema='broad-answer-transfer-v1',source=source_pin,source_step=saved['step'],mapping=mapping,
        retired=[n for n in source_names if n not in mapping],ages=ages,names=list(destination),
        moment_origins={dest:fingerprint(source_opt.state[source_names[n]]) for n,dest in mapping.items() if saved['step']},
        inherited=fingerprint(kept),new_heads=fingerprint({n:v for n,v in original.items() if n not in {PREFIX+k for k in kept}}))
    binding['pin']=fingerprint(binding)
    opt=torch.optim.AdamW(model.parameters(),lr=policy['learning_rate'],weight_decay=policy['weight_decay'],foreach=False,fused=False)
    for n,dest in mapping.items():
        if saved['step']:
            opt.state[destination[dest]]=deepcopy(source_opt.state[source_names[n]])
    validate(model,opt,policy,binding,0)
    model.load_state_dict(proposed,strict=True)
    need(fingerprint(model.state_dict())==fingerprint(proposed),'TRANSFER_INSTALL_EXACT')
    return opt,binding


def checkpoint(model,opt,binding,cursor,policy):
    validate(model,opt,policy,binding,cursor)
    return cpu(dict(schema='broad-answer-checkpoint-v1',model=model.state_dict(),optimizer=opt.state_dict(),
                    binding=binding,cursor=cursor))


def restore(model,opt,value,binding,policy,*,pin):
    need(fingerprint(value)==pin and set(value)=={'schema','model','optimizer','binding','cursor'}
         and value['schema']=='broad-answer-checkpoint-v1' and value['binding']==binding,'TRANSFER_RESTORE_BINDING')
    # Validate an owned candidate before touching the live model or its moments.
    candidate=deepcopy(model);candidate.load_state_dict(value['model'],strict=True)
    trial=torch.optim.AdamW(candidate.parameters(),lr=policy['learning_rate'],weight_decay=policy['weight_decay'],foreach=False,fused=False)
    trial.load_state_dict(deepcopy(value['optimizer']));validate(candidate,trial,policy,binding,value['cursor'])
    model.load_state_dict(value['model'],strict=True);opt.load_state_dict(deepcopy(value['optimizer']))
    need(fingerprint(checkpoint(model,opt,binding,value['cursor'],policy))==pin,'TRANSFER_RESTORE_EXACT')
    return value['cursor']


def backward(model,item,cursor,policy,request,target):
    first={};handles=[];phases={};tick=time.perf_counter()
    def capture(name):
        def hook(module,args,output):
            if name not in first:first[name]=output
        return hook
    for n,p in model.forecaster.predictor.projectors.items():handles.append(p.register_forward_hook(capture(n)))
    zero_initial=cursor==0 and all(not bool(p.detach().any()) for p in model.ending.parameters())
    try:
        total,end,records=history.forward(model,item,cursor,policy,request,target)
        phases['forward']=time.perf_counter()-tick;tick=time.perf_counter()
        need(set(first)==set(model.roster) and all(t.requires_grad for t in first.values()),'TRANSFER_EARLY_GRAPH')
        ends=torch.autograd.grad(end,tuple(first.values()),retain_graph=True)
        early=torch.autograd.grad(total,tuple(first.values()),retain_graph=True)
        credit={n:dict(value=cpu(t),requires_grad=True,end_gradient=cpu(e),gradient=cpu(g))
                for (n,t),e,g in zip(first.items(),ends,early)}
        need(all(bool(r['gradient'].abs().any()) for r in credit.values()),'TRANSFER_DISCONNECTED_CONTEXT')
        if zero_initial:
            # A zero linear END head has an analytically zero derivative to its input.
            # Verify that exact identity, then require actual END-head learning below.
            need(all(not bool(r['end_gradient'].any()) for r in credit.values()),'TRANSFER_INITIAL_END_DERIVATIVE')
        else:need(all(bool(r['end_gradient'].abs().any()) for r in credit.values()),'TRANSFER_DISCONNECTED_END')
        phases['credit_probe']=time.perf_counter()-tick;tick=time.perf_counter()
        total.backward();gradients=cpu({n:p.grad for n,p in model.named_parameters()})
        phases['backward']=time.perf_counter()-tick
        return dict(records=records,zero_initial_end=zero_initial),gradients,credit,phases
    finally:
        for handle in handles:handle.remove()
        first.clear()


def update(model,opt,item,cursor,policy,binding,*,request,target):
    check(item);validate(model,opt,policy,binding,cursor)
    before=fingerprint(model.state_dict());frozen=versions(target);opt.zero_grad(set_to_none=True)
    storage=SavedStorage(policy['working_bytes'],next(model.parameters()).device)
    try:
        with storage.hooks():
            row,gradients,credit,phases=backward(model,item,cursor,policy,request,target)
        need(storage.current==0,'TRANSFER_RETAINED_GRAPH')
        need(all(g is not None and bool(g.isfinite().all()) for g in gradients.values()),'TRANSFER_GRADIENTS')
        prefixes=['forecaster.predictor.projectors.'+n+'.' for n in model.roster]
        prefixes+=['request_projection.lanes.'+n+'.' for n in model.roster]
        prefixes+=['ending.'+p+'.'+n+'.' for p in model.proposals for n in model.roster]
        branches={}
        for prefix in prefixes:
            group={n:g for n,g in gradients.items() if n.startswith(prefix)}
            need(group and any(bool(g.abs().any()) for g in group.values()),'TRANSFER_BRANCH:'+prefix)
            branches[prefix]=fingerprint(group)
        need(versions(target)==frozen,'TRANSFER_FROZEN_CHANGED');check(item)
        opt.step();validate(model,opt,policy,binding,cursor+1)
        need(all(bool(x.isfinite().all()) for x in model.state_dict().values()),'TRANSFER_PARAMETERS')
    except BaseException:
        opt.zero_grad(set_to_none=True);raise
    return dict(document=item['document'],request=request.pin,native=item['native_pin'],labels=item['label_pin'],
        binding=item['binding'],before=before,after=fingerprint(model.state_dict()),optimizer=fingerprint(opt.state_dict()),
        gradients=fingerprint(gradients),branch_gradients=branches,records=row['records'],decisions=len(row['records']),
        storage=storage.report(),phases=phases,cursor=cursor+1,zero_initial_end=row['zero_initial_end']),gradients,credit
