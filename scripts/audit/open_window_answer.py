"""Matched native count-ending change in the actual complete-answer learner."""
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
from audit import requested_answer as old
from audit import legal_answer_sequence as sequence
from claims.process_conditions import write_committed
from model.native_open_window_answer import OpenPositionRQ,answer_model
from model.native_placement_checked import optimizer_for
from model.native_population_training import fingerprint,cpu
from model.native_requested_answer import update
from model.native_request_input import NativeRequest,legal
from model.native_window_end import physical_limits
from text.constellation_corpus import canonical,digest,file_sha,need

ROOT=sequence.ROOT
BASE=('requested-answer-cpu-v2-20261024','fc6c18e0468fd63a03c4bddcdc1aa49799cac1246ef05f9fbb7811664f4123b3')
CENSUS=('canonical-answer-consumer-plan-v2-20261024','9311040852b8301bc172efc8e16b0ac1133010393d8212724d3ddc4d2b3eaf50')
NEW=('model/native_window_end.py','model/native_open_window_answer.py','audit/open_window_answer.py',
     'audit/open_window_answer_readback.py')


def code(source):
    base=Path(__file__).resolve().parents[1]
    return dict(old.code(source),**{n:file_sha(base/n) for n in NEW})


def extension(bundle):
    target=OpenPositionRQ(bundle['dimensions'],bundle['config'],bundle['sizes'],'no_encoder_quantizer',bundle['statistics'])
    target.load_state_dict(bundle['target'],strict=True);target.eval().requires_grad_(False)
    base=Path(__file__).resolve().parents[1]
    pin=digest(canonical(dict(parent=bundle['target_ref'],runtime=file_sha(base/'model/native_open_window_answer.py'),
                             physical_policy=file_sha(base/'model/native_window_end.py'),
                             rule='preserve_TRAIN_position_prefix_then_TRAIN_mean')))
    return target,pin


def make(bundle,target,pin,architecture):
    model=answer_model(target,bundle,architecture,target_pin=pin)
    return model,optimizer_for(model,dict(bundle['policy'],device='cpu'))


def source(a):
    s,b,d=sequence.prepare(a.root)
    parent=sequence.runtime.pinned(a.root/BASE[0]/'report.json',BASE[1])
    need(parent['complete'] and parent['code']==old.code(s),'OPEN_ANSWER_PARENT')
    for name,pin in parent['files'].items():need(file_sha(a.root/BASE[0]/name)==pin,'OPEN_ANSWER_PARENT_BYTES')
    report=sequence.runtime.pinned(a.root/CENSUS[0]/'report.json',CENSUS[1])
    need(file_sha(a.root/CENSUS[0]/'census.json')==report['files']['census.json']['sha256'],'OPEN_ANSWER_CENSUS')
    rows=json.loads((a.root/CENSUS[0]/'census.json').read_bytes())['rows']
    unsupported=[r for r in rows if any(r['unsupported'].values())]
    need(len(unsupported)==1 and unsupported[0]['side']=='query','OPEN_ANSWER_CAPACITY_SCOPE')
    return s,b,d,unsupported[0]


def count_heads(model):
    p=model.forecaster.predictor
    if model.forecaster.architecture=='whole_proposals':return {n:r.counts for n,r in p.readouts.items()}
    return dict(assembled=p.counts)


def context(model,frames,request):
    stream=model.stream(request);state=hidden=None
    with torch.no_grad():
        for i,f in enumerate(frames):hidden,state=stream.step((f,),context='count-proof',offset=i,state=state,detach_history=True)
    return hidden


def observe(model,target,prefix,proposal,seed,decisions,request):
    stream=model.rollout(prefix,request=request,target=target,proposal=proposal,context='count-answer-proof',seed=seed)
    events=[];refusal=None
    try:
        for _ in range(decisions):
            row=next(stream);events.append(cpu(row))
            if row['kind']=='end':break
    except ValueError as exc:
        allowed=('WINDOW_END_PHYSICAL_MEMORY_EXHAUSTED_NO_TRUNCATION','PLACEMENT_OUTPUT_MEMORY',
                 'RQ_BATCH_PHYSICAL_MEMORY','PREFIX_STREAM_CURRENT_MEMORY','PREFIX_STREAM_CACHE_MEMORY')
        need(any(c in str(exc) for c in allowed),'OPEN_ANSWER_UNEXPECTED_REFUSAL:'+str(exc));refusal=str(exc)
    finally:stream.close()
    return dict(events=events,refusal=refusal)


def measured(result,future,roster,copy):
    events=result['events']
    if events:return dict(sequence.metrics(events,future,roster,copy),refusal=result['refusal'])
    return dict(expected_atoms=len(future),generated_atoms=0,ended=False,ended_at_reference_boundary=False,
        censored=result['refusal'] is None,matched_positions=0,missing=len(future),extra=0,
        per_slot={n:[-1.]*len(future) for n in roster},
        copy_current={n:[sequence.native_similarity(copy[n],f[n]) for f in future] for n in roster},
        missing_similarity=-1.,semantic_validity_established=False,refusal=result['refusal'])


def capacity(model,target,bundle,train,case):
    item,request=old.item(train);hidden=context(model,item['context'],request);codes=item['codes'][:1]
    counts={n:torch.tensor([case['counts'][n][0]],dtype=torch.int64) for n in model.roster}
    out={};frozen=fingerprint(model.state_dict())
    for proposal,(head,h) in model.forecaster.contexts(hidden).items():
        law={n:cpu(head.counts[n].loss(h[:,i],counts[n])) for i,n in enumerate(model.roster)}
        noise=torch.randn(1,1,8,generator=torch.Generator().manual_seed(93))
        with torch.no_grad():sample=model.forecaster.placement[proposal](h,codes,counts,noise,target=target)
        need(all(sample[0][n].shape==(1,int(counts[n][0]),bundle['dimensions'][n]) for n in model.roster),
             'OPEN_ANSWER_TAIL_SHAPE')
        out[proposal]=dict(hidden=cpu(h),codes=codes,counts=counts,noise=noise,losses=law,sample=cpu(sample))
    need(fingerprint(model.state_dict())==frozen,'OPEN_ANSWER_CAPACITY_CHANGED_WEIGHTS')
    return dict(source_task=case['task'],scope='attested_count_shape_intervention_not_task_prediction',proposals=out)


def controls(model,optimizer,target,bundle,train,held,step):
    item,request=old.item(train);bad,_=old.item(held);policy=dict(bundle['policy'],device='cpu')
    reject=old.original.reject;initial=fingerprint((model.state_dict(),optimizer.state_dict()))
    out=[reject('CONDITIONED_NONTRAIN',lambda:update(model,optimizer,bad,step,policy,request=request,target=target)),
         reject('PLACEMENT_OPTIMIZER_INCOMPLETE_STATE_RESTORE_FULL_CHECKPOINT',lambda:update(model,optimizer_for(model,policy),item,
                 step,policy,request=request,target=target))]
    name=model.roster[0];head=deepcopy(next(iter(count_heads(model).values()))[name]);h=torch.zeros(1,head.width)
    with torch.no_grad():head.out.weight.zero_();head.out.bias.fill_(-100.)
    out.append(reject('WINDOW_END_PHYSICAL_MEMORY_EXHAUSTED_NO_TRUNCATION',lambda:head.forecast(h)))
    out.append(reject('WINDOW_END_TARGET_PHYSICAL_SCOPE',lambda:head.loss(h,torch.tensor([head.limit+1]))))
    out.append(reject('WINDOW_END_HIDDEN',lambda:head.forecast(h+float('nan'))))
    out.append(reject('OPEN_POSITION_PHYSICAL_MEMORY',lambda:target.baseline(0,physical_limits(target,16<<20)[name]+1)))
    wrong=deepcopy(target);placement=next(iter(model.forecaster.placement.values()))
    out.append(reject('PLACEMENT_TARGET_CHANGED',lambda:placement.check_target(wrong)))
    def future():model.rollout(item['context'],request=request,target=target,proposal=model.proposals[0],context='negative',seed=1,target_length=2)
    try:future()
    except TypeError as exc:out.append(dict(code='FUTURE_LENGTH_ARGUMENT_REJECTED',observed=str(exc)))
    else:raise ValueError('OPEN_ANSWER_FUTURE_LENGTH_ACCEPTED')
    need(fingerprint((model.state_dict(),optimizer.state_dict()))==initial,'OPEN_ANSWER_CONTROL_MUTATION');return out


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();phases={}
    try:
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        s,b,data,case=source(a);pins=code(s);target,pin=extension(b);original=data['labels'].model
        need(fingerprint(target.state_dict())==fingerprint(original.state_dict()),'OPEN_ANSWER_FROZEN_WEIGHTS')
        (a.output/'nvidia-before.txt').write_text(__import__('subprocess').check_output(['nvidia-smi'],text=True))
        phases['authenticate']=time.perf_counter()-tick;tick=time.perf_counter()
        identities=[]
        with torch.no_grad():
            for item in data['items']:
                for f in item['frames']:
                    left=original.latent((f,));right=target.latent((f,));need(torch.equal(left,right),'OPEN_ANSWER_LATENT_CHANGED')
                    c=original.quantize(left)[0];need(torch.equal(c,target.quantize(right)[0]),'OPEN_ANSWER_CODES_CHANGED')
                    counts=[{n:len(f[n]) for n in original.roster}]
                    # Only known position support has an old decoder definition.
                    if all(counts[0][n]<=len(getattr(original,'position_'+str(i))) for i,n in enumerate(original.roster)):
                        x=original.reconstruct(original.values(c),counts);y=target.reconstruct(target.values(c),counts)
                        need(fingerprint(x)==fingerprint(y),'OPEN_ANSWER_SUPPORTED_DECODE_CHANGED')
                        identities.append(fingerprint(y))
        write_committed(a.output/'frozen.json',dict(state=fingerprint(target.state_dict()),codes_checked=215,
            supported_decodes=identities,extension_pin=pin,physical_limits=physical_limits(target,b['policy']['output_bytes'])))
        phases['frozen_code_and_supported_decode']=time.perf_counter()-tick
        fitting=[d for d in data['items'] if d['side']=='train'];held={d['document']:d for d in data['items'] if d['side']!='train'}
        results={}
        for architecture in sequence.runtime.original.ARCHITECTURES:
            tick=time.perf_counter();dest=a.output/architecture;dest.mkdir();model,opt=make(b,target,pin,architecture)
            prior=torch.load(a.root/BASE[0]/architecture/'initial.pt',map_location='cpu',weights_only=True)
            select=lambda state:{n:v for n,v in state.items() if '.counts.' not in n}
            need(fingerprint(select(model.state_dict()))==fingerprint(select(prior['model'])),'OPEN_ANSWER_UNMATCHED_INITIALIZATION')
            sequence.runtime.persist(dest/'initial.pt',dict(model=model.state_dict(),optimizer=opt.state_dict()))
            for step,d in enumerate(fitting):
                item,request=old.item(d);row,grad=update(model,opt,item,step,dict(b['policy'],device='cpu'),request=request,target=target)
                need(all(any('.counts.' in n and lane in n and bool(g.abs().any()) for n,g in grad.items())
                         for lane in model.roster),'OPEN_ANSWER_DISCONNECTED_COUNT')
                sequence.runtime.persist(dest/f'update-{step+1}.pt',dict(record=row,gradients=grad,model=model.state_dict(),optimizer=opt.state_dict()))
            negative=controls(model,opt,target,b,fitting[0],next(iter(held.values())),len(fitting))
            sequence.runtime.persist(dest/'capacity.pt',capacity(model,target,b,fitting[0],case))
            phases[architecture+'_train_and_capacity']=time.perf_counter()-tick;tick=time.perf_counter()
            baseline=torch.load(a.root/BASE[0]/architecture/'evaluation.pt',map_location='cpu',weights_only=True);rows=[]
            for oldrow in baseline:
                d=held[oldrow['document']];desc=legal(2)
                if oldrow['mode']=='wrong_operation':desc['operation']='respond_to_source_question'
                result=observe(model,target,tuple(d['frames'][:2]),oldrow['proposal'],20261024+d['document'],len(d['frames'])-1,NativeRequest(desc))
                rows.append(dict(document=d['document'],mode=oldrow['mode'],proposal=oldrow['proposal'],**result,
                    metrics=measured(result,d['frames'][2:],model.roster,d['frames'][1]),parent_metrics=oldrow['metrics']))
            sequence.runtime.persist(dest/'evaluation.pt',rows);write_committed(dest/'controls.json',negative)
            results[architecture]=dict(updates=len(fitting),rollouts=len(rows),ended=sum(r['metrics']['ended'] for r in rows),
                ended_at_reference_boundary=sum(r['metrics']['ended_at_reference_boundary'] for r in rows),
                refused=sum(r['refusal'] is not None for r in rows),parameters=sum(p.numel() for p in model.parameters()))
            phases[architecture+'_evaluation']=time.perf_counter()-tick;print(json.dumps(dict(architecture=architecture,**results[architecture])),flush=True)
        tick=time.perf_counter();data['close']();need(code(s)==pins and not torch.cuda.is_initialized(),'OPEN_ANSWER_FINAL_STATE')
        files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
        phases['close_and_hash']=time.perf_counter()-tick;elapsed=time.perf_counter()-began
        need(.9<=sum(phases.values())/elapsed<=1.001,'OPEN_ANSWER_PHASE_COVERAGE')
        r=dict(complete=True,schema='native-open-window-answer-v1',code=pins,files=files,extension_pin=pin,
            parent=dict(path=str(a.root/BASE[0]),pin=BASE[1]),census=dict(path=str(a.root/CENSUS[0]),pin=CENSUS[1]),
            results=results,phases=phases,seconds=elapsed,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            runtime='explicit_CPU_engineering_qualification',capability_demonstrated=False,scientific_admission=False,acceptance='author-checked')
        h=write_committed(a.output/'report.json',r);print(json.dumps(dict(pin=h,seconds=elapsed)),flush=True)
    except BaseException as e:write_committed(a.output/'failure.json',dict(error=repr(e),phases=phases));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
