"""Source request census and native complete-answer conditioning qualification."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import conditioned_answer as original
from audit import legal_answer_sequence as sequence
from claims.process_conditions import write_committed
from model.answer_training_schedule import AnswerSchedule
from model.native_conditioned_answer import document_item
from model.native_placement_checked import optimizer_for
from model.native_population_training import fingerprint
from model.native_request_input import NativeRequest,describe,legal,WIDTH
from model.native_requested_answer import RequestedAnswer,update
from text.constellation_corpus import file_sha,need

ROOT=sequence.ROOT
NEW=('model/native_request_input.py','model/native_request_stream.py','model/native_requested_answer.py',
     'audit/requested_answer.py','audit/requested_answer_readback.py')


def code(source):
    base=Path(__file__).resolve().parents[1];result=original.code(source)
    for n in (*NEW,'model/capability_kernel_learning.py','model/capability_structure_input.py'):
        result[n]=file_sha(base/n)
    return result


def make(bundle,target,architecture):
    old,_=sequence.make(bundle,target,architecture);model=RequestedAnswer(old.forecaster)
    model.ending.load_state_dict(old.ending.state_dict())
    return model,optimizer_for(model,dict(bundle['policy'],device='cpu'))


def item(original):
    request=NativeRequest(legal(2));value=document_item(original,2)
    value['provenance']['request']=request.pin
    value['binding']=fingerprint((value['document'],value['side'],value['provenance']))
    return value,request


def observe(model,target,prefix,proposal,seed,decisions,request):
    gen=model.rollout(prefix,request=request,target=target,proposal=proposal,context='request-proof',seed=seed);events=[]
    try:
        for _ in range(decisions):
            e=next(gen);events.append(e)
            if e['kind']=='end':break
    finally:gen.close();need(gen.gi_frame is None,'REQUEST_PROOF_GENERATOR')
    return events


def request_controls(descriptors):
    good=next(iter(descriptors.values()));out=[]
    bad=deepcopy(good);bad['answer_length']=2
    out.append(original.reject('REQUEST_INPUT_SCHEMA',lambda:NativeRequest(bad)))
    bad=deepcopy(good);bad['fields'][0]['role']='invented'
    out.append(original.reject('STRUCTURE_ROLE',lambda:NativeRequest(bad)))
    bad=legal(2);bad['fields'][0]['clock']={'id':'invented'}
    out.append(original.reject('REQUEST_INPUT_UNKNOWN_CLOCK',lambda:NativeRequest(bad)))
    bad=legal(2);bad['latest_input_end']='1'
    out.append(original.reject('REQUEST_INPUT_UNATTESTED_ORIGIN',lambda:NativeRequest(bad)))
    request=NativeRequest(good);request.features[0,0]+=1
    out.append(original.reject('REQUEST_INPUT_MUTATED',lambda:request.materialize('cpu')))
    # Clock IDs are provenance; consistent renaming cannot alter any feature.
    clocked=next(d for d in descriptors.values() if d['latest_input_end'] is not None)
    renamed=deepcopy(clocked)
    for f in renamed['fields']:f['clock']['id']='different-provenance-only'
    need(torch.equal(NativeRequest(clocked).features,NativeRequest(renamed).features),'REQUEST_PROOF_CLOCK_ID_FEATURE')
    out.append(dict(code='CLOCK_IDS_NOT_FEATURES',observed='all feature bytes unchanged'))
    return out


def boundary_controls(model,optimizer,target,bundle,train,held):
    value,request=item(train);initial=fingerprint((model.state_dict(),optimizer.state_dict()));out=[]
    wrong=NativeRequest(dict(legal(2),operation='respond_to_source_question'))
    out.append(original.reject('REQUESTED_ANSWER_SOURCE_BINDING',lambda:update(model,optimizer,value,0,
        dict(bundle['policy'],device='cpu'),request=wrong,target=target)))
    value,req=item(held)
    out.append(original.reject('CONDITIONED_NONTRAIN',lambda:update(model,optimizer,value,0,
        dict(bundle['policy'],device='cpu'),request=req,target=target)))
    stream=model.stream(request)
    with torch.no_grad():_,cache=stream.step((train['frames'][0],),context='probe',offset=0,detach_history=True)
    leaf=stream if hasattr(stream,'features') else next(iter(stream.streams.values()))
    leaf.features[0,0]+=1
    out.append(original.reject('REQUEST_STREAM_FEATURES_CHANGED',lambda:stream.step((train['frames'][1],),
        context='probe',offset=1,state=cache,detach_history=True)))
    stream=model.stream(request);other=model.stream(wrong)
    with torch.no_grad():_,cache=stream.step((train['frames'][0],),context='probe',offset=0,detach_history=True)
    reason='PREFIX_STREAM_WRONG_CONTEXT' if hasattr(stream,'features') else 'BRANCH_CACHE_OWNER'
    out.append(original.reject(reason,lambda:other.step((train['frames'][1],),context='probe',offset=1,state=cache,detach_history=True)))
    left=stream if hasattr(stream,'features') else next(iter(stream.streams.values()))
    right=other if hasattr(other,'features') else next(iter(other.streams.values()))
    single={n:train['frames'][0][n] for n in left.names}
    with torch.no_grad():_,leaf_cache=left.step((single,),context='checkpoint',offset=0,detach_history=True)
    payload=left.export(leaf_cache)
    out.append(original.reject('PREFIX_STREAM_CHECKPOINT_BINDING',lambda:right.restore(payload,context='checkpoint')))
    resumed=left.restore(payload,context='checkpoint')
    single={n:train['frames'][1][n] for n in left.names}
    with torch.no_grad():
        direct,_=left.step((single,),context='checkpoint',offset=1,state=leaf_cache,detach_history=True)
        replay,_=left.step((single,),context='checkpoint',offset=1,state=resumed,detach_history=True)
    need(torch.equal(direct,replay),'REQUEST_PROOF_CACHE_RESTORE')
    out.append(dict(code='REQUEST_CACHE_RESTORE_EXACT',observed='same request resumes exactly; different request refused'))
    probe=deepcopy(model);stream=probe.stream(request)
    with torch.no_grad():
        _,cache=stream.step((train['frames'][0],),context='probe',offset=0,detach_history=True)
        next(probe.request_projection.parameters()).add_(1)
    out.append(original.reject('PREFIX_STREAM_WEIGHTS_CHANGED',lambda:stream.step((train['frames'][1],),
        context='probe',offset=1,state=cache,detach_history=True)))
    need(fingerprint((model.state_dict(),optimizer.state_dict()))==initial,'REQUEST_PROOF_CONTROL_MUTATION')
    return out


def run(a):
    a.output.mkdir(exist_ok=False);started=tick=time.perf_counter()
    need(a.cpu_proof,'REQUEST_PROOF_CPU_ONLY')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    schedule=AnswerSchedule(original.PLAN['path'],original.PLAN['pin'])
    descriptors={k:describe(schedule,k) for k in schedule.rows};features={k:NativeRequest(v).features for k,v in descriptors.items()}
    write_committed(a.output/'requests.json',descriptors);sequence.runtime.persist(a.output/'features.pt',features)
    controls=request_controls(descriptors);write_committed(a.output/'request-controls.json',controls)
    count=dict(tasks=len(descriptors),fields=sum(len(d['fields']) for d in descriptors.values()),width=WIDTH,
        feature_bytes=sum(t.numel()*t.element_size() for t in features.values()),
        operations={op:sum(d['operation']==op for d in descriptors.values()) for op in sorted({d['operation'] for d in descriptors.values()})},
        clock_basis={k:sum(f['time_basis']==k for d in descriptors.values() for f in d['fields'])
                     for k in ['annotation_interval','unknown','receipt_only','base_before_patch']})
    phases=dict(requests=time.perf_counter()-tick);tick=time.perf_counter()
    source,bundle,data=sequence.prepare(a.root);target=data['labels'].model;pins=code(source)
    train=[d for d in data['items'] if d['side']=='train'];held=[d for d in data['items'] if d['side']!='train']
    phases['native_authentication']=time.perf_counter()-tick;results={}
    for architecture in sequence.runtime.original.ARCHITECTURES:
        tick=time.perf_counter();dest=a.output/architecture;dest.mkdir();model,optimizer=make(bundle,target,architecture)
        baseline,_=sequence.make(bundle,target,architecture);zero=[]
        for d in held:
            for proposal in model.proposals:
                expected=sequence.observe(baseline,target,tuple(d['frames'][:2]),proposal,20261024+d['document'],2)
                actual=observe(model,target,tuple(d['frames'][:2]),proposal,20261024+d['document'],2,NativeRequest(legal(2)))
                need(fingerprint(expected)==fingerprint(actual),'REQUEST_PROOF_ZERO_EQUIVALENCE')
                zero.append(dict(document=d['document'],proposal=proposal,expected=expected,actual=actual))
        sequence.runtime.persist(dest/'zero.pt',zero)
        negative=boundary_controls(model,optimizer,target,bundle,train[0],held[0])
        sequence.runtime.persist(dest/'initial.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict()))
        records=[]
        for step,d in enumerate(train):
            value,request=item(d);record,grad=update(model,optimizer,value,step,dict(bundle['policy'],device='cpu'),request=request,target=target)
            need(all(bool(grad['request_projection.lanes.'+n+'.weight'].abs().any()) for n in model.roster),
                 'REQUEST_PROOF_DISCONNECTED_METADATA')
            sequence.runtime.persist(dest/f'update-{step+1}.pt',dict(record=record,gradients=grad,
                model=model.state_dict(),optimizer=optimizer.state_dict()));records.append(record)
        events=[]
        for d in held:
            prefix=tuple(d['frames'][:2]);future=d['frames'][2:]
            for mode in ('correct','wrong_operation'):
                desc=legal(2)
                if mode=='wrong_operation':desc['operation']='respond_to_source_question'
                request=NativeRequest(desc)
                for proposal in model.proposals:
                    rows=observe(model,target,prefix,proposal,20261024+d['document'],len(future)+1,request)
                    events.append(dict(document=d['document'],proposal=proposal,mode=mode,request=request.pin,
                        events=rows,metrics=sequence.metrics(rows,future,model.roster,prefix[-1])))
        sequence.runtime.persist(dest/'evaluation.pt',events)
        source_only=False
        if architecture=='whole_proposals':
            d=held[0];p=model.proposals[0];prefix=tuple(d['frames'][:2]);req=NativeRequest(legal(2))
            mixed=tuple({n:(f[n] if n==p else train[0]['frames'][i][n]) for n in model.roster} for i,f in enumerate(prefix))
            source_only=fingerprint(observe(model,target,prefix,p,31,2,req))==fingerprint(observe(model,target,mixed,p,31,2,req))
            need(source_only,'REQUEST_PROOF_SOURCE_ONLY')
        results[architecture]=dict(parameters=sum(p.numel() for p in model.parameters()),updates=len(records),
            controls=negative,zero_equivalent=len(zero),source_only=source_only,trajectories=len(events),
            ended=sum(e['metrics']['ended'] for e in events),correct_end=sum(e['metrics']['ended_at_reference_boundary'] for e in events),
            changed_request_weights=all(bool(p.abs().any()) for p in model.request_projection.parameters()))
        phases[architecture]=time.perf_counter()-tick;print(json.dumps(dict(architecture=architecture,**results[architecture])),flush=True)
    tick=time.perf_counter();data['close']();need(pins==code(source),'REQUEST_PROOF_CODE_CHANGED')
    for n,h in pins.items():
        p=a.output/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((Path(__file__).resolve().parents[1]/n).read_bytes())
        need(file_sha(p)==h,'REQUEST_PROOF_SOURCE_COPY')
    write_committed(a.output/'results.json',results)
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['seal']=time.perf_counter()-tick;seconds=time.perf_counter()-started
    need(sum(phases.values())/seconds>=.9 and not torch.cuda.is_initialized(),'REQUEST_PROOF_PHASES')
    report=dict(complete=True,files=files,code=pins,plan=original.PLAN,counts=count,phases=phases,seconds=seconds,
        legal_runtime_qualification=True,capability_native_training=False,scientific_admission=False,
        meaning_understanding=False,serving=False,acceptance='author-checked')
    pin=write_committed(a.output/'report.json',report);print(json.dumps(dict(pin=pin,counts=count,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu-proof',action='store_true');a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(exc)))
        raise
