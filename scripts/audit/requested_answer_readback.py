"""Cold model/optimizer replay and independent observed-request feature reconstruction."""
import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import requested_answer as producer
from audit import legal_answer_sequence as sequence
from audit.surface_device_authority import pinned
from claims.process_conditions import write_committed
from model.answer_training_schedule import AnswerSchedule
from model.capability_structure_input import ROLES,SPEAKER_ROLES
from model.native_population_training import fingerprint
from model.native_request_input import NativeRequest,REQUESTS,legal
from model.native_requested_answer import update
from text.constellation_corpus import file_sha,need


def features(d):
    result=[]
    def log(x):v=float(x);return math.copysign(math.log1p(abs(v)),v)
    op=[float(d['operation']==s) for s in REQUESTS]
    for f in d['fields']:
        n=[float(f['document_position'] is not None),log(f['document_position'] or 0),
           float(f['revision']=='original'),float(f['revision']=='revised'),0.,0.,0.,0.,0.,
           float(f['time_basis']=='receipt_only'),float(f['time_basis']=='base_before_patch')]
        if f['time_basis']=='annotation_interval':
            a,b,z=Fraction(f['clock']['start']),Fraction(f['clock']['end']),Fraction(d['latest_input_end'])
            n[4:9]=[1.,log(a-z),log(b-z),log(b-a),1.]
        result.append(op+[float(f['role']==r) for r in ROLES]+[float(f['speaker_role']==r) for r in SPEAKER_ROLES]+n+[1.,0.])
    result.append(op+[0.]*(len(ROLES)+len(SPEAKER_ROLES)+11)+[0.,1.])
    return torch.tensor(result,dtype=torch.float32)


def source_descriptors(schedule,descriptors,saved):
    need(set(descriptors)==set(saved)==set(schedule.rows),'REQUEST_READBACK_TASKS')
    count=0
    for key,d in descriptors.items():
        pop,task,_=schedule.targets.source.selected(key);context=[pop.fields[k] for k in task['context']]
        need(d['operation']==task['kind'] and len(d['fields'])==len(context),'REQUEST_READBACK_SOURCE')
        for i,(row,f) in enumerate(zip(d['fields'],context)):
            need(row['role']==f['role'] and row['clock']==f['clock'],'REQUEST_READBACK_FIELD')
            interval=task['kind'] in ('respond_to_attested_act','continue_attested_dialogue')
            need(row['speaker_role']==(f['source']['speaker']['role'] if interval else None),'REQUEST_READBACK_SPEAKER')
            need(row['document_position']==(i if task['kind']=='continue_recorded_steps' else None),'REQUEST_READBACK_ORDER')
            revision=('original' if i<3 else 'revised') if task['kind']=='revise_with_source_evidence' else None
            need(row['revision']==revision,'REQUEST_READBACK_REVISION');count+=1
        need(torch.equal(features(d),saved[key]),'REQUEST_READBACK_FEATURE_VALUES')
        need(NativeRequest(d).observed==len(context),'REQUEST_READBACK_CONTEXT_COUNT')
    return count


def pack_check(model,request,frame,position):
    original=model.forecaster.predictor.stream();bound=model.stream(request)
    originals=original.streams if hasattr(original,'streams') else {'all':original}
    wrapped=bound.streams if hasattr(bound,'streams') else {'all':bound}
    metadata=features(json.loads(request.descriptor))[min(position,request.observed)]
    with torch.no_grad():
        for name,left in originals.items():
            right=wrapped[name];selected={n:frame[n] for n in left.names}
            base,atoms,queries=left.pack((selected,),position);actual,aa,qq=right.pack((selected,),position)
            expected=base.clone();cursor=0
            for n in left.names:
                vector=model.request_projection.lanes[n](metadata[None]);size=len(frame[n])+1
                expected[:,cursor:cursor+size]+=vector;cursor+=size
            need(torch.equal(expected,actual) and torch.equal(atoms,aa) and torch.equal(queries,qq),
                 'REQUEST_READBACK_TOKEN_BINDING')


def run(a):
    a.output.mkdir(exist_ok=False);started=tick=time.perf_counter()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    report=pinned(a.input,a.pin);schedule=AnswerSchedule(**dict(root=producer.original.PLAN['path'],pin=producer.original.PLAN['pin']))
    descriptors=json.loads((a.input/'requests.json').read_bytes())
    saved=torch.load(a.input/'features.pt',map_location='cpu',weights_only=True)
    count=source_descriptors(schedule,descriptors,saved)
    need(count==report['counts']['fields'] and len(descriptors)==report['counts']['tasks'],'REQUEST_READBACK_CENSUS')
    phases=dict(descriptors=time.perf_counter()-tick);tick=time.perf_counter()
    source,bundle,data=sequence.prepare(a.root);target=data['labels'].model;need(producer.code(source)==report['code'],'REQUEST_READBACK_CODE')
    phases['native_authentication']=time.perf_counter()-tick
    train=[d for d in data['items'] if d['side']=='train'];held={d['document']:d for d in data['items'] if d['side']!='train'}
    results={}
    for architecture in sequence.runtime.original.ARCHITECTURES:
        tick=time.perf_counter();base=a.input/architecture;model,optimizer=producer.make(bundle,target,architecture)
        initial=torch.load(base/'initial.pt',map_location='cpu',weights_only=True)
        need(fingerprint(dict(model=model.state_dict(),optimizer=optimizer.state_dict()))==fingerprint(initial),'REQUEST_READBACK_INITIAL')
        zeros=torch.load(base/'zero.pt',map_location='cpu',weights_only=True)
        for row in zeros:
            d=held[row['document']];actual=producer.observe(model,target,tuple(d['frames'][:2]),row['proposal'],
                20261024+d['document'],2,NativeRequest(legal(2)))
            need(fingerprint(actual)==fingerprint(row['actual'])==fingerprint(row['expected']),'REQUEST_READBACK_ZERO')
        ledger=[]
        for step,d in enumerate(train):
            value,request=producer.item(d);record,gradients=update(model,optimizer,value,step,dict(bundle['policy'],device='cpu'),request=request,target=target)
            expected=torch.load(base/f'update-{step+1}.pt',map_location='cpu',weights_only=True)
            actual=dict(record=record,gradients=gradients,model=model.state_dict(),optimizer=optimizer.state_dict())
            need(fingerprint(actual)==fingerprint(expected),'REQUEST_READBACK_UPDATE')
            ledger.append(dict(step=step,state=fingerprint(actual),metadata_gradients={n:fingerprint(g) for n,g in gradients.items() if n.startswith('request_projection.')}))
        rows=torch.load(base/'evaluation.pt',map_location='cpu',weights_only=True);evaluated=[]
        for row in rows:
            d=held[row['document']];desc=legal(2)
            if row['mode']=='wrong_operation':desc['operation']='respond_to_source_question'
            request=NativeRequest(desc);need(row['request']==request.pin,'REQUEST_READBACK_REQUEST_BINDING')
            events=producer.observe(model,target,tuple(d['frames'][:2]),row['proposal'],20261024+d['document'],len(d['frames'])-1,request)
            need(fingerprint(events)==fingerprint(row['events']),'REQUEST_READBACK_ROLLOUT')
            measured=sequence.metrics(events,d['frames'][2:],model.roster,d['frames'][1])
            need(measured==row['metrics'],'REQUEST_READBACK_METRICS')
            evaluated.append(dict(document=d['document'],mode=row['mode'],proposal=row['proposal'],events=fingerprint(events),
                                  ended=measured['ended'],extra=measured['extra'],missing=measured['missing']))
        for position in [0,1,2,7]:pack_check(model,NativeRequest(legal(2)),train[0]['frames'][0],position)
        write_committed(a.output/(architecture+'.json'),dict(updates=ledger,trajectories=evaluated,token_packing_exact=True))
        results[architecture]=dict(updates=len(ledger),rollouts=len(evaluated),zero_checks=len(zeros),
            ended=sum(r['ended'] for r in evaluated),all_cold_states_exact=True)
        phases[architecture]=time.perf_counter()-tick;print(json.dumps(dict(architecture=architecture,**results[architecture])),flush=True)
    tick=time.perf_counter();data['close']();need(producer.code(source)==report['code'],'REQUEST_READBACK_CODE_CHANGED')
    phases['closing']=time.perf_counter()-tick;seconds=time.perf_counter()-started
    need(sum(phases.values())/seconds>=.9 and not torch.cuda.is_initialized(),'REQUEST_READBACK_PHASES')
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    result=dict(complete=True,input_pin=a.pin,code=report['code'],files=files,counts=report['counts'],results=results,
        all_source_features_exact=True,all_token_packing_exact=True,all_cold_updates_exact=True,all_cold_rollouts_exact=True,
        phases=phases,seconds=seconds,scientific_admission=False,capability_native_training=False,
        meaning_understanding=False,serving=False,acceptance='author-checked')
    pin=write_committed(a.output/'report.json',result);print(json.dumps(dict(pin=pin,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=producer.ROOT)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--pin',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(exc)))
        raise
