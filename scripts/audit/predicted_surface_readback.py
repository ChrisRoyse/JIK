"""Cold source/gradient/AdamW/output replay without the new adaptation wrapper."""
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import predicted_surface_source as source
from audit.answer_surface_stream import rejected
from audit.answer_surface_stream_readback import baseline_atom
from model.native import NativeInput
from model.native_open_window_answer import answer_model
from model.native_population_training import cpu, fingerprint
from model.native_request_input import NativeRequest, legal
from model.stream_surface import StreamSurfaceDecoder
from model import surface_inversion_training as inversion
from model.surface_device_runtime import evaluate
from claims.process_conditions import write_committed
from text.constellation_corpus import canonical, digest, file_sha, need


def rebuild(state,traces,pin,receipts):
    groups={};lookup={(r['key'],r['document'],r['position']):r for r in receipts}
    need(len(lookup)==232,'ADAPTATION_COLD_RECEIPTS')
    rows=[(r,pin) for r in traces]
    for arm in source.ARCHITECTURES:
        path=state['root']/source.REFS['predictions'][0]/arm/'evaluation.pt'
        for r in torch.load(path,map_location='cpu',weights_only=True):
            if r['mode']=='correct' and r['document'] in source.EVALUATION:
                rows.append((dict(r,architecture=arm),file_sha(path)))
    for r,trace in rows:
        key=r['architecture']+'/'+r['proposal'];doc=r['document'];pairs=[]
        indices=state['groups'][doc][2:]
        need(len(r['events'])>=len(indices),'ADAPTATION_COLD_MISSING')
        for position,index in enumerate(indices):
            event=r['events'][position];e=state['examples'][index]
            need(event['kind']=='constellation' and event['index']==position
                 and event['proposal']==r['proposal'],'ADAPTATION_COLD_ORDER')
            row=digest(canonical(dict(trace=trace,proposal=r['proposal'],position=position)));slots={}
            for spec in state['specs']:
                v=event['slots'][spec.name]
                need(v.shape==(int(event['counts'][spec.name][0]),spec.dims),'ADAPTATION_COLD_COUNTS')
                slots[spec.name]=NativeInput(spec.identity,(row,),v[None].clone(),
                    torch.zeros((1,len(v)),dtype=torch.bool),torch.zeros(1,dtype=torch.bool))
            native=replace(e,native=slots)
            region=state['region_pin']+':'+'.'.join(str(x) for x in event['codes'][0].tolist())
            receipt=dict(document=doc,source_index=index,position=position,proposal=r['proposal'],trace_pin=trace,
                region_pin=state['region_pin'],regions=(region,),source_sha=hashlib.sha256(e.raw).hexdigest(),
                observed=fingerprint({n:x.values for n,x in e.native.items()}),
                predicted=fingerprint({n:x.values for n,x in slots.items()}),
                predicted_counts={n:len(x.values[0]) for n,x in slots.items()},scientific_admission=False)
            expected=dict(key=key,**receipt,side=e.side,train_present=state['metadata'][index]['train_present'])
            need(canonical(expected)==canonical(lookup[key,doc,position]),'ADAPTATION_COLD_PAIR')
            seal=digest(canonical(dict(receipt=receipt,occurrence=e.occurrence,uid=e.uid,side=e.side,
                                       raw=receipt['source_sha'])))
            pairs.append(SimpleNamespace(observed=e,predicted=native,receipt=receipt,seal=seal))
        groups.setdefault(key,{})[doc]=pairs
    return groups


def initial(state):
    model=StreamSurfaceDecoder(state['config'],state['specs'])
    model.load_state_dict(state['checkpoint']['after'],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,foreach=False)
    optimizer.load_state_dict(deepcopy(state['checkpoint']['optimizer']));inversion.check_state(model,optimizer,state['initial_step'])
    return model,optimizer


def replay_predictions(state,traces):
    decisions=0
    for arm in source.ARCHITECTURES:
        path=state['root']/source.REFS['predictions'][0]/arm/'update-5.pt'
        cp=torch.load(path,map_location='cpu',weights_only=True);predictor_pin=file_sha(path)
        model=answer_model(state['target'],state['bundle'],arm,target_pin=state['region_pin'])
        model.load_state_dict(cp['model'],strict=True);model.eval().requires_grad_(False)
        before=fingerprint(model.state_dict())
        for row in (r for r in traces if r['architecture']==arm):
            doc=row['document'];d=state['items'][doc];need(doc in source.TRAIN and d['side']=='train','ADAPTATION_COLD_TRAIN')
            need(set(row)=={'architecture','proposal','document','events','seed','request','predictor','input'}
                 and row['predictor']==predictor_pin and row['input']==fingerprint(d['frames'][:2]),
                 'ADAPTATION_COLD_PREDICTOR_INPUT_BINDING')
            request=NativeRequest(legal(2));need(row['request']==request.pin and row['seed']==20261024+doc,
                                             'ADAPTATION_COLD_REQUEST')
            gen=model.rollout(tuple(d['frames'][:2]),request=request,target=state['target'],proposal=row['proposal'],
                              context='request-proof',seed=20261024+doc);events=[]
            try:
                for _ in range(len(d['frames'])-1):
                    event=next(gen);events.append(cpu(event))
                    if event['kind']=='end':break
            finally:gen.close()
            need(fingerprint(events)==fingerprint(row['events']),'ADAPTATION_COLD_PREDICTION')
            decisions+=len(events)
        need(fingerprint(model.state_dict())==before,'ADAPTATION_COLD_PREDICTOR_CHANGED')
    return decisions


def fit(state,group,key,path,trace_pin,*,predicted):
    records=torch.load(path/'updates.pt',map_location='cpu',weights_only=True)
    refs=json.loads((path/'checkpoints.json').read_bytes());model,opt=initial(state);start=state['initial_step']
    binding=dict(code=state['code'],source=trace_pin,predictor=key,
        initial=fingerprint((state['checkpoint']['after'],state['checkpoint']['optimizer'])),train=source.TRAIN,
        rule='half_true_half_predicted' if predicted else 'two_true_views',policy=inversion.POLICY,
        scope='CPU_engineering_qualification',initial_step=start,decoder=state['decoder_receipt'])
    need(canonical(binding)==canonical(json.loads((path/'binding.json').read_bytes())),'ADAPTATION_COLD_BINDING')
    ledger=digest(canonical(binding));positions=0
    need(len(records)==len(refs)==2,'ADAPTATION_COLD_UPDATE_COUNT')
    for cursor,doc in enumerate(source.TRAIN):
        pairs=group[doc];examples=[]
        for pair in pairs:examples.extend([pair.observed,pair.predicted if predicted else pair.observed])
        row,_=inversion.update(model,opt,examples,start+cursor)
        row['conditioning']=dict(rule=binding['rule'],pairs=[p.seal for p in pairs],unique_targets=len(pairs))
        need(fingerprint(row)==fingerprint(records[cursor]['update']),'ADAPTATION_COLD_GRADIENT_UPDATE')
        ledger=digest(canonical(dict(previous=ledger,row=row)));ref=refs[cursor]
        need(file_sha(ref['path'])==ref['pin'],'ADAPTATION_COLD_CHECKPOINT_FILE')
        saved=torch.load(ref['path'],map_location='cpu',weights_only=True)
        need(fingerprint(saved)==ref['state'] and saved['binding']==binding and saved['step']==start+cursor+1
             and saved['cursor']==cursor+1 and saved['ledger']==ledger
             and fingerprint(saved['model'])==fingerprint(model.state_dict())
             and fingerprint(saved['optimizer'])==fingerprint(opt.state_dict()),'ADAPTATION_COLD_CHECKPOINT')
        positions+=row['positions']
        if cursor==0:
            model,opt=initial(state);pos,step,chain=inversion.restore(ref,model,opt,binding,size=2)
            need((pos,step,chain)==(1,start+1,ledger),'ADAPTATION_COLD_RESTART')
    return model,positions


def score(model,group,*,true_cache):
    result={};train=[p for doc in source.TRAIN for p in group[doc]];model.eval()
    with torch.no_grad():
        for doc in source.EVALUATION:
            pairs=group[doc];wrong=[];donors=[]
            for position,pair in enumerate(pairs):
                available=[p for p in train if p.receipt['document']!=doc and p.observed.uid!=pair.observed.uid]
                donor=available[position%len(available)];donors.append(donor.seal)
                wrong.append(replace(pair.predicted,native=donor.predicted.native))
            scores={}
            for name,examples in [('true',[p.observed for p in pairs]),('predicted',[p.predicted for p in pairs]),('shuffled',wrong)]:
                if name=='true' and doc in true_cache:scores[name]=true_cache[doc]
                else:
                    scores[name]=evaluate(model,examples,'gathered',128)[0]
                    if name=='true':true_cache[doc]=scores[name]
            free=[]
            for p in pairs:
                value=baseline_atom(model,p.predicted.native,96)
                for e in value['events']:e.pop('logits',None)
                free.append(value)
            result[doc]=dict(metadata=dict(atoms=len(pairs),donors=donors,positions=sum(len(p.observed.raw)+1 for p in pairs)),
                             scores=scores,free=free)
    return result


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();phases={}
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    need(file_sha(a.source/'report.json')==a.pin,'ADAPTATION_COLD_PIN')
    report=json.loads((a.source/'report.json').read_bytes())
    for n,h in report['files'].items():need(file_sha(a.source/n)==h,'ADAPTATION_COLD_FILE')
    state=source.load(a.root,trained_decoder=report['trained_decoder']);need(state['code']==report['code'],'ADAPTATION_COLD_CODE')
    need(state['initial_step']==report['initial_step'] and state['decoder_receipt']==report['initial_decoder'],
         'ADAPTATION_COLD_INITIAL_AUTHORITY')
    saved_initial=torch.load(a.source/'initial.pt',map_location='cpu',weights_only=True)
    need(fingerprint(saved_initial)==fingerprint(dict(after=state['checkpoint']['after'],
         optimizer=state['checkpoint']['optimizer'])),'ADAPTATION_COLD_INITIAL')
    traces=torch.load(a.source/'train-predictions.pt',map_location='cpu',weights_only=True)
    need(file_sha(a.source/'train-predictions.pt')==report['trace_pin'],'ADAPTATION_COLD_TRACE_PIN')
    receipts=json.loads((a.source/'pairs.json').read_bytes());groups=rebuild(state,traces,report['trace_pin'],receipts)
    phases['authentication_and_independent_joins']=time.perf_counter()-tick;tick=time.perf_counter()
    if report['prediction_reuse'] is not None:
        reused,_=source.retained_predictions(state)
        need(state['prediction_reuse']==report['prediction_reuse'] and fingerprint(reused)==fingerprint(traces),
             'ADAPTATION_COLD_RETAINED_TRACE')
        decisions=sum(len(r['events']) for r in traces)
    else:decisions=replay_predictions(state,traces)
    phases['TRAIN_trace_authentication_or_replay']=time.perf_counter()-tick;tick=time.perf_counter()
    positions=0;models={};key=report['baseline_key']
    model,count=fit(state,groups[key],key,a.source/'observed',report['trace_pin'],predicted=False)
    positions+=count;models['observed']=model
    for key,group in groups.items():
        model,count=fit(state,group,key,a.source/key.replace('/','--'),report['trace_pin'],predicted=True)
        positions+=count;models[key]=model
    phases['independent_gradient_parameter_optimizer_replay']=time.perf_counter()-tick;tick=time.perf_counter()
    model,_=initial(state);checked=0
    for label,model in [('initial',model),('observed',models['observed'])]:
        cache={}
        for key,group in groups.items():
            value=score(model,group,true_cache=cache)
            saved=torch.load(a.source/'evaluation'/(label+'-'+key.replace('/','--')+'.pt'),map_location='cpu',weights_only=True)
            need(fingerprint(value)==fingerprint(saved),'ADAPTATION_COLD_OUTPUT');checked+=1
    for key,group in groups.items():
        value=score(models[key],group,true_cache={})
        saved=torch.load(a.source/'evaluation'/('mixed-'+key.replace('/','--')+'.pt'),map_location='cpu',weights_only=True)
        need(fingerprint(value)==fingerprint(saved),'ADAPTATION_COLD_OUTPUT');checked+=1
    phases['all_evaluation_losses_and_direct_free_rendering']=time.perf_counter()-tick;tick=time.perf_counter()
    bad=deepcopy(traces);bad[0]['events']=[]
    negatives=dict(missing_prediction=rejected(lambda:rebuild(state,bad,report['trace_pin'],receipts),'ADAPTATION_COLD_MISSING'))
    bad=deepcopy(receipts);bad[0]['source_sha']='0'*64
    negatives['wrong_source']=rejected(lambda:rebuild(state,traces,report['trace_pin'],bad),'ADAPTATION_COLD_PAIR')
    bad=deepcopy(traces);bad[0]['events'][0]['index']=1
    negatives['wrong_order']=rejected(lambda:rebuild(state,bad,report['trace_pin'],receipts),'ADAPTATION_COLD_ORDER')
    state['fresh']();need(not torch.cuda.is_initialized(),'ADAPTATION_COLD_GPU')
    need(positions==7218 and checked==24 and decisions==report['predictor_decisions'],'ADAPTATION_COLD_DENOMINATOR')
    phases['controls_and_final_state']=time.perf_counter()-tick
    result=dict(complete=True,source=dict(path=str(a.source),pin=a.pin),code=state['code'],
        all_predictor_outputs_exact=True,all_native_source_pairs_exact=True,all_updates_exact=True,
        all_checkpoints_and_moments_exact=True,all_evaluation_and_free_outputs_exact=True,
        updates=18,positions=positions,predictor_decisions=decisions,evaluation_cells=checked,controls=negatives,
        initial_step=state['initial_step'],trained_decoder=report['trained_decoder'],
        predictor_outputs_replayed=report['prediction_reuse'] is None,prediction_reuse=report['prediction_reuse'],
        phases=phases,seconds=time.perf_counter()-began,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        acceptance='author-checked',number_status='measured',scientific_admission=False,capability_demonstrated=False)
    pin=write_committed(a.output/'report.json',result);print(json.dumps(dict(complete=True,pin=pin,seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=source.authority.ROOT)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error)))
        raise
