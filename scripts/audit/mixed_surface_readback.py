"""Read packed tensors independently and replay every optimizer/score artifact."""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import mixed_surface_proof as producer
from claims.process_conditions import write_committed
from model import mixed_surface_batches as mixed
from model.native_population_training import cpu,fingerprint
from model.stream_surface import StreamSurfaceDecoder
from text.constellation_corpus import file_sha,need


def check(states, examples, config, specs):
    ends=[0]*len(examples);rows=positions=windows=0
    for state in states:
        need(state['atoms']==len(examples),'MIXED_READBACK_ATOM_COUNT')
        p,y,offsets=state['prefix'],state['labels'],state['offsets']
        need(p.shape==y.shape and len(p)==len(state['records'])==len(offsets),'MIXED_READBACK_ROW_SHAPE')
        need(set(state['native'])=={s.name for s in specs},'MIXED_READBACK_ROSTER')
        for row,r in enumerate(state['records']):
            i=r['atom'];e=examples[i];a,b=r['scored'];lo=r['offset'];length=r['prefix_length']
            need(r['occurrence']==e.occurrence and r['uid']==e.uid and int(offsets[row])==lo,
                 'MIXED_READBACK_SOURCE_JOIN')
            need(a==ends[i] and 0<=lo<=a<b<=len(e.raw)+1 and b-lo==length<=config.max_prefix,
                 'MIXED_READBACK_COVERAGE')
            sequence=[256,*e.raw,257]
            need(p[row,:length].tolist()==sequence[lo:b] and not bool(p[row,length:].any()),
                 'MIXED_READBACK_FUTURE_PREFIX')
            need(y[row,:length].tolist()==[-100]*(a-lo)+sequence[a+1:b+1]
                 and bool((y[row,length:]==-100).all()),'MIXED_READBACK_TARGET')
            need(float(state['denominators'][row])==len(e.raw)+1,'MIXED_READBACK_DENOMINATOR')
            for spec in specs:
                x=state['native'][spec.name];original=e.native[spec.name].values[0];n=len(original)
                need(x['identity']==spec.identity and not bool(x['absent'][row])
                     and torch.equal(x['values'][row,:n],original) and not bool(x['values'][row,n:].any())
                     and not bool(x['padding'][row,:n].any()) and bool(x['padding'][row,n:].all()),
                     'MIXED_READBACK_NATIVE_JOIN')
                windows+=n
            ends[i]=b;rows+=1;positions+=b-a
    need(ends==[len(e.raw)+1 for e in examples],'MIXED_READBACK_MISSING_TAIL')
    return dict(atoms=len(examples),rows=rows,scored_positions=positions,native_window_copies=windows)


def rejected(fn):
    try:fn()
    except (ValueError,RuntimeError) as e:return str(e)
    raise ValueError('MIXED_READBACK_NEGATIVE_ACCEPTED')


def run(a):
    began=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    need(file_sha(a.input/'report.json')==a.input_pin,'MIXED_READBACK_PIN')
    r=json.loads((a.input/'report.json').read_bytes())
    need(r['complete'] and r['code']==producer.code(),'MIXED_READBACK_CODE')
    for name,pin in r['files'].items():need(file_sha(a.input/name)==pin,'MIXED_READBACK_PHYSICAL')
    prior,config,specs,examples,metadata,initial=producer.load(a.root)
    results=json.loads((a.input/'results.json').read_bytes());train=[e for e in examples if e.side=='train']
    selected=[next(e for e in train if e.occurrence==i) for i in results['selected']]
    packing={};controls={}
    for rows in producer.COHORTS:
        states=torch.load(a.input/f'packed-{rows}.pt',map_location='cpu',weights_only=True)
        packing[str(rows)]=check(states,selected,config,specs)
        actual=[producer.packed_state(b) for b in mixed.batches(selected,config,specs,rows=rows)]
        need(fingerprint(states)==fingerprint(actual),'MIXED_READBACK_PACK_REPLAY')
        records=json.loads((a.input/f'schedule-{rows}.json').read_bytes())
        def full_states():
            cursor=0
            for batch in producer.source_batches(examples,config,specs,rows):
                expected=records[cursor];state=producer.packed_state(batch)
                need(fingerprint(state)==expected['state']
                     and json.loads(json.dumps(batch.records))==expected['records'],'MIXED_READBACK_FULL_MANIFEST')
                cursor+=1;yield state
            need(cursor==len(records)==results['schedules'][str(rows)],'MIXED_READBACK_MANIFEST_COUNT')
        packing[str(rows)]['full_source_rows']=check(full_states(),examples,config,specs)
    good=torch.load(a.input/'packed-32.pt',map_location='cpu',weights_only=True)
    for mode in ['future_prefix','changed_target','wrong_uid','wrong_native','wrong_identity','denominator','missing_tail']:
        bad=deepcopy(good);first=bad[0];name=specs[0].name
        if mode=='future_prefix':first['prefix'][0,0]=selected[0].raw[0]
        elif mode=='changed_target':first['labels'][0,0]=(int(first['labels'][0,0])+1)%256
        elif mode=='wrong_uid':first['records'][0]['uid']+=1
        elif mode=='wrong_native':first['native'][name]['values'][0,0,0]+=1
        elif mode=='wrong_identity':first['native'][name]['identity']='0'*64
        elif mode=='denominator':first['denominators'][0]+=1
        else:bad.pop()
        controls[mode]=rejected(lambda b=bad:check(b,selected,config,specs))
    for limit in [0,-1,True,513]:controls['rows_'+str(limit)]=rejected(lambda v=limit:list(mixed.batches(selected,config,specs,rows=v)))
    tiny=replace(config,max_attention_elements=1)
    controls['physical_memory']=rejected(lambda:list(mixed.batches(selected,tiny,specs,rows=32)))
    for label,selection in [('reversed',list(reversed(selected))),('duplicates',[*selected,selected[0]])]:
        states=[producer.packed_state(b) for b in mixed.batches(selection,config,specs,rows=32)]
        packing[label]=check(states,selection,config,specs)
    controls['held_gradient']=rejected(lambda:producer.evaluate(StreamSurfaceDecoder(config,specs),
        [next(e for e in examples if e.side=='query')],32,backward=True))
    shuffled=[replace(e,native=examples[j].native) for e,j in zip(examples,results['donors'])]
    for i,j in enumerate(results['donors']):
        need(examples[j].side=='train' and metadata[i]['document']!=metadata[j]['document']
             and examples[i].uid!=examples[j].uid,'MIXED_READBACK_DONOR')
    replay={}
    for label,rows in [('reference',None)]+[(f'rows-{v}',v) for v in producer.COHORTS]:
        model=StreamSurfaceDecoder(config,specs);model.load_state_dict(initial,strict=True);model.train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,foreach=False);model.zero_grad(set_to_none=True)
        fitting,_=producer.evaluate(model,train,rows,backward=True)
        gradients={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
        optimizer.step();after=cpu(model.state_dict());moments=cpu(optimizer.state_dict())
        with torch.no_grad():
            evaluation,_=producer.evaluate(model,examples,rows);control,_=producer.evaluate(model,shuffled,rows)
        actual=dict(training=fitting,evaluation=evaluation,shuffled=control,gradients=gradients,after=after,optimizer=moments)
        saved=torch.load(a.input/(label+'.pt'),map_location='cpu',weights_only=True)
        need(fingerprint(actual)==fingerprint(saved)==results['results'][label]['state'],'MIXED_READBACK_OPTIMIZER_REPLAY')
        replay[label]=dict(state=fingerprint(actual),all_tensors_exact=True)
        print(json.dumps(dict(replayed=label,exact=True)),flush=True)
    need(not torch.cuda.is_initialized() and producer.code()==r['code'],'MIXED_READBACK_ENVIRONMENT')
    result=dict(complete=True,input=str(a.input),input_pin=a.input_pin,packing=packing,controls=controls,replay=replay,
        code=r['code'],seconds=time.perf_counter()-began,all_persisted_native_values_exact=True,
        all_cold_arm_states_exact=True,reference_candidate_equivalence=r['numerical_equivalence'],promoted=False,
        number_status='measured',acceptance='author-checked',scientific_admission=False)
    pin=write_committed(a.output/'report.json',result);print(json.dumps(dict(complete=True,pin=pin)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--input',type=Path,required=True);p.add_argument('--input-pin',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error)))
        raise
