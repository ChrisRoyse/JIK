"""Cold decoder replay and independent full-denominator answer likelihood readback."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from torch.nn import functional as F
from audit import answer_surface as producer
from audit import surface_device_authority as authority
from claims.process_conditions import write_committed
from model.native_population_training import fingerprint
from model.stream_surface import batches
from text.constellation_corpus import file_sha,need


def cold_score(model,examples):
    result=[]
    for e in examples:
        losses=torch.full((len(e.raw)+1,),float('nan'));argmax=torch.full((len(e.raw)+1,),-1,dtype=torch.int64)
        for prefix,labels,offsets,native,items in batches(e.raw,e.native,model.config):
            logits=model(prefix,native,offsets=offsets)['logits']
            token=F.cross_entropy(logits.reshape(-1,model.config.vocabulary),labels.reshape(-1),
                                  ignore_index=-100,reduction='none').reshape(labels.shape)
            chosen=logits.argmax(-1)
            for row,part in enumerate(items):
                a,b=part['scored'];lo=a-part['offset'];hi=b-part['offset']
                need(bool(torch.isnan(losses[a:b]).all()),'ANSWER_READBACK_REPEATED_BYTE')
                losses[a:b]=token[row,lo:hi];argmax[a:b]=chosen[row,lo:hi]
        need(bool(losses.isfinite().all()) and bool((argmax>=0).all()),'ANSWER_READBACK_MISSING_BYTE')
        result.append(dict(losses=losses,argmax=argmax))
    return result


def independent_metrics(scores,indices,state):
    predicted=[v['losses'].tolist() for v in scores]
    true=[state['checkpoint']['evaluation'][i]['losses'].tolist() for i in indices]
    shuffled=[state['checkpoint']['shuffled'][i]['losses'].tolist() for i in indices]
    n=sum(len(v)-1 for v in predicted)
    sums={k:math.fsum(x for v in values for x in v[:-1])
          for k,values in [('predicted',predicted),('true',true),('shuffled',shuffled)]}
    sums.update(first_byte=math.fsum(v[0] for v in predicted),eos=math.fsum(v[-1] for v in predicted))
    return dict(byte_positions=n,atoms=len(predicted),sums=sums,
        nats_per_byte={k:sums[k]/n for k in ['predicted','true','shuffled']} if n else None,
        predicted_better_than_shuffled_atoms=sum(math.fsum(p[:-1])<math.fsum(s[:-1]) for p,s in zip(predicted,shuffled)),
        predicted_equal_shuffled_atoms=sum(math.fsum(p[:-1])==math.fsum(s[:-1]) for p,s in zip(predicted,shuffled)))


def equal(a,b):
    if isinstance(a,dict):
        need(a.keys()==b.keys(),'ANSWER_READBACK_KEYS')
        for k in a:equal(a[k],b[k])
    elif isinstance(a,list):
        need(len(a)==len(b),'ANSWER_READBACK_LENGTH')
        for x,y in zip(a,b):equal(x,y)
    elif isinstance(a,float):need(math.isclose(a,b,rel_tol=0,abs_tol=1e-10),'ANSWER_READBACK_FLOAT')
    else:need(a==b,'ANSWER_READBACK_VALUE')


def aggregate(rows):
    cells={}
    for r in rows:cells.setdefault('/'.join(r[k] for k in ['architecture','proposal','mode']),[]).append(r)
    output={}
    for key,group in cells.items():
        v=dict(documents=len(group),**{k:sum(int(r['coverage'][k]) for r in group) for k in group[0]['coverage']})
        v['byte_positions']=sum(r['metrics']['byte_positions'] for r in group)
        v['sums']={k:math.fsum(r['metrics']['sums'][k] for r in group) for k in group[0]['metrics']['sums']}
        v['nats_per_byte']={k:v['sums'][k]/v['byte_positions'] for k in ['predicted','true','shuffled']}
        v['prediction_penalty']=v['nats_per_byte']['predicted']-v['nats_per_byte']['true']
        v['shuffled_minus_predicted']=v['nats_per_byte']['shuffled']-v['nats_per_byte']['predicted'];output[key]=v
    return output


def pairs(rows):
    lookup={(r['architecture'],r['proposal'],r['document'],r['mode']):r for r in rows};out=[]
    for key,r in lookup.items():
        if key[-1]!='correct':continue
        for mode in ['reversed','train_donor']:
            other=lookup[(*key[:-1],mode)]
            need(r['coverage']['matched']==other['coverage']['matched']
                 and r['metrics']['byte_positions']==other['metrics']['byte_positions'],
                 'ANSWER_READBACK_PAIRED_SUPPORT')
            a=r['metrics']['nats_per_byte']['predicted'];b=other['metrics']['nats_per_byte']['predicted']
            out.append(dict(architecture=key[0],proposal=key[1],document=key[2],control=mode,
                control_minus_correct=b-a,correct_wins=b>a,tie=b==a))
    return out


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    report=authority.pinned(a.input,a.input_pin);state=producer.load(a.root);model=state['model']
    need(report['code']==state['code'] and report['sources']=={k:list(v) for k,v in producer.REFS.items()},
         'ANSWER_READBACK_BINDING')
    need(report['decoder_state']==fingerprint(model.state_dict()),'ANSWER_READBACK_CHECKPOINT')
    need(json.loads((a.input/'joins.json').read_bytes())==state['joins'],'ANSWER_READBACK_JOIN')
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter();allrows=[];counts=dict(events=0,scored_atoms=0,byte_positions=0)
    with torch.no_grad():
        for key,examples in [('evaluation',state['examples']),('shuffled',state['shuffled'])]:
            cold=cold_score(model,[examples[i] for i in state['query']])
            need(fingerprint(cold)==fingerprint([state['checkpoint'][key][i] for i in state['query']]),
                 'ANSWER_READBACK_BASELINE')
        phases['baselines']=time.perf_counter()-tick
        for architecture in producer.ARCHITECTURES:
            tick=time.perf_counter();path=a.root/producer.REFS['predictions'][0]/architecture/'evaluation.pt'
            trace_pin=file_sha(path);actual=torch.load(path,map_location='cpu',weights_only=True)
            saved=torch.load(a.input/(architecture+'.pt'),map_location='cpu',weights_only=True)
            need(len(actual)==len(saved),'ANSWER_READBACK_TRAJECTORIES')
            for source,stored in zip(actual,saved):
                examples,indices,inputs,coverage=producer.adapted_events(source,state,trace_pin)
                need(stored['indices']==indices and stored['inputs']==inputs and stored['row']['coverage']==coverage,
                     'ANSWER_READBACK_ADAPTER')
                for event,adapted in zip(source['events'],examples):
                    for n,slot in adapted.native.items():
                        need(torch.equal(slot.values[0],event['slots'][n]) and not bool(slot.padding.any())
                             and not bool(slot.absent.any()),'ANSWER_READBACK_NATIVE_PRESERVATION')
                cold=cold_score(model,examples)
                need(fingerprint(cold)==fingerprint(stored['scores']),'ANSWER_READBACK_BYTE_REPLAY')
                measured=independent_metrics(stored['scores'],indices,state);equal(measured,stored['row']['metrics'])
                row=dict(stored['row'],metrics=measured);allrows.append(row)
                counts['events']+=len(inputs);counts['scored_atoms']+=len(indices);counts['byte_positions']+=measured['byte_positions']
            phases[architecture]=time.perf_counter()-tick
            print(json.dumps(dict(architecture=architecture,cold_exact=True,seconds=phases[architecture])),flush=True)
    tick=time.perf_counter();equal(allrows,json.loads((a.input/'rows.json').read_bytes()))
    cells=aggregate(allrows);equal(cells,json.loads((a.input/'cells.json').read_bytes()))
    controls=json.loads((a.input/'controls.json').read_bytes());need(len(controls)==9,'ANSWER_READBACK_CONTROLS')
    comparisons=pairs(allrows);write_committed(a.output/'paired-contexts.json',comparisons)
    write_committed(a.output/'cells.json',cells);write_committed(a.output/'rows.json',allrows)
    state['fresh']();need(fingerprint(model.state_dict())==report['decoder_state'],'ANSWER_READBACK_MUTATION')
    need(not torch.cuda.is_initialized(),'ANSWER_READBACK_CUDA')
    phases['readback']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(sum(phases.values())/seconds>=.9,'ANSWER_READBACK_PHASE_COVERAGE')
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    result=dict(complete=True,input_pin=a.input_pin,code=state['code'],files=files,counts=counts,
        all_cold_byte_scores_exact=True,all_native_slots_preserved=True,all_independent_aggregates_match=True,
        controls=len(controls),phases=phases,seconds=seconds,scientific_admission=False,
        free_generation=False,semantic_correctness=False,serving=False,acceptance='author-checked')
    pin=write_committed(a.output/'report.json',result);print(json.dumps(dict(pin=pin,counts=counts,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=producer.ROOT)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--input-pin',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(exc)))
        raise
