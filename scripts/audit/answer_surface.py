"""Frozen-decoder likelihood of actual multi-constellation answer predictions."""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import legal_answer_sequence as sequence
from audit import surface_device_authority as authority
from claims.process_conditions import write_committed
from model.answer_surface_input import predicted_atom
from model.native_population_training import fingerprint
from model.stream_surface import StreamSurfaceDecoder
from model.surface_device_runtime import evaluate
from text.constellation_corpus import file_sha, need

ROOT = authority.ROOT
REFS = dict(
    predictions=('conditioned-answer-cpu-v2-20261023','f3c23bf3b18e58cb89ab505f5d5a457081378648d31d71a55ee2e510b58b4b07'),
    predictions_readback=('conditioned-answer-readback-v2-20261023','b184928707e1651db3f1e12b62fefb9d60ed6404bd54bd42b0b98ba5c028c8fc'),
    decoder=('legal-surface-device-cpu-v2-20261023','033ef5b9313b85e89fabe0fa663dacd2c5735978d18310ebddd3c48e4846ef99'),
    decoder_readback=('legal-surface-device-readback-v2-20261023','ed9fcc7e33634e6ec4b9ace7f5278c9c40103926e37f6e6ed725c4cc122ceec9'))
NEW = ('model/answer_surface_input.py','audit/answer_surface.py','audit/answer_surface_readback.py')
ARCHITECTURES = ('shared','own_slot','whole_proposals')


def load(root):
    records = {k:authority.pinned(root/n,h) for k,(n,h) in REFS.items()}
    for name in ['predictions','decoder']:
        need(records[name+'_readback']['input_pin']==REFS[name][1], 'ANSWER_SURFACE_READBACK')
    need(records['predictions_readback']['all_cold_updates_exact']
         and records['predictions_readback']['all_independent_rollouts_exact']
         and records['decoder_readback']['all_cold_arm_states_exact'], 'ANSWER_SURFACE_PRIOR_EXACT')
    config,specs,examples,meta,_,shuffled,saved,code,fresh_surface = authority.load(root)
    source,bundle,data = sequence.prepare(root)
    need(fingerprint(data['labels'].model.state_dict())==records['predictions']['frozen_target'],
         'ANSWER_SURFACE_FROZEN_REGION')
    by_doc = {}
    for i,m in enumerate(meta): by_doc.setdefault(m['document'],[]).append(i)
    need(set(by_doc)=={i['document'] for i in data['items']}, 'ANSWER_SURFACE_DOCUMENT_JOIN')
    joins = []
    for item in data['items']:
        indices = by_doc[item['document']]
        need(len(indices)==len(item['frames']), 'ANSWER_SURFACE_ATOM_JOIN')
        for position,(index,frame) in enumerate(zip(indices,item['frames'])):
            e=examples[index]
            need(e.side==item['side'] and e.occurrence==item['start']+position
                 and set(frame)=={s.name for s in specs}, 'ANSWER_SURFACE_SOURCE_JOIN')
            need(all(torch.equal(e.native[n].values[0],v) for n,v in frame.items()),
                 'ANSWER_SURFACE_NATIVE_JOIN')
            joins.append(dict(document=item['document'],position=position,index=index,
                occurrence=e.occurrence,uid=e.uid,side=e.side,native=fingerprint(frame)))
    code.update(sequence.code(source))
    for r in records.values():
        for n,h in r['code'].items():
            need(file_sha(Path(__file__).resolve().parents[1]/n)==h,'ANSWER_SURFACE_PRIOR_CODE')
            need(n not in code or code[n]==h,'ANSWER_SURFACE_CODE_CONFLICT');code[n]=h
    for n in NEW: code[n]=file_sha(Path(__file__).resolve().parents[1]/n)
    stamps={root/n/f:authority.stamp(root/n/f) for n,_ in REFS.values()
            for f in ['report.json',*json.loads((root/n/'report.json').read_bytes()).get('files',{})]}
    def fresh():
        fresh_surface();data['close']()
        for p,s in stamps.items():need(authority.stamp(p)==s,'ANSWER_SURFACE_SOURCE_CHANGED')
        for n,h in code.items():need(file_sha(Path(__file__).resolve().parents[1]/n)==h,'ANSWER_SURFACE_CODE_CHANGED')
    checkpoint=torch.load(root/REFS['decoder'][0]/'reference.pt',map_location='cpu',weights_only=True)
    model=StreamSurfaceDecoder(config,specs);model.load_state_dict(checkpoint['after'],strict=True)
    model.eval().requires_grad_(False)
    query=[i for d,indices in by_doc.items() if examples[indices[0]].side!='train' for i in indices[2:]]
    need(len(query)==124 and len(joins)==215,'ANSWER_SURFACE_CENSUS')
    return dict(config=config,specs=specs,examples=examples,metadata=meta,shuffled=shuffled,
        checkpoint=checkpoint,model=model,query=query,by_doc=by_doc,joins=joins,code=code,fresh=fresh,
        adapter=dict(region_pin=bundle['target_ref']['pin'],region_sizes=tuple(bundle['sizes'])))


def controls(event, specs, config, binding):
    output=[]
    def reject(code,bad,**extra):
        try:predicted_atom(bad,specs,config,**binding,position=0,**extra)
        except (ValueError,RuntimeError,TypeError) as exc:
            need(code in str(exc),'ANSWER_SURFACE_CONTROL_REASON:'+str(exc));output.append(dict(code=code,observed=str(exc)))
        else:raise ValueError('ANSWER_SURFACE_CONTROL_ACCEPTED:'+code)
    bad=deepcopy(event);bad['slots'].pop(next(iter(bad['slots'])));reject('ANSWER_SURFACE_ROSTER',bad)
    bad=deepcopy(event);bad['counts'][next(iter(bad['counts']))]+=1;reject('ANSWER_SURFACE_COUNT',bad)
    bad=deepcopy(event);bad['index']=1;reject('ANSWER_SURFACE_POSITION',bad)
    bad=deepcopy(event);bad['slots'][next(iter(bad['slots']))][0,0]=float('nan');reject('ANSWER_SURFACE_VALUES',bad)
    bad=deepcopy(event);bad['codes'][0,0]=binding['region_sizes'][0];reject('ANSWER_SURFACE_REGION',bad)
    bad=deepcopy(event);bad['stop_logits'][0]=torch.tensor([-1.,1.]);reject('ANSWER_SURFACE_STOP_CONFLICT',bad)
    bad=deepcopy(event);bad['raw']=b'private target';reject('ANSWER_SURFACE_EVENT_FIELDS',bad)
    reject('unexpected keyword argument',event,target_length=17)
    original=fingerprint(event);a=predicted_atom(event,specs,config,**binding,position=0)
    next(iter(a.slots.values())).values.fill_(42)
    need(fingerprint(event)==original,'ANSWER_SURFACE_ALIAS')
    output.append(dict(code='OWNED_PREDICTED_SLOTS',observed='caller mutation leaves saved event unchanged'))
    return output


def adapted_events(record, state, trace_pin):
    row,events=record['row'],record['events'];generated=[];inputs=[];end=False
    need(row['mode'] in ('correct','reversed','train_donor'),'ANSWER_SURFACE_CONTEXT')
    indices=state['by_doc'][row['document']][2:]
    need(state['examples'][indices[0]].side!='train','ANSWER_SURFACE_QUERY_ONLY')
    for e in events:
        need(not end,'ANSWER_SURFACE_AFTER_END')
        if e['kind']=='end':
            need(e['generated']==len(generated) and int(e['stop_logits'].argmax(-1))==1,
                 'ANSWER_SURFACE_END_CONFLICT');end=True;continue
        need(e['proposal']==row['proposal'],'ANSWER_SURFACE_PROPOSAL')
        value=predicted_atom(e,state['specs'],state['config'],**state['adapter'],trace_pin=trace_pin,position=len(generated))
        generated.append(value)
        inputs.append(dict(position=e['index'],regions=value.regions,rows=next(iter(value.slots.values())).rows,
            native=fingerprint({n:v.values[0] for n,v in value.slots.items()}),
            counts={n:int(v[0]) for n,v in e['counts'].items()}))
    count=min(len(indices),len(generated))
    examples=[replace(state['examples'][i],native=c.slots) for i,c in zip(indices,generated)]
    coverage=dict(expected=len(indices),generated=len(generated),matched=count,missing=len(indices)-count,
                  extra=max(0,len(generated)-len(indices)),ended=end,censored=not end,
                  correct_end=end and len(generated)==len(indices))
    return examples,indices[:count],inputs,coverage


def metrics(scores,indices,state):
    sums={k:0. for k in ['predicted','true','shuffled','first_byte','eos']};positions=0;wins=0;ties=0
    for score,i in zip(scores,indices):
        p=score['losses'].double();t=state['checkpoint']['evaluation'][i]['losses'].double()
        s=state['checkpoint']['shuffled'][i]['losses'].double()
        need(p.shape==t.shape==s.shape,'ANSWER_SURFACE_SCORE_SHAPE')
        for k,v in [('predicted',p),('true',t),('shuffled',s)]:sums[k]+=float(v[:-1].sum())
        sums['first_byte']+=float(p[0]);sums['eos']+=float(p[-1]);positions+=len(p)-1
        wins+=int(p[:-1].sum()<s[:-1].sum());ties+=int(p[:-1].sum()==s[:-1].sum())
    return dict(byte_positions=positions,atoms=len(scores),sums=sums,
        nats_per_byte={k:sums[k]/positions for k in ['predicted','true','shuffled']} if positions else None,
        predicted_better_than_shuffled_atoms=wins,predicted_equal_shuffled_atoms=ties)


def summarize(rows):
    cells={}
    for r in rows:
        key='/'.join(r[k] for k in ['architecture','proposal','mode'])
        v=cells.setdefault(key,dict(documents=0,expected=0,generated=0,matched=0,missing=0,extra=0,ended=0,
            censored=0,correct_end=0,byte_positions=0,sums={k:0. for k in ['predicted','true','shuffled','first_byte','eos']}))
        v['documents']+=1
        for k,x in r['coverage'].items():v[k]+=int(x)
        v['byte_positions']+=r['metrics']['byte_positions']
        for k,x in r['metrics']['sums'].items():v['sums'][k]+=x
    for v in cells.values():
        v['nats_per_byte']={k:v['sums'][k]/v['byte_positions'] for k in ['predicted','true','shuffled']}
        v['prediction_penalty']=v['nats_per_byte']['predicted']-v['nats_per_byte']['true']
        v['shuffled_minus_predicted']=v['nats_per_byte']['shuffled']-v['nats_per_byte']['predicted']
    return cells


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();cpu=time.process_time()
    need(a.cpu_proof,'ANSWER_SURFACE_CPU_QUALIFICATION_ONLY')
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    state=load(a.root);model=state['model'];initial=fingerprint(model.state_dict())
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    write_committed(a.output/'joins.json',state['joins']);negative=[];rows=[]
    with torch.no_grad():
        baseline={}
        for name,examples in [('evaluation',state['examples']),('shuffled',state['shuffled'])]:
            actual,info=evaluate(model,[examples[i] for i in state['query']],'reference',None)
            expected=[state['checkpoint'][name][i] for i in state['query']]
            need(fingerprint(actual)==fingerprint(expected),'ANSWER_SURFACE_BASELINE_REPLAY')
            baseline[name]=dict(exact=True,states=fingerprint(actual),info=info)
        write_committed(a.output/'baseline.json',baseline)
        phases['baseline_replay']=time.perf_counter()-tick
        for architecture in ARCHITECTURES:
            tick=time.perf_counter();path=a.root/REFS['predictions'][0]/architecture/'evaluation.pt'
            trace_pin=file_sha(path);trajectories=torch.load(path,map_location='cpu',weights_only=True);outputs=[]
            if not negative:negative=controls(trajectories[0]['events'][0],state['specs'],state['config'],dict(state['adapter'],trace_pin=trace_pin))
            for ordinal,record in enumerate(trajectories):
                examples,indices,inputs,coverage=adapted_events(record,state,trace_pin)
                scores,info=evaluate(model,examples,'reference',None) if examples else ([],{})
                r=record['row'];row={k:r[k] for k in ['document','mode','proposal','side','component']}
                row.update(architecture=architecture,ordinal=ordinal,coverage=coverage,metrics=metrics(scores,indices,state))
                rows.append(row);outputs.append(dict(row=row,indices=indices,inputs=inputs,scores=scores,info=info))
            sequence.runtime.persist(a.output/(architecture+'.pt'),outputs)
            phases[architecture]=time.perf_counter()-tick
            print(json.dumps(dict(architecture=architecture,trajectories=len(outputs),seconds=phases[architecture])),flush=True)
    tick=time.perf_counter();state['fresh']();need(initial==fingerprint(model.state_dict()),'ANSWER_SURFACE_DECODER_MUTATION')
    need(not torch.cuda.is_initialized(),'ANSWER_SURFACE_UNEXPECTED_CUDA')
    write_committed(a.output/'controls.json',negative);write_committed(a.output/'rows.json',rows)
    cells=summarize(rows);write_committed(a.output/'cells.json',cells)
    for n,h in state['code'].items():
        p=a.output/'source'/n;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((Path(__file__).resolve().parents[1]/n).read_bytes());need(file_sha(p)==h,'ANSWER_SURFACE_SOURCE_COPY')
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['seal']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(sum(phases.values())/seconds>=.9,'ANSWER_SURFACE_PHASE_COVERAGE')
    report=dict(complete=True,files=files,code=state['code'],sources=REFS,decoder_state=initial,
        region=state['adapter'],reference_atoms=len(state['query']),trajectories=len(rows),cells=len(cells),
        phases=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        metric='teacher_forced_byte_nll; per_atom_EOS_separate_from_answer_END',cpu_qualification_only=True,
        free_generation=False,semantic_correctness=False,scientific_admission=False,serving=False,
        number_status='measured',acceptance='author-checked')
    pin=write_committed(a.output/'report.json',report);print(json.dumps(dict(pin=pin,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu-proof',action='store_true');a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(exc)))
        raise
