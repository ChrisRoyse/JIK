"""Cold numeric replay through the original uncached StreamSurfaceDecoder."""
import argparse
import codecs
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import answer_surface_stream as proof
from claims.process_conditions import write_committed
from model.native import NativeInput
from model.native_population_training import fingerprint
from model.stream_surface import BOS, EOS
from text.constellation_corpus import canonical, digest, file_sha, need


def native(event,state,trace):
    """Rebuild directly from persisted predictor outputs, without the adapter or cache."""
    row=digest(canonical(dict(trace=trace,proposal=event['proposal'],position=event['index'])))
    result={}
    for s in state['specs']:
        v=event['slots'][s.name]
        need(int(event['counts'][s.name][0])==len(v)>0 and v.shape[1]==s.dims,'STREAM_COLD_NATIVE')
        result[s.name]=NativeInput(s.identity,(row,),v.clone()[None],torch.zeros((1,len(v)),dtype=torch.bool),
                                  torch.zeros(1,dtype=torch.bool))
    return result


def logits(model,slots,prefix,offset):
    with torch.no_grad():
        return model(torch.tensor([prefix],dtype=torch.int64),slots,
                     offsets=torch.tensor([offset],dtype=torch.int64))['logits'][0,-1].detach().clone()


def baseline_atom(model,slots,limit):
    raw=bytearray();events=[];utf8=codecs.getincrementaldecoder('utf-8')('strict');valid=True
    for position in range(limit):
        window_start=(position//16)*16
        left=max(0,window_start-model.config.max_prefix+16)
        prefix=([BOS]+list(raw))[left:position+1]
        score=logits(model,slots,prefix,left);token=int(torch.argmax(score))
        row=dict(position=position,token=token,offset=left,prefix=prefix,
                 validity='unassessed',serving=False,logits=score)
        if token==EOS:
            try:utf8.decode(b'',final=True);valid=len(raw)>0
            except UnicodeDecodeError:valid=False
            events.append(dict(row,kind='atom_end' if valid else 'refuse',
                code=None if valid else ('SURFACE_STREAM_EMPTY' if not raw else 'SURFACE_STREAM_UTF8'),
                bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
            return dict(events=events,decisions=position+1,censored=False)
        if token>=256:
            events.append(dict(row,kind='refuse',code='SURFACE_STREAM_SYMBOL',bytes=len(raw),
                               sha256=hashlib.sha256(raw).hexdigest()))
            return dict(events=events,decisions=position+1,censored=False)
        raw.append(token)
        try:utf8.decode(bytes([token]),final=False)
        except UnicodeDecodeError:valid=False
        events.append(dict(row,kind='byte',value=token,utf8_prefix_valid=valid))
        if not valid:
            events.append(dict(kind='refuse',code='SURFACE_STREAM_UTF8',bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),validity='unassessed',serving=False))
            return dict(events=events,decisions=position+1,censored=False)
    return dict(events=events,decisions=limit,censored=True)


def check_teacher(saved,r,state):
    raw=bytes(r['reference'].tolist());tokens=[BOS,*raw,EOS];m=state['model']
    need(len(saved)==len(raw)+1,'STREAM_COLD_TAIL')
    for i,row in enumerate(saved):
        left=max(0,16*(i//16)-(m.config.max_prefix-16))
        need(row['position']==i and row['offset']==left and row['prefix']==tokens[left:i+1]
             and row['target']==tokens[i+1],'STREAM_COLD_PREFIX')


def bridge(model,r,state,limit):
    result=[];decisions=0;completed=0
    for e in r['events']:
        if e['kind']=='end':
            need(e['generated']==completed and int(e['stop_logits'].argmax(-1))==1,'STREAM_COLD_END')
            result.append(dict(kind='answer_end',atoms=completed,proposal=r['proposal'],
                               validity='unassessed',serving=False))
            return dict(events=result,decisions=decisions,censored=False)
        need(e['index']==completed,'STREAM_COLD_ORDER')
        region=state['binding']['region_pin']+':'+'.'.join(str(v) for v in e['codes'][0].tolist())
        result.append(dict(kind='atom_start',index=completed,proposal=r['proposal'],regions=(region,),
                           validity='unassessed',serving=False))
        output=baseline_atom(model,native(e,state,r['trace_pin']),limit-decisions)
        for event in output['events']:
            item={k:v for k,v in event.items() if k!='logits'}
            result.append(dict(item,atom=completed,proposal=r['proposal']))
        decisions+=output['decisions']
        if output['censored'] or output['events'][-1]['kind']=='refuse':
            return dict(events=result,decisions=decisions,censored=output['censored'])
        completed+=1
        if decisions==limit:return dict(events=result,decisions=decisions,censored=True)
    raise ValueError('STREAM_COLD_MISSING_END')


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    need(file_sha(a.source/'report.json')==a.pin,'STREAM_COLD_PIN')
    report=json.loads((a.source/'report.json').read_bytes())
    for n,h in report['files'].items():need(file_sha(a.source/n)==h,'STREAM_COLD_FILE')
    state=proof.load(a.root);need(state['code']==report['code'],'STREAM_COLD_CODE')
    need(state['binding']==dict(report['binding'],region_sizes=tuple(report['binding']['region_sizes'])),
         'STREAM_COLD_BINDING')
    def read(n):return torch.load(a.source/n,map_location='cpu',weights_only=True)
    inputs=read('inputs.pt');teacher=read('teacher.pt');free=read('free.pt');proposals=read('proposals.pt')
    need(fingerprint(inputs)==fingerprint(state['inputs']) and fingerprint(proposals)==fingerprint(state['proposals']),
         'STREAM_COLD_SOURCE_JOIN')
    need(fingerprint(read('decoder.pt'))==fingerprint(state['model'].state_dict())==report['decoder_state'],
         'STREAM_COLD_WEIGHTS')
    need(len(inputs)==len(teacher)==len(free)==16,'STREAM_COLD_DENOMINATOR')
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter();positions=0;free_positions=0
    for ordinal,(r,rows,saved) in enumerate(zip(inputs,teacher,free)):
        slots=native(r['event'],state,r['trace_pin'])
        need(fingerprint({n:v.values[0] for n,v in slots.items()})==r['native'],'STREAM_COLD_ADAPTER')
        check_teacher(rows,r,state)
        for row in rows:
            value=logits(state['model'],slots,row['prefix'],row['offset'])
            need(torch.equal(value,row['logits']),'STREAM_COLD_LOGIT');positions+=1
        generated=baseline_atom(state['model'],slots,report['observer_decisions'])
        need(fingerprint(generated)==fingerprint(saved),'STREAM_COLD_FREE')
        free_positions+=generated['decisions']
        print(json.dumps(dict(atom=ordinal,positions=positions,free_positions=free_positions)),flush=True)
    phases['independent_all_logits_and_free']=time.perf_counter()-tick;tick=time.perf_counter()
    bridges=read('bridges.pt');need(len(bridges)==len(proposals)==8,'STREAM_COLD_BRIDGES')
    for r,saved in zip(proposals,bridges):
        value=bridge(state['model'],r,state,report['observer_decisions'])
        need(fingerprint(dict(architecture=r['architecture'],proposal=r['proposal'],**value))==fingerprint(saved),
             'STREAM_COLD_BRIDGE')
    bad=read('teacher.pt')[0];bad[0]['prefix']=[0]
    controls=dict(wrong_prefix=proof.rejected(lambda:check_teacher(bad,inputs[0],state),'STREAM_COLD_PREFIX'),
        missing_tail=proof.rejected(lambda:check_teacher(teacher[0][:-1],inputs[0],state),'STREAM_COLD_TAIL'),
        stale_pin=proof.rejected(lambda:need(file_sha(a.source/'report.json')=='0'*64,'STREAM_COLD_PIN'),'STREAM_COLD_PIN'))
    state['fresh']();need(not torch.cuda.is_initialized(),'STREAM_COLD_GPU')
    need(positions==1352,'STREAM_COLD_POSITIONS')
    need(fingerprint(state['model'].state_dict())==report['decoder_state'],'STREAM_COLD_MODEL_CHANGED')
    phases['bridges_controls_and_state']=time.perf_counter()-tick
    result=dict(complete=True,source=dict(path=str(a.source),pin=a.pin),code=state['code'],
        acceptance='author-checked',number_status='measured',all_original_forward_logits_exact=True,
        all_free_paths_exact=True,all_bridge_paths_exact=True,teacher_positions=positions,
        free_positions=free_positions,controls=controls,phases=phases,seconds=time.perf_counter()-began,
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        capability_demonstrated=False,scientific_admission=False,promoted_to_serving=False)
    pin=write_committed(a.output/'report.json',result);print(json.dumps(dict(complete=True,pin=pin)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=proof.authority.ROOT)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
