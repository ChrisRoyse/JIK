"""Real retained prediction streams through the larger decoder; CPU engineering only."""
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
from audit import trained_answer_render as source
from audit.answer_surface_stream import rejected
from audit.decoder_prefix_cohort import code,save
from claims.process_conditions import write_committed
from model.answer_surface_input import predicted_atom
from model.answer_surface_stream import SurfaceStepper,render_answer as direct_answer
from model.native_population_training import fingerprint
from model.protected_answer_render import TrainedProtectedDecoder,render_answer
from model.protected_surface_stream import ProtectedSurfaceStepper
from model.stream_surface import BOS,EOS,schedule
from text.constellation_corpus import file_sha,need

PRODUCER=('protected-surface-cpu-v2-20261025','3886e234652885db6725b0f526697c9d8d9b2f64347edaee3dcc096180f2ef0e')
COLD=('protected-surface-cold-v2-20261025','b5861d5abe725a580946ebc5e0e17a140a68f5dd94b53fe3af5d585a11b881c6')


def load(root):
    s=source.load(root)
    adapter=TrainedProtectedDecoder(s['trained'],root/PRODUCER[0],PRODUCER[1],root/COLD[0],COLD[1],cpu_diagnostic=True)
    return s,adapter


def teacher(s,model,*,direct=False):
    output=[];timings=[]
    for r in s['inputs']:
        native=predicted_atom(r['event'],model.specs,model.config,**s['binding'],trace_pin=r['trace_pin'],position=r['position']).slots
        need(fingerprint({n:v.values[0] for n,v in native.items()})==r['native'],'PROTECTED_ANSWER_NATIVE')
        raw=bytes(r['reference'].tolist());rows=[]
        cached=None if direct else ProtectedSurfaceStepper(model,native)
        plain=SurfaceStepper(model,native,cached=False);elapsed=dict(cached=0.,direct=0.);positions=0
        try:
            for part in schedule(raw,width=model.config.max_prefix):
                for position in range(*part['scored']):
                    prefix=part['prefix'][:position-part['offset']+1];logits={}
                    order=['direct'] if direct else (['direct','cached'] if position%2 else ['cached','direct'])
                    for key in order:
                        start=time.perf_counter();logits[key]=(plain if key=='direct' else cached).logits(prefix,part['offset'])
                        elapsed[key]+=time.perf_counter()-start
                    if not direct:need(torch.equal(logits['cached'],logits['direct']),'PROTECTED_ANSWER_LOGITS')
                    rows.append(dict(position=position,offset=part['offset'],prefix=prefix,
                                     target=raw[position] if position<len(raw) else EOS,logits=logits['direct']))
                    positions+=1
            if not direct:
                need(cached.parent.projection_calls==cached.correction.projection_calls==1,'PROTECTED_ANSWER_CACHE')
            timings.append(dict(seconds=elapsed,positions=positions,direct_calls=plain.calls,
                                cached_projections=None if direct else [cached.parent.projection_calls,cached.correction.projection_calls]))
        finally:
            plain.close()
            if cached is not None:cached.close()
        output.append(rows)
    need(sum(len(x) for x in output)==1352,'PROTECTED_ANSWER_TEACHER_DENOMINATOR')
    return output,timings


def bridges(s,adapter,*,direct=False):
    output=[];timings=[]
    for ordinal,r in enumerate(s['proposals']):
        result={};elapsed={}
        order=['direct'] if direct else (['direct','cached'] if ordinal%2 else ['cached','direct'])
        for key in order:
            adapter.fresh();start=time.perf_counter()
            binding=dict(s['binding'],trace_pin=r['trace_pin'])
            stream=(direct_answer(iter(r['events']),adapter.model,cached=False,**binding) if key=='direct'
                    else adapter.render(iter(r['events']),**binding))
            result[key]=source.observe(stream,96,atom_only=False)
            elapsed[key]=time.perf_counter()-start;adapter.fresh()
        if not direct:need(fingerprint(result['cached'])==fingerprint(result['direct']),'PROTECTED_ANSWER_STREAM')
        output.append(dict(architecture=r['architecture'],proposal=r['proposal'],**result['direct']));timings.append(elapsed)
        print(json.dumps(dict(proposal=r['proposal'],censored=result['direct']['censored'],
                              last=result['direct']['events'][-1]['kind'])),flush=True)
    return output,timings


def controls(s,adapter):
    model=adapter.model;r=s['inputs'][0];binding=dict(s['binding'],trace_pin=r['trace_pin']);out={}
    bad=deepcopy(r['event']);bad['index']=1
    out['order']=rejected(lambda:next(render_answer([bad],model,**binding)),'ANSWER_SURFACE_POSITION')
    bad=deepcopy(r['event']);bad['raw']=b'forbidden target'
    out['future_target']=rejected(lambda:next(render_answer([bad],model,**binding)),'ANSWER_SURFACE_EVENT_FIELDS')
    bad=deepcopy(r['event']);bad['slots'].pop(next(iter(bad['slots'])))
    out['missing_slot']=rejected(lambda:next(render_answer([bad],model,**binding)),'ANSWER_SURFACE_ROSTER')
    out['missing_end']=rejected(lambda:next(render_answer([],model,**binding)),'ANSWER_RENDER_MISSING_END')
    # Altered/synthetic ENDs are rejection probes only, never positive capability evidence.
    end=dict(kind='end',proposal='negative-control',generated=0,stop_logits=torch.tensor([[-1.,1.]]),validity='unassessed')
    bad=deepcopy(end);bad['generated']=1
    out['end_count']=rejected(lambda:list(render_answer([bad],model,**binding)),'ANSWER_RENDER_END')
    bad=deepcopy(end);bad['stop_logits']=torch.tensor([[1.,-1.]])
    out['false_end']=rejected(lambda:list(render_answer([bad],model,**binding)),'ANSWER_RENDER_END')
    out['trailing_end']=rejected(lambda:list(render_answer([end,None],model,**binding)),'ANSWER_RENDER_AFTER_END')
    closed=[]
    def actual_source():
        try:yield from s['proposals'][0]['events']
        finally:closed.append(True)
    b=dict(s['binding'],trace_pin=s['proposals'][0]['trace_pin']);g=adapter.render(actual_source(),**b)
    need(next(g)['kind']=='atom_start','PROTECTED_ANSWER_REAL_START');next(g)
    answer=g.gi_yieldfrom;atom=answer.gi_frame.f_locals['surface'];step=atom.gi_frame.f_locals['stepper']
    g.close()
    need(closed==[True] and step.closed and step.model is None and step.parent.closed and step.correction.closed
         and all(x.memories is None and x.native is None and x.model is None for x in (step.parent,step.correction)),
         'PROTECTED_ANSWER_CANCEL_RELEASE')
    out['closed_cache']=rejected(lambda:step.logits([BOS],0),'PROTECTED_STREAM_CLOSED')
    out['cancel_released']=True
    native=predicted_atom(r['event'],model.specs,model.config,**binding,position=0).slots
    cache=ProtectedSurfaceStepper(model,native);p=next(model.correction.parameters());old=p.detach().clone()
    with torch.no_grad():p.add_(1)
    out['stale_adapter']=rejected(adapter.fresh,'PROTECTED_RENDER_WEIGHTS_CHANGED')
    out['stale_cache']=rejected(lambda:cache.logits([BOS],0),'SURFACE_STREAM_WEIGHTS_CHANGED')
    with torch.no_grad():p.copy_(old)
    out['restored_but_stale']=rejected(adapter.fresh,'PROTECTED_RENDER_WEIGHTS_CHANGED');cache.close()
    adapter.close();out['closed_adapter']=rejected(adapter.fresh,'PROTECTED_RENDER_CLOSED')
    s['fresh']();return out


def run(a):
    began=tick=time.perf_counter();c0=time.process_time();a.output.mkdir(exist_ok=False)
    s,adapter=load(a.root);phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    files={'inputs.pt':save(a.output/'inputs.pt',dict(inputs=s['inputs'],proposals=s['proposals'],binding=s['binding']))}
    t,tt=teacher(s,adapter.model);b,bt=bridges(s,adapter)
    files['outputs.pt']=save(a.output/'outputs.pt',dict(teacher=t,bridges=b))
    phases['matched_rendering']=time.perf_counter()-tick;tick=time.perf_counter()
    receipt=adapter.receipt;consumed=dict(adapter.consumed)
    consumed.update({str(p):file_sha(p) for p in s['stamps']})
    negatives=controls(s,adapter);s['trained'].close();need(not torch.cuda.is_initialized(),'PROTECTED_ANSWER_CPU_ONLY')
    phases['controls_freshness']=time.perf_counter()-tick
    report=dict(complete=True,receipt=receipt,consumed=consumed,files=files,code=code(),phases=phases,
                teacher_timings=tt,bridge_timings=bt,teacher_positions=sum(len(x) for x in t),proposals=len(b),
                atoms=len(t),controls=negatives,cached_direct_exact=True,summary=source.summary(t,[],b),
                wall=time.perf_counter()-began,cpu=time.process_time()-c0,rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                cuda_qualified=False,promoted=False,capability_gain_demonstrated=False,acceptance='author-checked')
    pin=write_committed(a.output/'report.json',report);print(json.dumps(dict(pin=pin,wall=report['wall'])),flush=True)


def cold(a):
    began=tick=time.perf_counter();c0=time.process_time();a.output.mkdir(exist_ok=False)
    need(file_sha(a.input/'report.json')==a.pin,'PROTECTED_ANSWER_REPORT');r=json.loads((a.input/'report.json').read_bytes())
    base=Path(__file__).resolve().parents[1]
    need(all(file_sha(base/n)==h for n,h in r['code'].items()),'PROTECTED_ANSWER_CODE')
    need(all(file_sha(p)==h for p,h in r['consumed'].items()),'PROTECTED_ANSWER_SOURCE_PIN')
    def saved(name):
        path=a.input/name;ref=r['files'][name];need(file_sha(path)==ref['sha256'],'PROTECTED_ANSWER_FILE')
        value=torch.load(path,map_location='cpu',weights_only=True);need(fingerprint(value)==ref['state'],'PROTECTED_ANSWER_PAYLOAD');return value
    s,adapter=load(a.root)
    need(adapter.receipt==r['receipt'] and fingerprint(saved('inputs.pt'))==fingerprint(dict(inputs=s['inputs'],proposals=s['proposals'],binding=s['binding'])),
         'PROTECTED_ANSWER_INPUT_READBACK')
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    t,_=teacher(s,adapter.model,direct=True);b,_=bridges(s,adapter,direct=True)
    need(fingerprint(dict(teacher=t,bridges=b))==fingerprint(saved('outputs.pt')),'PROTECTED_ANSWER_OUTPUT_READBACK')
    need(source.summary(t,[],b)==r['summary'],'PROTECTED_ANSWER_DENOMINATORS')
    adapter.fresh();s['fresh']();adapter.close();s['trained'].close()
    need(not torch.cuda.is_initialized(),'PROTECTED_ANSWER_COLD_CPU_ONLY');phases['direct_replay']=time.perf_counter()-tick
    result=dict(complete=True,input_pin=a.pin,source_and_weights_exact=True,all_teacher_logits_exact=True,
                all_full_stream_events_exact=True,denominators_recounted=True,code=code(),phases=phases,
                wall=time.perf_counter()-began,cpu=time.process_time()-c0,rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                cuda_qualified=False,promoted=False,acceptance='author-checked')
    pin=write_committed(a.output/'report.json',result);print(json.dumps(dict(pin=pin,wall=result['wall'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['run','cold'],required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input',type=Path);p.add_argument('--pin');a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    try:(run if a.mode=='run' else cold)(a)
    except BaseException as e:
        if a.output.is_dir():write_committed(a.output/'failure.json',dict(error=type(e).__name__,message=str(e)))
        raise
