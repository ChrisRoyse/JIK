"""Real free rendering plus explicitly observed controls of the ordered guard."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import ordered_answer_guard_source as source
from audit.answer_surface_stream import REFS, rejected
from audit.predicted_surface_training import persist
from claims.process_conditions import write_committed
from model.answer_surface_stream import render_answer
from model.ordered_answer_guard import OrderedAnswerGuard
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha, need


def consumer(s, predictions=None, *, origin='observed_surface_control', measure=None, buffer_bytes=16<<20, trace=None):
    return OrderedAnswerGuard(s['trace']['events'] if predictions is None else predictions,
        s['state']['specs'],s['state']['config'],**s['binding'],trace_pin=s['trace_pin'] if trace is None else trace,
        measurements=s['measure'] if measure is None else measure,origin=origin,buffer_bytes=buffer_bytes)


def consume(c, events):
    out=[]
    try:
        for event in events:
            row=c.observe(event)
            if row is not None: out.append(row)
        c.finish()
        return out
    finally: c.close()


def geometry_predictions(s):
    events=deepcopy(s['trace']['events'])
    for event,e in zip(events,s['examples']):
        event['slots']={n:v.values[0].clone() for n,v in e.native.items()}
        event['counts']={n:torch.tensor([len(v)]) for n,v in event['slots'].items()}
    return events


def controls(s, events):
    out={}
    def check(changed=events, predictions=None, measure=None, limit=16<<20):
        return consume(consumer(s,predictions,measure=measure,buffer_bytes=limit),changed)
    bad=deepcopy(events); bad[0]['index']=1
    out['wrong_order']=rejected(lambda:check(bad),'ANSWER_GUARD_ATOM_ORDER')
    bad=deepcopy(events); bad[1]['proposal']='another-proposal'
    out['wrong_proposal']=rejected(lambda:check(bad),'ANSWER_GUARD_PROPOSAL')
    bad=deepcopy(events); bad[1]['value']=bad[1]['token']=32 if bad[1]['value']!=32 else 65
    out['changed_bytes']=rejected(lambda:check(bad),'ANSWER_GUARD_EOS_BYTES')
    out['missing_eos']=rejected(lambda:check([e for e in events if not(e['kind']=='atom_end' and e['atom']==2)]),
                                 'ANSWER_GUARD_MISSING_ATOM_EOS')
    out['missing_end']=rejected(lambda:check(events[:-1]),'ANSWER_GUARD_INCOMPLETE_STREAM')
    bad=deepcopy(events); bad[-1]['atoms']=2
    out['wrong_end_count']=rejected(lambda:check(bad),'ANSWER_GUARD_END_COUNT')
    out['extra_after_end']=rejected(lambda:check([*events,events[-1]]),'ANSWER_GUARD_AFTER_TERMINAL')
    out['prediction_after_end']=rejected(lambda:check(predictions=[*s['trace']['events'],s['trace']['events'][-1]]),
                                          'ANSWER_GUARD_PREDICTION_AFTER_END')
    out['buffer']=rejected(lambda:check(limit=1),'ANSWER_GUARD_PHYSICAL_BUFFER_EXHAUSTED')
    def missing(raw): raise ValueError('ANSWER_GUARD_MEASUREMENT_REQUIRED')
    out['missing_measurement']=rejected(lambda:check(measure=missing),'ANSWER_GUARD_MEASUREMENT_REQUIRED')
    def damaged(raw):
        values,receipt=s['measure'](raw); values[next(iter(values))][0,0]+=1; return values,receipt
    out['native_corruption']=rejected(lambda:check(measure=damaged),'SURFACE_GUARD_NATIVE_BYTES')
    def absent(raw):
        values,receipt=s['measure'](raw); values.pop(next(iter(values))); return values,receipt
    out['missing_slot']=rejected(lambda:check(measure=absent),'SURFACE_GUARD_MISSING_SLOT')
    def identity(raw):
        values,receipt=s['measure'](raw); receipt=deepcopy(receipt); receipt['instruments'][next(iter(values))]='0'*64
        return values,receipt
    out['wrong_identity']=rejected(lambda:check(measure=identity),'SURFACE_GUARD_INSTRUMENT')
    from model.capability_surface_guard import evaluate,POLICY
    raw=s['examples'][0].raw; values,receipt=s['measure'](raw)
    out['threshold_change']=rejected(lambda:evaluate(raw,values,values,{v.name:v.identity for v in s['state']['specs']},
        receipt,policy=dict(POLICY,threshold=.69)),'SURFACE_GUARD_POLICY')
    return out


def run(a):
    a.output.mkdir(exist_ok=False); began=tick=time.perf_counter(); phases={}
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    s=source.load(a.root); source_events=source.observed_events(s)
    persist(a.output/'control-source.pt',dict(predictions=s['trace']['events'],surface=source_events,
        raw=[torch.tensor(list(e.raw),dtype=torch.uint8) for e in s['examples']],
        native=[{n:v.values[0] for n,v in e.native.items()} for e in s['examples']],
        receipts=[s['measure'](e.raw)[1] for e in s['examples']]))
    phases['authentication']=time.perf_counter()-tick; tick=time.perf_counter()
    cases={}
    for origin,predictions in [('observed_surface_control',s['trace']['events']),('observed_geometry_control',geometry_predictions(s))]:
        result=consume(consumer(s,predictions,origin=origin),source_events)
        need(result[-1]['structural_complete'] and not result[-1]['accepted'],'ORDERED_GUARD_OUTER_GATE')
        if origin=='observed_geometry_control': need(result[-1]['geometric_conjunction'],'ORDERED_GUARD_GEOMETRY_CONTROL')
        write_committed(a.output/(origin+'.json'),result); cases[origin]=result[-1]
    phases['observed_controls']=time.perf_counter()-tick; tick=time.perf_counter(); generated=[]
    for arch in ('shared','own_slot','whole_proposals'):
        path=a.root/REFS['predictions'][0]/arch/'evaluation.pt'; pin=file_sha(path)
        for row in torch.load(path,map_location='cpu',weights_only=True):
            if row['document']!=22924 or row['mode']!='correct': continue
            stream=render_answer(row['events'],s['model'],**s['binding'],trace_pin=pin)
            c=consumer(s,row['events'],origin='generated',trace=pin); observed=[]; output=[]; censored=True
            try:
                # This observer stops inspection, never supplies a target length to the model.
                for _ in range(96):
                    event=next(stream); observed.append(event); decision=c.observe(event)
                    if decision is not None: output.append(decision)
                    if c.terminal: c.finish(); censored=False; break
            finally: stream.close(); c.close()
            generated.append(dict(architecture=arch,proposal=row['proposal'],prediction_trace=pin,
                surface_events=observed,guard=output,censored=censored))
    need(len(generated)==8,'ORDERED_GUARD_GENERATED_CENSUS'); persist(a.output/'generated.pt',generated)
    phases['free_render_guard']=time.perf_counter()-tick; tick=time.perf_counter()
    negative=controls(s,source_events); write_committed(a.output/'controls.json',negative); s['fresh']()
    need(not torch.cuda.is_initialized(),'ORDERED_GUARD_GPU_UNEXPECTED')
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['controls_hashes']=time.perf_counter()-tick; seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.001,'ORDERED_GUARD_PHASES')
    report=dict(complete=True,code=s['code'],files=files,source=source.PARENT,cold=source.COLD,cases=cases,
        generated_cases=8,generated=[dict(architecture=r['architecture'],proposal=r['proposal'],guard=r['guard'],censored=r['censored']) for r in generated],
        controls=negative,seconds=seconds,phases=phases,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        acceptance='author-checked',scientific_admission=False,serving=False,capability_demonstrated=False)
    pin=write_committed(a.output/'report.json',report); print(json.dumps(dict(pin=pin,seconds=seconds,cases=cases)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=source.ROOT)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
