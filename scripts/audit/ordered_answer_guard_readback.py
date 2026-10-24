"""Separate byte/order/score readback without invoking OrderedAnswerGuard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import ordered_answer_guard_source as source
from audit.ordered_answer_guard import geometry_predictions
from audit.capability_surface_guard_fsv import independent_scores
from audit.answer_surface_stream_readback import bridge
from claims.process_conditions import write_committed
from claims.episodic_memory import digest as record_digest
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha,digest,need


def run(a):
    a.output.mkdir(exist_ok=False); began=tick=time.perf_counter(); phases={}
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    need(file_sha(a.input/'report.json')==a.pin,'ORDERED_COLD_PIN')
    report=json.loads((a.input/'report.json').read_bytes());need(report['complete'],'ORDERED_COLD_COMPLETE')
    for n,h in report['files'].items():need(file_sha(a.input/n)==h,'ORDERED_COLD_BYTES')
    s=source.load(a.root);need(s['code']==report['code'],'ORDERED_COLD_CODE')
    saved=torch.load(a.input/'control-source.pt',map_location='cpu',weights_only=True)
    need(fingerprint(saved['predictions'])==fingerprint(s['trace']['events']) and
         saved['surface']==source.observed_events(s),'ORDERED_COLD_SOURCE')
    phases['authentication']=time.perf_counter()-tick;tick=time.perf_counter();counts={}
    for origin,predictions in [('observed_surface_control',s['trace']['events']),('observed_geometry_control',geometry_predictions(s))]:
        results=json.loads((a.input/(origin+'.json')).read_bytes()); chain=hashlib.sha256(); errors=[]; total=0; passing=[]
        for i,(e,prediction,row) in enumerate(zip(s['examples'],predictions,results[:-1])):
            raw=bytes(saved['raw'][i].tolist());need(raw==e.raw,'ORDERED_COLD_RAW')
            measured,receipt=s['measure'](raw);need(saved['receipts'][i]==receipt,'ORDERED_COLD_RECEIPT')
            need(row['index']==i and row['prediction']==fingerprint(prediction) and row['origin']==origin,'ORDERED_COLD_ORDER')
            guard=row['guard'];need(guard['receipt']==record_digest(receipt) and guard['source_sha256']==digest(raw),'ORDERED_COLD_GUARD_BINDING')
            for n,values in measured.items():
                need(torch.equal(torch.from_numpy(values),saved['native'][i][n]),'ORDERED_COLD_NATIVE')
                errors.append(independent_scores(values,prediction['slots'][n].numpy(),guard['slots'][n]))
                need(guard['slots'][n]['passes']==(guard['slots'][n]['score']>=.7),'ORDERED_COLD_THRESHOLD')
            conjunction=all(v['passes'] for v in guard['slots'].values());passing.append(conjunction)
            need(conjunction==guard['geometric_conjunction'] and not guard['accepted'] and not row['serving'],'ORDERED_COLD_ATOM_GATE')
            chain.update(bytes.fromhex(fingerprint(row)));total+=len(raw)
        end=results[-1];need(len(results)==4 and s['trace']['events'][-1]['generated']==3 and end['atoms']==3
             and end['bytes']==total and end['sequence']==chain.hexdigest() and end['geometric_conjunction']==all(passing)
             and end['structural_complete'] and not end['accepted'] and not end['semantic_validity_established'], 'ORDERED_COLD_END')
        counts[origin]=dict(atoms=3,bytes=total,maximum_score_difference=max(errors),geometric=all(passing))
    phases['ordered_score_readback']=time.perf_counter()-tick;tick=time.perf_counter()
    rows=torch.load(a.input/'generated.pt',map_location='cpu',weights_only=True)
    source_rows={}
    from audit.answer_surface_stream import REFS
    for arch in ('shared','own_slot','whole_proposals'):
        path=a.root/REFS['predictions'][0]/arch/'evaluation.pt'
        for r in torch.load(path,map_location='cpu',weights_only=True):
            if r['document']==22924 and r['mode']=='correct':source_rows[arch,r['proposal']]=dict(r,trace_pin=file_sha(path))
    for row in rows:
        r=source_rows[row['architecture'],row['proposal']]
        replay=bridge(s['model'],r,dict(specs=s['state']['specs'],binding=s['binding']),96)
        # The new guard detects invalid UTF-8 on the byte before the old renderer emits its refusal.
        expected=replay['events'][:len(row['surface_events'])]
        need(fingerprint(expected)==fingerprint(row['surface_events']),'ORDERED_COLD_FREE_RENDER')
        raw=bytes(e['value'] for e in expected if e['kind']=='byte')
        try:raw.decode('utf-8',errors='strict')
        except UnicodeDecodeError:pass
        else:raise ValueError('ORDERED_COLD_EXPECTED_INVALID_UTF8')
        need(len(row['guard'])==1 and row['guard'][0]['code']=='ANSWER_GUARD_UTF8'
             and row['guard'][0]['measurement_calls']==0 and not row['guard'][0]['accepted'] and not row['censored'],
             'ORDERED_COLD_FREE_REFUSAL')
    need(len(rows)==8,'ORDERED_COLD_FREE_CENSUS');phases['free_render_replay']=time.perf_counter()-tick;tick=time.perf_counter()
    s['fresh']();need(not torch.cuda.is_initialized(),'ORDERED_COLD_GPU');phases['source_fresh']=time.perf_counter()-tick
    seconds=time.perf_counter()-began;need(.9<=sum(phases.values())/seconds<=1.001,'ORDERED_COLD_PHASES')
    result=dict(complete=True,source=dict(path=str(a.input),pin=a.pin),code=s['code'],controls=counts,generated_utf8_refusals=8,
        seconds=seconds,phases=phases,acceptance='author-checked',scientific_admission=False,capability_demonstrated=False)
    pin=write_committed(a.output/'report.json',result);print(json.dumps(dict(pin=pin,seconds=seconds,controls=counts)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=source.ROOT)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
