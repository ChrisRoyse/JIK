"""Cold replay using original forward arithmetic and an independent byte loop."""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import trained_answer_render as producer
from audit.answer_surface_stream_readback import native, logits, baseline_atom, check_teacher
from claims.process_conditions import write_committed
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha, need


def answer(model,r,state,limit):
    events=[];decisions=0;completed=0
    for index,e in enumerate(r['events']):
        if e['kind']=='end':
            need(index==len(r['events'])-1 and e['generated']==completed and int(e['stop_logits'].argmax())==1,
                 'TRAINED_COLD_END')
            events.append(dict(kind='answer_end',atoms=completed,proposal=r['proposal'],validity='unassessed',serving=False))
            return dict(events=events,decisions=decisions,censored=False)
        need(e['index']==completed,'TRAINED_COLD_ORDER')
        region=state['binding']['region_pin']+':'+'.'.join(map(str,e['codes'][0].tolist()))
        events.append(dict(kind='atom_start',index=completed,proposal=r['proposal'],regions=(region,),
                           validity='unassessed',serving=False))
        generated=baseline_atom(model,native(e,state,r['trace_pin']),limit-decisions)
        events.extend(dict({k:v for k,v in row.items() if k!='logits'},atom=completed,proposal=r['proposal'])
                      for row in generated['events'])
        decisions+=generated['decisions']
        if generated['censored'] or generated['events'][-1]['kind']=='refuse':
            return dict(events=events,decisions=decisions,censored=generated['censored'])
        completed+=1
        if decisions==limit:return dict(events=events,decisions=decisions,censored=True)
    return dict(events=events,decisions=decisions,censored=False,source_error='MNLM_ANSWER_RENDER_MISSING_END')


def recount(saved):
    correct=positions=0;loss=0.
    for rows in saved['teacher']:
        for row in rows:
            value=row['logits'].double()
            loss-=float(torch.log_softmax(value,dim=0)[row['target']])
            correct+=int(int(value.argmax())==row['target']);positions+=1
    out=dict(teacher_positions=positions,teacher_correct=correct,teacher_nll_sum=loss,teacher_mean_nll=loss/positions)
    for key in ('free','bridges'):
        out[key]=[]
        for run in saved[key]:
            need(run['decisions']==sum('token' in e for e in run['events']),'TRAINED_COLD_DECISIONS')
            need(not run['censored'] or all(e['kind']!='answer_end' for e in run['events']),'TRAINED_COLD_FALSE_END')
            row=run['events'][-1]
            out[key].append(dict(decisions=run['decisions'],censored=run['censored'],last=row['kind'],
                code=run.get('source_error',row.get('code')),bytes=sum(e['kind']=='byte' for e in run['events']),
                atoms=sum(e['kind']=='atom_end' for e in run['events'])))
    return out


def run(a):
    a.output.mkdir(exist_ok=False);began=time.perf_counter();cpu=time.process_time()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    need(file_sha(a.source/'report.json')==a.pin,'TRAINED_COLD_PIN')
    report=json.loads((a.source/'report.json').read_bytes());base=Path(__file__).resolve().parents[1]
    for name,pin in report['code'].items():need(file_sha(base/name)==pin,'TRAINED_COLD_CODE')
    for name,pin in report['files'].items():need(file_sha(a.source/name)==pin,'TRAINED_COLD_FILE')
    state=producer.load(a.root)
    need(state['trained'].receipt==report['decoder'],'TRAINED_COLD_DECODER')
    counts={};summaries={}
    arms=[('earlier_proof',state['model']),('corpus_trained',state['trained'].model)]
    if a.reverse:arms.reverse()
    for label,model in arms:
        saved=torch.load(a.source/(label+'.pt'),map_location='cpu',weights_only=True)
        need(len(saved['teacher'])==len(saved['free'])==16 and len(saved['bridges'])==8,'TRAINED_COLD_DENOMINATOR')
        local=dict(state,model=model);positions=free_positions=0
        indices=list(range(16))
        if a.reverse:indices.reverse()
        for ordinal in indices:
            r=state['inputs'][ordinal];rows=saved['teacher'][ordinal]
            slots=native(r['event'],local,r['trace_pin']);check_teacher(rows,r,local)
            for row in (reversed(rows) if a.reverse else rows):
                value=logits(model,slots,row['prefix'],row['offset'])
                need(torch.equal(value,row['logits']),'TRAINED_COLD_LOGITS');positions+=1
            generated=baseline_atom(model,slots,96)
            need(fingerprint(generated)==fingerprint(saved['free'][ordinal]),'TRAINED_COLD_FREE')
            free_positions+=generated['decisions']
            print(json.dumps(dict(decoder=label,atom=ordinal,positions=positions)),flush=True)
        for ordinal in (reversed(range(8)) if a.reverse else range(8)):
            r=state['proposals'][ordinal];value=answer(model,r,local,96)
            need(fingerprint(dict(architecture=r['architecture'],proposal=r['proposal'],**value))
                 ==fingerprint(saved['bridges'][ordinal]),'TRAINED_COLD_ANSWER')
        summaries[label]=recount(saved)
        for key,value in summaries[label].items():
            target=report['summary'][label][key]
            need(math.isclose(value,target,rel_tol=1e-14,abs_tol=1e-12) if type(value) is float else value==target,
                 'TRAINED_COLD_SUMMARY')
        counts[label]=dict(teacher_positions=positions,free_positions=free_positions)
    state['fresh']();state['trained'].close();need(not torch.cuda.is_initialized(),'TRAINED_COLD_GPU')
    result=dict(complete=True,source=dict(path=str(a.source),pin=a.pin),acceptance='author-checked',number_status='measured',
        reverse=a.reverse,all_original_forward_logits_exact=True,all_free_paths_exact=True,all_answer_paths_exact=True,
        source_and_weights_unchanged=True,summary=summaries,counts=counts,code=report['code'],
        seconds=time.perf_counter()-began,cpu_seconds=time.process_time()-cpu,
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,independent_adjudication=False,
        semantic_validity_established=False,promoted_to_serving=False)
    pin=write_committed(a.output/'report.json',result);print(json.dumps(dict(complete=True,pin=pin,counts=counts)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=producer.ROOT)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--reverse',action='store_true');p.add_argument('--output',type=Path,required=True);run(p.parse_args())
