"""Original-model cold replay and denominator recount for paired task scores."""
import argparse
from copy import deepcopy
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time

if '--cuda' not in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from torch.nn import functional as F
from audit import paired_surface_scoring as producer
from claims.process_conditions import write_committed as write
from lenscommon import take_lease,release_lease
from model import protected_corpus_training as training
from model import task_surface_scoring as original
from model.answer_surface_input import predicted_atom
from model.complete_answer_trace import replay
from model.contracts import ModelConfig
from model.corpus_protected_decoder import scope
from model.native_population_training import fingerprint
from model.protected_corpus_lineage import runtime
from model.stream_surface import EOS
from model.trained_surface_decoder import TrainedSurfaceDecoder
from text.constellation_corpus import digest,file_sha,need

ARMS=('parent','corrected')


def recount(value,raw):
    target=torch.tensor([*raw,EOS],dtype=torch.int64)
    need(value['raw_sha256']==digest(raw) and value['raw_bytes']==len(raw) and torch.equal(value['targets'],target)
         and value['logits'].shape==(len(raw)+1,258) and bool(value['logits'].isfinite().all()),'PAIRED_READBACK_POSITIONS')
    loss=F.cross_entropy(value['logits'].double(),target,reduction='none');hit=value['logits'].argmax(-1)==target
    need(torch.equal(loss,value['losses']) and torch.equal(hit,value['correct']),'PAIRED_READBACK_SCORES')
    return dict(nll=float(loss.sum()),correct=int(hit.sum()))


def read_pair(left,right,root):
    right_atoms={v['index']:v for v in right['atoms'] if v['status']=='scored'}
    positions=0;deltas={key:0. for key in ARMS}
    for a in left['atoms']:
        if a['status']!='scored' or a['index'] not in right_atoms:continue
        b=right_atoms[a['index']]
        need(a['reference']==b['reference'] and a['positions']==b['positions'],'PAIRED_READBACK_PAIR_REFERENCE')
        values=[]
        for arm,record in ((left,a),(right,b)):
            path=root/arm['path']/record['path'];need(file_sha(path)==record['pin'],'PAIRED_READBACK_PAIR_BYTES')
            values.append(torch.load(path,map_location='cpu',weights_only=True))
        for key in ARMS:deltas[key]+=float((values[1][key]['losses']-values[0][key]['losses']).sum())
        positions+=a['positions']
    need(left['reference_positions']==right['reference_positions'],'PAIRED_READBACK_PAIR_SCOPE')
    return dict(common_positions=positions,reference_positions=left['reference_positions'],
        excluded_positions=left['reference_positions']-positions,
        scores={key:dict(control_minus_correct_nll_sum=v,mean_control_minus_correct_nll=None if not positions else v/positions)
                for key,v in deltas.items()})


def run(a):
    producer.contract.owner_fd(a.owner_fd)
    began=tick=time.perf_counter();a.output.mkdir(exist_ok=False);parent=decoder=None;leased=False
    try:
        r=producer.previous.contract.read(a.input/'report.json',a.pin)
        need(r['complete'] and r['gpu_used']==a.cuda and r['qualification']==a.qualification
             and r['teacher_forced'] and not r['free_generation_tested'],
             'PAIRED_READBACK_SCOPE')
        for n,h in r['code'].items():need(file_sha(producer.BASE/n)==h,'PAIRED_READBACK_CODE')
        m,generated,cases,binding=producer.previous.authenticate(Path(r['request']['path']),r['request']['pin'],
            Path(r['generated']['path']),r['generated']['pin'],[0,1,3] if not a.cuda or a.qualification else None)
        producer.contract.worker_scope(m,cuda=a.cuda,qualification=a.qualification)
        need(r['full_query']==(a.cuda and not a.qualification) and r['full_training']==m['full_training'],
             'PAIRED_READBACK_DEVICE')
        schedule=producer.AnswerSchedule(binding['plan']['path'],binding['plan']['pin'])
        texts={v['id']:v['text'] for v in schedule.plan.records}
        if a.cuda:producer.previous.gpu_ready(a.output);take_lease('29','paired-surface-readback');leased=True
        args=[]
        for path,pin in producer.previous.DECODER:args.extend((producer.P/path,pin))
        parent=TrainedSurfaceDecoder(*args,device='cuda:0' if a.cuda else 'cpu',cpu_diagnostic=not a.cuda)
        cfg=r['corpus'];p=producer.previous.contract.read(Path(cfg['plan'])/'report.json',cfg['plan_pin'])
        run=producer.previous.contract.read(Path(cfg['run'])/'report.json',cfg['run_pin'])
        c=producer.previous.contract.read(Path(cfg['readback'])/'report.json',cfg['readback_pin'])
        ref=scope(parent.receipt,p,run,c,plan_pin=cfg['plan_pin'],run_pin=cfg['run_pin'],full=a.cuda,device=m['device'])
        need(runtime(m['device'])==run['runtime'] and r['decoder']['parent']==parent.receipt
             and r['decoder']['checkpoint']==ref['pin'] and r['decoder']['correction']==ref['after']
             and r['decoder']['train_atoms']==ref['cursor'] and r['decoder']['full_training']==a.cuda,
             'PAIRED_READBACK_CORPUS_BINDING')
        for n,h in run['implementation'].items():need(file_sha(producer.BASE/n)==h,'PAIRED_READBACK_CORPUS_CODE')
        decoder,opt=training.create(parent.model,ModelConfig(**p['config']))
        need(training.restore(ref,decoder,opt,run['binding'],size=ref['size'])==(ref['cursor'],ref['step'],ref['ledger'])
             and fingerprint(decoder.correction.state_dict())==ref['after']
             and fingerprint(opt.state_dict())==ref['optimizer'],'PAIRED_READBACK_RESTORE')
        del opt;gc.collect();decoder.eval().requires_grad_(False)
        phases=dict(authentication_restore=time.perf_counter()-tick);tick=time.perf_counter()
        trajectories=events=atoms=positions=missing=extras=refused=calls=0;summary=[];negative=None
        need(len(cases)==len(r['results'])==r['evaluated_cases'],'PAIRED_READBACK_CASES')
        for (path,case),case_ref in zip(cases,r['results'],strict=True):
            root=a.input/case_ref['path'];report=producer.previous.contract.read(root/'report.json',case_ref['pin'])
            need(report['case']==case['case'] and report['unavailable']==case['unavailable']
                 and len(report['arms'])==len(case['traces']),'PAIRED_READBACK_CASE')
            references=producer.previous.references(schedule,case['case']['task'],texts);counts=[]
            for arm,trace_ref in zip(report['arms'],case['traces'],strict=True):
                need(all(arm[k]==trace_ref[k] for k in ('architecture','mode','proposal')),'PAIRED_READBACK_GRID')
                stored=producer.previous.contract.read(root/arm['path']/'report.json',arm['pin'])
                need(all(arm[k]==v for k,v in stored.items()) and arm['trace']==trace_ref['pin'],'PAIRED_READBACK_TRACE')
                generated=seen=matched=refusal=0;ended=False;delta=0.;nll={key:[] for key in ARMS};hits={key:0 for key in ARMS}
                for event in replay(path/trace_ref['path'],trace_ref['pin']):
                    seen+=1
                    if event['kind']=='end':ended=True;continue
                    native=predicted_atom(event,decoder.specs,decoder.config,region_pin=arm['region_pin'],
                        region_sizes=arm['region_sizes'],trace_pin=trace_ref['pin'],position=generated).slots
                    if generated<len(references):
                        entry=arm['atoms'][generated];raw=references[generated]['raw']
                        need(entry['index']==generated and entry['reference']==references[generated]['id']
                             and entry['positions']==len(raw)+1,'PAIRED_READBACK_REFERENCE')
                        if entry['status']=='physical_refusal':
                            need(entry['refusal']==producer.scoring.PHYSICAL,'PAIRED_READBACK_REFUSAL');refusal+=len(raw)+1
                        else:
                            need(entry['status']=='scored','PAIRED_READBACK_STATUS');f=root/arm['path']/entry['path']
                            need(file_sha(f)==entry['pin'],'PAIRED_READBACK_BYTES');saved=torch.load(f,map_location='cpu',weights_only=True)
                            need(set(saved)==set(ARMS),'PAIRED_READBACK_ARMS')
                            for key,model in (('parent',parent.model),('corrected',decoder)):
                                count=recount(saved[key],raw);direct=original.atom_score(model,native,raw)
                                need(fingerprint(direct)==fingerprint(saved[key]),'PAIRED_READBACK_ORIGINAL_FORWARD')
                                need(count==entry['scores'][key] and direct['native']==entry['native'],'PAIRED_READBACK_RECOUNT')
                                nll[key].append(count['nll']);hits[key]+=count['correct']
                            need(saved['parent']['calls']==saved['corrected']['calls']==entry['calls'],'PAIRED_READBACK_CALLS')
                            change=float((saved['parent']['losses']-saved['corrected']['losses']).sum())
                            need(entry['parent_minus_corrected_nll_sum']==change,'PAIRED_READBACK_DECODER_DELTA');delta+=change
                            atoms+=1;matched+=len(raw)+1;calls+=entry['calls']
                            if negative is None:negative=(saved,raw,path/trace_ref['path'],trace_ref,arm,references,native)
                    generated+=1
                total=sum(len(v['raw'])+1 for v in references);lost=sum(len(v['raw'])+1 for v in references[generated:])
                need(arm['events']==seen and arm['generated_atoms']==generated and arm['model_ended']==ended
                     and len(arm['atoms'])==min(generated,len(references)) and arm['matched_positions']==matched
                     and arm['reference_positions']==total==matched+lost+refusal and arm['missing_positions']==lost
                     and arm['refused_positions']==refusal and arm['extra_atoms']==max(0,generated-len(references))
                     and arm['missing_atoms']==max(0,len(references)-generated),'PAIRED_READBACK_DENOMINATORS')
                for key in ARMS:
                    need(arm['scores'][key]==dict(matched_nll_sum=sum(nll[key]),correct=hits[key],full_reference_accuracy=hits[key]/total),
                         'PAIRED_READBACK_TOTALS')
                need(arm['parent_minus_corrected_nll_sum']==delta and arm['mean_parent_minus_corrected_nll']==(None if not matched else delta/matched),
                     'PAIRED_READBACK_DECODER_CONTRAST')
                events+=seen;trajectories+=1;positions+=matched;missing+=lost;extras+=arm['extra_atoms'];refused+=refusal
                counts.append(dict(architecture=arm['architecture'],mode=arm['mode'],proposal=arm['proposal'],
                    positions=matched,parent_minus_corrected_nll_sum=delta,scores=arm['scores']))
            lookup={(v['architecture'],v['proposal'],v['mode']):v for v in report['arms']}
            for pair in report['paired']:
                left=lookup[pair['architecture'],pair['proposal'],'correct'];right=lookup.get((pair['architecture'],pair['proposal'],pair['control']))
                need(pair['available']==(right is not None) and pair['scores']==(None if right is None else read_pair(left,right,root)),
                     'PAIRED_READBACK_CONTEXT_CONTRAST')
            summary.append(dict(case=case['case'],counts=counts,paired=report['paired']));schedule.fresh(case['case']['task'])
        phases['original_forward_and_recount']=time.perf_counter()-tick;tick=time.perf_counter();controls={}
        if negative is not None:
            saved,raw,path,ref_trace,arm,refs,native=negative
            def reject(name,fn,code):
                try:fn()
                except (ValueError,RuntimeError) as exc:need(code in str(exc),'PAIRED_READBACK_WRONG_REFUSAL');controls[name]=str(exc)
                else:raise ValueError('PAIRED_READBACK_CONTROL_ACCEPTED:'+name)
            bad=deepcopy(saved['parent']);bad['logits']=bad['logits'][:-1]
            reject('missing_eos',lambda:recount(bad,raw),'PAIRED_READBACK_POSITIONS')
            bad=deepcopy(saved['corrected']);bad['targets'][0]=(int(bad['targets'][0])+1)%256
            reject('wrong_reference_byte',lambda:recount(bad,raw),'PAIRED_READBACK_POSITIONS')
            bad=deepcopy(saved['corrected']);bad['losses'][0]+=1
            reject('wrong_loss',lambda:recount(bad,raw),'PAIRED_READBACK_SCORES')
            kw=dict(region_pin=arm['region_pin'],region_sizes=arm['region_sizes'],expected_request=arm['trace_request'])
            reject('wrong_trace_request',lambda:producer.scoring.score_trace(path,ref_trace['pin'],decoder,refs,
                output=a.output/'wrong-request',**dict(kw,expected_request='0'*64)),'PAIRED_SURFACE_REQUEST')
            changed=deepcopy(refs);changed[0]['id']='0'*64
            reject('wrong_reference_identity',lambda:producer.scoring.score_trace(path,ref_trace['pin'],decoder,changed,
                output=a.output/'wrong-reference',**kw),'PAIRED_SURFACE_REFERENCE')
            specs=decoder.specs
            try:
                decoder.specs=(replace(specs[0],identity='0'*64),*specs[1:])
                reject('wrong_instrument',lambda:producer.scoring.score_trace(path,ref_trace['pin'],decoder,refs,
                    output=a.output/'wrong-instrument',**kw),'MNLM_MODEL_IDENTITY')
            finally:decoder.specs=specs
            bad=deepcopy(run);bad['full_training']=not a.cuda
            reject('wrong_training_scope',lambda:scope(parent.receipt,p,bad,c,plan_pin=cfg['plan_pin'],run_pin=cfg['run_pin'],
                full=a.cuda,device=m['device']),'CORPUS_DECODER_TRAINING_SCOPE')
        parent.fresh();decoder.fresh()
        need(calls==r['parent_forward_calls']==r['correction_forward_calls'],'PAIRED_READBACK_COMPONENT_CALLS')
        need(fingerprint(decoder.correction.state_dict())==r['decoder']['correction']
             and fingerprint(parent.model.state_dict())==r['decoder']['parent']['model'],'PAIRED_READBACK_WEIGHTS')
        for n,h in r['code'].items():need(file_sha(producer.BASE/n)==h,'PAIRED_READBACK_CODE_CHANGED')
        if not a.cuda:need(not torch.cuda.is_initialized(),'PAIRED_READBACK_CPU_ONLY')
        phases['controls_and_freshness']=time.perf_counter()-tick;seconds=time.perf_counter()-began
        need(.9<=sum(phases.values())/seconds<=1.001,'PAIRED_READBACK_PHASES')
        result=dict(complete=True,input_pin=a.pin,code=r['code'],trajectories=trajectories,events=events,matched_atoms=atoms,
            positions_per_decoder=positions,missing_positions_per_decoder=missing,extra_atoms=extras,refused_positions=refused,
            calls_per_component=calls,all_scored_logits_exact=True,all_reference_denominators_recounted=True,
            all_context_and_decoder_contrasts_exact=True,weights_unchanged=True,controls=controls,summary=summary,
            free_generation_tested=False,capability_demonstrated=False,gpu_used=a.cuda,
            qualification=a.qualification,full_query=r['full_query'],evaluated_cases=len(cases),
            phases=phases,seconds=seconds,
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,acceptance='author-checked',number_status='measured')
        pin=write(a.output/'report.json',result);print(json.dumps(dict(pin=pin,positions=positions,trajectories=trajectories,seconds=seconds)),flush=True)
    finally:
        if parent:parent.close()
        if leased:release_lease()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--cpu-proof',action='store_true');g.add_argument('--cuda',action='store_true')
    p.add_argument('--qualification',action='store_true');p.add_argument('--owner-fd',type=int)
    for n in ('input','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--pin',required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write(a.output/'failure.json',dict(error=repr(exc)))
        raise
