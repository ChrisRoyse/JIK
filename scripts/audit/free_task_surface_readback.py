"""Cold original-forward replay of every freely emitted byte and terminal decision."""
import argparse
from collections import deque
from copy import deepcopy
import codecs
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

if '--cuda' not in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import free_task_surface as producer
from claims.process_conditions import write_committed as write
from lenscommon import take_lease,release_lease
from model.answer_surface_input import predicted_atom
from model.complete_answer_trace import replay
from model.native_population_training import fingerprint
from model.stream_surface import BOS,EOS
from model.task_surface_scoring import native_pin
from text.constellation_corpus import file_sha,need


def decision(direct,saved,token):
    need(torch.equal(direct,saved),'FREE_READBACK_LOGITS')
    need(int(direct.argmax())==token,'FREE_READBACK_TOKEN')


def summary(r,*,attempted,completed,size,decisions,refusal,source):
    expected=dict(attempted_atoms=attempted,completed_atoms=completed,generated_bytes=size,byte_decisions=decisions,
        surface_refusal=refusal,native_events=source['events'],native_atoms=source['generated_atoms'],
        native_ended=source['model_ended'],native_status=source['status'],native_refusal=source['refusal'],
        answer_ended=source['model_ended'] and refusal is None and completed==source['generated_atoms'])
    need(all(r[k]==v for k,v in expected.items()),'FREE_READBACK_TERMINAL');return expected


def atom(model,slots,rows,surface,index):
    prefix=deque([BOS],maxlen=model.config.max_prefix);utf8=codecs.getincrementaldecoder('utf-8')('strict')
    digest=hashlib.sha256();position=decisions=0;pending=False;last=None
    while True:
        row,logits=next(rows);record={k:v for k,v in row.items() if k not in ('ordinal','previous','logits')}
        if pending:
            expected=dict(kind='refuse',code='SURFACE_STREAM_UTF8',bytes=position,sha256=digest.hexdigest(),
                          validity='unassessed',serving=False,atom=index)
            need(logits is None and record==expected,'FREE_READBACK_UTF8_REFUSAL')
            return False,expected['code'],position,decisions,last
        offset=max(0,(position//16)*16-(model.config.max_prefix-16));count=position+1-offset
        need(0<count<=len(prefix),'FREE_READBACK_PREFIX');ids=list(prefix)[-count:]
        device=next(model.parameters()).device
        with torch.no_grad():
            direct=model(torch.tensor([ids],dtype=torch.int64,device=device),slots,
                offsets=torch.tensor([offset],dtype=torch.int64,device=device))['logits'][0,-1].cpu()
        need(logits is not None,'FREE_READBACK_MISSING_LOGITS');token=int(direct.argmax());decision(direct,logits,row['token'])
        decisions+=1;last=(direct,logits,token)
        expected=dict(position=position,token=token,offset=offset,prefix=ids,validity='unassessed',serving=False,atom=index)
        if token==EOS:
            try:utf8.decode(b'',final=True);valid=position>0
            except UnicodeDecodeError:valid=False
            code=None if valid else ('SURFACE_STREAM_EMPTY' if not position else 'SURFACE_STREAM_UTF8')
            expected.update(kind='atom_end' if valid else 'refuse',code=code,bytes=position,sha256=digest.hexdigest())
            need(record==expected,'FREE_READBACK_EOS');return valid,code,position,decisions,last
        if not 0<=token<256:
            expected.update(kind='refuse',code='SURFACE_STREAM_SYMBOL',bytes=position,sha256=digest.hexdigest())
            need(record==expected,'FREE_READBACK_SYMBOL');return False,expected['code'],position,decisions,last
        raw=bytes([token]);need(surface.read(1)==raw,'FREE_READBACK_SURFACE_BYTE');digest.update(raw);prefix.append(token);position+=1
        try:utf8.decode(raw,final=False);valid=True
        except UnicodeDecodeError:valid=False
        expected.update(kind='byte',value=token,utf8_prefix_valid=valid)
        need(record==expected,'FREE_READBACK_BYTE_RECORD');pending=not valid


def run(a):
    began=tick=time.perf_counter();a.output.mkdir(exist_ok=False);parent=correction=None;leased=False
    try:
        r=producer.paired.previous.contract.read(a.input/'report.json',a.pin)
        need(r['complete'] and r['gpu_used']==a.cuda and r['qualification']==a.qualification
             and r['reference_visible'] is False and r['reference_open_refused'] and not r['teacher_forced'],
             'FREE_READBACK_SCOPE')
        for name,h in r['code'].items():need(file_sha(producer.paired.BASE/name)==h,'FREE_READBACK_CODE')
        args=SimpleNamespace(cuda=a.cuda,qualification=a.qualification,request=Path(r['request']['path']),pin=r['request']['pin'],
            generated=Path(r['generated']['path']),generated_pin=r['generated']['pin'],
            **{'corpus_'+k:Path(v) if k in ('plan','run','readback') else v for k,v in r['corpus'].items() if k!='full'})
        if a.cuda:producer.paired.previous.gpu_ready(a.output);take_lease('29','free-task-readback');leased=True
        m,cases,binding,parent,correction,sizes=producer.load(args);models=dict(parent=parent.model,corrected=correction.model)
        need(r['decoder']==correction.receipt and binding==r['source'] and r['evaluated_cases']==len(cases)
             and r['full_query']==(a.cuda and not a.qualification) and r['full_training']==m['full_training'],
             'FREE_READBACK_BINDING')
        need(all(fingerprint(model.state_dict())==r['decoder_pins'][name] for name,model in models.items()),'FREE_READBACK_WEIGHTS')
        allowed=[a.input/'report.json'];reports=[]
        for (path,case),ref in zip(cases,r['results'],strict=True):
            root=a.input/ref['path'];allowed.append(root/'report.json')
            case_report=producer.paired.previous.contract.read(root/'report.json',ref['pin'])
            need(case_report['case']==case['case'] and case_report['unavailable']==case['unavailable']
                 and len(case_report['traces'])==len(case['traces']),'FREE_READBACK_CASE')
            traces=[]
            for trace,original in zip(case_report['traces'],case['traces'],strict=True):
                need(all(trace[k]==original[k] for k in ('architecture','mode','proposal')),'FREE_READBACK_GRID')
                dest=root/trace['path'];allowed.append(dest/'report.json')
                value=producer.paired.previous.contract.read(dest/'report.json',trace['pin']);traces.append((trace,original,dest,value))
                for name in models:allowed += [dest/name/n for n in ('report.json','request.json','events.jsonl','surface.bin','logits.bin')]
            reports.append((path,case,traces))
        opened=producer.private_guard(cases,a.output,allowed)
        phases=dict(authenticate_and_restore=time.perf_counter()-tick);tick=time.perf_counter();results=[];last=None;negative=None
        for path,case,traces in reports:
            for trace,original,dest,value in traces:
                source_path=path/original['path'];source=producer.paired.previous.contract.read(source_path/'report.json',original['pin'])
                need(value['trace']==original['pin'] and value['events']==source['events']
                     and value['generated_atoms']==source['generated_atoms'] and value['model_ended']==source['model_ended'],
                     'FREE_READBACK_NATIVE_COUNTS')
                arms={}
                for name,model in models.items():
                    ref=value['arms'][name];root=dest/name;own=producer.paired.previous.contract.read(root/'report.json',ref['pin'])
                    need(all(ref[k]==v for k,v in own.items()),'FREE_READBACK_ARM')
                    request=producer.paired.previous.contract.read(root/'request.json',own['request'])
                    need(request['trace']==original['pin'] and request['trace_request']==source['files']['request.json']
                         and request['region_pin']==case['checkpoints'][original['architecture']]['target'] and request['region_sizes']==sizes
                         and request['decoder']==r['decoder_pins'][name] and request['stride']==16
                         and request['max_prefix']==model.config.max_prefix and request['reference_visible'] is False,'FREE_READBACK_REQUEST')
                    stream=producer.render.rows(root,ref['pin']);attempted=completed=size=decisions=0;refusal=None;generated=0
                    with (root/'surface.bin').open('rb') as surface:
                        for event in replay(source_path,original['pin']):
                            if event['kind']=='end':continue
                            if refusal is None:
                                native=predicted_atom(event,model.specs,model.config,region_pin=request['region_pin'],
                                    region_sizes=sizes,trace_pin=original['pin'],position=generated)
                                slots={n:v.to(next(model.parameters()).device) for n,v in native.slots.items()}
                                row,logits=next(stream);content={k:v for k,v in row.items() if k not in ('ordinal','previous','logits')}
                                need(logits is None and content==dict(kind='atom_start',atom=generated,
                                    regions=list(native.regions),native=native_pin(slots)),'FREE_READBACK_ATOM_START')
                                attempted+=1;ended,code,count,n,last=atom(model,slots,stream,surface,generated)
                                completed+=int(ended);size+=count;decisions+=n;refusal=code
                            generated+=1
                        need(next(stream,None) is None and surface.read(1)==b'','FREE_READBACK_TRAILING')
                    counts=dict(attempted=attempted,completed=completed,size=size,decisions=decisions,refusal=refusal,source=source)
                    arms[name]=summary(own,**counts)
                    need(trace['arms'][name]=={k:own[k] for k in trace['arms'][name]},'FREE_READBACK_CASE_SUMMARY')
                    if negative is None:negative=(own,counts,root,ref['pin'])
                results.append(dict(case=case['case'],architecture=original['architecture'],proposal=original['proposal'],
                    mode=original['mode'],arms=arms))
        phases['original_free_forward_replay']=time.perf_counter()-tick;tick=time.perf_counter();controls={}
        def reject(name,fn,code):
            try:fn()
            except ValueError as exc:need(code in str(exc),'FREE_READBACK_WRONG_REFUSAL');controls[name]=str(exc)
            else:raise ValueError('FREE_READBACK_CONTROL_PASSED:'+name)
        if last is not None:
            direct,saved,token=last;bad=saved.clone();bad[0]+=1
            reject('changed_logit',lambda:decision(direct,bad,token),'FREE_READBACK_LOGITS')
            reject('wrong_byte',lambda:decision(direct,saved,(token+1)%258),'FREE_READBACK_TOKEN')
        if negative is not None:
            own,counts,root,pin=negative;bad=deepcopy(own);bad['answer_ended']=not bad['answer_ended']
            reject('false_completion',lambda:summary(bad,**counts),'FREE_READBACK_TERMINAL')
            reject('wrong_artifact_pin',lambda:next(producer.render.rows(root,'0'*64)),'FREE_SURFACE_REPORT_PIN')
        forbidden=producer.paired.P/'answer-native-plan-v1-20261023/report.json'
        reject('private_reference_open',forbidden.read_bytes,'FREE_TASK_PRIVATE_OPEN')
        parent.fresh();correction.fresh()
        need(all(fingerprint(model.state_dict())==r['decoder_pins'][name] for name,model in models.items()),'FREE_READBACK_FINAL_WEIGHTS')
        if not a.cuda:need(not torch.cuda.is_initialized(),'FREE_READBACK_CPU_ONLY')
        phases['negative_controls_and_seal']=time.perf_counter()-tick;seconds=time.perf_counter()-began
        need(.9<=sum(phases.values())/seconds<=1.001,'FREE_READBACK_PHASES')
        result=dict(complete=True,input_pin=a.pin,code=r['code'],results=results,trajectories=len(results),
            all_free_logits_exact=True,all_emitted_bytes_exact=True,all_terminals_recounted=True,controls=controls,
            source_traces_drained=True,reference_visible=False,gpu_used=a.cuda,qualification=a.qualification,
            full_training=r['full_training'],capability_demonstrated=False,phases=phases,seconds=seconds,
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,acceptance='author-checked')
        pin=write(a.output/'report.json',result);print(json.dumps(dict(complete=True,pin=pin,seconds=seconds)),flush=True)
    finally:
        if correction:correction.close()
        if parent:parent.close()
        if leased:release_lease()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--cpu-proof',action='store_true');g.add_argument('--cuda',action='store_true');p.add_argument('--qualification',action='store_true')
    for name in ('input','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pin',required=True);a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write(a.output/'failure.json',dict(error=repr(exc)))
        raise
