"""Physical CPU characterization of target-free complete-answer byte rendering."""
import argparse
from copy import deepcopy
import hashlib
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
from audit import surface_device_authority as authority
from audit.legal_grouped_placement_training import source_bundle
from claims.process_conditions import write_committed
from model.answer_surface_input import predicted_atom
from model.answer_surface_stream import SurfaceStepper, render_atom, render_answer
from model.native_population_training import fingerprint
from model.stream_surface import BOS, EOS, StreamSurfaceDecoder, schedule
from text.constellation_corpus import file_sha, need

REFS = dict(
    predictions=('open-window-answer-cpu-v2-20261024','a62c0402a911f7e7c7cafaf3f88e760c1fd6c332159931e10923028a3d3ae339'),
    predictions_readback=('open-window-answer-readback-v2-20261024','abcd6121a90bc83118995db0877fc380f29d0581a18542d9d3b4f9a63b166242'),
    decoder=('legal-surface-device-cpu-v2-20261023','033ef5b9313b85e89fabe0fa663dacd2c5735978d18310ebddd3c48e4846ef99'),
    decoder_readback=('legal-surface-device-readback-v2-20261023','ed9fcc7e33634e6ec4b9ace7f5278c9c40103926e37f6e6ed725c4cc122ceec9'))
NEW = ('model/answer_surface_stream.py','audit/answer_surface_stream.py','audit/answer_surface_stream_readback.py')
DOCUMENT, OBSERVATIONS = 22924, 96


def load(root):
    records = {k:authority.pinned(root/n,h) for k,(n,h) in REFS.items()}
    need(records['predictions_readback']['source']['pin']==REFS['predictions'][1]
         and records['decoder_readback']['input_pin']==REFS['decoder'][1], 'STREAM_PROOF_PRIOR_BINDING')
    config,specs,examples,meta,_,_,_,code,surface_fresh = authority.load(root)
    source,bundle = source_bundle(root)
    base = Path(__file__).resolve().parents[1]
    for record in [*records.values(),source]:
        for n,h in record.get('code',{}).items():
            need(file_sha(base/n)==h and (n not in code or code[n]==h), 'STREAM_PROOF_PRIOR_CODE')
            code[n]=h
    for n in NEW: code[n]=file_sha(base/n)
    checkpoint = torch.load(root/REFS['decoder'][0]/'reference.pt',map_location='cpu',weights_only=True)
    model = StreamSurfaceDecoder(config,specs);model.load_state_dict(checkpoint['after'],strict=True)
    model.eval().requires_grad_(False)
    indices = [i for i,m in enumerate(meta) if m['document']==DOCUMENT]
    item = next(x for x in bundle['items'] if x['document']==DOCUMENT)
    need(len(indices)==len(item['frames']) and len(indices)>3,'STREAM_PROOF_JOIN')
    for position,index in enumerate(indices):
        e = examples[index]
        need(e.occurrence==item['start']+position and e.side==item['side']
             and all(torch.equal(e.native[n].values[0],v) for n,v in item['frames'][position].items()),
             'STREAM_PROOF_NATIVE_JOIN')
    binding = dict(region_pin=records['predictions']['extension_pin'],region_sizes=tuple(bundle['sizes']))
    inputs=[]; proposals=[]
    for arm in ['shared','own_slot','whole_proposals']:
        path=root/REFS['predictions'][0]/arm/'evaluation.pt';trace=file_sha(path)
        rows=torch.load(path,map_location='cpu',weights_only=True)
        for r in rows:
            if r['document']!=DOCUMENT or r['mode']!='correct':continue
            need(len(r['events'])>=2,'STREAM_PROOF_PREDICTED_TAIL')
            proposals.append(dict(architecture=arm,proposal=r['proposal'],events=r['events'],trace_pin=trace))
            for position,event in enumerate(r['events'][:2]):
                atom=predicted_atom(event,specs,config,**binding,trace_pin=trace,position=position)
                index=indices[2+position];e=examples[index]
                inputs.append(dict(architecture=arm,proposal=r['proposal'],position=position,event=event,
                    trace_pin=trace,reference=torch.tensor(list(e.raw),dtype=torch.uint8),uid=e.uid,
                    occurrence=e.occurrence,train_present=meta[index]['train_present'],side=e.side,
                    native=fingerprint({n:v.values[0] for n,v in atom.slots.items()})))
    need(len(inputs)==16 and len(proposals)==8,'STREAM_PROOF_CENSUS')
    stamps={root/n/f:authority.stamp(root/n/f) for n,_ in REFS.values()
            for f in ['report.json',*json.loads((root/n/'report.json').read_bytes()).get('files',{})]}
    def fresh():
        surface_fresh()
        for p,s in stamps.items():need(authority.stamp(p)==s,'STREAM_PROOF_SOURCE_CHANGED')
        for n,h in code.items():need(file_sha(base/n)==h,'STREAM_PROOF_CODE_CHANGED')
    fresh()
    return dict(model=model,config=config,specs=specs,inputs=inputs,proposals=proposals,
                binding=binding,code=code,fresh=fresh)


def observe(generator, limit=OBSERVATIONS, *, atom_only=True):
    result=[]; decisions=0;terminal=False
    try:
        while decisions<limit or (result and result[-1].get('utf8_prefix_valid') is False):
            try:item=next(generator)
            except StopIteration:break
            result.append(item);decisions+=int('token' in item)
            if item['kind'] in ('answer_end','refuse') or (atom_only and item['kind']=='atom_end'):
                terminal=True;break
    finally:generator.close()
    return dict(events=result,decisions=decisions,censored=not terminal)


def rejected(call,code):
    try:call()
    except (ValueError,RuntimeError,TypeError) as exc:
        need(code in str(exc),'STREAM_PROOF_WRONG_REFUSAL:'+str(exc));return str(exc)
    raise ValueError('STREAM_PROOF_CONTROL_ACCEPTED:'+code)


def controls(state):
    model=state['model'];r=state['inputs'][0];binding=dict(state['binding'],trace_pin=r['trace_pin'])
    def slots():return predicted_atom(r['event'],state['specs'],state['config'],**binding,position=0).slots
    out={};native=slots();step=SurfaceStepper(model,native)
    next(iter(native.values())).values.add_(1)
    out['native_changed']=rejected(lambda:step.logits([BOS],0),'SURFACE_STREAM_INPUT_CHANGED');step.close()
    step=SurfaceStepper(model,slots());next(iter(step.memories.values())).zero_()
    out['memory_changed']=rejected(lambda:step.logits([BOS],0),'SURFACE_STREAM_MEMORY_CHANGED');step.close()
    copied=deepcopy(model);step=SurfaceStepper(copied,slots())
    with torch.no_grad():next(copied.parameters()).add_(1)
    out['weights_changed']=rejected(lambda:step.logits([BOS],0),'SURFACE_STREAM_WEIGHTS_CHANGED');step.close()
    step=SurfaceStepper(model,slots());step.close();step.close()
    need(step.native is None and step.model is None and step.memories is None,'STREAM_PROOF_TEARDOWN')
    out['closed']=rejected(lambda:step.logits([BOS],0),'SURFACE_STREAM_CLOSED')
    step=SurfaceStepper(model,slots())
    for name,prefix,offset in [('empty',[],0),('symbol',[258],0),('negative',[BOS],-1),('offset',[BOS],2**24)]:
        out[name]=rejected(lambda:step.logits(prefix,offset),'SURFACE_STREAM_PREFIX')
    step.close()
    out['stride']=rejected(lambda:next(render_atom(model,slots(),stride=0)),'SURFACE_STREAM_STRIDE')
    out['target_length']=rejected(lambda:render_atom(model,slots(),target_length=12),'unexpected keyword')
    out['target_text']=rejected(lambda:render_atom(model,slots(),target_text=b'private'),'unexpected keyword')
    bad=deepcopy(r['event']);bad['index']=1
    out['order']=rejected(lambda:next(render_answer([bad],model,**binding)),'ANSWER_SURFACE_POSITION')
    bad=deepcopy(r['event']);bad['raw']=b'private'
    out['future_target']=rejected(lambda:next(render_answer([bad],model,**binding)),'ANSWER_SURFACE_EVENT_FIELDS')
    bad=deepcopy(r['event']);bad['slots'].pop(next(iter(bad['slots'])))
    out['missing_slot']=rejected(lambda:next(render_answer([bad],model,**binding)),'ANSWER_SURFACE_ROSTER')
    out['missing_end']=rejected(lambda:next(render_answer([],model,**binding)),'ANSWER_RENDER_MISSING_END')
    end=dict(kind='end',proposal='control',generated=0,stop_logits=torch.tensor([[-1.,1.]]),validity='unassessed')
    bad=deepcopy(end);bad['generated']=1
    out['end_count']=rejected(lambda:list(render_answer([bad],model,**binding)),'ANSWER_RENDER_END')
    out['after_end']=rejected(lambda:list(render_answer([end,end],model,**binding)),'ANSWER_RENDER_AFTER_END')
    out['null_after_end']=rejected(lambda:list(render_answer([end,None],model,**binding)),'ANSWER_RENDER_AFTER_END')
    bad=deepcopy(end);bad['generated']=False
    out['boolean_end']=rejected(lambda:list(render_answer([bad],model,**binding)),'ANSWER_RENDER_END')
    bad=deepcopy(end);bad['stop_logits']=torch.tensor([[1.,-1.]])
    out['end_stop']=rejected(lambda:list(render_answer([bad],model,**binding)),'ANSWER_RENDER_END')
    # Explicit synthetic boundary control, never a generated answer or a model result.
    empty=list(render_answer([end],model,**binding))
    need(len(empty)==1 and empty[0]['atoms']==0 and not empty[0]['serving'],'STREAM_PROOF_END_CONTROL')
    # Close a real interrupted render and inspect its formerly resident atom cache.
    generator=render_atom(model,slots());next(generator)
    resident=generator.gi_frame.f_locals['stepper'];generator.close()
    need(resident.closed and resident.memories is None and resident.native is None
         and resident.model is None,'STREAM_PROOF_CANCEL_LEAK')
    return out


def run(a):
    a.output.mkdir(exist_ok=False);start=tick=time.perf_counter();cpu0=time.process_time()
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    state=load(a.root);model=state['model'];initial=fingerprint(model.state_dict())
    torch.save(model.state_dict(),a.output/'decoder.pt');torch.save(state['inputs'],a.output/'inputs.pt')
    torch.save(state['proposals'],a.output/'proposals.pt')
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    teacher=[];free=[];timings=[];counts=[]
    for ordinal,r in enumerate(state['inputs']):
        native=predicted_atom(r['event'],state['specs'],state['config'],**state['binding'],
                              trace_pin=r['trace_pin'],position=r['position']).slots
        raw=bytes(r['reference'].tolist());rows=[]
        steps={k:SurfaceStepper(model,native,cached=k=='cached') for k in ['reference','cached']}
        elapsed={k:0. for k in steps}
        # Same prefix shape and original arithmetic for both paths, including every EOS.
        for part in schedule(raw,width=model.config.max_prefix):
            for absolute in range(*part['scored']):
                prefix=part['prefix'][:absolute-part['offset']+1];answers={}
                for k in (['reference','cached'] if absolute%2==0 else ['cached','reference']):
                    began=time.perf_counter();answers[k]=steps[k].logits(prefix,part['offset'])
                    elapsed[k]+=time.perf_counter()-began
                need(torch.equal(answers['reference'],answers['cached']),'STREAM_PROOF_LOGITS')
                rows.append(dict(position=absolute,offset=part['offset'],prefix=prefix,
                    target=raw[absolute] if absolute<len(raw) else EOS,logits=answers['cached']))
        teacher.append(rows);timings.append(elapsed)
        counts.append({k:dict(calls=s.calls,projection_calls=s.projection_calls) for k,s in steps.items()})
        for s in steps.values():s.close()
        runs={k:observe(render_atom(model,native,cached=k=='cached',capture_logits=True)) for k in steps}
        need(fingerprint(runs['cached'])==fingerprint(runs['reference']),'STREAM_PROOF_FREE_EXACT')
        free.append(runs['cached'])
        print(json.dumps(dict(atom=ordinal,teacher_positions=len(rows),free_decisions=runs['cached']['decisions'],
            censored=runs['cached']['censored'])),flush=True)
    need(sum(map(len,teacher))==1352,'STREAM_PROOF_POSITIONS')
    torch.save(teacher,a.output/'teacher.pt');torch.save(free,a.output/'free.pt')
    phases['all_logits_and_free_rendering']=time.perf_counter()-tick;tick=time.perf_counter()
    bridges=[]
    for r in state['proposals']:
        result=observe(render_answer(iter(r['events']),model,**state['binding'],trace_pin=r['trace_pin']),atom_only=False)
        bridges.append(dict(architecture=r['architecture'],proposal=r['proposal'],**result))
    torch.save(bridges,a.output/'bridges.pt')
    negatives=controls(state);state['fresh']()
    need(fingerprint(model.state_dict())==initial and not torch.cuda.is_initialized(),'STREAM_PROOF_STATE_CHANGED')
    phases['bridges_controls_and_readback']=time.perf_counter()-tick;tick=time.perf_counter()
    def outcome(r):
        return dict(decisions=r['decisions'],censored=r['censored'],last_kind=r['events'][-1]['kind'],
                    code=r['events'][-1].get('code'),bytes=sum(e['kind']=='byte' for e in r['events']))
    files={p.name:file_sha(p) for p in a.output.iterdir() if p.is_file()}
    phases['hashes']=time.perf_counter()-tick;seconds=time.perf_counter()-start
    need(sum(phases.values())/seconds>=.9,'STREAM_PROOF_PHASE_ATTRIBUTION')
    result=dict(complete=True,acceptance='author-checked',number_status='measured',schema=1,
        sources=REFS,code=state['code'],files=files,binding=state['binding'],decoder_state=initial,
        config=vars(state['config']),instruments=[vars(s) for s in state['specs']],
        document=DOCUMENT,atoms=16,proposals=8,teacher_positions=1352,observer_decisions=OBSERVATIONS,
        timings=timings,counts=counts,free=[outcome(r) for r in free],bridges=[outcome(r) for r in bridges],
        controls=negatives,all_logits_exact=True,all_free_paths_exact=True,source_and_weights_unchanged=True,
        observer_censoring_is_not_end=True,scientific_admission=False,capability_demonstrated=False,
        promoted_to_serving=False,seconds=seconds,cpu_seconds=time.process_time()-cpu0,phases=phases,
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    pin=write_committed(a.output/'report.json',result)
    print(json.dumps(dict(complete=True,pin=pin,seconds=seconds,free=result['free'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=authority.ROOT)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
