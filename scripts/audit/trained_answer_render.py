"""Matched, explicitly CPU-only diagnostic of two decoders on saved predictions."""
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
from audit.answer_surface_stream import controls, rejected
from claims.process_conditions import write_committed
from model.answer_surface_input import predicted_atom
from model.answer_surface_stream import SurfaceStepper, render_atom, render_answer
from model.contracts import ModelConfig, Instrument
from model.native_population_training import fingerprint
from model.stream_surface import StreamSurfaceDecoder, schedule, EOS
from model.trained_surface_decoder import TrainedSurfaceDecoder, check_state
from text.constellation_corpus import file_sha, need
from text.constellation_input import stamp

ROOT=Path('/zfs/archive/mnlm/evidence/fsv/29')
REFS=dict(
    full=('phase115-complete-answer-recovery-v1-20261024/inversion-full','1fe1850ea1cedfe834259073f8ba71b56b969808499492bc7aaf83a98dc2b125'),
    plan=('legal-inversion-plan-v3-20261024','2d548670cf1b40f859262c3789d1defefc9a977649cbbdeeff4fdd481b91668b'),
    cold=('inversion-canonical-repair-v2-20261024/readback','2680402106463fbfe867f307946cb85f0e94400a129706f6c89ada471f7b1b60'),
    final=('legal-inversion-final-readback-v1-20261024','02e63b0d17f059a83dd43541a8590d9842a627c825ca12e2a33a252c53018582'),
    cohort=('answer-surface-stream-cpu-v3-20261024','b7fc6eadcfcfedc89efe995c681a888100b6d32c979235c55b3ceb69962568ee'),
    cohort_readback=('answer-surface-stream-readback-v3-20261024','2c69df76d0ba26f660892ee1415e2510a16f1e8e59707b50cf98fba2d9b1e217'))


def observe(generator,limit,*,atom_only=True):
    events=[];decisions=0;terminal=False;error=None
    try:
        while decisions<limit or (events and events[-1].get('utf8_prefix_valid') is False):
            try:item=next(generator)
            except StopIteration:break
            except ValueError as exc:
                if str(exc)!='MNLM_ANSWER_RENDER_MISSING_END':raise
                error=str(exc);terminal=True;break
            events.append(item);decisions+=int('token' in item)
            if item['kind'] in ('answer_end','refuse') or (atom_only and item['kind']=='atom_end'):
                terminal=True;break
    finally:generator.close()
    result=dict(events=events,decisions=decisions,censored=not terminal)
    if error:result['source_error']=error
    return result


def load(root):
    stamps={};records={}
    def checked(path,pin):
        before=stamp(path);need(file_sha(path)==pin,'TRAINED_RENDER_PIN')
        raw=path.read_bytes();need(stamp(path)==before,'TRAINED_RENDER_CHANGED')
        stamps[path]=before;return raw
    for name,(directory,pin) in REFS.items():
        records[name]=json.loads(checked(root/directory/'report.json',pin))
    cohort=records['cohort']; rb=records['cohort_readback']
    need(rb['complete'] and rb['source']['pin']==REFS['cohort'][1]
         and rb['all_original_forward_logits_exact'] and rb['all_free_paths_exact']
         and rb['all_bridge_paths_exact'],'TRAINED_RENDER_PRIOR_READBACK')
    need(cohort['atoms']==16 and cohort['proposals']==8 and cohort['teacher_positions']==1352
         and cohort['observer_decisions']==96,'TRAINED_RENDER_COHORT')
    def payload(name):
        path=root/REFS['cohort'][0]/name
        checked(path,cohort['files'][name])
        value=torch.load(path,map_location='cpu',weights_only=True)
        need(stamp(path)==stamps[path],'TRAINED_RENDER_CHANGED');return value
    config=ModelConfig(**cohort['config']);specs=tuple(Instrument(**s).validate() for s in cohort['instruments'])
    state=payload('decoder.pt');need(fingerprint(state)==cohort['decoder_state'],'TRAINED_RENDER_OLD_STATE')
    old=StreamSurfaceDecoder(config,specs);old.load_state_dict(state,strict=True);del state
    old.eval().requires_grad_(False)
    args=[]
    for key in ('full','plan','cold','final'):args.extend((root/REFS[key][0],REFS[key][1]))
    trained=TrainedSurfaceDecoder(*args,device='cpu',cpu_diagnostic=True)
    need(config==trained.config and specs==trained.specs,'TRAINED_RENDER_ROSTER')
    inputs=payload('inputs.pt');proposals=payload('proposals.pt')
    oldteacher=payload('teacher.pt');oldfree=payload('free.pt');oldbridges=payload('bridges.pt')
    need(len(inputs)==len(oldteacher)==len(oldfree)==16 and len(proposals)==len(oldbridges)==8,
         'TRAINED_RENDER_DENOMINATOR')
    initial=fingerprint(inputs),fingerprint(proposals)
    def fresh():
        trained.fresh()
        need(all(stamp(p)==s for p,s in stamps.items()),'TRAINED_RENDER_CHANGED')
        need((fingerprint(inputs),fingerprint(proposals))==initial,'TRAINED_RENDER_INPUT_CHANGED')
        need(fingerprint(old.state_dict())==cohort['decoder_state'],'TRAINED_RENDER_OLD_CHANGED')
    return dict(model=old,trained=trained,config=config,specs=specs,inputs=inputs,proposals=proposals,
                binding=cohort['binding'],records=records,fresh=fresh,stamps=stamps,args=args,
                oldteacher=oldteacher,oldfree=oldfree,oldbridges=oldbridges)


def summary(teacher,free,bridges):
    positions=correct=0;total=0.
    for rows in teacher:
        for row in rows:
            score=row['logits'].double();target=row['target']
            total+=float(torch.logsumexp(score,0)-score[target]);correct+=int(int(score.argmax())==target);positions+=1
    def outcome(run):
        return dict(decisions=run['decisions'],censored=run['censored'],last=run['events'][-1]['kind'],
                    code=run.get('source_error',run['events'][-1].get('code')),bytes=sum(e['kind']=='byte' for e in run['events']),
                    atoms=sum(e['kind']=='atom_end' for e in run['events']))
    return dict(teacher_positions=positions,teacher_correct=correct,teacher_nll_sum=total,
                teacher_mean_nll=total/positions,free=[outcome(x) for x in free],bridges=[outcome(x) for x in bridges])


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();cpu=time.process_time()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    state=load(a.root);phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    receipt=state['trained'].receipt;results={};timings={}
    for label,model in [('earlier_proof',state['model']),('corpus_trained',state['trained'].model)]:
        teacher=[];free=[];bridges=[];times=[]
        for ordinal,r in enumerate(state['inputs']):
            native=predicted_atom(r['event'],state['specs'],state['config'],**state['binding'],
                                  trace_pin=r['trace_pin'],position=r['position']).slots
            need(fingerprint({n:v.values[0] for n,v in native.items()})==r['native'],'TRAINED_RENDER_NATIVE')
            raw=bytes(r['reference'].tolist());rows=[]
            steps={key:SurfaceStepper(model,native,cached=key=='cached') for key in ('direct','cached')}
            elapsed={key:0. for key in steps}
            try:
                for part in schedule(raw,width=model.config.max_prefix):
                    for position in range(*part['scored']):
                        prefix=part['prefix'][:position-part['offset']+1];scores={}
                        for key in (('direct','cached') if position%2 else ('cached','direct')):
                            start=time.perf_counter();scores[key]=steps[key].logits(prefix,part['offset'])
                            elapsed[key]+=time.perf_counter()-start
                        need(torch.equal(scores['direct'],scores['cached']),'TRAINED_RENDER_CACHE_NUMERIC')
                        rows.append(dict(position=position,offset=part['offset'],prefix=prefix,
                                         target=raw[position] if position<len(raw) else EOS,logits=scores['cached']))
                times.append(dict(seconds=elapsed,counts={k:dict(calls=s.calls,projection_calls=s.projection_calls)
                                                          for k,s in steps.items()}))
            finally:
                for step in steps.values():step.close()
            teacher.append(rows)
            generated=observe(render_atom(model,native,capture_logits=True),limit=96)
            direct=observe(render_atom(model,native,cached=False,capture_logits=True),limit=96)
            need(fingerprint(generated)==fingerprint(direct),'TRAINED_RENDER_FREE_NUMERIC');free.append(generated)
            if label=='earlier_proof':
                need(fingerprint(rows)==fingerprint(state['oldteacher'][ordinal])
                     and fingerprint(generated)==fingerprint(state['oldfree'][ordinal]),'TRAINED_RENDER_OLD_REPLAY')
            print(json.dumps(dict(decoder=label,atom=ordinal,decisions=generated['decisions'],
                                  last=generated['events'][-1]['kind'],censored=generated['censored'])),flush=True)
        for r in state['proposals']:
            result=observe(render_answer(iter(r['events']),model,**state['binding'],trace_pin=r['trace_pin']),
                           limit=96,atom_only=False)
            bridges.append(dict(architecture=r['architecture'],proposal=r['proposal'],**result))
        if label=='earlier_proof':need(fingerprint(bridges)==fingerprint(state['oldbridges']),'TRAINED_RENDER_OLD_BRIDGES')
        torch.save(dict(teacher=teacher,free=free,bridges=bridges),a.output/(label+'.pt'))
        results[label]=summary(teacher,free,bridges);timings[label]=times
    phases['matched_rendering']=time.perf_counter()-tick;tick=time.perf_counter()
    negatives=controls(dict(state,model=state['trained'].model))
    full=state['records']['full'];ref=full['checkpoints'][-1]
    checkpoint=torch.load(ref['path'],map_location='cpu',weights_only=True)
    for key in ('cursor','step','ledger','binding','model'):
        bad=deepcopy(checkpoint)
        if key in ('cursor','step'):bad[key]+=1
        elif key=='ledger':bad[key]='0'*64
        elif key=='binding':bad[key]['plan']='0'*64
        else:next(iter(bad['model'].values())).add_(1)
        negatives['checkpoint_'+key]=rejected(lambda:check_state(bad,full,ref),'TRAINED_DECODER_CHECKPOINT_STATE')
    wrong=list(state['args']);wrong[1]='0'*64
    negatives['source_pin']=rejected(lambda:TrainedSurfaceDecoder(*wrong,device='cpu',cpu_diagnostic=True),'TRAINED_DECODER_PIN')
    negatives['implicit_cpu']=rejected(lambda:TrainedSurfaceDecoder(*state['args'],device='cpu'),'TRAINED_DECODER_EXPLICIT_CPU_DIAGNOSTIC')
    negatives['cuda_without_lease']=rejected(lambda:TrainedSurfaceDecoder(*state['args'],device='cuda:0'),'TRAINED_DECODER_GPU_LEASE')
    state['fresh']();need(not torch.cuda.is_initialized(),'TRAINED_RENDER_GPU')
    state['trained'].close();state['trained'].close()
    negatives['closed_loader']=rejected(state['trained'].fresh,'TRAINED_DECODER_CLOSED')
    phases['controls_and_freshness']=time.perf_counter()-tick
    base=Path(__file__).resolve().parents[1]
    names={Path(m.__file__).resolve().relative_to(base).as_posix() for m in list(sys.modules.values())
           if getattr(m,'__file__',None) and Path(m.__file__).resolve().is_relative_to(base)}
    names.add('audit/trained_answer_render_readback.py')
    code={name:file_sha(base/name) for name in sorted(names)}
    files={p.name:file_sha(p) for p in a.output.iterdir() if p.is_file()}
    report=dict(complete=True,acceptance='author-checked',number_status='measured',sources=REFS,code=code,files=files,
        device='cpu',cpu_diagnostic=True,decoder=receipt,summary=results,timings=timings,controls=negatives,
        atoms=16,proposals=8,observer_decisions=96,phases=phases,seconds=time.perf_counter()-began,
        cpu_seconds=time.process_time()-cpu,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        old_outputs_replayed_exactly=True,cached_and_direct_logits_exact=True,all_inputs_unchanged=True,
        free_generation_tested=True,predicted_conditioning_tested=True,teacher_diagnostic_is_not_free_generation=True,
        observer_censoring_is_not_end=True,semantic_validity_established=False,scientific_admission=False,
        independent_generalization_established=False,promoted_to_serving=False)
    pin=write_committed(a.output/'report.json',report)
    print(json.dumps(dict(complete=True,pin=pin,summary=results,controls=len(negatives))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
