"""Matched native-conditioning and free-reconstruction diagnosis; no training."""
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
from audit import surface_condition_inputs as inputs
from audit.trained_answer_render import ROOT, REFS, observe, rejected
from claims.process_conditions import write_committed
from model.answer_surface_stream import SurfaceStepper, render_atom
from model.native import NativeInput
from model.native_population_training import fingerprint
from model.stream_surface import BOS,EOS,schedule
from text.constellation_corpus import file_sha,need

MODES=('true','shuffled','no_cross')


def score(teacher,free,raw):
    targets=torch.tensor([*raw,EOS],dtype=torch.int64)
    need(teacher.shape==(len(targets),258) and bool(teacher.isfinite().all()),'CONDITION_TEACHER_TAIL')
    double=teacher.double();loss=torch.logsumexp(double,1)-double.gather(1,targets[:,None])[:,0]
    generated=bytes(e['value'] for e in free['events'] if e['kind']=='byte');last=free['events'][-1]
    completed=last['kind']=='atom_end'
    need(not (completed and free['censored']),'CONDITION_FALSE_COMPLETION')
    return dict(positions=len(targets),correct=int((teacher.argmax(1)==targets).sum()),loss_sum=float(loss.sum()),
        mean_loss=float(loss.mean()),source_bytes=len(raw),generated_bytes=len(generated),decisions=free['decisions'],
        last=last['kind'],code=last.get('code'),censored=free['censored'],completed=completed,
        reference_fits_observer=len(raw)+1<=96,exact=completed and generated==raw,
        generated_sha256=__import__('hashlib').sha256(generated).hexdigest())


def teacher(model,native,raw):
    values=torch.empty((len(raw)+1,model.config.vocabulary),dtype=torch.float32)
    seen=torch.zeros(len(raw)+1,dtype=torch.bool);step=SurfaceStepper(model,native)
    try:
        for part in schedule(raw,width=model.config.max_prefix):
            for position in range(*part['scored']):
                need(not seen[position],'CONDITION_DUPLICATE_POSITION')
                prefix=part['prefix'][:position-part['offset']+1]
                values[position]=step.logits(prefix,part['offset']);seen[position]=True
        need(bool(seen.all()),'CONDITION_TEACHER_TAIL')
    finally:step.close()
    return values


def controls(state):
    model=state['decoder'].model;rows=state['rows'];target=rows[0];donor=rows[state['donors'][0]]
    bad=deepcopy(target);bad['native'].pop(next(iter(bad['native'])))
    result=dict(missing_slot=rejected(lambda:inputs.measured_native(bad,model),'CONDITION_ROSTER'),
        self_donor=rejected(lambda:inputs.paired(target,target),'CONDITION_DONOR'),
        evaluation_donor=rejected(lambda:inputs.paired(target,dict(donor,side='query')),'CONDITION_DONOR'))
    badmodel=deepcopy(state['ablated'])
    with torch.no_grad():badmodel.tokens.weight[0,0]+=1
    result['ablation_non_gain_weight']=rejected(lambda:inputs.check_ablation(model,badmodel),'CONDITION_ABLATION_STATE')
    native=inputs.measured_native(target,model)
    absent={n:NativeInput(x.identity,x.rows,torch.zeros_like(x.values),torch.ones_like(x.padding),torch.ones_like(x.absent))
            for n,x in native.items()}
    result['all_absent']=rejected(lambda:model.validate_inputs(absent),'MNLM_MODEL_NO_OBSERVATION')
    step=SurfaceStepper(model,native);next(iter(native.values())).values.add_(1)
    result['native_mutation']=rejected(lambda:step.logits([BOS],0),'SURFACE_STREAM_INPUT_CHANGED');step.close()
    badlogits=torch.zeros((1,258));free=dict(events=[dict(kind='atom_end')],censored=True)
    result['lost_teacher_tail']=rejected(lambda:score(badlogits,free,b'ab'),'CONDITION_TEACHER_TAIL')
    result['false_completion']=rejected(lambda:score(torch.zeros((3,258)),free,b'ab'),'CONDITION_FALSE_COMPLETION')
    reference=dict(occurrence=target['occurrence'],uid=target['uid'],side=target['side'],reference=target['raw'].clone())
    inputs.check_reference(reference,target)
    for key in ('occurrence','uid','reference'):
        bad=deepcopy(reference)
        if key=='reference':bad[key][0]^=1
        else:bad[key]+=1
        result['source_'+key]=rejected(lambda:inputs.check_reference(bad,target),'CONDITION_PREDICTED_SOURCE_JOIN')
    return result


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();cpu=time.process_time()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    state=inputs.load(a.root);model=state['decoder'].model
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    need(max(len(r['raw']) for r in state['rows'])*258*4*16<available//4,'CONDITION_PHYSICAL_MEMORY')
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    files={};summaries=[];frozen_free=None
    (a.output/'atoms').mkdir()
    for ordinal,row in enumerate(state['rows']):
        raw=bytes(row['raw'].tolist());native=inputs.measured_native(row,model)
        donor=inputs.measured_native(state['rows'][state['donors'][ordinal]],model)
        outputs={};stats={}
        for mode in MODES:
            current=state['ablated'] if mode=='no_cross' else model
            slots=donor if mode=='shuffled' else native
            logits=teacher(current,slots,raw)
            free=observe(render_atom(current,slots,capture_logits=True),limit=96)
            outputs[mode]=dict(teacher=logits,free=free);stats[mode]=score(logits,free,raw)
            if mode=='no_cross':
                value=fingerprint(free)
                need(frozen_free is None or frozen_free==value,'CONDITION_ABLATION_FREE_DEPENDENCE');frozen_free=value
                # First/last true prefix against a different actual native input: exact no-conditioning control.
                step=SurfaceStepper(current,donor)
                try:
                    for position in (0,len(raw)):
                        left=max(0,(position//16)*16-16);prefix=([BOS,*raw])[left:position+1]
                        need(torch.equal(step.logits(prefix,left),logits[position]),'CONDITION_ABLATION_NATIVE_DEPENDENCE')
                finally:step.close()
        relative=f'atoms/{ordinal:04d}.pt';torch.save(outputs,a.output/relative);files[relative]=file_sha(a.output/relative)
        summaries.append(dict(ordinal=ordinal,document=row['document'],side=row['side'],train_present=row['train_present'],
            occurrence=row['occurrence'],uid=row['uid'],donor=state['donors'][ordinal],scores=stats))
        if ordinal%10==0 or ordinal==len(state['rows'])-1:
            print(json.dumps(dict(atom=ordinal+1,atoms=len(state['rows']),seconds=time.perf_counter()-began)),flush=True)
        del outputs,logits,free
    phases['teacher_and_free']=time.perf_counter()-tick;tick=time.perf_counter()
    negatives=controls(state);state['fresh']();need(not torch.cuda.is_initialized(),'CONDITION_GPU')
    binding=dict(decoder=state['decoder'].receipt,ablation_state=state['ablation_pin'],zeroed_gains=state['gains'],
                 native_population=state['input_pin'],consumed=state['consumed'])
    state['decoder'].close();phases['controls_and_freshness']=time.perf_counter()-tick
    base=Path(__file__).resolve().parents[1]
    names={Path(m.__file__).resolve().relative_to(base).as_posix() for m in list(sys.modules.values())
           if getattr(m,'__file__',None) and Path(m.__file__).resolve().is_relative_to(base)}
    names.add('audit/trained_surface_conditions_readback.py')
    code={name:file_sha(base/name) for name in sorted(names)}
    seconds=time.perf_counter()-began;need(sum(phases.values())/seconds>=.9,'CONDITION_PHASE_ATTRIBUTION')
    files['nvidia-before.txt']=file_sha(a.output/'nvidia-before.txt')
    report=dict(complete=True,acceptance='author-checked',number_status='measured',code=code,files=files,binding=binding,
        atoms=215,documents=15,teacher_positions_per_condition=42738,observer_decisions=96,modes=MODES,summary=summaries,
        controls=negatives,phases=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu,
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,cpu_diagnostic=True,optimizer_steps=0,
        native_encoder_calls=0,scientific_admission=False,semantic_validity_established=False,promoted_to_serving=False,
        no_cross_is_separate_diagnostic=True,all_absent_still_refuses=True,observer_censoring_is_not_end=True)
    pin=write_committed(a.output/'report.json',report);print(json.dumps(dict(complete=True,pin=pin,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error)))
        raise
