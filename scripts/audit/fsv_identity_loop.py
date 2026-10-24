"""Physical intrinsic-loop proof on source-profile readings; no conduct claim.

The phase sequence exercises software with real source fields. It is not a
reconstruction of five historical events or evidence of learned compliance.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lens'))
from lens.lenscommon import run_cli, take_lease
from model.contracts import ModelConfig
from model.identity_loop import IdentityLoopPredictor, PHASES
from model.loop_checkpoint import save, load
from model.loop_input import LoopInputs
from senses.contract import canonical
from senses.trajectory_population import publish


def check(ok,code):
    if not ok:
        raise ValueError('MNLM_LOOP_FSV_'+code)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def event_ids(sequence, units, inputs):
    # This records our actual development software invocation, not event time in
    # the Gospel account. Repeated source content is not an independent witness.
    return tuple(hashlib.sha256(canonical(dict(purpose='engineering_phase_execution',sequence=sequence,
                    context='development:C'+str(i+1),source=inputs.refs[u]['text_sha256'],source_id=inputs.source_id))).hexdigest()
                 for i,u in enumerate(units))


def step(model, inputs, states, phase, device):
    field=('echo','echo','not_chosen','response','response')[phase]
    units=[inputs.lookup[('C'+str(i+1),field)] for i in range(len(states))]
    native=inputs.batch(units,model.config,device)
    return model(native,states,phase=phase,event_ids=event_ids(states[0].sequence,units,inputs),
                 evidence_origin=['source_observation']*len(states))


def outputs(out):
    result={'states':torch.stack([s.hidden for s in out['states']]).detach().cpu().numpy()}
    for kind in ('regions','slots','predicted_echo','predicted_verdict'):
        for name,value in out[kind].items():
            result[kind+'__'+name]=value.detach().cpu().numpy()
    return result


def configure():
    check(torch.cuda.is_available(),'NO_GPU')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(20261020)


def cold(args):
    started=time.perf_counter();configure()
    resident=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True)
    take_lease('10','intrinsic-loop-cold-readback')
    plan=json.loads((args.output/'plan.json').read_bytes())
    inputs=LoopInputs(args.population,args.native,args.source)
    model,states,_=load(args.output/'pending',expected_checkpoint=plan['pending'],expected_standard=plan['standard_id'],device='cuda')
    check([s.sequence for s in states]==[3,3], 'PENDING_PHASE')
    with torch.no_grad():out=step(model,inputs,states,3,'cuda')
    with np.load(args.output/'expected.npz',allow_pickle=False) as z:
        observed=outputs(out)
        check(set(z.files)==set(observed) and all(np.array_equal(z[k],v) for k,v in observed.items()),'COLD_REPLAY')
    check(sha(args.output/'expected.npz')==plan['expected_sha256'],'EXPECTED_PIN')
    model.check_reference()
    publish(args.output/'cold-readback.json',dict(complete=True,acceptance='author-checked',all_outputs_exact=True,
            arrays=len(observed),contexts=2,pending_sequence=3,source=inputs.source_readback,
            native=inputs.window_readback,wall_seconds=time.perf_counter()-started,gpu_before=resident))


def main(args):
    started=time.perf_counter();configure()
    check(not args.output.exists(),'EXISTS');args.output.mkdir(parents=True)
    resident=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True)
    take_lease('10','intrinsic-loop-physical-proof')
    (args.output/'gpu-before.txt').write_text(resident,encoding='utf-8')
    inputs=LoopInputs(args.population,args.native,args.source)
    loaded=time.perf_counter()
    config=ModelConfig(width=32,heads=4,layers=1)
    core=inputs.batch(inputs.standard_units,config,'cpu')
    model=IdentityLoopPredictor(config,inputs.specs,core,source_id=inputs.source_id).cuda()
    states=tuple(model.start('development:C'+str(i+1)) for i in range(2))
    initial=save(model,states,args.output/'initial')
    baseline={n:p.detach().clone() for n,p in model.named_parameters()}
    trace=[]
    for phase in range(3):
        out=step(model,inputs,states,phase,'cuda');states=out['states']
        trace.append(dict(phase=PHASES[phase],sequence=states[0].sequence,events=[s.last_event for s in states]))
    pending=save(model,states,args.output/'pending',parent=initial)
    with torch.no_grad():echo=step(model,inputs,states,3,'cuda')
    np.savez(args.output/'expected.npz',**outputs(echo))
    # Actual from-scratch gradient path, using the observed source responses as
    # geometry targets. This is one engineering update, not a learning benchmark.
    echo=step(model,inputs,states,3,'cuda')
    corrected=step(model,inputs,echo['states'],4,'cuda')
    target=inputs.batch([inputs.lookup[('C1','response')],inputs.lookup[('C2','response')]],config,'cuda')
    losses=[]
    for i,(name,x) in enumerate(target.items()):
        # Every target window contributes separately; no native vector pooling.
        scores=(corrected['slots'][name][:,None,:]*x.values).sum(-1)
        losses.append((1-scores)[~x.padding].mean())
        book=getattr(model,f'ref_values_{i}')[~getattr(model,f'ref_padding_{i}')]
        agreement=(x.values@book.T).masked_fill(x.padding[...,None],-torch.inf).amax(1)
        # A geometric reference-region target, not a compliant-conduct oracle.
        losses.append(F.cross_entropy(corrected['regions'][name],agreement.argmax(1)))
    loss=torch.stack(losses).sum()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    opt.zero_grad(set_to_none=True);loss.backward()
    grads={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
    check(bool(grads) and all(np.isfinite(v) for v in grads.values()),'GRADIENTS')
    check(any(n.startswith('update.') and v>0 for n,v in grads.items()),'RECURRENT_GRADIENT')
    check(any(n.startswith('anchor.') and v>0 for n,v in grads.items()),'CORE_ATTENTION_GRADIENT')
    opt.step();model.check_reference()
    check(any(not torch.equal(p,baseline[n]) for n,p in model.named_parameters()),'NO_LEARNING_UPDATE')
    # Derive new states through the updated model; do not publish stale states.
    with torch.no_grad():
        revised=tuple(model.start('development:C'+str(i+1)) for i in range(2))
        for phase in range(5):
            final=step(model,inputs,revised,phase,'cuda');revised=final['states']
    updated=save(model,revised,args.output/'updated',parent=pending)
    ran=time.perf_counter()
    controls=[]
    def refuses(name,fn,code):
        try:fn()
        except (ValueError,RuntimeError) as exc:
            check(code in str(exc),'WRONG_ERROR_'+name)
            controls.append(dict(name=name,code=code,refused=True));return
        raise ValueError('MNLM_LOOP_FSV_ACCEPTED_'+name)
    restored,pending_states,_=load(args.output/'pending',expected_checkpoint=pending,expected_standard=model.standard_id,device='cuda')
    units=[inputs.lookup[('C1','response')],inputs.lookup[('C2','response')]]
    native=inputs.batch(units,config,'cuda')
    kw=dict(phase=3,event_ids=event_ids(3,units,inputs),evidence_origin=['source_observation']*2)
    refuses('skip_echo',lambda:restored(native,pending_states,**{**kw,'phase':4}),'MNLM_LOOP_PHASE_ORDER')
    refuses('generated_echo',lambda:restored(native,pending_states,**{**kw,'evidence_origin':['generated_proposal']*2}),'MNLM_LOOP_CAPTURED_ECHO')
    refuses('duplicate_context',lambda:restored(native,(pending_states[0],pending_states[0]),**kw),'MNLM_LOOP_CONTEXT')
    refuses('foreign_standard_context',lambda:restored(native,(replace(pending_states[0],standard_id='0'*64),pending_states[1]),**kw),'MNLM_LOOP_STANDARD_ID')
    refuses('repeat_event',lambda:restored(native,pending_states,**{**kw,'event_ids':[s.last_event for s in pending_states]}),'MNLM_LOOP_EVENT')
    first=inputs.specs[0].name
    wrong={**native,first:replace(native[first],identity='0'*64)}
    refuses('changed_instrument',lambda:restored(wrong,pending_states,**kw),'MNLM_MODEL_IDENTITY')
    missing={n:x for n,x in native.items() if n!=first}
    refuses('missing_instrument_record',lambda:restored(missing,pending_states,**kw),'MNLM_LOOP_INPUT_ROSTER')
    empty={n:replace(x,values=torch.zeros_like(x.values),padding=torch.ones_like(x.padding),absent=torch.ones_like(x.absent)) for n,x in native.items()}
    refuses('idle_is_not_experience',lambda:restored(empty,pending_states,**kw),'MNLM_LOOP_NO_OBSERVATION')
    bad=native[first].values.clone();bad[0,0,0]=float('nan')
    refuses('nonfinite_input',lambda:restored({**native,first:replace(native[first],values=bad)},pending_states,**kw),'MNLM_MODEL_VALUES')
    with torch.no_grad():
        original=restored(native,pending_states,**kw)
        # Alter one context using another real context's recurrent history.
        altered=(replace(pending_states[0],hidden=pending_states[1].hidden.clone()),pending_states[1])
        changed=restored(native,altered,**kw)
        check(any(not torch.equal(original['slots'][n][0],changed['slots'][n][0]) for n in native),'STATE_NOT_INTRINSIC')
        check(all(torch.equal(original['slots'][n][1],changed['slots'][n][1]) for n in native),'CONTEXT_LEAK')
        check(torch.equal(original['states'][1].hidden,changed['states'][1].hidden),'CONTEXT_STATE_LEAK')
        with np.load(args.output/'expected.npz',allow_pickle=False) as z:
            check(all(np.array_equal(z[k],v) for k,v in outputs(original).items()),'ROLLBACK_REPLAY')
        controls.append(dict(name='recurrent_state_affects_own_output_only',passed=True))
        ref=restored.ref_values_0.clone();restored.ref_values_0[0,0,0]+=.01
        refuses('reference_mutation',restored.check_reference,'MNLM_LOOP_STANDARD_DRIFT')
        restored.ref_values_0.copy_(ref);restored.check_reference()
    refuses('foreign_checkpoint_pin',lambda:load(args.output/'pending',expected_checkpoint='0'*64,expected_standard=model.standard_id,device='cuda'),'MNLM_LOOP_CHECKPOINT_CORRUPT')
    refuses('foreign_core_pin',lambda:load(args.output/'pending',expected_checkpoint=pending,expected_standard='0'*64,device='cuda'),'MNLM_LOOP_CHECKPOINT_CONTRACT')
    refuses('immutable_checkpoint',lambda:save(model,revised,args.output/'updated'),'MNLM_LOOP_CHECKPOINT_EXISTS')
    # Corrupt only owned copies. Preserve the originals and their retained pins.
    shadow=args.output/'corrupt-copy';shutil.copytree(args.output/'pending',shadow)
    body=(shadow/'contexts.npy').read_bytes();changed=bytearray(body);changed[-1]^=1
    (shadow/'contexts.npy').write_bytes(changed)
    refuses('context_file_corruption',lambda:load(shadow,expected_checkpoint=pending,expected_standard=model.standard_id,device='cuda'),'MNLM_LOOP_CHECKPOINT_CORRUPT')
    (shadow/'contexts.npy').write_bytes(body)
    marker=(shadow/'COMMITTED').read_bytes();(shadow/'COMMITTED').unlink()
    refuses('interrupted_publication',lambda:load(shadow,expected_checkpoint=pending,expected_standard=model.standard_id,device='cuda'),'MNLM_LOOP_CHECKPOINT_INCOMPLETE')
    (shadow/'COMMITTED').write_bytes(marker)
    check(all(sha(shadow/n)==sha(args.output/'pending'/n) for n in ('COMMITTED','manifest.json','contexts.npy','weights.pt')),'SHADOW_NOT_RESTORED')
    load(shadow,expected_checkpoint=pending,expected_standard=model.standard_id,device='cuda')
    model.check_reference()
    checked=time.perf_counter()
    plan=dict(initial=initial,pending=pending,updated=updated,standard_id=model.standard_id,
              expected_sha256=sha(args.output/'expected.npz'),source_id=inputs.source_id)
    publish(args.output/'plan.json',plan)
    wall=time.perf_counter()-started
    report=dict(complete=True,acceptance='author-checked',purpose='intrinsic_model_state_and_persistence_engineering',
                historical_sequence_reconstruction=False,behavioural_compliance_established=False,
                truth_discrimination_established=False,meaning_asserted=False,scientific_admission=False,
                prototype_only=True,serving_version_changed=False,context_count=2,source_records=['C1','C2'],
                source=inputs.source_readback,native=inputs.window_readback,plan=plan,phase_trace=trace,
                parameters=sum(p.numel() for p in model.parameters()),one_update_loss=float(loss.detach()),
                gradients=grads,controls=controls,source_standard_unchanged=True,
                files={n:sha(args.output/n) for n in ('plan.json','expected.npz')},
                timing=dict(readback_seconds=loaded-started,model_execution_seconds=ran-loaded,
                            negative_controls_seconds=checked-ran,publication_seconds=wall-(checked-started),wall_seconds=wall))
    publish(args.output/'report.json',report)
    print(json.dumps(dict(complete=True,parameters=report['parameters'],controls=len(controls),wall_seconds=wall)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('population','native','source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cold',action='store_true')
    a=p.parse_args()
    raise SystemExit(run_cli(lambda:cold(a) if a.cold else main(a)))
