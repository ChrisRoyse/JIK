"""Six matched true/noised/actual-predicted byte renderers; no served assertions."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.qasc_context import POLICIES,check
from model.qasc_generation_data import load,TARGETS,TARGET_NATIVE
from audit.qasc_region_readback import read,read_json
from model.qasc_surface import CONFIG,ARMS,conditions,plan,evaluate,render,summarize
from model.surface_renderer import Renderer
from model.strict_surface import encode_complete,losses_encoded
from model.renderer_conditioning import gpu_lease
from model.train_qasc_regions import files
from model.scifact_readout import array_sha
from text.evaluation_multiscale import file_sha
from senses.trajectory_population import publish
from senses.contract import canonical
from lens.lenscommon import run_cli

def lines(path,rows):
    with path.open('xb') as f:
        for row in rows:f.write(canonical(row)+b'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('population','native','targets','target-native','raw','proof','region-model','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();check(not a.output.exists(),'SURFACE_OUTPUT_EXISTS');started=time.perf_counter();phases={};torch.set_num_threads(32)
    ctx=load(a.population,a.native,a.targets,a.target_native,a.raw,a.proof);cold=read(a.region_model,ctx)
    proof=read_json(a.proof/'region-fsv-linux-v1/report.json');replay=read_json(a.proof/'region-replay-linux-v1/report.json')
    check(proof['source_report_sha256']==replay['model_report_sha256']==cold['report_sha256']
          and len(proof['controls'])==11 and all(x['refused'] for x in proof['controls']) and proof['shadow_restored']
          and proof['originals_unchanged'] and replay['originals_unchanged']
          and replay['all_cases_and_saved_interventions_replayed'] and set(replay['fits'])==set(cold['summary']),'SURFACE_REGION_PROOF')
    texts=[ctx['texts'][i] for i in ctx['target_index']];packed_cpu=encode_complete(texts,'cpu',512)
    lengths=np.array([len(t.encode())+2 for t in texts]);check(lengths.max()==152,'SURFACE_LENGTH')
    a.output.mkdir(parents=True);before={**ctx['before'],str(a.region_model/'report.json'):cold['report_sha256']}
    for name,pin in read_json(a.region_model/'report.json')['files'].items():before[str(a.region_model/name)]=pin['sha256']
    for n,v in [('before',before),('region-cold',cold),('config',CONFIG),('policies',ctx['policies'])]:publish(a.output/(n+'.json'),v)
    np.savez(a.output/'complete-target-bytes.npz',ids=packed_cpu[0].numpy(),pad=packed_cpu[1].numpy(),lengths=lengths)
    phases['source_region_proofs_and_complete_codec']=time.perf_counter()-started;tick=time.perf_counter()
    residents=gpu_lease(None,'qasc-matched-complete-byte-renderers');(a.output/'gpu-before.txt').write_text(residents,encoding='utf-8')
    packed=tuple(x.cuda() for x in packed_cpu)
    code={Path(sys.modules[n].__file__).name:file_sha(Path(sys.modules[n].__file__)) for n in ('model.qasc_surface','model.surface_renderer','model.strict_surface')}
    code[Path(__file__).name]=file_sha(Path(__file__))
    publish(a.output/'runtime.json',dict(torch=torch.__version__,cuda=torch.version.cuda,platform=sys.platform,device=torch.cuda.get_device_name(0),source_code=code))
    phases['lease_upload_and_runtime']=time.perf_counter()-tick;results={};errors=[]
    for policy in POLICIES:
        tick=time.perf_counter();folder=a.output/policy;folder.mkdir();rr=ctx['policies'][policy];source=a.region_model/policy
        values=conditions(ctx,source);order,heldout,selected=plan(rr);donors=np.load(source/'donors.npy',allow_pickle=False)
        np.save(folder/'training-rows.npy',order);np.save(folder/'donors.npy',donors);np.save(folder/'heldout.npy',heldout)
        np.savez(folder/'selected.npz',**selected);np.savez(folder/'gaussian.npz',**{f'slots_{k}':v.cpu().numpy() for k,v in enumerate(values['gaussian'])})
        condition_hashes={name:[array_sha(v.cpu().numpy()) for v in arrays] for name,arrays in values.items()}
        publish(folder/'conditioning.json',dict(hashes=condition_hashes,codebook_sha256=file_sha(source/'codebook.npz'),
                predictor_outputs_sha256=file_sha(source/'seed-25/region/outputs.npz'),actual_predictor_seed=25,
                train_predictions_in_sample=True,predictions_out_of_fold=False,heldout_predictions_target_free=True,unsnapped=True))
        torch.manual_seed(25);torch.cuda.manual_seed_all(25);model=Renderer([768]*6,256,4,8)
        initial={k:v.detach().clone() for k,v in model.state_dict().items()};parameters=sum(v.numel() for v in model.parameters());del model
        torch.save(initial,folder/'initial.pt');phases[policy+'/conditioning_and_plan']=time.perf_counter()-tick
        for arm in ARMS:
            tick=time.perf_counter();root=folder/arm;root.mkdir();local={};model=opt=None
            try:
                model=Renderer([768]*6,256,4,8).cuda();model.load_state_dict(initial);model.train()
                opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.05,betas=(.9,.95));losses=[]
                torch.cuda.reset_peak_memory_stats();local['initialization']=time.perf_counter()-tick;at=time.perf_counter()
                for step,ix in enumerate(order):
                    opt.zero_grad(set_to_none=True);length=int(lengths[ix].max());s=[x[ix] for x in values[arm]]
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss,bb,eb,counts=losses_encoded(model,s,packed[0][ix,:length],packed[1][ix,:length])
                    check(bool(torch.isfinite(loss)),'SURFACE_NONFINITE_LOSS');loss.backward()
                    if step==0:
                        gradients={k:float(v.grad.norm()) if v.grad is not None else None for k,v in model.named_parameters()}
                        check(all(v is not None and np.isfinite(v) for v in gradients.values()),'SURFACE_GRADIENTS')
                        np.savez(root/'objective.npz',rows=ix,byte_nll_bits=bb.detach().cpu().numpy(),eos_nll_bits=eb.detach().cpu().numpy(),bytes=counts.cpu().numpy())
                        publish(root/'objective.json',dict(loss=float(loss.detach()),gradients=gradients))
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
                torch.cuda.synchronize();local['training']=time.perf_counter()-at;at=time.perf_counter()
                np.save(root/'losses.npy',np.array(losses));torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},root/'renderer.pt')
                publish(root/'training-time.json',dict(seconds=local['training'],updates=600,case_updates=19200,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
                model.requires_grad_(False).eval();records=evaluate(model,values,packed,lengths,heldout,rr,donors)
                lines(root/'scores.jsonl',records);metrics=summarize(records);publish(root/'metrics.json',metrics)
                local['checkpoint_and_complete_likelihood']=time.perf_counter()-at;at=time.perf_counter()
                generated=render(model,values,selected,rr,texts,arm,policy);lines(root/'renders.jsonl',generated)
                counts=dict(Counter(r['status'] for r in generated));publish(root/'surface-statuses.json',counts)
                local['strict_greedy_generation']=time.perf_counter()-at;wall=time.perf_counter()-tick
                publish(root/'report.json',dict(policy=policy,arm=arm,parameters=parameters,initial_sha256=file_sha(folder/'initial.pt'),
                        files=files(root),phases_seconds=local,wall_seconds=wall,phase_coverage=sum(local.values())/wall,
                        scientific_admission=False,served=False,guard_evaluated=False,free_text_composition_proven=False))
                results[policy+'/'+arm]=file_sha(root/'report.json')
                print(json.dumps(dict(policy=policy,arm=arm,evaluation=metrics['evaluation'],statuses=counts,seconds=wall)),flush=True)
            except (ValueError,RuntimeError) as exc:
                error=dict(policy=policy,arm=arm,error=str(exc));publish(root/'failure.json',error);errors.append(error);print(json.dumps(error),flush=True)
            finally:del model,opt;torch.cuda.empty_cache()
            phases[policy+'/'+arm]=time.perf_counter()-tick
        tick=time.perf_counter();check({name:[array_sha(v.cpu().numpy()) for v in arrays] for name,arrays in values.items()}==condition_hashes,'SURFACE_CONDITION_DRIFT')
        del values;torch.cuda.empty_cache();phases[policy+'/frozen_conditioning_check']=time.perf_counter()-tick
    tick=time.perf_counter();after={p:file_sha(Path(p)) for p in before};check(after==before,'SURFACE_PARENT_DRIFT');publish(a.output/'after.json',after)
    metadata=files(a.output);phases['final_parent_and_result_hashes']=time.perf_counter()-tick;wall=time.perf_counter()-started
    publish(a.output/'report.json',dict(schema='qasc-complete-surfaces-v1',region_report_sha256=cold['report_sha256'],targets_sha256=TARGETS,
            target_native_sha256=TARGET_NATIVE,files=metadata,results=results,errors=errors,complete=not errors,
            scientific_admission=False,served=False,sealed_evaluation=False,free_text_composition_proven=False,guard_evaluated=False,
            production_assertions=0,assertion_verdict='REFUSE_UNADMITTED_PANEL',phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall))
    print(json.dumps(dict(complete=not errors,report_sha256=file_sha(a.output/'report.json'),wall_seconds=wall)),flush=True)
    return 2 if errors else 0

if __name__=='__main__':sys.exit(run_cli(main))
