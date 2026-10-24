"""Replay every initial objective, final checkpoint and saved binding intervention."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.multirc_context import load,check
from model.multirc_binding import ARMS,Store,Binding,infer,controls
from model.renderer_conditioning import gpu_lease
from audit.multirc_result_readback import read
from audit.multirc_binding_readback import read_json
from model.scifact_readout import array_sha
from text.evaluation_multiscale import file_sha
from senses.trajectory_population import publish
from lens.lenscommon import run_cli

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('population','native','quality','raw','model','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();check(not a.output.exists(),'REPLAY_EXISTS');start=time.perf_counter();phases={};torch.set_num_threads(32)
    ctx=load(a.population,a.native,a.quality,a.raw);cold=read(a.model,ctx,a.native,a.raw);runtime=read_json(a.model/'runtime.json')
    check(runtime['torch']==torch.__version__ and runtime['cuda']==torch.version.cuda and runtime['platform']==sys.platform,'REPLAY_RUNTIME')
    for name,pin in runtime['source_code'].items():check(file_sha(Path(__file__).parents[1]/(name.replace('.','/')+'.py'))==pin,'REPLAY_SOURCE_CODE')
    a.output.mkdir(parents=True);publish(a.output/'cold.json',cold);phases['complete_source_result_and_metric_readback']=time.perf_counter()-start;tick=time.perf_counter()
    residents=gpu_lease(None,'multirc-binding-full-checkpoint-replay');(a.output/'gpu-before.txt').write_text(residents,encoding='utf-8')
    check(torch.cuda.get_device_name(0)==runtime['device'],'REPLAY_DEVICE');store=Store(ctx)
    with np.load(a.model/'evaluation-scope.npz',allow_pickle=False) as z:ids=z['ids'];heldout=z['heldout'];donor=z['donors']
    phases['lease_and_upload']=time.perf_counter()-tick;results={}
    for seed in (25,26,27):
        for arm in (*ARMS,*(['shuffled'] if seed==25 else [])):
            tick=time.perf_counter();key=f'seed-{seed}/{arm}';root=a.model/key;scope='joint' if arm=='shuffled' else arm
            model=Binding().cuda();model.load_state_dict(torch.load(root.parent/'initial.pt',map_location='cpu',weights_only=True))
            with np.load(root/'objective.npz',allow_pickle=False) as z:
                value=model(store.batch(z['rows']),scope);check(np.array_equal(value.detach().cpu().numpy(),z['logits']),'REPLAY_INITIAL')
                loss=torch.nn.functional.cross_entropy(value,torch.as_tensor(z['labels'],device='cuda'));loss.backward()
            gradients={k:float(v.grad.norm()) if v.grad is not None else None for k,v in model.named_parameters()}
            check(gradients==read_json(root/'objective.json')['gradients'],'REPLAY_GRADIENTS')
            model.load_state_dict(torch.load(root/'model.pt',map_location='cpu',weights_only=True));model.requires_grad_(False).eval()
            values=infer(model,store,scope,ids);check(np.array_equal(values,np.load(root/'logits.npy',allow_pickle=False)),'REPLAY_FINAL')
            hashes={'primary':array_sha(values)}
            if arm=='joint':
                with np.load(root/'interventions.npz',allow_pickle=False) as z:
                    for kind in ('remove-context','question-swap','passage-swap'):
                        v=infer(model,store,scope,heldout,kind,donor);check(np.array_equal(v,z[kind]),'REPLAY_INTERVENTION');hashes[kind]=array_sha(v)
            check(controls(model,store,ids)==read_json(root/'controls.json'),'REPLAY_FORWARD_CONTROLS')
            results[key]=dict(checkpoint_sha256=file_sha(root/'model.pt'),primary_options=len(ids),intervention_options=3*len(heldout) if arm=='joint' else 0,
                initial_logits_and_gradients_bitwise_equal=True,all_final_logits_and_interventions_bitwise_equal=True,hashes=hashes)
            del model;torch.cuda.empty_cache();phases[key]=time.perf_counter()-tick;print(json.dumps(dict(replayed=key,seconds=phases[key])),flush=True)
    tick=time.perf_counter();check(file_sha(a.model/'report.json')==cold['report_sha256'],'RESULT_DRIFT')
    check(all(file_sha(Path(p))==h for p,h in ctx['before'].items()),'PARENT_DRIFT');phases['immutable_parent_hashes']=time.perf_counter()-tick;wall=time.perf_counter()-start
    publish(a.output/'report.json',dict(model_report_sha256=cold['report_sha256'],fits=results,all_saved_outputs_replayed=True,originals_unchanged=True,
            acceptance='author-checked',scientific_admission=False,served=False,phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,
            code_sha256=file_sha(Path(__file__))))
    print(json.dumps(dict(fits=len(results),report_sha256=file_sha(a.output/'report.json'),wall_seconds=wall)),flush=True)

if __name__=='__main__':sys.exit(run_cli(main))
