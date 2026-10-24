"""Complete same-runtime replay of paired objectives, gradients and checkpoints."""
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
from model.multirc_binding import Store,Binding,infer,controls
from model.multirc_contrast import ARMS,scope,objective,objective_controls
from model.renderer_conditioning import gpu_lease
from model.scifact_readout import array_sha
from audit.multirc_binding_readback import read_json
from audit.multirc_contrast_readback import read
from text.evaluation_multiscale import file_sha
from senses.trajectory_population import publish
from lens.lenscommon import run_cli

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('population','native','quality','raw','baseline','model','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();check(not a.output.exists(),'CONTRAST_REPLAY_EXISTS');start=time.perf_counter();phases={};torch.set_num_threads(32)
    ctx=load(a.population,a.native,a.quality,a.raw);cold=read(a.model,ctx,a.native,a.raw,a.baseline);runtime=read_json(a.model/'runtime.json')
    check(runtime['torch']==torch.__version__ and runtime['cuda']==torch.version.cuda and runtime['platform']==sys.platform,'CONTRAST_REPLAY_RUNTIME')
    for name,pin in runtime['source_code'].items():check(file_sha(Path(__file__).parents[1]/(name.replace('.','/')+'.py'))==pin,'CONTRAST_REPLAY_CODE')
    a.output.mkdir(parents=True);publish(a.output/'cold.json',cold);phases['source_result_pairs_and_arithmetic_readback']=time.perf_counter()-start;tick=time.perf_counter()
    residents=gpu_lease(None,'multirc-contrast-complete-checkpoint-replay');(a.output/'gpu-before.txt').write_text(residents,encoding='utf-8')
    check(torch.cuda.get_device_name(0)==runtime['device'],'CONTRAST_REPLAY_DEVICE');store=Store(ctx)
    with np.load(a.model/'evaluation-scope.npz',allow_pickle=False) as z:ids=z['ids'];heldout=z['heldout'];donor=z['donors']
    phases['lease_and_upload']=time.perf_counter()-tick;results={}
    for seed in (25,26,27):
        for arm in (*ARMS,*(['shuffled-rank'] if seed==25 else [])):
            tick=time.perf_counter();key=f'seed-{seed}/{arm}';root=a.model/key;model_scope=scope(arm);ranked=arm!='joint-ce'
            model=Binding().cuda();model.load_state_dict(torch.load(root.parent/'initial.pt',map_location='cpu',weights_only=True))
            with np.load(root/'objective.npz',allow_pickle=False) as z:
                logits=model(store.batch(z['rows']),model_scope);check(np.array_equal(logits.detach().cpu().numpy(),z['logits']),'CONTRAST_REPLAY_INITIAL')
                labels=torch.as_tensor(z['labels'],device='cuda');pairs=torch.as_tensor(z['pairs'],device='cuda')
                loss,terms,counts=objective(logits,labels,pairs,ranked);loss.backward();obj=read_json(root/'objective.json')
                check(float(loss.detach())==obj['loss'] and terms.detach().cpu().tolist()==obj['terms'] and counts.cpu().tolist()==obj['counts'],'CONTRAST_REPLAY_OBJECTIVE')
                check(objective_controls(logits.detach(),labels,pairs,ranked)==read_json(root/'objective-controls.json'),'CONTRAST_REPLAY_OBJECTIVE_CONTROLS')
            gradients={k:float(v.grad.norm()) if v.grad is not None else None for k,v in model.named_parameters()}
            check(gradients==obj['gradients'],'CONTRAST_REPLAY_GRADIENTS')
            model.load_state_dict(torch.load(root/'model.pt',map_location='cpu',weights_only=True));model.requires_grad_(False).eval()
            values=infer(model,store,model_scope,ids);check(np.array_equal(values,np.load(root/'logits.npy',allow_pickle=False)),'CONTRAST_REPLAY_FINAL')
            hashes={'primary':array_sha(values)}
            if arm=='joint-rank':
                with np.load(root/'interventions.npz',allow_pickle=False) as z:
                    for kind in ('remove-context','question-swap','passage-swap'):
                        value=infer(model,store,model_scope,heldout,kind,donor);check(np.array_equal(value,z[kind]),'CONTRAST_REPLAY_INTERVENTION');hashes[kind]=array_sha(value)
            check(controls(model,store,ids)==read_json(root/'controls.json'),'CONTRAST_REPLAY_FORWARD_CONTROLS')
            results[key]=dict(checkpoint_sha256=file_sha(root/'model.pt'),primary_options=len(ids),intervention_options=3*len(heldout) if arm=='joint-rank' else 0,
                initial_objective_and_gradients_bitwise_equal=True,all_final_outputs_bitwise_equal=True,hashes=hashes)
            del model;torch.cuda.empty_cache();phases[key]=time.perf_counter()-tick;print(json.dumps(dict(replayed=key,seconds=phases[key])),flush=True)
    tick=time.perf_counter();check(file_sha(a.model/'report.json')==cold['report_sha256'],'CONTRAST_REPLAY_RESULT_DRIFT')
    check(all(file_sha(Path(p))==v for p,v in read_json(a.model/'source-before.json').items()),'CONTRAST_REPLAY_PARENT_DRIFT')
    phases['immutable_parent_hashes']=time.perf_counter()-tick;wall=time.perf_counter()-start
    publish(a.output/'report.json',dict(model_report_sha256=cold['report_sha256'],fits=results,all_saved_outputs_replayed=True,originals_unchanged=True,
        acceptance='author-checked',scientific_admission=False,served=False,phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,code_sha256=file_sha(Path(__file__))))
    print(json.dumps(dict(fits=len(results),report_sha256=file_sha(a.output/'report.json'),wall_seconds=wall)),flush=True)

if __name__=='__main__':sys.exit(run_cli(main))
