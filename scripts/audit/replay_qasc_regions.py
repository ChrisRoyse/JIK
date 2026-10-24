"""Same-runtime complete checkpoint and train-only codebook replay, without retraining."""
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
from model.qasc_generation_data import load
from model.qasc_context import LANES,POLICIES,check
from model.typed_regions import Store,Predictor,build_books,infer,controls
from model.qasc_region_metrics import scores
from model.renderer_conditioning import gpu_lease
from model.scifact_readout import array_sha
from audit.qasc_region_readback import read,read_json
from text.evaluation_multiscale import file_sha
from senses.trajectory_population import publish
from lens.lenscommon import run_cli

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('population','native','targets','target-native','raw','proof','model','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();check(not a.output.exists(),'REGION_REPLAY_EXISTS');start=time.perf_counter();phases={};torch.set_num_threads(32)
    ctx=load(a.population,a.native,a.targets,a.target_native,a.raw,a.proof);cold=read(a.model,ctx)
    runtime=read_json(a.model/'runtime.json');check(runtime['torch']==torch.__version__ and runtime['cuda']==torch.version.cuda
          and runtime['platform']==sys.platform,'REGION_REPLAY_RUNTIME')
    for name,pin in runtime['source_code'].items():check(file_sha(Path(__file__).parents[1]/'model'/name)==pin,'REGION_REPLAY_CODE')
    a.output.mkdir(parents=True);publish(a.output/'cold.json',cold);phases['full_input_and_result_cold_read']=time.perf_counter()-start;tick=time.perf_counter()
    residents=gpu_lease(None,'qasc-region-full-checkpoint-replay');(a.output/'gpu-before.txt').write_text(residents,encoding='utf-8')
    check(torch.cuda.get_device_name(0)==runtime['device'],'REGION_REPLAY_DEVICE');store=Store(ctx)
    phases['runtime_lease_and_upload']=time.perf_counter()-tick;results={};book_proof={}
    for policy in POLICIES:
        tick=time.perf_counter();folder=a.model/policy;rr=ctx['policies'][policy]
        train=np.array([i for i,r in enumerate(rr) if r['partition']=='train'],np.int64);unique=store.prepare(train)
        books,codes,histories,initial=build_books(store,unique)
        with np.load(folder/'codebook.npz',allow_pickle=False) as z:
            check(np.array_equal(z['codes'],codes.cpu().numpy()) and np.array_equal(z['initial_positions'],initial),'REGION_REPLAY_CODES')
            for k,name in enumerate(LANES):
                check(np.array_equal(z[name+'_mean'],store.means[k].cpu().numpy()) and np.array_equal(z[name+'_centers'],books[k].cpu().numpy())
                      and np.array_equal(z[name+'_counts'],histories[k]),'REGION_REPLAY_BOOK')
        book_proof[policy]=dict(all_means_centers_assignments_counts_bitwise_equal=True,train_only=True)
        truth=[v.cpu().numpy() for v in store.y];code_array=codes.cpu().numpy();donor=np.load(folder/'donors.npy',allow_pickle=False)
        phases[policy+'/codebook_rebuild']=time.perf_counter()-tick
        for seed in (25,26,27):
            for arm in (('region','direct','shuffled') if seed==25 else ('region','direct')):
                tick=time.perf_counter();key=f'{policy}/seed-{seed}/{arm}';root=a.model/key;placement='region' if arm=='shuffled' else arm
                model=Predictor(books).cuda().train();model.load_state_dict(torch.load(root.parent/'initial.pt',map_location='cpu',weights_only=True))
                with np.load(root/'objective.npz',allow_pickle=False) as z:
                    v,l=model(store.batch(z['rows']),placement)
                    check(all(np.array_equal(v[k].detach().cpu().numpy(),z[f'slots_{k}']) and np.array_equal(l[k].detach().cpu().numpy(),z[f'logits_{k}']) for k in range(6)),'REGION_REPLAY_INITIAL')
                model.load_state_dict(torch.load(root/'model.pt',map_location='cpu',weights_only=True));model.requires_grad_(False).eval();v,l=infer(model,store,placement)
                hashes={}
                with np.load(root/'outputs.npz',allow_pickle=False) as z:
                    for k in range(6):
                        for kind,value in (('slots',v[k]),('logits',l[k])):
                            check(np.array_equal(value,z[f'{kind}_{k}']),'REGION_REPLAY_OUTPUT_DRIFT');hashes[f'{kind}_{k}']=array_sha(value)
                if arm=='region':
                    with np.load(root/'interventions.npz',allow_pickle=False) as z:
                        for scope in ('fact1-only','fact2-only','question-only','facts-only','other-group-facts'):
                            v,l=infer(model,store,placement,'both-facts' if scope=='other-group-facts' else scope,donor if scope=='other-group-facts' else None)
                            c,ok=scores(v,truth,ctx['target_index'],l,code_array)
                            check(np.array_equal(c,z[scope+'_cosine']) and np.array_equal(ok,z[scope+'_correct']),'REGION_REPLAY_INTERVENTION')
                check(controls(model,store,placement)==read_json(root/'controls.json'),'REGION_REPLAY_CONTROLS')
                results[key]=dict(checkpoint_sha256=file_sha(root/'model.pt'),all_8134_cases_replayed=True,
                        all_initial_and_final_values_bitwise_equal=True,all_saved_interventions_bitwise_equal=True,output_hashes=hashes)
                del model;torch.cuda.empty_cache();phases[key]=time.perf_counter()-tick
                print(json.dumps(dict(replayed=key,seconds=phases[key])),flush=True)
    tick=time.perf_counter();check(file_sha(a.model/'report.json')==cold['report_sha256'],'REGION_REPLAY_RESULT_DRIFT')
    check(all(file_sha(Path(p))==h for p,h in ctx['before'].items()),'REGION_REPLAY_PARENT_DRIFT')
    phases['final_hashes']=time.perf_counter()-tick;wall=time.perf_counter()-start
    publish(a.output/'report.json',dict(model_report_sha256=cold['report_sha256'],fits=results,codebooks=book_proof,
            all_cases_and_saved_interventions_replayed=True,originals_unchanged=True,acceptance='author-checked',
            scientific_admission=False,served=False,phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,
            code_sha256=file_sha(Path(__file__))))
    print(json.dumps(dict(fits=len(results),report_sha256=file_sha(a.output/'report.json'),wall_seconds=wall)),flush=True)

if __name__=='__main__':sys.exit(run_cli(main))
