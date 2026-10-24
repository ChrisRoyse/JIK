"""Fresh leased replay of all saved choice distributions, plus checkpoint drift control."""
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
from model.qasc_context import load,check
from model.qasc_combination import Store,Combination,evaluate,controls
from model.renderer_conditioning import gpu_lease
from audit.qasc_combination_readback import read
from model.scifact_readout import array_sha
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha
from lens.lenscommon import run_cli

def main():
    p=argparse.ArgumentParser()
    for n in ('population','native','raw','result','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();check(not a.output.exists(),'REPLAY_EXISTS');torch.set_num_threads(32);start=time.perf_counter();phases={}
    ctx=load(a.population,a.native,a.raw);result=read(a.result,ctx);runtime=json.loads((a.result/'runtime.json').read_bytes())
    check(runtime['torch']==torch.__version__ and runtime['cuda']==torch.version.cuda and runtime['platform']==sys.platform,'REPLAY_RUNTIME')
    a.output.mkdir(parents=True);publish(a.output/'cold.json',result['cold']);before={**ctx['before'],str(a.result/'report.json'):file_sha(a.result/'report.json')}
    rep=json.loads((a.result/'report.json').read_bytes())
    for name,v in rep['files'].items():before[str(a.result/name)]=v['sha256']
    publish(a.output/'before.json',before);phases['source_and_result_cold_read']=time.perf_counter()-start;tick=time.perf_counter()
    residents=gpu_lease(None,'qasc-32-checkpoint-full-output-replay');(a.output/'gpu-before.txt').write_text(residents,encoding='utf-8')
    store=Store(ctx,'cuda');phases['resident_check_lease_and_native_upload']=time.perf_counter()-tick;arms={};drift_done=False
    for key,expected in result['outputs'].items():
        tick=time.perf_counter();policy,seed,name=key.split('/');folder=a.result/key;scope='both-facts' if name=='shuffled' else name
        state=torch.load(folder/'model.pt',map_location='cpu',weights_only=True);model=Combination().cuda();model.load_state_dict(state);model.eval()
        donor=np.load(a.result/policy/'donors.npy');outputs={}
        for channel in expected:
            mode=scope if channel=='probabilities' else 'both-facts' if channel=='other-group-facts' else channel
            actual=evaluate(model,store,8134,mode,donor if channel=='other-group-facts' else None)
            check(np.array_equal(actual,expected[channel]),'REPLAY_OUTPUT_DRIFT');outputs[channel]=actual
        proof=controls(model,store);check(proof==json.loads((folder/'controls.json').read_bytes()),'REPLAY_CONTROLS_DRIFT')
        if not drift_done:
            owned=a.output/'checkpoint-control.pt';owned.write_bytes((folder/'model.pt').read_bytes());before_hash=file_sha(owned)
            changed={k:v.clone() for k,v in state.items()};changed['head.1.weight'][0,0]+=.25;torch.save(changed,owned);action_hash=file_sha(owned)
            model.load_state_dict(torch.load(owned,map_location='cpu',weights_only=True))
            with torch.inference_mode():mutated=model(store.batch(np.arange(32)),scope).softmax(1).cpu().numpy()
            check(not np.array_equal(mutated,expected['probabilities'][:32]),'REPLAY_DRIFT_CONTROL_INSENSITIVE')
            owned.write_bytes((folder/'model.pt').read_bytes());model.load_state_dict(torch.load(owned,map_location='cpu',weights_only=True))
            with torch.inference_mode():restored=model(store.batch(np.arange(32)),scope).softmax(1).cpu().numpy()
            check(np.array_equal(restored,expected['probabilities'][:32]) and file_sha(owned)==before_hash,'REPLAY_CHECKPOINT_RESTORE')
            np.savez(a.output/'checkpoint-control.npz',changed=mutated,restored=restored)
            publish(a.output/'checkpoint-control.json',dict(before_sha256=before_hash,action_sha256=action_hash,after_sha256=file_sha(owned),changed_predictions_refused=True,restored_exact=True));drift_done=True
        dest=a.output/key;dest.mkdir(parents=True);np.savez(dest/'outputs.npz',**outputs)
        proof=dict(model_sha256=file_sha(folder/'model.pt'),cases=8134,arrays=len(outputs),all_outputs_bitwise_equal=True,output_sha256={n:array_sha(v) for n,v in outputs.items()})
        publish(dest/'report.json',proof);arms[key]=file_sha(dest/'report.json');del model;torch.cuda.empty_cache();phases[key]=time.perf_counter()-tick
        print(json.dumps(dict(arm=key,arrays=len(outputs),all_exact=True)),flush=True)
    tick=time.perf_counter();after={p:file_sha(Path(p)) for p in before};check(after==before,'REPLAY_SOURCE_DRIFT');publish(a.output/'after.json',after)
    files={p.relative_to(a.output).as_posix():dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in a.output.rglob('*') if p.is_file()}
    phases['immutable_originals_and_result_hashes']=time.perf_counter()-tick;wall=time.perf_counter()-start
    publish(a.output/'report.json',dict(schema='qasc-combination-replay-v1',result_sha256=result['cold']['report_sha256'],arms=arms,files=files,complete=True,
        scientific_admission=False,served=False,production_assertions=0,all_outputs_replayed=True,phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall))
    print(json.dumps(dict(complete=True,checkpoints=len(arms),wall_seconds=wall,sha256=file_sha(a.output/'report.json'))))

if __name__=='__main__':sys.exit(run_cli(main))
