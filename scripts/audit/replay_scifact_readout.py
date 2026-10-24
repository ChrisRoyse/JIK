"""Fresh-process GPU replay of every final checkpoint and every source-bound case."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.scifact_population_readback import reference
from audit.scifact_result_readback import read,check
from model.scifact_readout import NativeStore,RelationReadout,ARMS,probabilities,array_sha
from model.renderer_conditioning import gpu_lease
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha
from lens.lenscommon import run_cli


def main():
    p=argparse.ArgumentParser()
    for n in ("population","inventory","result","output"):p.add_argument("--"+n,type=Path,required=True)
    for n in ("population-sha256","result-sha256"):p.add_argument("--"+n,required=True)
    a=p.parse_args();check(not a.output.exists(),"OUTPUT_EXISTS");start=time.perf_counter();phases={}
    torch.set_num_threads(min(32,os.cpu_count() or 1));context=reference(a.inventory)
    cold=read(a.result,a.result_sha256,a.population,a.population_sha256,a.inventory,context)
    before={str(path):file_sha(path) for path in a.result.rglob('*') if path.is_file()};before.update(context["sources"])
    a.output.mkdir(parents=True);publish(a.output/"before.json",before);publish(a.output/"cold-readback.json",cold)
    phases["source_checkpoint_and_result_readback"]=time.perf_counter()-start;tick=time.perf_counter()
    residents=gpu_lease(None,"scifact-all-final-checkpoints-fresh-process-replay");(a.output/"gpu-before.txt").write_text(residents,encoding="utf-8")
    runtime=json.loads((a.result/"runtime.json").read_bytes())
    check(runtime["torch"]==torch.__version__ and runtime["cuda"]==torch.version.cuda and runtime["device"]==torch.cuda.get_device_name(0),"REPLAY_RUNTIME")
    store=NativeStore(context,"cuda");dims=[l["dimension"] for l in context["lanes"]];results={}
    phases["lease_and_native_upload"]=time.perf_counter()-tick;tick=time.perf_counter()
    for key in cold["arms"]:
        seed,name=key.split('/');root=a.result/("seed-"+seed)/name;active,scope=ARMS[name];at=time.perf_counter()
        model=RelationReadout(dims).cuda().eval();model.load_state_dict(torch.load(root/"model.pt",map_location="cpu",weights_only=True))
        # Preserve training-checkpoint parameter flags: changing requires_grad
        # selects a different PyTorch attention path, even under inference_mode.
        # probabilities is inference-only; immutable checkpoint hashes gate drift.
        prob=probabilities(model,store,context["rows"],active,scope)
        np.save(a.output/(seed+"-"+name+".npy"),prob)
        expected=np.load(root/"probabilities.npy",allow_pickle=False);check(np.array_equal(expected,prob),"FRESH_REPLAY")
        results[key]=dict(cases=len(prob),exact=True,sha256=array_sha(prob),wall_seconds=time.perf_counter()-at)
        print(json.dumps(dict(arm=key,**results[key])),flush=True);del model;torch.cuda.empty_cache()
    phases["sixteen_complete_checkpoint_replays"]=time.perf_counter()-tick;tick=time.perf_counter()
    check(all(array_sha(x.cpu().numpy())==array_sha(y) for x,y in zip(store.native,context["native"])),"NATIVE_DRIFT")
    after={n:file_sha(Path(n)) for n in before};check(before==after,"SOURCE_CHECKPOINT_DRIFT");publish(a.output/"after.json",after)
    files={p.name:dict(sha256=file_sha(p),bytes=p.stat().st_size) for p in a.output.iterdir() if p.is_file()}
    phases["source_native_checkpoint_drift_and_hashes"]=time.perf_counter()-tick;wall=time.perf_counter()-start
    publish(a.output/"report.json",dict(schema="scifact-fresh-replay-v1",result_sha256=a.result_sha256,files=files,arms=results,
        cases=sum(v["cases"] for v in results.values()),exact=True,source_checkpoint_native_unchanged=True,scientific_admission=False,
        checkpoint_parameter_flags_preserved=True,inference_mode=True,
        phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,code_sha256=file_sha(Path(__file__))))
    print(json.dumps(dict(exact=True,arms=len(results),wall_seconds=wall)),flush=True)


if __name__=="__main__":sys.exit(run_cli(main))
