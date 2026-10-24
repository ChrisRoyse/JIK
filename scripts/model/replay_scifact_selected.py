"""Fresh same-runtime checkpoint and actual-input intervention replay."""
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.scifact_selected_source import arguments,load,PINS
from model.scifact_selected import SelectedReadout,SelectedStore,arm_plan,check,controls
from model.train_scifact_selected import all_outputs
from model.renderer_conditioning import gpu_lease
from audit.scifact_selected_readback import read_result
from audit.scifact_result_readback import array_sha
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha
from lens.lenscommon import run_cli


def main():
    a=arguments(("result",));check(not a.output.exists(),"OUTPUT_EXISTS");start=time.perf_counter();torch.set_num_threads(32);phases={}
    source=load(a);pin=file_sha(a.result/"report.json");readback=read_result(a.result,pin,source);ctx=source["ctx"]
    before={**source["before"],**{str(p):file_sha(p) for p in a.result.rglob('*') if p.is_file()}}
    a.output.mkdir(parents=True);publish(a.output/"before.json",before);publish(a.output/"cold-readback.json",readback["cold"])
    phases["source_selector_gathers_and_independent_results"]=time.perf_counter()-start;tick=time.perf_counter()
    resident=gpu_lease(None,"scifact-selected-thirteen-fresh-replays");(a.output/"gpu-before.txt").write_text(resident,encoding="utf-8")
    prior=readback["runtime"];runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),platform=sys.platform)
    check(all(runtime[k]==prior[k] for k in runtime),"RUNTIME");publish(a.output/"runtime.json",runtime)
    store=SelectedStore(ctx,source["plans"],"cuda");dims=[l["dimension"] for l in ctx["lanes"]];phases["lease_and_upload"]=time.perf_counter()-tick;tick=time.perf_counter();arms={}
    for key,old in readback["outputs"].items():
        seed,name=key.split('/');plan=arm_plan(int(seed),name);folder=a.output/(seed+'-'+name);folder.mkdir();at=time.perf_counter()
        model=SelectedReadout(dims).cuda().eval();model.load_state_dict(torch.load(a.result/f"seed-{seed}"/name/"model.pt",map_location="cpu",weights_only=True))
        torch.cuda.reset_peak_memory_stats();actual=all_outputs(model,store,ctx["rows"],plan,name,source["donors"])
        check(set(old)==set(actual) and all(np.array_equal(old[n],v) for n,v in actual.items()),"FRESH_REPLAY")
        np.savez(folder/"outputs.npz",**actual);c=controls(model,store,ctx,plan) if name=="selected" else None
        check(c==json.loads((a.result/f"seed-{seed}"/name/"controls.json").read_bytes()),"CONTROLS_REPLAY")
        torch.cuda.synchronize();publish(folder/"report.json",dict(model_sha256=file_sha(a.result/f"seed-{seed}"/name/"model.pt"),controls=c,
            output_sha256={n:array_sha(v) for n,v in actual.items()},cases=len(ctx["rows"]),output_arrays=len(actual),all_outputs_exact=True,
            inference_seconds=time.perf_counter()-at,inference_peak_allocated_bytes=torch.cuda.max_memory_allocated(),inference_peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        arms[key]=file_sha(folder/"report.json");del model;torch.cuda.empty_cache();print(json.dumps(dict(arm=key,exact=True)),flush=True)
    phases["thirteen_replays_and_source_mutation_controls"]=time.perf_counter()-tick;tick=time.perf_counter()
    check(all(array_sha(x.cpu().numpy())==array_sha(y) for x,y in zip(store.native,ctx["native"])),"NATIVE_DRIFT")
    after={p:file_sha(Path(p)) for p in before};check(after==before,"SOURCE_DRIFT");publish(a.output/"after.json",after)
    files={p.relative_to(a.output).as_posix():dict(sha256=file_sha(p),bytes=p.stat().st_size) for p in a.output.rglob('*') if p.is_file()}
    phases["source_result_and_replay_pins"]=time.perf_counter()-tick;wall=time.perf_counter()-start
    publish(a.output/"report.json",dict(schema="scifact-selected-replay-v1",result_sha256=pin,files=files,arms=arms,complete=True,
        phase_coverage=sum(phases.values())/wall,phases_seconds=phases,wall_seconds=wall,parameters_require_grad_preserved=True,
        inference_mode=True,scientific_admission=False,served=False,assertion_verdict="REFUSE_UNADMITTED_PANEL"))
    print(json.dumps(dict(report_sha256=file_sha(a.output/"report.json"),wall_seconds=wall)),flush=True)


if __name__=="__main__":sys.exit(run_cli(main))
