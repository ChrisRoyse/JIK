"""Fresh-process GPU checkpoint replay with exact runtime and native population pins."""
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
from audit.anchor_readout_readback import read
from model.anchor_readout import AnchorReadout,prepare,check
from model.train_anchor_readout import probabilities
from model.renderer_conditioning import gpu_lease
from lens.lenscommon import run_cli
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha


def main():
    p=argparse.ArgumentParser()
    for n in ("population","result","output"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--report-sha256",required=True);p.add_argument("--reviewed-desktop-residents",type=Path)
    a=p.parse_args();check(not a.output.exists(),"OUTPUT_EXISTS");started=time.perf_counter();phases={}
    read(a.population,a.result,a.report_sha256);r=json.loads((a.result/"report.json").read_bytes())
    for n,h in r["code_sha256"].items():check(file_sha(Path(__file__).parent/n)==h,"REPLAY_CODE")
    rows,memories,masks,questions,_,_,pins=prepare(a.population)
    check(pins==json.loads((a.result/"input-pins.json").read_bytes()),"REPLAY_INPUTS")
    phases["cold_source_input_and_runtime_readback"]=time.perf_counter()-started;tick=time.perf_counter()
    residents=gpu_lease(a.reviewed_desktop_residents,"source-anchor-fresh-process-checkpoint-replay")
    runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,platform=sys.platform,device=torch.cuda.get_device_name(0))
    check(runtime==r["runtime"],"REPLAY_RUNTIME")
    a.output.mkdir(parents=True);(a.output/"gpu-before.txt").write_text(residents,encoding="utf-8")
    torch.set_num_threads(min(32,os.cpu_count() or 1));memories=[x.cuda() for x in memories];masks=[x.cuda() for x in masks];questions=questions.cuda()
    phases["lease_and_native_upload"]=time.perf_counter()-tick;tick=time.perf_counter();outcomes={}
    for key,arm in r["arms"].items():
        seed,name=key.split('/');source=a.result/("seed-"+seed)/name
        model=AnchorReadout([l["dimension"] for l in r["lanes"]]).cuda().eval()
        model.load_state_dict(torch.load(source/"model.pt",map_location="cpu",weights_only=True))
        replay=probabilities(model,memories,masks,questions,arm["active_lanes"])
        prior=np.load(source/"probabilities.npy",allow_pickle=False);same=np.array_equal(prior,replay)
        output=a.output/(seed+"-"+name+".npy");np.save(output,replay)
        outcomes[key]=dict(exact=same,rows=len(replay),sha256=file_sha(output));check(same,"FRESH_CHECKPOINT_REPLAY")
        del model
    torch.cuda.synchronize();phases["seven_checkpoint_replays"]=time.perf_counter()-tick;tick=time.perf_counter()
    read(a.population,a.result,a.report_sha256,audit_inputs=False)
    phases["source_and_checkpoint_drift_check"]=time.perf_counter()-tick;wall=time.perf_counter()-started
    publish(a.output/"report.json",dict(result_sha256=a.report_sha256,author_checked=True,arms=outcomes,exact=True,runtime=runtime,
        scientific_admission=False,phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall))
    print(json.dumps(dict(arms=len(outcomes),rows=len(rows)*len(outcomes),exact=True,wall_seconds=wall)))


if __name__=="__main__":sys.exit(run_cli(main))
