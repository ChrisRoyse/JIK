"""Complete a retained serialization-failed capture by cold audit and fresh GPU replay."""
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
from audit.scifact_rationale_result_readback import context,inspect_training,check
from audit.scifact_result_readback import equal,array_sha
from audit.scifact_rationale_comparisons import paired,diagnostics
from model.scifact_rationale import RationaleStore,RationaleReadout,evaluate
from model.scifact_rationale_metrics import evidence_metrics
from model.train_scifact_rationale import controls
from model.renderer_conditioning import gpu_lease
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha
from lens.lenscommon import run_cli


def main():
    p=argparse.ArgumentParser()
    for n in ("population","inventory","rationale","baseline","capture","output"):p.add_argument("--"+n,type=Path,required=True)
    for n in ("population-sha256","rationale-sha256","baseline-sha256","capture-sha256"):p.add_argument("--"+n,required=True)
    a=p.parse_args();check(not a.output.exists(),"OUTPUT_EXISTS");start=time.perf_counter();phases={};torch.set_num_threads(32)
    context_=context(a.population,a.population_sha256,a.inventory,a.rationale,a.rationale_sha256,a.baseline,a.baseline_sha256)
    inspected=inspect_training(a.capture,a.capture_sha256,context_);ctx=context_["native"];truth=context_["truth"]
    before=dict(ctx["sources"])
    for root in (a.population,a.rationale,a.baseline,a.capture):before.update({str(p):file_sha(p) for p in root.rglob('*') if p.is_file()})
    a.output.mkdir(parents=True);publish(a.output/"before.json",before);publish(a.output/"capture-readback.json",inspected["cold"])
    phases["independent_complete_capture_readback"]=time.perf_counter()-start;tick=time.perf_counter()
    residents=gpu_lease(None,"scifact-rationale-seven-final-checkpoints-full-replay");(a.output/"gpu-before.txt").write_text(residents,encoding="utf-8")
    prior=json.loads((a.capture/"runtime.json").read_bytes())
    check(prior["torch"]==torch.__version__ and prior["cuda"]==torch.version.cuda and prior["device"]==torch.cuda.get_device_name(0),"RUNTIME")
    publish(a.output/"runtime.json",dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),platform=sys.platform,
        model_code_sha256=file_sha(Path(__file__).with_name("scifact_rationale.py")),metrics_code_sha256=file_sha(Path(__file__).with_name("scifact_rationale_metrics.py")),
        checkpoint_parameter_flags_preserved=True,inference_mode=True))
    store=RationaleStore(ctx,truth,"cuda");dims=[l["dimension"] for l in ctx["lanes"]];results={}
    phases["lease_and_immutable_native_upload"]=time.perf_counter()-tick;tick=time.perf_counter()
    for key,cold in inspected["cold"]["arms"].items():
        seed,name=key.split('/');source=a.capture/("seed-"+seed)/name;folder=a.output/(seed+'-'+name);folder.mkdir();at=time.perf_counter()
        model=RationaleReadout(dims).cuda().eval();model.load_state_dict(torch.load(source/"model.pt",map_location="cpu",weights_only=True));torch.cuda.reset_peak_memory_stats()
        fresh=evaluate(model,store,ctx["rows"],truth);check(all(np.array_equal(v,inspected["outputs"][key][k]) for k,v in fresh.items()),"FRESH_REPLAY")
        np.savez_compressed(folder/"replay.npz",**fresh);scope=controls(model,store)
        score,agreement=evidence_metrics(fresh,ctx,truth);check(equal(score,cold["metrics"]) and equal(agreement,inspected["agreements"][key]),"INDEPENDENT_METRICS")
        publish(folder/"metrics.json",score);publish(folder/"agreements.json",agreement)
        torch.cuda.synchronize();rep=dict(model_sha256=file_sha(source/"model.pt"),cases=len(ctx["rows"]),all_output_arrays_exact=True,
            input_controls=scope,inference_peak_allocated_bytes=torch.cuda.max_memory_allocated(),inference_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            wall_seconds=time.perf_counter()-at,output_sha256={k:array_sha(v) for k,v in fresh.items()})
        publish(folder/"report.json",rep);results[key]=file_sha(folder/"report.json");print(json.dumps(dict(arm=key,exact=True,seconds=rep["wall_seconds"])),flush=True)
        del model;torch.cuda.empty_cache()
    phases["seven_full_checkpoint_replays_and_metric_reconstruction"]=time.perf_counter()-tick;tick=time.perf_counter()
    plan,samples,comparison=paired(ctx,inspected);np.savez(a.output/"bootstrap-plan.npz",**plan);np.savez(a.output/"bootstrap-samples.npz",**samples);publish(a.output/"paired-comparisons.json",comparison)
    position,multi,pairs=diagnostics(ctx,truth,inspected);publish(a.output/"position-controls.json",position);publish(a.output/"multi-sentence-only.json",multi);publish(a.output/"same-document-contrasts.json",pairs)
    phases["fixed_position_controls_conjunctions_and_bootstrap"]=time.perf_counter()-tick;tick=time.perf_counter()
    check(all(array_sha(x.cpu().numpy())==array_sha(y) for x,y in zip(store.native,ctx["native"])),"NATIVE_DRIFT")
    after={p:file_sha(Path(p)) for p in before};check(before==after,"SOURCE_DRIFT");publish(a.output/"after.json",after)
    files={p.relative_to(a.output).as_posix():dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in a.output.rglob('*') if p.is_file()}
    phases["original_source_capture_and_output_pins"]=time.perf_counter()-tick;wall=time.perf_counter()-start
    rep=dict(schema="scifact-rationale-completed-v1",capture_sha256=a.capture_sha256,parent_population_sha256=a.population_sha256,rationale_sha256=a.rationale_sha256,
        baseline_sha256=a.baseline_sha256,files=files,arms=results,all_output_arrays_replayed=True,cases_replayed=8806,complete=True,
        failed_capture_retained=True,training_repeated=False,per_arm_training_time_and_vram="unavailable:failed-before-publication",scientific_admission=False,served=False,
        sealed_evaluation=False,assertion_verdict="REFUSE_UNADMITTED_PANEL",phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(a.output/"report.json",rep);print(json.dumps(dict(report_sha256=file_sha(a.output/"report.json"),wall_seconds=wall)),flush=True)


if __name__=="__main__":sys.exit(run_cli(main))
