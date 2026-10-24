"""Full source membership and OR-of-AND target FSV, with real consumer gathers."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.scifact_population_readback import reference
from audit.scifact_rationale_readback import expected,read
from audit.scifact_readback import jsonl
from model.scifact_rationale import RationaleStore
from model.scifact_readout import array_sha
from senses.contract import canonical
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha


def main():
    p=argparse.ArgumentParser()
    for n in ("population","inventory","rationale","output"):p.add_argument("--"+n,type=Path,required=True)
    for n in ("population-sha256","rationale-sha256"):p.add_argument("--"+n,required=True)
    a=p.parse_args();torch.set_num_threads(1);start=time.perf_counter();phases={};a.output.mkdir(parents=True,exist_ok=False)
    ctx=reference(a.inventory);truth=expected(ctx)
    before=dict(ctx["sources"])
    for root in (a.population,a.rationale):before.update({str(p):file_sha(p) for p in root.rglob('*') if p.is_file()})
    publish(a.output/"before.json",before)
    def cold(root,pin):return read(root,pin,a.population,a.population_sha256,a.inventory,ctx,truth)
    initial=cold(a.rationale,a.rationale_sha256);publish(a.output/"cold-readback.json",initial)
    checks=[dict(name="B1/B2 every original source sentence, native-window occurrence and complete alternative path",passed=True)]
    phases["complete_source_and_supervision_readback"]=time.perf_counter()-start;tick=time.perf_counter()
    shadow=a.output/"owned-shadow";shutil.copytree(a.rationale,shadow);controls=a.output/"controls";controls.mkdir()
    def fault(name,path,fn,code,kind="json"):
        file=shadow/path;restore={path:file.read_bytes(),"report.json":(shadow/"report.json").read_bytes()}
        if kind=="npz":
            with np.load(file,allow_pickle=False) as z:value={k:z[k].copy() for k in z.files}
        elif kind=="npy":value=np.load(file,allow_pickle=False).copy()
        elif kind=="jsonl":value=jsonl(file)
        else:value=json.loads(file.read_bytes())
        changed=fn(value)
        if changed is not None:value=changed
        if kind=="npz":np.savez_compressed(file,**value)
        elif kind=="npy":np.save(file,value)
        else:file.write_bytes(b''.join(canonical(v)+b'\n' for v in value) if kind=="jsonl" else canonical(value))
        if path!="report.json":
            rep=json.loads(restore["report.json"]);rep["files"][path]=dict(bytes=file.stat().st_size,sha256=file_sha(file));(shadow/"report.json").write_bytes(canonical(rep))
        publish(controls/(name+".json"),dict(path=path,mutant_sha256=file_sha(file)))
        try:cold(shadow,file_sha(shadow/"report.json"));observed="accepted"
        except ValueError as exc:observed=str(exc)
        checks.append(dict(name=name,expected=code,observed=observed,passed=code in observed))
        for n,raw in restore.items():(shadow/n).write_bytes(raw)
    fault("B1_wrong_parent","report.json",lambda r:r.update(parent_population_sha256="0"*64),"PARENT")
    fault("B1_missing_sentence","sentences.jsonl",lambda r:r[0].update(sentences=r[0]["sentences"][:-1]),"SENTENCES","jsonl")
    fault("B1_wrong_partition","paths.jsonl",lambda r:r[0].update(partition="wrong"),"PATHS","jsonl")
    multi,alternative=next((i,j) for i,r in enumerate(truth["paths"]) for j,p in enumerate(r["alternatives"]) if len(p)>2)
    alternatives=next(i for i,r in enumerate(truth["paths"]) if len(r["alternatives"])>1)
    def partial(r):del r[multi]["alternatives"][alternative][0]
    fault("B2_partial_AND","paths.jsonl",partial,"PATHS","jsonl")
    def merge(r):r[alternatives]["alternatives"]=[sorted(set(v for p in r[alternatives]["alternatives"] for v in p))]
    fault("B2_merged_OR","paths.jsonl",merge,"PATHS","jsonl")
    path="membership/"+ctx["lanes"][0]["name"]+".npz"
    fault("B2_missing_native_window",path,lambda r:r["membership"].__setitem__((0,0),False),"MEMBERSHIP","npz")
    fault("B2_shifted_source_window",path,lambda r:r.update(membership=np.roll(r["membership"],1,axis=2)),"MEMBERSHIP","npz")
    fault("B2_changed_validity","valid-sentences.npy",lambda r:r.__setitem__((0,0),False),"VALID","npy")
    for key in ("choices","steps","present"):
        def change(r,k=key):r[k].flat[0]=r[k].flat[0]-1 if k=="choices" else not r[k].flat[0]
        fault("B2_wrong_"+key,"targets.npz",change,"TARGET_"+key.upper(),"npz")
    fault("B2_injected_gold_input","targets.npz",lambda r:r.update(oracle=np.zeros(1258)),"TARGET_KEYS","npz")
    for flag in ("served","scientific_admission","sealed_evaluation"):
        fault("B3_false_"+flag,"report.json",lambda r,k=flag:r.update({k:True}),"CLAIM")
    phases["persisted_semantic_fault_controls"]=time.perf_counter()-tick;tick=time.perf_counter()
    store=RationaleStore(ctx,truth,"cpu");ids=sorted({0,multi,alternatives,1257});memories,valid=store.batch(ids);doc=[ctx["rows"][i]["document_index"] for i in ids];bindings=[]
    for k,m in enumerate(memories):
        same=np.array_equal(m["membership"].numpy(),truth["members"][k][doc]);checks.append(dict(name="B3 actual occurrence membership "+ctx["lanes"][k]["name"],passed=same))
        bindings.append(dict(lane=ctx["lanes"][k]["name"],membership_sha256=array_sha(m["membership"].numpy())))
    checks.append(dict(name="B3 actual valid source sentences",passed=np.array_equal(valid.numpy(),truth["valid"][doc])))
    publish(a.output/"consumer-readback.json",dict(cases=ids,bindings=bindings,valid_sha256=array_sha(valid.numpy()),model_forward=False))
    phases["real_typed_consumer"]=time.perf_counter()-tick;tick=time.perf_counter();after={p:file_sha(Path(p)) for p in before};publish(a.output/"after.json",after)
    checks.append(dict(name="B1/B2/B3 originals unchanged and exact shadow restoration",passed=before==after and cold(shadow,a.rationale_sha256)==initial))
    phases["drift_and_restoration"]=time.perf_counter()-tick;wall=time.perf_counter()-start
    result=dict(passed=all(c["passed"] for c in checks),checks=checks,acceptance="author-checked",scientific_admission=False,phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(a.output/"fsv.json",result);print(json.dumps(dict(passed=result["passed"],checks=len(checks),failed=[c for c in checks if not c["passed"]],wall_seconds=wall)))
    return 0 if result["passed"] else 2


if __name__=="__main__":raise SystemExit(main())
