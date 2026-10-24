"""Source, reference-cache and actual window-loader FSV on the full population."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.scifact_input_readback import read
from audit.scifact_readback import raw_source,jsonl
from senses.contract import canonical
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha


def inventory(root):
    return {p.relative_to(root).as_posix():file_sha(p) for p in root.rglob('*') if p.is_file()}


def main():
    p=argparse.ArgumentParser()
    for n in ("source","publication","inputs","output"):p.add_argument("--"+n,type=Path,required=True)
    for n in ("source","publication","inputs"):p.add_argument("--"+n+"-sha256",required=True)
    a=p.parse_args();started=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False);phases={}
    before={n:inventory(getattr(a,n)) for n in ("source","publication","inputs")};publish(a.output/"before.json",before)
    context=raw_source(a.source,a.source_sha256)
    def audit(root,pin):return read(a.source,a.source_sha256,a.publication,a.publication_sha256,root,pin,context)
    cold=audit(a.inputs,a.inputs_sha256);publish(a.output/"cold-readback.json",cold)
    checks=[dict(name="B1/B2 original source independently reconstructs every native input and view reference",passed=True)]
    phases["source_and_native_readback"]=time.perf_counter()-started;tick=time.perf_counter()
    shadow=a.output/"owned-shadow";shutil.copytree(a.inputs,shadow);controls=a.output/"controls";controls.mkdir()
    originals={n:(shadow/n).read_bytes() for n in before["inputs"]}
    report=json.loads(originals["report.json"])
    def trial(name,code,fn):
        try:fn();observed="accepted"
        except (ValueError,RuntimeError) as exc:observed=str(exc)
        checks.append(dict(name=name,expected=code,observed=observed,passed=code in observed))
    def fault(name,file,mutation,code,kind="jsonl"):
        path=shadow/file;value=pq.read_table(path).to_pylist() if kind=="parquet" else jsonl(path) if kind=="jsonl" else json.loads(path.read_bytes())
        mutation(value)
        if kind=="parquet":pq.write_table(pa.Table.from_pylist(value),path,compression="zstd")
        else:path.write_bytes(b''.join(canonical(v)+b'\n' for v in value) if kind=="jsonl" else canonical(value))
        r=copy.deepcopy(report)
        if file=="scifact/chunks/atoms.parquet":
            summary=json.loads(originals["scifact/summary.json"]);summary["chunks_sha256"]=file_sha(path)
            (shadow/"scifact/summary.json").write_bytes(canonical(summary))
            r["families"]["scifact"]["summary_sha256"]=file_sha(shadow/"scifact/summary.json")
        if file!="report.json":
            for n in r["files"]:r["files"][n]=dict(bytes=(shadow/n).stat().st_size,sha256=file_sha(shadow/n))
            (shadow/"report.json").write_bytes(canonical(r))
        publish(controls/(name+".json"),dict(file=file,mutant_sha256=file_sha(path)))
        trial(name,code,lambda:audit(shadow,file_sha(shadow/"report.json")))
        for n,raw in originals.items():(shadow/n).write_bytes(raw)
    fault("B1_wrong_parent","report.json",lambda r:r.update(parent_publication_sha256="0"*64),"MNLM_SCIFACT_INPUT_READBACK_PARENT","json")
    fault("B2_missing_view","references.jsonl",lambda r:r.pop(),"MNLM_SCIFACT_INPUT_READBACK_REFERENCE_POPULATION")
    fault("B2_wrong_reuse","references.jsonl",lambda r:r[0].update(unit=1),"MNLM_SCIFACT_INPUT_READBACK_REFERENCE_BINDING")
    fault("B2_fabricated_page","references.jsonl",lambda r:r[0].update(kind="page"),"MNLM_SCIFACT_INPUT_READBACK_REFERENCE_BINDING")
    fault("B2_changed_text","scifact/chunks/atoms.parquet",lambda r:r[0].update(text="changed"),"MNLM_SCIFACT_INPUT_READBACK_NATIVE_SOURCE","parquet")
    fault("B2_label_as_input","scifact/chunks/atoms.parquet",lambda r:[v.update(label="SUPPORT") for v in r],"MNLM_SCIFACT_INPUT_READBACK_NATIVE_SOURCE","parquet")
    fault("B2_missing_unit","scifact/chunks/atoms.parquet",lambda r:r.pop(),"MNLM_SCIFACT_INPUT_READBACK_NATIVE_SOURCE","parquet")
    for flag in ("gold_in_inputs","citation_context_in_inputs","scientific_admission","served"):
        fault("B3_false_"+flag,"report.json",lambda r,k=flag:r.update({k:True}),"MNLM_SCIFACT_INPUT_READBACK_CLAIM","json")
    fault("B3_lower_floor","report.json",lambda r:r.update(native_text_floor=4),"MNLM_SCIFACT_INPUT_READBACK_CLAIM","json")
    phases["persisted_reference_and_native_corruptions"]=time.perf_counter()-tick;tick=time.perf_counter()
    from text.window_readings import load_chunks
    records,sources=load_chunks(a.inputs,a.inputs_sha256)
    expected=pq.read_table(a.inputs/"scifact/chunks/atoms.parquet").to_pylist()
    checks.append(dict(name="B3 existing frozen-window consumer returns all exact label-free observations",
        passed=records==[{**r,"family":"scifact"} for r in expected]
        and all(set(r)=={"row","chunk","entity","text","family"} for r in records)))
    trial("B3 wrong source pin before any encoder load","MNLM_WINDOW_INPUT_REPORT",lambda:load_chunks(a.inputs,"0"*64))
    publish(a.output/"window-loader.json",dict(inputs=len(records),sources=sources,keys=sorted(records[0]),encoder_forward=False))
    phases["actual_window_consumer"]=time.perf_counter()-tick;tick=time.perf_counter()
    after={n:inventory(getattr(a,n)) for n in ("source","publication","inputs")};publish(a.output/"after.json",after)
    final=read(a.source,a.source_sha256,a.publication,a.publication_sha256,shadow,a.inputs_sha256)
    checks.append(dict(name="B1/B2/B3 unchanged originals, exact restoration and final cold readback",passed=before==after and inventory(shadow)==before["inputs"] and final==cold))
    phases["final_full_source_and_restoration"]=time.perf_counter()-tick;wall=time.perf_counter()-started
    result=dict(passed=all(r["passed"] for r in checks),checks=checks,author_checked=True,scientific_admission=False,
                phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(a.output/"fsv.json",result)
    print(json.dumps(dict(passed=result["passed"],checks=len(checks),failed=[r for r in checks if not r["passed"]],
                         wall_seconds=wall,phase_coverage=result["phase_coverage"])))
    return 0 if result["passed"] else 2


if __name__=="__main__":raise SystemExit(main())
