"""Preserve all source scales in conservative relation-task overlap components."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.scifact_input_readback import read
from senses.contract import canonical
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha
from text.evaluation_overlap import run
from text.scifact_source import rows, require

SCHEMA="scifact-relational-overlap-v1"


def partition(group):
    n=int(hashlib.sha256((SCHEMA+"|25|"+group).encode()).hexdigest()[:8],16)%100
    return "train" if n<70 else "calibration" if n<85 else "evaluation"


def project(source,publication,inputs,pins,output):
    started=time.perf_counter();output.mkdir(parents=True,exist_ok=False);phases={}
    read(source,pins[0],publication,pins[1],inputs,pins[2])
    units=pq.read_table(inputs/"scifact/chunks/atoms.parquet").to_pylist()
    refs=rows(inputs/"references.jsonl");cases=rows(publication/"cases.jsonl")
    phases["parent_readback"]=time.perf_counter()-started;tick=time.perf_counter()
    capture=output/"capture";capture.mkdir()
    with (capture/"scifact.jsonl").open("xb") as f:
        for ref in refs:
            text=units[ref["unit"]]["text"]
            field=dict(name=ref["kind"],text=text,sha256=hashlib.sha256(text.encode()).hexdigest(),characters=len(text),utf8_bytes=len(text.encode()))
            f.write(canonical(dict(id=ref["id"],domain="scifact",group_keys=[ref["group"]],fields=[field]))+b"\n")
    publish(capture/"report.json",dict(schema="evaluation-source-capture-v1",complete=True,errors=[],sealed_evaluation=False,
            sources={"scifact":dict(file="scifact.jsonl",sha256=file_sha(capture/"scifact.jsonl"),
                                   bytes=(capture/"scifact.jsonl").stat().st_size,counts=dict(selected_documents=len(refs)))},
            parent_inputs_sha256=pins[2],scientific_admission=False,served=False))
    phases["all_scale_capture"]=time.perf_counter()-tick;tick=time.perf_counter()
    overlap=run(capture,file_sha(capture/"report.json"),output/"overlap")
    require(overlap["complete"] and not overlap["errors"],"OVERLAP_INCOMPLETE")
    groups=rows(output/"overlap/groups.jsonl");by_old={}
    for group in groups:
        for i in group["members"]:
            old=refs[i]["group"]
            require(old not in by_old or by_old[old]==group["id"],"OVERLAP_SPLIT_SOURCE")
            by_old[old]=group["id"]
    phases["existing_complete_overlap"]=time.perf_counter()-tick;tick=time.perf_counter()
    mapped=[]
    for c in cases:
        group=by_old[c["group"]]
        mapped.append({**c,"parent_group":c["group"],"parent_partition":c["partition"],"group":group,"partition":partition(group)})
    with (output/"cases.jsonl").open("xb") as f:
        for c in mapped:f.write(canonical(c)+b"\n")
    publish(output/"source-groups.json",by_old)
    labels={r["case"]:r["label"] for r in rows(publication/"labels.jsonl")}
    support={s:dict(Counter(labels[c["id"]] for c in mapped if c["partition"]==s)) for s in ("train","calibration","evaluation")}
    counts=dict(references=len(refs),cases=len(cases),source_components=len(by_old),overlap_components=len(groups),
                cases_by_partition=dict(Counter(c["partition"] for c in mapped)),support=support,
                reassigned_cases=sum(c["parent_partition"]!=c["partition"] for c in mapped),
                merged_source_components=sum(len({refs[i]["group"] for i in g["members"]})>1 for g in groups))
    phases["case_partition_and_support"]=time.perf_counter()-tick;tick=time.perf_counter()
    require([file_sha(p/"report.json") for p in (source,publication,inputs)]==pins,"OVERLAP_PARENT_DRIFT")
    files={p.relative_to(output).as_posix():dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in output.rglob('*') if p.is_file()}
    phases["final_pins_and_publication"]=time.perf_counter()-tick;wall=time.perf_counter()-started
    report=dict(schema=SCHEMA,parent_source_sha256=pins[0],parent_publication_sha256=pins[1],parent_inputs_sha256=pins[2],
                files=files,counts=counts,overlap_counts=overlap["counts"],threshold=dict(numerator=3,denominator=5),shingle_words=5,
                partition_rule="SHA256(scifact-relational-overlap-v1|25|component), first8hex mod100; train<70 calibration<85 else evaluation",
                observations_removed=0,labels_used_for_grouping=False,scientific_admission=False,sealed_evaluation=False,
                source_time_composition_separated=False,encoder_exposure_certified=False,served=False,native_text_floor=5,
                code_sha256=file_sha(Path(__file__)),phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(output/"report.json",report);return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("source","publication","inputs","output"):p.add_argument("--"+n,type=Path,required=True)
    for n in ("source","publication","inputs"):p.add_argument("--"+n+"-sha256",required=True)
    a=p.parse_args();r=project(a.source,a.publication,a.inputs,[a.source_sha256,a.publication_sha256,a.inputs_sha256],a.output)
    print(json.dumps({k:r[k] for k in ("counts","overlap_counts","wall_seconds","phase_coverage")}))
