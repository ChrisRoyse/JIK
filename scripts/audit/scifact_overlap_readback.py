"""Independent source-view, lexical-component and relation split readback."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.scifact_input_readback import read as input_read
from audit.scifact_readback import jsonl
from audit.evaluation_overlap_readback import audit
from text.evaluation_multiscale import file_sha


def check(ok,code):
    if not ok:raise ValueError("MNLM_SCIFACT_OVERLAP_READBACK_"+code)


def read(source,publication,inputs,pins,root,pin):
    check(file_sha(root/"report.json")==pin,"PIN");r=json.loads((root/"report.json").read_bytes())
    check(r["schema"]=="scifact-relational-overlap-v1" and [r["parent_"+k+"_sha256"] for k in ("source","publication","inputs")]==pins,"PARENT")
    input_read(source,pins[0],publication,pins[1],inputs,pins[2])
    for k in ("labels_used_for_grouping","scientific_admission","sealed_evaluation","source_time_composition_separated","encoder_exposure_certified","served"):
        check(r[k] is False,"CLAIM")
    check(r["observations_removed"]==0 and r["native_text_floor"]==5,"CLAIM")
    check(r["threshold"]==dict(numerator=3,denominator=5) and r["shingle_words"]==5,"CONTRACT")
    check(r["partition_rule"]=="SHA256(scifact-relational-overlap-v1|25|component), first8hex mod100; train<70 calibration<85 else evaluation","CONTRACT")
    expected_files={"capture/report.json","capture/scifact.jsonl","cases.jsonl","source-groups.json","overlap/report.json"}
    expected_files.update("overlap/"+n for n in ("words.json","shingles.npy","indptr.npy","members.npy","documents.jsonl","candidates.npy","links.jsonl","groups.jsonl"))
    check(set(r["files"])==expected_files and {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==expected_files|{"report.json"},"INVENTORY")
    for name,v in r["files"].items():check(file_sha(root/name)==v["sha256"] and (root/name).stat().st_size==v["bytes"],"FILE_HASH")
    refs=jsonl(inputs/"references.jsonl");units=pq.read_table(inputs/"scifact/chunks/atoms.parquet").to_pylist()
    capture=jsonl(root/"capture/scifact.jsonl");check(len(capture)==len(refs),"CAPTURE_POPULATION")
    for ref,row in zip(refs,capture):
        text=units[ref["unit"]]["text"]
        expected=dict(id=ref["id"],domain="scifact",group_keys=[ref["group"]],fields=[dict(name=ref["kind"],text=text,
                      sha256=hashlib.sha256(text.encode()).hexdigest(),characters=len(text),utf8_bytes=len(text.encode()))])
        check(row==expected,"CAPTURE_SOURCE")
    cm=json.loads((root/"capture/report.json").read_bytes())
    check(cm["parent_inputs_sha256"]==pins[2] and cm["scientific_admission"] is False and cm["served"] is False,"CAPTURE_PARENT")
    cold=audit(root/"overlap",root/"capture")
    groups=jsonl(root/"overlap/groups.jsonl");membership={}
    for g in groups:
        for i in g["members"]:
            old=refs[i]["group"];check(old not in membership or membership[old]==g["id"],"GROUP_CLOSURE");membership[old]=g["id"]
    check(json.loads((root/"source-groups.json").read_bytes())==membership,"GROUP_MAP")
    parent=jsonl(publication/"cases.jsonl");cases=jsonl(root/"cases.jsonl");check(len(cases)==len(parent),"CASE_POPULATION")
    for c,s in zip(cases,parent):
        group=membership[s["group"]];v=int(hashlib.sha256(("scifact-relational-overlap-v1|25|"+group).encode()).hexdigest()[:8],16)%100
        part="train" if v<70 else "calibration" if v<85 else "evaluation"
        check(c=={**s,"parent_group":s["group"],"parent_partition":s["partition"],"group":group,"partition":part},"CASE_BINDING")
    labels={row["case"]:row["label"] for row in jsonl(publication/"labels.jsonl")}
    support={s:dict(Counter(labels[c["id"]] for c in cases if c["partition"]==s)) for s in ("train","calibration","evaluation")}
    counts=dict(references=len(refs),cases=len(cases),source_components=len(membership),overlap_components=len(groups),
                cases_by_partition=dict(Counter(c["partition"] for c in cases)),support=support,
                reassigned_cases=sum(c["parent_partition"]!=c["partition"] for c in cases),
                merged_source_components=sum(len({refs[i]["group"] for i in g["members"]})>1 for g in groups))
    check(r["counts"]==counts and r["overlap_counts"]==cold["counts"],"COUNTS")
    return dict(report_sha256=pin,counts=counts,overlap_counts=cold["counts"],source_views_complete=True,
                candidate_components_reconstructed=True,all_cases_group_separated=True,scientific_admission=False,sealed_evaluation=False)
