"""Reconstruct every native input from original provider rows, not export code."""
from collections import Counter
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.scifact_readback import read as relation_read,raw_source,ident,jsonl
from text.evaluation_multiscale import file_sha


def check(ok,code):
    if not ok:raise ValueError("MNLM_SCIFACT_INPUT_READBACK_"+code)


def read(source,source_pin,publication,publication_pin,root,pin,context=None):
    context=context if context is not None else raw_source(source,source_pin)
    relation_read(source,source_pin,publication,publication_pin,context)
    check(file_sha(root/"report.json")==pin,"REPORT_PIN"); report=json.loads((root/"report.json").read_bytes())
    check(report["schema"]=="scifact-native-input-v1" and report["parent_source_sha256"]==source_pin
          and report["parent_publication_sha256"]==publication_pin,"PARENT")
    check(report["complete"] is True and report["calibration_only"] is True and report["errors"]==[]
          and all(report[k] is False for k in ("gold_in_inputs","citation_context_in_inputs","scientific_admission","served"))
          and report["native_text_floor"]==5 and report["source_views_removed"]==0,"CLAIM")
    check(report["text_composition"]=="newline join of unchanged provider sentence strings; no child-vector pooling","JOIN_RULE")
    expected_files={"references.jsonl","scifact/summary.json","scifact/chunks/atoms.parquet"}
    check(set(report["files"])==expected_files and {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
          ==expected_files|{"report.json"},"FILES")
    for name,saved in report["files"].items():
        check(file_sha(root/name)==saved["sha256"] and (root/name).stat().st_size==saved["bytes"],"FILE_PIN")
    summary=json.loads((root/"scifact/summary.json").read_bytes())
    check(report["families"]=={"scifact":dict(ready=True,summary_sha256=file_sha(root/"scifact/summary.json"))}
          and summary["parent_publication_sha256"]==publication_pin
          and summary["chunks_sha256"]==file_sha(root/"scifact/chunks/atoms.parquet"),"SUMMARY")
    native=pq.read_table(root/"scifact/chunks/atoms.parquet").to_pylist(); refs=jsonl(root/"references.jsonl")
    raw_docs,raw_claims,_=context
    claim_groups={r["id"]:r["group"] for r in jsonl(publication/"claims.jsonl")}
    doc_groups={r["id"]:r["group"] for r in jsonl(publication/"documents.jsonl")}
    observations=[]
    for _,c in raw_claims:observations.append(("claim",c["id"],"claim",claim_groups[c["id"]],c["claim"],None))
    for key in sorted(doc_groups):observations.append(("title",key,"title",doc_groups[key],raw_docs[key]["title"],None))
    for key in sorted(doc_groups):
        d=raw_docs[key];n=len(d["abstract"])
        scopes=[("source_sentence",i,i+1) for i in range(n)]+[("sentence_pair",i,min(n,i+2)) for i in range(0,n,2)]+[("abstract_context",0,n)]
        for level,a,b in scopes:
            observations.append((level,key,ident([ident(d),level,a,b]),doc_groups[key],"\n".join(d["abstract"][a:b]),[a,b]))
    check(len(refs)==len(observations),"REFERENCE_POPULATION")
    counts=Counter();unit_for={};expected_native=[]
    for ref,(kind,owner,view,group,text,span) in zip(refs,observations):
        digest=ident(text)
        if digest not in unit_for:
            unit_for[digest]=len(expected_native)
            expected_native.append(dict(row=len(expected_native),chunk=0,entity=digest,text=text))
        check(ref==dict(id=ident(["scifact-native-input-v1",kind,owner,view]),kind=kind,owner=owner,view=view,
                        group=group,unit=unit_for[digest],source_sentence_range=span,text_sha256=digest),"REFERENCE_BINDING")
        counts[kind]+=1;counts["reference_characters"]+=len(text)
    check(native==expected_native and summary["chunks"]==len(native),"NATIVE_SOURCE")
    counts.update(references=len(refs),unique_inputs=len(native),reused_references=len(refs)-len(native),
                  unique_characters=sum(len(u["text"]) for u in native),max_characters=max(len(u["text"]) for u in native))
    check(report["counts"]==dict(counts),"COUNTS")
    return dict(counts=dict(counts),parent_source_sha256=source_pin,parent_publication_sha256=publication_pin,
                input_sha256=pin,all_original_text_and_reference_bindings_equal=True,scientific_admission=False)
