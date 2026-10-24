"""Exact, label-free source views for the existing frozen-window pipeline."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.scifact_readback import read, raw_source
from senses.contract import canonical
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha
from text.multiscale_spans import identity
from text.scifact_source import require, rows

SCHEMA = "scifact-native-input-v1"


def project(source, source_pin, publication, publication_pin, output):
    started = time.perf_counter(); output.mkdir(parents=True, exist_ok=False); phases = {}
    context = raw_source(source, source_pin)
    read(source, source_pin, publication, publication_pin, context)
    claims = rows(publication/"claims.jsonl"); docs = rows(publication/"documents.jsonl")
    phases["source_and_relation_readback"] = time.perf_counter()-started; tick = time.perf_counter()
    units = []; refs = []; seen = {}; counts = Counter()

    def add(kind, owner, view, group, pieces, span):
        text = "\n".join(pieces); key = identity(text)
        if key not in seen:
            seen[key] = len(units)
            units.append(dict(row=len(units), chunk=0, entity=key, text=text))
        require(units[seen[key]]["text"] == text, "INPUT_HASH_COLLISION")
        refs.append(dict(id=identity([SCHEMA,kind,owner,view]), kind=kind, owner=owner, view=view,
                         group=group, unit=seen[key], source_sentence_range=span, text_sha256=identity(text)))
        counts[kind] += 1; counts["reference_characters"] += len(text)

    for c in claims:
        add("claim",c["id"],"claim",c["group"],[c["text"]],None)
    for d in docs:
        add("title",d["id"],"title",d["group"],[d["title"]],None)
    for d in docs:
        for v in d["views"]["nodes"]:
            add(v["level"],d["id"],v["id"],d["group"],
                d["views"]["source_sentences"][slice(*v["sentence_range"])],v["sentence_range"])
    phases["exact_observation_index"] = time.perf_counter()-tick; tick = time.perf_counter()
    family = output/"scifact"; (family/"chunks").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(units),family/"chunks/atoms.parquet",compression="zstd")
    with (output/"references.jsonl").open("xb") as f:
        for r in refs:f.write(canonical(r)+b"\n")
    counts.update(references=len(refs),unique_inputs=len(units),reused_references=len(refs)-len(units),
                  unique_characters=sum(len(u["text"]) for u in units),max_characters=max(len(u["text"]) for u in units))
    publish(family/"summary.json",dict(chunks=len(units),chunks_sha256=file_sha(family/"chunks/atoms.parquet"),
                                       parent_publication_sha256=publication_pin))
    files = {p.relative_to(output).as_posix():dict(bytes=p.stat().st_size,sha256=file_sha(p))
             for p in output.rglob('*') if p.is_file()}
    phases["native_table_and_reference_publication"] = time.perf_counter()-tick; tick = time.perf_counter()
    require(file_sha(source/"report.json") == source_pin and file_sha(publication/"report.json") == publication_pin,
            "INPUT_PARENT_DRIFT")
    for parent in (source,publication):
        for name,saved in json.loads((parent/"report.json").read_bytes())["files"].items():
            require(file_sha(parent/name)==saved["sha256"],"INPUT_PARENT_DRIFT")
    phases["final_parent_drift"] = time.perf_counter()-tick; wall = time.perf_counter()-started
    result = dict(schema=SCHEMA,parent_source_sha256=source_pin,parent_publication_sha256=publication_pin,
                  files=files,counts=dict(counts),complete=True,errors=[],calibration_only=True,
                  research_scope="provisional-frozen-sense-research; source development partitions retained in reference groups",
                  families={"scifact":dict(ready=True,summary_sha256=file_sha(family/"summary.json"))},
                  text_composition="newline join of unchanged provider sentence strings; no child-vector pooling",
                  gold_in_inputs=False,citation_context_in_inputs=False,scientific_admission=False,served=False,
                  native_text_floor=5,source_views_removed=0,
                  code_sha256=file_sha(Path(__file__)),phases_seconds=phases,wall_seconds=wall,
                  phase_coverage=sum(phases.values())/wall)
    publish(output/"report.json",result); return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("source","publication","output"):p.add_argument("--"+k,type=Path,required=True)
    for k in ("source-sha256","publication-sha256"):p.add_argument("--"+k,required=True)
    a = p.parse_args(); r=project(a.source,a.source_sha256,a.publication,a.publication_sha256,a.output)
    print(json.dumps({k:r[k] for k in ("counts","wall_seconds","phase_coverage")}))
