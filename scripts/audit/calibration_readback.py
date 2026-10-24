"""Independent calibration selection and full-source coverage readback (#12).

Re-select within each class then merge by rank, independent of the publisher's
single pass over global ordering. No publisher or chunker imports.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path

import pyarrow.parquet as pq


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def check(ok, code):
    if not ok:
        raise ValueError(code)


def readback(source, output, contract):
    contract_raw = contract.read_bytes()
    cfg = json.loads(contract_raw)
    report_raw = (output/"report.json").read_bytes()
    report = json.loads(report_raw)
    check(report["complete"] and not report["errors"] and set(report["families"]) == set(cfg["families"]),
          "MNLM_CALPOP_READBACK_INCOMPLETE")
    check(report["contract_sha256"] == sha(contract_raw) and (output/"contract.json").read_bytes() == contract_raw,
          "MNLM_CALPOP_READBACK_CONTRACT")
    check(report["source_report_sha256"] == cfg["source_report_sha256"] == sha((source/"report.json").read_bytes()),
          "MNLM_CALPOP_READBACK_SOURCE")
    check(report["served"] is False and report["scientific_admission"] is False and report["calibration_only"] is True,
          "MNLM_CALPOP_READBACK_CLAIM")
    families, ready_count, all_available, groups = {}, 0, True, set()
    for family, pin in cfg["families"].items():
        raw = (source/"records"/family/"rows.jsonl").read_bytes()
        meta_raw = (source/"records"/family/"summary.json").read_bytes()
        old_raw = (source/"chunked"/family/"anchors.json").read_bytes()
        check(sha(raw) == pin["rows_sha256"] and sha(meta_raw) == pin["summary_sha256"]
              and sha(old_raw) == pin["original_anchors_sha256"], "MNLM_CALPOP_READBACK_INPUT_HASH")
        rows = [json.loads(l) for l in raw.decode().split("\n") if l]
        meta, original = json.loads(meta_raw), json.loads(old_raw)
        dest = output/family
        sm_raw = (dest/"summary.json").read_bytes()
        sm, an_raw = json.loads(sm_raw), (dest/"anchors.json").read_bytes()
        an = json.loads(an_raw)
        check(sha(sm_raw) == report["families"][family]["summary_sha256"] and sha(an_raw) == sm["anchors_sha256"],
              "MNLM_CALPOP_READBACK_HASH")
        check(sm["source_pin"] == pin and sm["parent_rows_sha256"] == pin["rows_sha256"]
              and sm["served"] is False and sm["old_slot_reuse_allowed"] is False
              and sm["calibration_only"] is True and sm["anchor_family"] == "A2", "MNLM_CALPOP_READBACK_CLAIM")
        check(set(sm["power"]) == set(original) == set(pin["questions"]), "MNLM_CALPOP_READBACK_QUESTIONS")
        ready, used, question_counts = [], set(), {}
        rank = lambda i: (sha(f"{cfg['seed']}|chunked|{rows[i]['entity']}".encode()), rows[i]["entity"])
        for prop, classes in pin["questions"].items():
            check(classes == original[prop]["classes"], "MNLM_CALPOP_READBACK_CLASSES")
            vocab = meta["values"][prop]
            per = (cfg["sample_target"]+len(classes)-1)//len(classes)
            pools = [[i for i,r in enumerate(rows) if r["y"][prop] == vocab.index(c)] for c in classes]
            counts = list(map(len, pools))
            available = min(counts) >= per
            expected_power = {"classes": classes, "available_per_class": counts, "required_per_class": per,
                              "planned_n": per*len(classes), "available": available,
                              "code": None if available else "MNLM_CALPOP_UNDERPOWERED"}
            check(sm["power"][prop] == expected_power, "MNLM_CALPOP_READBACK_POWER")
            if not available:
                all_available = False
                check(prop not in an, "MNLM_CALPOP_READBACK_UNAVAILABLE")
                continue
            ready.append(prop)
            chosen = sorted((i for pool in pools for i in sorted(pool, key=rank)[:per]), key=rank)
            labels = [classes.index(vocab[rows[i]["y"][prop]]) for i in chosen]
            obs = an[prop]
            check(obs["classes"] == classes and obs["rows"] == chosen and obs["labels"] == labels
                  and obs["per_class"] == per and len(chosen) >= 500, "MNLM_CALPOP_READBACK_SAMPLE")
            check(len({rows[i]["entity"] for i in chosen}) == len(chosen), "MNLM_CALPOP_READBACK_ENTITY")
            check(obs["overlap_with_historical_sample"] == len(set(chosen)&set(original[prop]["rows"])),
                  "MNLM_CALPOP_READBACK_OVERLAP")
            used.update(chosen)
            question_counts[prop] = len(chosen)
        check(sorted(ready) == sm["available_questions"] == report["families"][family]["ready"]
              and set(an) == set(ready) and set(sm["anchors"]) == set(ready), "MNLM_CALPOP_READBACK_READY")
        check(report["families"][family]["unavailable"] == sorted(set(original)-set(ready)), "MNLM_CALPOP_READBACK_UNAVAILABLE")
        ready_count += len(ready)
        for q in ready:
            check(sm["anchors"][q] == {"n": len(an[q]["rows"]), "classes": an[q]["classes"]}, "MNLM_CALPOP_READBACK_ANCHOR_SUMMARY")
        cursor, chunks, chars = {}, [], sum(len(rows[i]["text"]) for i in used)
        if used:
            chunk_raw = (dest/"chunks/atoms.parquet").read_bytes()
            check(sha(chunk_raw) == sm["chunks_sha256"], "MNLM_CALPOP_READBACK_CHUNK_HASH")
            chunks = pq.read_table(io.BytesIO(chunk_raw)).to_pylist()
            for c in chunks:
                i = c["row"]
                check(i in used, "MNLM_CALPOP_READBACK_CHUNK_POPULATION")
                end, ordinal = cursor.get(i, (0, 0))
                check(c["start"] == end and c["chunk"] == ordinal and end < c["end"] <= len(rows[i]["text"]),
                      "MNLM_CALPOP_READBACK_SPAN")
                check(c["entity"] == rows[i]["entity"] and c["text"] == rows[i]["text"][end:c["end"]]
                      and len(c["text"]) <= 600 and c["source_sha256"] == sha(rows[i]["text"].encode()), "MNLM_CALPOP_READBACK_TEXT")
                cursor[i] = (c["end"], ordinal+1)
            check(set(cursor) == used and all(cursor[i][0] == len(rows[i]["text"]) for i in used), "MNLM_CALPOP_READBACK_COVERAGE")
        else:
            check(sm["chunks_sha256"] is None and not (dest/"chunks/atoms.parquet").exists(), "MNLM_CALPOP_READBACK_EMPTY")
        check(sm["entities"] == len(used) and sm["chunks"] == len(chunks)
              and sm["source_characters"] == sm["chunk_characters"] == chars, "MNLM_CALPOP_READBACK_COUNTS")
        for key, val in (("entities", len(used)), ("chunks", len(chunks)), ("source_characters", chars)):
            check(report["families"][family][key] == val, "MNLM_CALPOP_READBACK_COUNTS")
        entities = {rows[i]["entity"] for i in used}
        check(not entities&groups, "MNLM_CALPOP_READBACK_CROSS_DOMAIN_ENTITY")
        groups.update(entities)
        families[family] = {"entities": len(used), "chunks": len(chunks), "characters": chars,
                            "question_n": question_counts, "unavailable": len(original)-len(ready)}
    check(report["all_questions_available"] is all_available, "MNLM_CALPOP_READBACK_ALL_AVAILABLE")
    return {"acceptance": "author-checked", "families": families, "ready_questions": ready_count,
            "entities": len(groups), "report_sha256": sha(report_raw), "source_report_sha256": report["source_report_sha256"],
            "all_source_spans_equal": True, "all_samples_match_fixed_hash_order": True,
            "scientific_admission": False, "calibration_only": True}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "publication", "contract", "output"):
        p.add_argument("--"+arg, type=Path, required=True)
    a = p.parse_args()
    result = readback(a.source, a.publication, a.contract)
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, sort_keys=True)
    print(json.dumps(result))
