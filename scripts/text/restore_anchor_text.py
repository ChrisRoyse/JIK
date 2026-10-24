"""Restore full pinned canonical descriptions without changing anchor labels (#12).

One digest-bound source stream for every requested family (PC-04/07/13).
An old description must be exactly the historical cut of the canonical text.
Every selected entity must occur exactly once. No source fallback or label edit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import blake3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "canon"))
from canon.common import iter_canon_docs
from senses.trajectory_population import publish
from text.anchor_domains import cut
from text.rechunk_anchor import build as rechunk, require, sha

FAMILIES = ("biomedical", "finance", "legal", "science", "code")
SHARD = "wikidata5m_text"


def load_inputs(records, chunked, families):
    loaded, errors, entities = {}, {}, {}
    for family in families:
        try:
            root, old_chunks = records / family, chunked / family
            raw = (root / "rows.jsonl").read_bytes()
            summary_raw = (root / "summary.json").read_bytes()
            summary = json.loads(summary_raw)
            require(sha(raw) == summary["rows_sha256"], "MNLM_RESTORE_ROWS_HASH", family)
            rows = [json.loads(line) for line in raw.decode("utf-8").split("\n") if line]
            require(len(rows) == summary["rows"] and bool(rows), "MNLM_RESTORE_ROWS", family)
            ids = [r["entity"] for r in rows]
            require(len(set(ids)) == len(ids) and all(isinstance(q, str) and q for q in ids),
                    "MNLM_RESTORE_ENTITIES", family)
            require(not set(ids).intersection(entities), "MNLM_RESTORE_FAMILY_OVERLAP", family)
            chunk_summary_raw = (old_chunks / "summary.json").read_bytes()
            chunk_summary = json.loads(chunk_summary_raw)
            anchors_raw = (old_chunks / "anchors.json").read_bytes()
            require(chunk_summary["parent_rows_sha256"] == sha(raw)
                    and chunk_summary["anchors_sha256"] == sha(anchors_raw),
                    "MNLM_RESTORE_ANCHOR_BINDING", family)
            loaded[family] = {"rows": rows, "summary": summary, "chunk_summary": chunk_summary,
                              "anchors_raw": anchors_raw, "original_rows_sha256": sha(raw),
                              "original_summary_sha256": sha(summary_raw),
                              "original_chunk_summary_sha256": sha(chunk_summary_raw)}
            for i, q in enumerate(ids):
                entities[q] = (family, i)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors[family] = f"{type(exc).__name__}: {exc}"
    return loaded, entities, errors


def source_contract(canon, manifest_sha):
    raw = (canon / "manifest.jsonl").read_bytes()
    require(sha(raw) == manifest_sha, "MNLM_RESTORE_MANIFEST_HASH", "canonical manifest changed")
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    matches = [r for r in rows if r.get("shard") == SHARD]
    require(len(matches) == 1, "MNLM_RESTORE_MANIFEST_AMBIGUOUS", "one explicit canonical publication required")
    row = matches[0]
    require(row["source"] == "wikidata5m" and row["kind"] == "prose"
            and row["contract"] == "canon-v1" and row["docs"] > 0,
            "MNLM_RESTORE_SOURCE_CONTRACT", "wrong source/kind/contract")
    path = canon / (SHARD + ".jsonl.zst")
    require(path.stat().st_size == row["out_bytes"], "MNLM_RESTORE_SOURCE_SIZE", str(path))
    return row


def restore(canon, records, chunked, output, manifest_sha, families=FAMILIES):
    started = time.perf_counter()
    require(not output.exists(), "MNLM_RESTORE_EXISTS", "new immutable generation required")
    require(bool(families) and len(families) == len(set(families)) and set(families) <= set(FAMILIES),
            "MNLM_RESTORE_FAMILIES", "unique known family roster required")
    publication = source_contract(canon, manifest_sha)
    loaded, entities, errors = load_inputs(records, chunked, families)
    require(bool(loaded), "MNLM_RESTORE_NO_INPUTS", json.dumps(errors))
    output.mkdir(parents=True, exist_ok=False)
    loaded_at = time.perf_counter()
    full, seen, count = {}, set(), 0
    for rec in iter_canon_docs(canon, SHARD, expected_blake3=publication["out_blake3"]):
        count += 1
        q = rec["f"].get("entity_id")
        if q not in entities:
            continue
        family, i = entities[q]
        if q in seen:
            errors[family] = f"MNLM_RESTORE_DUPLICATE_SOURCE: {q}"
            continue
        seen.add(q)
        text = rec["text"]
        raw_text = text.encode("utf-8")
        require(rec["n"] == len(raw_text) and rec["b3"] == blake3.blake3(raw_text).hexdigest()[:32],
                "MNLM_RESTORE_CANONICAL_RECORD", q)
        if loaded[family]["rows"][i]["text"] != cut(text):
            errors[family] = f"MNLM_RESTORE_NOT_HISTORICAL_CUT: {q}"
            continue
        full[q] = {"text": text, "text_source": {"source": "wikidata5m", "shard": SHARD,
                   "canonical_row": rec["row"], "text_sha256": sha(raw_text),
                   "canonical_shard_blake3": publication["out_blake3"]}}
    require(count == publication["docs"], "MNLM_RESTORE_SOURCE_ROWS", f"{count} != {publication['docs']}")
    for q in sorted(set(entities)-seen):
        errors[entities[q][0]] = f"MNLM_RESTORE_MISSING_ENTITY: {q}"
    scanned_at = time.perf_counter()
    reports = {}
    for family, src in loaded.items():
        if family in errors:
            continue
        try:
            rows = src["rows"]
            old_chars = sum(len(r["text"]) for r in rows)
            restored = [{**r, **full[r["entity"]]} for r in rows]
            new_chars = sum(len(r["text"]) for r in restored)
            changed = sum(r["text"] != s["text"] for r, s in zip(rows, restored))
            dest = output / "records" / family
            dest.mkdir(parents=True)
            raw = b"".join((json.dumps(r, sort_keys=True, ensure_ascii=False)+"\n").encode() for r in restored)
            (dest / "rows.jsonl").write_bytes(raw)
            summary = {**src["summary"], "schema": "a2-domain-full-source-v1", "rows_sha256": sha(raw),
                       "text_source": publication, "canonical_manifest_sha256": manifest_sha,
                       "original_rows_sha256": src["original_rows_sha256"],
                       "original_summary_sha256": src["original_summary_sha256"],
                       "rules": {**src["summary"]["rules"], "max_chars": None, "text_scope": "full-canonical"},
                       "served": False}
            publish(dest / "summary.json", summary)
            parent = output / "anchor-bindings" / family
            parent.mkdir(parents=True)
            (parent / "anchors.json").write_bytes(src["anchors_raw"])
            binding = {**src["chunk_summary"], "schema": "a2-full-source-anchor-binding-v1",
                       "parent_rows_sha256": sha(raw),
                       "original_chunk_summary_sha256": src["original_chunk_summary_sha256"],
                       "original_rows_sha256": src["original_rows_sha256"]}
            publish(parent / "summary.json", binding)
            chunks = rechunk(parent, dest, output / "chunked" / family)
            reports[family] = {"entities": len(rows), "changed_descriptions": changed,
                               "old_characters": old_chars, "full_characters": new_chars,
                               "recovered_characters": new_chars-old_chars,
                               "rows_sha256": sha(raw), "chunked_entities": chunks["entities"],
                               "chunks": chunks["chunks"], "chunk_characters": chunks["chunk_characters"],
                               "chunks_sha256": chunks["chunks_sha256"]}
            print(json.dumps({"family": family, **reports[family]}), flush=True)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors[family] = f"{type(exc).__name__}: {exc}"
    result = {"schema": "full-source-anchor-restoration-v1", "families_requested": list(families),
              "families": reports, "errors": errors, "complete": not errors,
              "canonical_manifest_sha256": manifest_sha, "source_publication": publication,
              "source_records_read": count, "selected_entities": len(entities), "served": False,
              "old_slot_reuse_allowed": False, "acceptance": "author-checked",
              "code_sha256": sha(Path(__file__).read_bytes()),
              "timing": {"load_hash_seconds": loaded_at-started, "source_scan_seconds": scanned_at-loaded_at,
                         "publication_seconds": time.perf_counter()-scanned_at,
                         "wall_seconds": time.perf_counter()-started}}
    publish(output / "report.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("canon", "records", "chunked", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--canonical-manifest-sha256", required=True)
    p.add_argument("--families", default=",".join(FAMILIES))
    a = p.parse_args()
    r = restore(a.canon, a.records, a.chunked, a.output, a.canonical_manifest_sha256, tuple(a.families.split(",")))
    if not r["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
