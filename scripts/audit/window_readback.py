"""Cold validation of every saved window against untruncated tokenizer output.

Reads source, instrument and native values independently of window_readings.
Tokenization normalization gaps stay explicit; no literal byte-coverage claim.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"lens"))
from learned import snapshot_dir, weights_sha256, preprocessing_sha256


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def check(ok, code):
    if not ok:
        raise ValueError(code)


def canonical_assets(values):
    """Asset separators are host syntax; retain exact names and content hashes."""
    result = {}
    for name, digest in values.items():
        key = name.replace("\\", "/")
        parts = PurePosixPath(key).parts
        check(bool(parts) and not key.startswith("/") and ":" not in key
              and all(p not in ("", ".", "..") for p in key.split("/"))
              and key not in result, "MNLM_WINDOW_READBACK_ASSET_PATH")
        result[key] = digest
    return result


def validate_vectors(vectors, indices, inputs, windows, dims):
    check(vectors.dtype == np.float32 and vectors.shape == (windows, dims) and np.isfinite(vectors).all()
          and np.allclose(np.linalg.norm(vectors, axis=1), 1, rtol=0, atol=3e-6), "MNLM_WINDOW_READBACK_VECTORS")
    check(indices.dtype == np.int64 and len(indices) == inputs+1 and indices[0] == 0 and indices[-1] == len(vectors)
          and np.all(np.diff(indices) > 0), "MNLM_WINDOW_READBACK_INDICES")


def validate_trace(trace, text, reference, max_length):
    prefix = trace["prefix"]
    check(trace["surface_sha256"] == sha(text.encode()) and trace["surface_chars"] == len(text)
          and trace["surface_utf8_bytes"] == len(text.encode()) and trace["input_sha256"] == sha((prefix+text).encode()),
          "MNLM_WINDOW_READBACK_TEXT")
    ids, offsets, covered = [], [], np.zeros(len(text), dtype=bool)
    check(bool(trace["windows"]), "MNLM_WINDOW_READBACK_EMPTY")
    for ordinal, w in enumerate(trace["windows"]):
        n = len(w["input_ids"])
        check(w["ordinal"] == ordinal and n <= max_length
              and all(len(w[k]) == n for k in ("attention_mask", "special_tokens_mask", "offset_mapping")),
              "MNLM_WINDOW_READBACK_ALIGNMENT")
        local = []
        for token, span, attended, special in zip(w["input_ids"], w["offset_mapping"], w["attention_mask"], w["special_tokens_mask"]):
            check(attended in (0,1) and special in (0,1) and len(span) == 2 and 0 <= span[0] <= span[1] <= len(prefix)+len(text),
                  "MNLM_WINDOW_READBACK_OFFSET")
            if attended and not special:
                ids.append(token); offsets.append(tuple(span))
                lo, hi = max(0, span[0]-len(prefix)), min(len(text), max(0, span[1]-len(prefix)))
                if lo < hi:
                    covered[lo:hi] = True; local.append([lo, hi])
            else:
                check(span == [0,0], "MNLM_WINDOW_READBACK_SPECIAL_OFFSET")
        expected = [min(s[0] for s in local), max(s[1] for s in local)] if local else None
        check(w["source_char_span"] == expected, "MNLM_WINDOW_READBACK_EXTENT")
    check(ids == reference["input_ids"] and offsets == list(map(tuple, reference["offset_mapping"])),
          "MNLM_WINDOW_READBACK_TOKENS")
    gaps, begin = [], None
    for i, seen in enumerate(list(covered)+[True]):
        if not seen and begin is None:
            begin = i
        if seen and begin is not None:
            gaps.append([begin, i]); begin = None
    check(trace["unmapped_source_spans"] == gaps, "MNLM_WINDOW_READBACK_NORMALIZATION")
    return len(ids)


def readback(root, calibration):
    raw = (root/"report.json").read_bytes()
    report = json.loads(raw)
    check(report["complete"] and not report["errors"] and set(report["requested"]) == set(report["lenses"])
          and report["served"] is False and report["scientific_admission"] is False, "MNLM_WINDOW_READBACK_INCOMPLETE")
    cal_raw = (calibration/"report.json").read_bytes()
    check(report["calibration_report_sha256"] == sha(cal_raw), "MNLM_WINDOW_READBACK_CALIBRATION")
    cal = json.loads(cal_raw)
    records, documents = [], defaultdict(list)
    check(set(report["sources"]) == {f for f,v in cal["families"].items() if v["ready"]}, "MNLM_WINDOW_READBACK_SOURCES")
    for family, pin in sorted(report["sources"].items()):
        summary_raw = (calibration/family/"summary.json").read_bytes()
        chunk_raw = (calibration/family/"chunks/atoms.parquet").read_bytes()
        check(sha(summary_raw) == pin["summary_sha256"] == cal["families"][family]["summary_sha256"]
              and sha(chunk_raw) == pin["chunks_sha256"] == json.loads(summary_raw)["chunks_sha256"], "MNLM_WINDOW_READBACK_INPUT_HASH")
        for r in pq.read_table(io.BytesIO(chunk_raw)).to_pylist():
            r["family"] = family
            records.append(r)
            documents[(family, r["row"])].append(r["text"])
    check(len(records) == report["inputs"], "MNLM_WINDOW_READBACK_POPULATION")
    results = {}
    for name, item in report["lenses"].items():
        dest = root/name
        lens_raw = (dest/"report.json").read_bytes()
        check(sha(lens_raw) == item["report_sha256"], "MNLM_WINDOW_READBACK_LENS_HASH")
        lens = json.loads(lens_raw); contract = lens["contract"]
        check(lens["served"] is False and lens["scientific_admission"] is False
              and contract["window_reduction"] == "none:ordered-window-vectors", "MNLM_WINDOW_READBACK_CLAIM")
        check(sha(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()) == item["lens_id"] == lens["lens_id"],
              "MNLM_WINDOW_READBACK_ID")
        snap = snapshot_dir(contract["repo"])
        check(snap.name == contract["revision"] and weights_sha256(snap)[0] == contract["weights_sha256"]
              and canonical_assets(preprocessing_sha256(snap)) == canonical_assets(contract["preprocessing_sha256"]),
              "MNLM_WINDOW_READBACK_INSTRUMENT")
        for filename, pin in lens["files"].items():
            body = (dest/filename).read_bytes()
            check(sha(body) == pin["sha256"] and len(body) == pin["bytes"], "MNLM_WINDOW_READBACK_FILE_HASH")
        tok = AutoTokenizer.from_pretrained(str(snap), trust_remote_code=False)
        with np.load(dest/"slots.npz") as z:
            vectors, indices = z["vectors"], z["offsets"]
        dims = AutoConfig.from_pretrained(str(snap), trust_remote_code=False).hidden_size
        validate_vectors(vectors, indices, len(records), lens["windows"], dims)
        check(len(vectors) == item["windows"] and contract["shape"] == f"MultiWindows({dims})", "MNLM_WINDOW_READBACK_INDICES")
        with gzip.open(dest/"traces.jsonl.gz", "rt", encoding="utf-8") as f:
            traces = [json.loads(l) for l in f]
        check(len(traces) == len(records), "MNLM_WINDOW_READBACK_TRACE_COUNT")
        token_count, gap_chars = 0, 0
        for start in range(0, len(records), 256):
            batch = records[start:start+256]
            full = tok([contract["prefix"]+r["text"] for r in batch], add_special_tokens=False, truncation=False,
                       return_offsets_mapping=True)
            for j, r in enumerate(batch):
                i = start+j; tr = traces[i]
                check(tr["input"] == i and all(tr[k] == r[k] for k in ("family", "row", "entity", "chunk"))
                      and tr["vector_span"] == [int(indices[i]), int(indices[i+1])]
                      and len(tr["trace"]["windows"]) == indices[i+1]-indices[i]
                      and tr["trace"]["prefix"] == contract["prefix"], "MNLM_WINDOW_READBACK_BINDING")
                token_count += validate_trace(tr["trace"], r["text"], {k: full[k][j] for k in ("input_ids", "offset_mapping")}, contract["max_len"])
                gap_chars += sum(b-a for a,b in tr["trace"]["unmapped_source_spans"])
        probe = json.loads((dest/"probe.json").read_bytes())
        text = "".join(documents[(probe["family"], probe["row"])])
        check(sha(text.encode()) == probe["text_sha256"], "MNLM_WINDOW_READBACK_PROBE_SOURCE")
        ref = tok(contract["prefix"]+text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
        validate_trace(probe["trace"], text, ref, contract["max_len"])
        with np.load(dest/"probe.npz") as z:
            check(z["offsets"].tolist() == [0, len(z["vectors"])] and len(z["vectors"]) == len(probe["trace"]["windows"])
                  == lens["probe_windows"] and np.isfinite(z["vectors"]).all(), "MNLM_WINDOW_READBACK_PROBE")
            validate_vectors(z["vectors"], z["offsets"], 1, lens["probe_windows"], dims)
        results[name] = {"inputs": len(records), "windows": len(vectors), "untruncated_tokens": token_count,
                         "normalization_gap_characters": gap_chars, "probe_windows": lens["probe_windows"],
                         "all_token_ids_offsets_equal": True, "lens_id": lens["lens_id"]}
    return {"acceptance": "author-checked", "report_sha256": sha(raw), "lenses": results,
            "scientific_admission": False, "served": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("publication", "calibration", "output"):
        p.add_argument("--"+arg, type=Path, required=True)
    a = p.parse_args()
    result = readback(a.publication, a.calibration)
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, sort_keys=True)
    print(json.dumps(result))
