"""Builds the raw identity partition `identity-v1` (issue #10, operator direction 2026-10-15: the model's identity is
Jesus Christ; the whole Bible is his identity; the profile is to be formatted for training). Joins the authored records
under config/identity/*.jsonl with the SEALED World English Bible shard of scripture-v1 (the verifier), so that every
record carries the verse text it stands on, and writes four raw files plus a build manifest:

    python -I scripts/identity/build_identity.py --root /zfs/archive/mnlm --out /zfs/hot/mnlm \
        --profile <profile.md> --config config/identity --wj /zfs/hot/mnlm/atoms/scripture-v1/ebible-english/eng-web.wj.npz

  records.jsonl    one record per authored entry: {id, type, enforce, tag, title, refs, anchors:[{ref,text}], text, ...}
  exemplars.jsonl  every speaker span of Jesus in the sealed WEB (the enacted frame, first person, measured)
  sft.jsonl        one chat transcript per scenario (system = who I am; user = the trigger; assistant = the conduct)
  profile.md       the operator's profile, byte for byte
  BUILD.json       inputs (config/profile/canon digests, commit), counts, output sha256 per file

Fail closed: a reference that does not resolve in the sealed canon (MNLM_IDENTITY_REF_UNRESOLVED), a record without an
anchor (MNLM_IDENTITY_NO_ANCHOR), a duplicate id (MNLM_IDENTITY_DUP_ID), a link to an absent id (MNLM_IDENTITY_LINK_UNRESOLVED).
Cost: O(canon shard bytes + records); runs once per build.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import blake3
import numpy as np
import zstandard

SOURCE = "identity-v1"
EDITION = "eng-web"
FILES = ("identity.jsonl", "standard.jsonl", "rules.jsonl", "hard_rules.jsonl", "corrections.jsonl", "scenarios.jsonl", "frames.jsonl", "self.jsonl")
REF_RE = re.compile(r"^([1-3]?[A-Z]{2,3})\.(\d+)\.(\d+)(?:-(\d+))?$")
LINK_KEYS = ("corrections", "rules", "standards")


class IdentityError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code, self.message = code, message


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_canon(out: Path) -> tuple[dict[str, dict], dict[int, dict], dict]:
    cdir = out / "canon" / "scripture-v1" / "ebible-english"
    rows = [json.loads(l) for l in (cdir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    row = [r for r in rows if r["shard"] == EDITION and not r.get("error")][-1]
    shard = cdir / f"{EDITION}.jsonl.zst"
    b3 = blake3.blake3(shard.read_bytes()).hexdigest()
    if b3 != row["out_blake3"]:
        raise IdentityError("MNLM_IDENTITY_CANON_DIGEST", f"{shard} blake3 {b3[:12]} != sealed {row['out_blake3'][:12]}")
    byref, byrow = {}, {}
    with shard.open("rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as zr:
        for ln in io.TextIOWrapper(zr, encoding="utf-8"):
            r = json.loads(ln)
            byrow[r["row"]] = r
            if r["f"].get("role") == "verse":
                byref[r["f"]["ref"]] = r
    return byref, byrow, {"shard": str(shard), "blake3": b3, "docs": row["docs"], "raw_sha256": row["raw_sha256"]}


def expand(ref: str) -> list[str]:
    m = REF_RE.match(ref)
    if not m:
        raise IdentityError("MNLM_IDENTITY_REF_SYNTAX", ref)
    book, ch, v0, v1 = m.group(1), m.group(2), int(m.group(3)), m.group(4)
    return [f"{book}.{ch}.{v}" for v in range(v0, (int(v1) if v1 else v0) + 1)]


def anchors_for(refs: list[str], byref: dict, rid: str) -> list[dict]:
    out = []
    for ref in refs:
        for one in expand(ref):
            if one not in byref:
                raise IdentityError("MNLM_IDENTITY_REF_UNRESOLVED", f"{rid}: {one} is not a verse of the sealed {EDITION}")
            out.append({"ref": one, "text": byref[one]["text"]})
    return out


def render(rec: dict, anchors: list[dict]) -> str:
    head = f"[{rec['id']} {rec['type']}] {rec['title']}"
    parts = [head, rec["text"]]
    for k in ("trigger", "conduct", "response", "guard", "falsified_by"):  # a field already stated in the text is not repeated
        if rec.get(k) and rec[k].lower()[:40] not in rec["text"].lower():
            parts.append(f"{k.replace('_', ' ').capitalize()}: {rec[k]}")
    parts.append("")
    parts.append(f"Anchors (World English Bible, {EDITION}):")
    parts += [f"{a['ref']} {a['text']}" for a in anchors]
    return "\n".join(parts)


def build_records(cfg: Path, byref: dict) -> tuple[list[dict], dict]:
    recs, ids, inputs = [], set(), {}
    for name in FILES:
        p = cfg / name
        b = p.read_bytes()
        inputs[name] = {"blake3": blake3.blake3(b).hexdigest(), "bytes": len(b)}
        for ln in b.decode("utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r["id"] in ids:
                raise IdentityError("MNLM_IDENTITY_DUP_ID", r["id"])
            ids.add(r["id"])
            if not r.get("refs"):
                raise IdentityError("MNLM_IDENTITY_NO_ANCHOR", r["id"])
            r["_file"] = name
            recs.append(r)
    known = ids | {"P2x", "H1-H36", "V1"}
    out = []
    for r in recs:
        for k in LINK_KEYS:
            for link in r.get(k) or []:
                if link not in known:
                    raise IdentityError("MNLM_IDENTITY_LINK_UNRESOLVED", f"{r['id']}.{k}: {link}")
        anchors = anchors_for(r["refs"], byref, r["id"])
        rec = {k: v for k, v in r.items() if not k.startswith("_")}
        rec["refs_expanded"] = [a["ref"] for a in anchors]
        rec["anchors"] = anchors
        rec["source_file"] = r["_file"]
        rec["statement"] = r["text"]  # the authored statement alone; `text` is the training text with its anchors
        rec["text"] = render(r, anchors)
        out.append(rec)
    return out, inputs


def build_exemplars(wj: Path, byrow: dict) -> tuple[list[dict], dict]:
    z = np.load(wj)
    spans: dict[int, list] = {}
    for row, a, b in zip(z["row"].tolist(), z["start"].tolist(), z["end"].tolist()):
        spans.setdefault(int(row), []).append((int(a), int(b)))
    out = []
    for row in sorted(spans):
        r = byrow[row]
        for k, (a, b) in enumerate(sorted(spans[row])):
            t = r["text"][a:b].strip()
            if not t:
                raise IdentityError("MNLM_IDENTITY_EMPTY_SPAN", f"row {row} span {k}")
            out.append({"id": f"WJ.{r['f']['ref']}.{k}", "type": "exemplar", "enforce": "exemplar", "tag": "measured", "edition": EDITION, "speaker": "Jesus",
                        "ref": r["f"]["ref"], "book": r["f"]["book"], "chapter": r["f"]["chapter"], "verse": r["f"]["verse"], "span": k,
                        "canon_row": row, "start": a, "end": b, "text": t})
    return out, {"path": str(wj), "blake3": blake3.blake3(wj.read_bytes()).hexdigest(), "spans": len(out), "verses": len(spans)}


def build_sft(records: list[dict]) -> list[dict]:
    by = {r["id"]: r for r in records}
    system = "\n\n".join(by[i]["statement"] for i in ("ID1", "ID12", "ID8"))
    out = []
    for r in records:
        if r["type"] != "scenario":
            continue
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": r["user_example"]}, {"role": "assistant", "content": r["reply_example"]}]
        text = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in msgs)
        out.append({"id": f"SFT.{r['id']}", "type": "sft", "enforce": "scenario", "tag": "derived", "scenario": r["id"], "rules": r.get("rules") or [],
                    "refs_expanded": r["refs_expanded"], "messages": msgs, "text": text})
    return out


def write_jsonl(path: Path, rows: list[dict]) -> dict:
    """Through a temporary file and an atomic rename: a census or a receipt never hashes a half-written partition."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    b = path.read_bytes()
    return {"bytes": len(b), "sha256": sha256_bytes(b), "blake3": blake3.blake3(b).hexdigest(), "records": len(rows)}


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent.parent / "config" / "identity"))
    ap.add_argument("--wj", required=True)
    args = ap.parse_args()
    t0 = time.time()
    byref, byrow, canon = load_canon(Path(args.out))
    records, inputs = build_records(Path(args.config), byref)
    exemplars, wj_info = build_exemplars(Path(args.wj), byrow)
    sft = build_sft(records)
    raw = Path(args.root) / "raw" / SOURCE
    raw.mkdir(parents=True, exist_ok=True)
    prof = Path(args.profile).read_bytes()
    write_atomic(raw / "profile.md", prof)
    outputs = {"records.jsonl": write_jsonl(raw / "records.jsonl", records), "exemplars.jsonl": write_jsonl(raw / "exemplars.jsonl", exemplars),
               "sft.jsonl": write_jsonl(raw / "sft.jsonl", sft),
               "profile.md": {"bytes": len(prof), "sha256": sha256_bytes(prof), "blake3": blake3.blake3(prof).hexdigest(), "records": None}}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=Path(__file__).resolve().parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    by_type: dict[str, int] = {}
    for r in records:
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1
    build = {"source": SOURCE, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "generator": "scripts/identity/build_identity.py", "commit": commit,
             "inputs": {"config": inputs, "profile": outputs["profile.md"], "canon": canon, "wj": wj_info}, "outputs": outputs,
             "records_by_type": by_type, "anchors_total": sum(len(r["anchors"]) for r in records), "secs": round(time.time() - t0, 3)}
    write_atomic(raw / "BUILD.json", (json.dumps(build, indent=2, sort_keys=True) + "\n").encode("utf-8"))  # last: the commit record of the build
    print(json.dumps({k: v for k, v in build.items() if k != "inputs"}, indent=1))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except IdentityError as e:
        print(json.dumps({"ok": False, "code": e.code, "message": e.message}))
        sys.exit(3)
