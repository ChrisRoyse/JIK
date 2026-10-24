"""Stage 2 explicit field `speaker` (issues #10, #11): character spans of the USFM words-of-Jesus markup (\\wj ... \\wj*)
inside each canonical verse, re-extracted from the raw USFM with the same reader. Method: the \\wj markers are replaced
by private-use sentinels before parsing, the reader runs unchanged (sentinels are plain characters to it), the
sentinels are removed with an offset map, whitespace is collapsed through the same map, and the result must equal the
sealed canonical text byte for byte, else the verse is refused with MNLM_WJ_TEXT_MISMATCH (typed, counted). Output per
shard: `<out>/atoms/<version>/<source>/<shard>.wj.npz` (row, start, end) and a manifest line. Cost: O(bytes) per edition.
    python -I scripts/canon/wj_extract.py --root R --out O --version scripture-v1 --sources ebible-english,berean-bsb,open-english-bible,unfoldingword-ult,unfoldingword-ust
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scripture  # noqa: E402
from common import HOST, Paths, iter_canon_docs, now_iso, read_jsonl, write_json  # noqa: E402
from contracts import canon_text  # noqa: E402

OPEN, CLOSE = "", ""
_WJ_OPEN = re.compile(r"\\\+?wj\s")
_WJ_CLOSE = re.compile(r"\\\+?wj\*")
_SPACES = re.compile(r"[ \t]+")


def _strip_sentinels(s: str, carry_open: bool) -> tuple[str, list[tuple[int, int]], bool]:
    """Remove sentinels, then reproduce usfm_clean + canon_text on the result while mapping every surviving character
    to its final offset. Returns (final text, wj spans as final char offsets, whether a run is still open at the end).
    A run may open in one verse and close in a later one (Berean), so the open state is carried across records."""
    spans, out, depth, start = [], [], (1 if carry_open else 0), (0 if carry_open else None)
    for ch in s:
        if ch == OPEN:
            if depth == 0:
                start = len(out)
            depth += 1
        elif ch == CLOSE:
            depth = max(0, depth - 1)
            if depth == 0 and start is not None:
                spans.append((start, len(out)))
                start = None
        else:
            out.append(ch)
    stripped = "".join(out)
    if depth > 0 and start is not None:
        spans.append((start, len(stripped)))
    if unicodedata.normalize("NFC", stripped) != stripped:
        return "\0NFC", [], depth > 0  # a length-changing normalisation would move the offsets; refuse
    # usfm_clean collapses [ \t]+ to one space and strips the whole verse; canon_text then rstrips every line and strips
    # outer blank lines. Each step only deletes characters, so a two-pointer walk maps old offsets to new ones.
    collapsed = _SPACES.sub(" ", stripped).strip()
    final = canon_text(collapsed) or ""
    mapping = _delete_map(stripped, collapsed)
    mapping2 = _delete_map(collapsed, final)
    moved = []
    for a0, b0 in spans:
        ca, cb = mapping2[mapping[a0]], mapping2[mapping[b0]]
        while ca < cb and final[ca].isspace():  # a run's edge whitespace is not part of the words
            ca += 1
        while cb > ca and final[cb - 1].isspace():
            cb -= 1
        if cb > ca:
            moved.append((ca, cb))
    return final, moved, depth > 0


def _delete_map(src: str, dst: str) -> list[int]:
    """For a dst obtained from src by deleting characters only: map[i] = offset in dst of src[i] (or of the next kept char); map[len(src)] = len(dst)."""
    m, j = [], 0
    for ch in src:
        if j < len(dst) and dst[j] == ch:
            m.append(j)
            j += 1
        else:
            m.append(j)
    if j != len(dst):
        raise RuntimeError("MNLM_WJ_MAP: destination is not a deletion of the source")
    m.append(len(dst))
    return m


def _usfm_texts(source: str, shard: dict) -> list[tuple[str, str, str, str]]:
    """(file label, usfm text, edition, lang) for every USFM file of the shard, with wj markers replaced by sentinels."""
    path: Path = shard["files"][0]
    out = []
    if source in scripture.USFM_GIT:
        out.append((path.name, path.read_text(encoding="utf-8"), source.split("-")[1].upper(), shard["lang"]))
    else:
        z = zipfile.ZipFile(path)
        edition = shard["shard"] if source != "berean-bsb" else "BSB"
        lang = "English"
        for name in sorted(n for n in z.namelist() if n.lower().endswith(".usfm") and "__MACOSX/" not in n and not Path(n).name.startswith("._")):
            out.append((f"{path.name}/{name}", z.read(name).decode("utf-8", "replace"), edition, lang))
    kept = []
    for lbl, t, ed, lg in out:
        head = t.lstrip("\ufeff")
        if head.startswith("\\id FRT") or head.startswith("\\id BAK") or head.startswith("\\id GLO"):
            continue  # front and back matter are skipped by the reader; the same files must be skipped here or rows shift
        kept.append((lbl, _implicit_close(_WJ_CLOSE.sub(CLOSE, _WJ_OPEN.sub(OPEN, t))), ed, lg))
    return kept


_PARA_LINE = re.compile(r"^\\(?:p|m|b|nb|pi\d?|q\d?|qr|qc|li\d?|s\d?|ms\d?|mr|r|d|c|sp)(?:\s|$)")
IMPLICIT_CLOSES = {"n": 0}


def _implicit_close(t: str) -> str:
    """USFM ends a character style at the next paragraph, poetry, heading or chapter marker: an open run that reaches such a
    line is closed there (the Open English Bible leaves one \\wj unclosed in Luke 11:2). Counted in IMPLICIT_CLOSES."""
    out, depth = [], 0
    for ln in t.split("\n"):
        if depth > 0 and _PARA_LINE.match(ln):
            out[-1] = out[-1] + CLOSE * depth
            IMPLICIT_CLOSES["n"] += depth
            depth = 0
        depth += ln.count(OPEN) - ln.count(CLOSE)
        depth = max(depth, 0)
        out.append(ln)
    return "\n".join(out)


def extract_shard(source: str, shard: dict, canon_dir: Path, out_dir: Path) -> dict:
    t0 = time.time()
    canon = {r["row"]: r["text"] for r in iter_canon_docs(canon_dir, shard["shard"])}
    rows, starts, ends = [], [], []
    verses = with_wj = mismatch = nfc = 0
    row = 0
    for lbl, text, edition, lang in _usfm_texts(source, shard):
        carry = False
        for r, t, f, k in scripture.usfm_docs(text, edition, lang, lbl, row):
            if t is None:
                continue
            row = r + 1
            try:
                final, spans, carry = _strip_sentinels(t, carry)
            except RuntimeError:
                mismatch += 1
                continue
            if f.get("role") == "verse":
                verses += 1
            if final == "\0NFC":
                nfc += 1
                continue
            if final != canon.get(r):
                mismatch += 1
                continue
            if spans:
                with_wj += 1
                for a, b in spans:
                    rows.append(r)
                    starts.append(a)
                    ends.append(b)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{shard['shard']}.wj.npz"
    tmp = out_dir / f"{shard['shard']}.wj.npz.tmp"
    with tmp.open("wb") as fh:
        np.savez(fh, row=np.asarray(rows, dtype=np.int32), start=np.asarray(starts, dtype=np.int32), end=np.asarray(ends, dtype=np.int32))
    tmp.replace(final_path)
    closes, IMPLICIT_CLOSES["n"] = IMPLICIT_CLOSES["n"], 0
    return {"source": source, "shard": shard["shard"], "field": "speaker:wj", "records": len(canon), "verses": verses, "records_with_wj": with_wj, "wj_spans": len(rows),
            "implicit_closes": closes, "refused_text_mismatch": mismatch, "refused_nfc": nfc, "host": HOST, "ts": now_iso(), "secs": round(time.time() - t0, 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--sources", required=True)
    args = ap.parse_args()
    paths = Paths(args.root, args.out, args.version)
    adir = paths.out / "atoms" / args.version
    totals = {"shards": 0, "records": 0, "records_with_wj": 0, "wj_spans": 0, "implicit_closes": 0, "refused_text_mismatch": 0, "refused_nfc": 0}
    with (adir / "wj.manifest.jsonl").open("a", encoding="utf-8") as mf:
        for src in args.sources.split(","):
            raw_dir = paths.root / "raw" / scripture.RAW_DIR_ALIAS.get(src, src)
            receipts = {r["path"]: r for r in read_jsonl(raw_dir / "receipts.jsonl")}
            sealed = {r["shard"] for r in read_jsonl(paths.canon(src) / "manifest.jsonl") if "error" not in r}
            for shard in scripture.list_shards(src, raw_dir, receipts):
                if shard["shard"] not in sealed:
                    continue
                try:
                    res = extract_shard(src, shard, paths.canon(src), adir / src)
                except Exception as e:  # noqa: BLE001
                    res = {"source": src, "shard": shard["shard"], "error": f"{type(e).__name__}: {e}"[:300], "host": HOST, "ts": now_iso()}
                mf.write(json.dumps(res, sort_keys=True) + "\n")
                if "error" in res:
                    print(f"  REFUSED {src}/{shard['shard']}: {res['error']}", flush=True)
                    continue
                totals["shards"] += 1
                for k in ("records", "records_with_wj", "wj_spans", "implicit_closes", "refused_text_mismatch", "refused_nfc"):
                    totals[k] += res[k]
    summary = {**totals, "version": args.version, "host": HOST, "ts": now_iso()}
    write_json(adir / "wj.summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
