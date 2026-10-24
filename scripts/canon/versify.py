"""Stage 2 versification remap (issue #11 deferred item; spec §18 cross-edition alignment): every edition's verse
numbering scheme is inferred, never declared, because no edition in the corpus declares one. Inference: an edition's
per-chapter maximum verse (align/edition_chapters.json) is compared with each ubsicap scheme's `maxverses` rows; the
scheme with the most agreeing chapters wins when it agrees on at least MIN_AGREE of the compared chapters and beats
the runner-up; otherwise the edition is `unresolved` and keeps its own numbering. The winning scheme's `mapping` rows
(`BOOK c:v[-v2] = BOOK c:v[-v2]`, scheme on the left, org on the right) are applied verse by verse to the edition's verse
atoms, so verses that differ only by numbering tradition share one org key.
Output: align/org_index.npz (org_key_id parallel to verse_index.npz), align/org_keys.jsonl, align/versification.json
(per edition: scheme, agree, disagree, compared, mapped verses), one ledger entry `atoms.versify`.
Cost: O(editions x schemes x chapters) for inference, O(verse atoms) for the remap.
    python -I scripts/canon/versify.py --root R --out O --version scripture-v1
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HOST, Paths, iter_canon_docs, ledger_append, now_iso, read_jsonl, write_json  # noqa: E402

SOURCE = "ubsicap-versification"
MIN_AGREE = 0.95
_MAP = re.compile(r"^([1-9A-Z]{2,3}) (\d+):(\d+)(?:-(\d+))? = ([1-9A-Z]{2,3}) (\d+):(\d+)(?:-(\d+))?$")


class VersifyError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def load_schemes(paths: Paths) -> tuple[dict, dict, dict]:
    """maxverses[scheme][book][chapter] = max verse; mapping[scheme][(book, c, v)] = (book, c, v) in org; unparsed per scheme."""
    maxverses: dict = collections.defaultdict(lambda: collections.defaultdict(dict))
    mapping: dict = collections.defaultdict(dict)
    unparsed = collections.Counter()
    meta = json.loads((paths.dedup / "population.json").read_text(encoding="utf-8"))
    for sh in (s for s in meta["shards"] if s["source"] == SOURCE):
        for rec in iter_canon_docs(paths.canon(SOURCE), sh["shard"]):
            scheme, kind, text = rec["f"].get("scheme"), rec["f"].get("kind"), rec["text"].strip()
            if kind == "maxverses":
                parts = text.split()
                if len(parts) < 2 or not all(":" in p for p in parts[1:]):
                    unparsed[scheme] += 1
                    continue
                for p in parts[1:]:
                    c, v = p.split(":")
                    maxverses[scheme][parts[0]][int(c)] = int(v)
            elif kind == "mapping":
                m = _MAP.match(text)
                if not m:
                    unparsed[scheme] += 1
                    continue
                b1, c1, v1, v1b, b2, c2, v2, v2b = m.groups()
                left = range(int(v1), int(v1b or v1) + 1)
                right = range(int(v2), int(v2b or v2) + 1)
                if len(left) != len(right):
                    unparsed[scheme] += 1
                    continue
                for lv, rv in zip(left, right):
                    mapping[scheme].setdefault((b1, int(c1), lv), (b2, int(c2), rv))
    return maxverses, mapping, unparsed


def infer(chapters: dict, maxverses: dict) -> dict:
    """edition -> {scheme, agree, disagree, compared, runner_up}; scheme is 'unresolved' when the evidence is short."""
    out = {}
    for ed, books in chapters.items():
        scores = []
        for scheme, sbooks in maxverses.items():
            agree = disagree = 0
            for book, chs in books.items():
                for c, mv in chs.items():
                    want = sbooks.get(book, {}).get(int(c))
                    if want is None:
                        continue
                    agree += mv == want
                    disagree += mv != want
            scores.append((agree, -disagree, scheme))
        scores.sort(reverse=True)
        best = scores[0] if scores else (0, 0, "none")
        runner = scores[1] if len(scores) > 1 else (0, 0, "none")
        compared = best[0] - best[1]
        ok = compared > 0 and best[0] / compared >= MIN_AGREE and best[0] > runner[0]
        out[ed] = {"scheme": best[2] if ok else "unresolved", "best": best[2], "agree": best[0], "disagree": -best[1], "compared": compared,
                   "runner_up": runner[2], "runner_up_agree": runner[0]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", required=True)
    args = ap.parse_args()
    paths = Paths(args.root, args.out, args.version)
    align = paths.out / "atoms" / args.version / "align"
    chapters = json.loads((align / "edition_chapters.json").read_text(encoding="utf-8"))
    shard_edition = {int(k): v for k, v in json.loads((align / "shard_editions.json").read_text(encoding="utf-8")).items()}
    maxverses, mapping, unparsed = load_schemes(paths)
    if not maxverses:
        raise VersifyError("MNLM_VERSIFY_NO_SCHEMES", f"no maxverses rows under {SOURCE}")
    inferred = infer(chapters, maxverses)
    keys = {r["key_id"]: (r["book"], r["chapter"], r["verse"]) for r in read_jsonl(align / "verse_keys.jsonl")}
    vi = np.load(align / "verse_index.npz")
    key_id, shard = vi["key_id"], vi["shard"]
    org_keys: dict[tuple, int] = {}
    org_of = np.empty(len(key_id), dtype=np.int32)
    mapped = collections.Counter()
    for i in range(len(key_id)):
        ed = shard_edition[int(shard[i])]
        scheme = inferred[ed]["scheme"]
        k = keys[int(key_id[i])]
        if scheme not in ("unresolved", "org"):
            tgt = mapping.get(scheme, {}).get(k)
            if tgt is not None:
                mapped[ed] += 1
                k = tgt
        org_of[i] = org_keys.setdefault(k, len(org_keys))
    with (align / "org_index.npz.tmp").open("wb") as fh:
        np.savez(fh, org_key_id=org_of)
    (align / "org_index.npz.tmp").replace(align / "org_index.npz")
    with (align / "org_keys.jsonl").open("w", encoding="utf-8") as fh:
        for k, oid in sorted(org_keys.items(), key=lambda kv: kv[1]):
            fh.write(json.dumps({"org_key_id": oid, "book": k[0], "chapter": k[1], "verse": k[2]}) + "\n")
    for ed in inferred:
        inferred[ed]["mapped_verses"] = mapped[ed]
    write_json(align / "versification.json", inferred)
    by_scheme = collections.Counter(v["scheme"] for v in inferred.values())
    org_tuple = [None] * len(org_keys)
    for k, oid in org_keys.items():
        org_tuple[oid] = k
    changed = int(sum(1 for i in range(len(key_id)) if keys[int(key_id[i])] != org_tuple[org_of[i]]))
    summary = {"version": args.version, "editions": len(inferred), "editions_by_scheme": dict(by_scheme), "schemes": sorted(maxverses), "mapping_rows_parsed": {s: len(m) for s, m in mapping.items()},
               "rows_unparsed": dict(unparsed), "verse_atoms": int(len(key_id)), "verse_atoms_remapped": changed, "as_written_keys": len(keys), "org_keys": len(org_keys),
               "keys_merged": len(keys) - len(org_keys), "min_agree": MIN_AGREE, "host": HOST, "ts": now_iso()}
    write_json(align / "versify.summary.json", summary)
    e = ledger_append(paths, f"versify@{HOST}", "atoms.versify", f"stage2/{args.version}", summary)
    summary["ledger_seq"], summary["ledger_head"] = e["seq"], e["hash"]
    write_json(align / "versify.summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
