"""Stage 2 cross-edition alignment (issue #11 §3 item 4; anchor family A4 for #10): every verse atom of the scripture
partition is keyed by (book, chapter, verse) parsed from its explicit `ref`, so the same verse across editions and
languages is one key. First version: a hash join on the key as written by each edition; versification remapping through
the ubsicap tables is deferred and stated in the summary (keys that differ only by numbering stay separate).
Output: `<out>/atoms/<version>/align/verse_index.npz` (key_id, shard, row sorted by key), `verse_keys.jsonl` (key_id ->
book, chapter, verse, editions, testament, genre from config/book_classes.json; an unknown book is reported as unclassified),
`edition_chapters.json` (edition -> book -> chapter -> max verse, the input of versify.py), `align.summary.json`. Cost: O(verse atoms) one pass over the atom manifests and the
canon shards' field column; no all-pairs step.
    python -I scripts/canon/align_verses.py --root R --out O --version scripture-v1
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

ROLE_VERSE = 5
_REF = re.compile(r"^([1-4]? ?[A-Za-z0-9]{2,5})\.(\d+)\.(\d+)[a-z]?(?:-\d+[a-z]?)?$")  # BOOK.C.V, with "1 Chr", S3Y, PRO.3.5a and V-V2 ranges (keyed on the first verse)
_BOOK_ALIAS = {  # SWORD/OSIS book codes used by crosswire-sword-pd and the SBLGNT/Nestle names, mapped to USFM codes
    "Gen": "GEN", "Exod": "EXO", "Lev": "LEV", "Num": "NUM", "Deut": "DEU", "Josh": "JOS", "Judg": "JDG", "Ruth": "RUT", "1Sam": "1SA", "2Sam": "2SA",
    "1Kgs": "1KI", "2Kgs": "2KI", "1Chr": "1CH", "2Chr": "2CH", "Ezra": "EZR", "Neh": "NEH", "Esth": "EST", "Job": "JOB", "Ps": "PSA", "Prov": "PRO",
    "Eccl": "ECC", "Song": "SNG", "Isa": "ISA", "Jer": "JER", "Lam": "LAM", "Ezek": "EZK", "Dan": "DAN", "Hos": "HOS", "Joel": "JOL", "Amos": "AMO",
    "Obad": "OBA", "Jonah": "JON", "Mic": "MIC", "Nah": "NAM", "Hab": "HAB", "Zeph": "ZEP", "Hag": "HAG", "Zech": "ZEC", "Mal": "MAL",
    "Matt": "MAT", "Mark": "MRK", "Luke": "LUK", "John": "JHN", "Acts": "ACT", "Rom": "ROM", "1Cor": "1CO", "2Cor": "2CO", "Gal": "GAL", "Eph": "EPH",
    "Phil": "PHP", "Col": "COL", "1Thess": "1TH", "2Thess": "2TH", "1Tim": "1TI", "2Tim": "2TI", "Titus": "TIT", "Phlm": "PHM", "Heb": "HEB",
    "Jas": "JAS", "1Pet": "1PE", "2Pet": "2PE", "1John": "1JN", "2John": "2JN", "3John": "3JN", "Jude": "JUD", "Rev": "REV",
    "Tob": "TOB", "Jdt": "JDT", "Wis": "WIS", "Sir": "SIR", "Bar": "BAR", "1Macc": "1MA", "2Macc": "2MA", "3Macc": "3MA", "4Macc": "4MA",
    "1JO": "1JN", "2JO": "2JN", "3JO": "3JN", "JAM": "JAS", "JOH": "JHN", "MAR": "MRK", "EX": "EXO", "AM": "AMO", "OB": "OBA", "EPLAO": "LAO",  # codes seen unclassified on 2026-10-15
    "1Kings": "1KI", "2Kings": "2KI", "PrMan": "MAN", "1Esd": "1ES", "2Esd": "2ES", "EpJer": "LJE", "PrAzar": "S3Y", "Sus": "SUS", "Bel": "BEL", "AddEsth": "ESG", "AddPs": "PS2",
}

_BOOK_ALIAS_UPPER = {k.upper(): v for k, v in _BOOK_ALIAS.items()}  # editions spell a code Ex, EX or ex


def norm_book(b: str) -> str:
    b = b.replace(" ", "")
    return _BOOK_ALIAS.get(b) or _BOOK_ALIAS_UPPER.get(b.upper()) or b.upper()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", required=True)
    args = ap.parse_args()
    paths = Paths(args.root, args.out, args.version)
    adir = paths.out / "atoms" / args.version
    meta = json.loads((paths.dedup / "population.json").read_text(encoding="utf-8"))
    shard_index = {(s["source"], s["shard"]): i for i, s in enumerate(meta["shards"])}
    keys: dict[tuple, int] = {}
    key_rows: list[tuple[int, int, int]] = []  # (key_id, shard_idx, row)
    editions_of: dict[int, set] = collections.defaultdict(set)
    unparsed = collections.Counter()
    verse_atoms = 0
    classes = json.loads((Path(__file__).resolve().parents[2] / "config" / "book_classes.json").read_text(encoding="utf-8"))
    testament_of = {b: t for t, bs in classes["testament"].items() for b in bs}
    genre_of = {b: g for g, bs in classes["genre"].items() for b in bs}
    chapters: dict[str, dict[str, dict[str, int]]] = collections.defaultdict(lambda: collections.defaultdict(dict))  # edition -> book -> chapter -> max verse
    shard_edition: dict[int, str] = {}  # one edition per shard (versify.py keys the remap on it)
    for m in read_jsonl(adir / "manifest.jsonl"):
        if "error" in m or m["roles"].get("5", 0) == 0:
            continue
        a = np.load(adir / m["source"] / f"{m['shard']}.atoms.npz")
        verse_rows = set(a["row"][a["role"] == ROLE_VERSE].tolist())
        si = shard_index[(m["source"], m["shard"])]
        for rec in iter_canon_docs(paths.canon(m["source"]), m["shard"]):
            if rec["row"] not in verse_rows or rec["f"].get("role") != "verse":
                continue
            verse_atoms += 1
            mt = _REF.match(rec["f"].get("ref") or "")
            if not mt:
                unparsed[m["source"]] += 1
                continue
            key = (norm_book(mt.group(1)), int(mt.group(2)), int(mt.group(3)))
            kid = keys.setdefault(key, len(keys))
            key_rows.append((kid, si, rec["row"]))
            ed = f"{m['source']}/{rec['f'].get('edition') or m['shard']}"
            if shard_edition.setdefault(si, ed) != ed:
                raise RuntimeError(f"MNLM_ALIGN_SHARD_MULTI_EDITION: {m['source']}/{m['shard']} holds {shard_edition[si]} and {ed}")
            editions_of[kid].add(ed)
            ch = chapters[ed][key[0]]
            ch[str(key[1])] = max(ch.get(str(key[1]), 0), key[2])
    out = adir / "align"
    out.mkdir(parents=True, exist_ok=True)
    arr = np.array(sorted(key_rows), dtype=np.int64)
    np.savez(out / "verse_index.npz", key_id=arr[:, 0].astype(np.int32), shard=arr[:, 1].astype(np.int32), row=arr[:, 2].astype(np.int32))
    with (out / "verse_keys.jsonl").open("w", encoding="utf-8") as fh:
        for key, kid in sorted(keys.items(), key=lambda kv: kv[1]):
            fh.write(json.dumps({"key_id": kid, "book": key[0], "chapter": key[1], "verse": key[2], "editions": len(editions_of[kid]),
                                 "testament": testament_of.get(key[0], "unclassified"), "genre": genre_of.get(key[0], "unclassified")}) + "\n")
    write_json(out / "edition_chapters.json", chapters)
    write_json(out / "shard_editions.json", {str(k): v for k, v in shard_edition.items()})
    unclassified = sorted({k[0] for k in keys if k[0] not in testament_of})
    per_edition_count = collections.Counter(len(v) for v in editions_of.values())
    summary = {"version": args.version, "verse_atoms": verse_atoms, "aligned_atoms": int(len(arr)), "distinct_keys": len(keys),
               "keys_in_ge2_editions": sum(1 for v in editions_of.values() if len(v) >= 2), "keys_in_ge5_editions": sum(1 for v in editions_of.values() if len(v) >= 5),
               "max_editions_per_key": max((len(v) for v in editions_of.values()), default=0), "editions_per_key_histogram": {str(k): v for k, v in sorted(per_edition_count.items())[:40]},
               "unparsed_refs_per_source": dict(unparsed), "editions": len(chapters), "unclassified_books": unclassified,
               "keys_by_testament": dict(collections.Counter(testament_of.get(k[0], "unclassified") for k in keys)),
               "versification_remap": "keys are as each edition numbers them; versify.py infers each edition's scheme from edition_chapters.json and maps keys to org",
               "host": HOST, "ts": now_iso()}
    write_json(out / "align.summary.json", summary)
    e = ledger_append(paths, f"align_verses@{HOST}", "atoms.align", f"stage2/{args.version}", summary)
    summary["ledger_seq"], summary["ledger_head"] = e["seq"], e["hash"]
    write_json(out / "align.summary.json", summary)
    print(json.dumps(summary)[:1500], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
