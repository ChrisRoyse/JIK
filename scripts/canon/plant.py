"""Stage 1 negative-control corpus (issue #8, spec §17 B3): writes <root>/raw/plant-v1/plant.jsonl deterministically.

    python -I scripts/canon/plant.py --root R

Labels and the verdict each layer must return (checked by fsv_stage1.py --plant):
  exact_a / exact_b      byte-identical            -> exact layer removes exact_b, dup_of exact_a
  near_a / near_b        one template, 8 fields    -> near layer removes near_b, dup_of near_a
  span_a / span_b        distinct docs, 300-byte shared passage -> exact-substring layer cuts the passage in span_b, dup_of span_a
  short_a / short_b      distinct docs, 150-byte shared passage -> no span (below the 200-byte threshold)
  miss_a / miss_b        about 30% shingle overlap, no shared run over 7 words -> both survive every layer
  distinct               unique                    -> survives every layer
  filler_000..filler_199 random, unique            -> survive; give the LSH and the suffix array a population
  empty                  whitespace only           -> counted as empty, never a doc
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

VOCAB_SEED = 20261015


def vocab(rng: random.Random, n: int = 3000) -> list[str]:
    syl = ["ba", "ko", "ri", "ten", "mu", "la", "vor", "si", "ne", "dol", "pra", "um", "tek", "oš", "ya", "gan", "fe", "lio", "cur", "wex"]
    out = set()
    while len(out) < n:
        out.add("".join(rng.choice(syl) for _ in range(rng.randint(2, 4))))
    return sorted(out)


def words(rng: random.Random, v: list[str], n: int) -> list[str]:
    return [rng.choice(v) for _ in range(n)]


def prose(ws: list[str]) -> str:
    out, i = [], 0
    while i < len(ws):
        k = min(len(ws) - i, 9 + (i % 7))
        out.append(" ".join(ws[i:i + k]).capitalize() + ".")
        i += k
    lines, para = [], []
    for j, s in enumerate(out):
        para.append(s)
        if j % 5 == 4:
            lines.append(" ".join(para))
            para = []
    if para:
        lines.append(" ".join(para))
    return "\n\n".join(lines)


def passage(rng: random.Random, v: list[str], nbytes: int) -> str:
    ws = []
    while len(" ".join(ws).encode()) < nbytes:
        ws.append(rng.choice(v))
    return " ".join(ws)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    rng = random.Random(VOCAB_SEED)
    v = vocab(rng)
    docs: list[dict] = []
    exact = prose(words(rng, v, 400))
    docs += [{"label": "exact_a", "text": exact}, {"label": "exact_b", "text": exact}]
    template = words(rng, v, 400)
    slots = sorted(rng.sample(range(400), 8))
    near_a, near_b = list(template), list(template)
    for s in slots:
        near_a[s], near_b[s] = f"alpha{s}", f"beta{s}"
    docs += [{"label": "near_a", "text": prose(near_a)}, {"label": "near_b", "text": prose(near_b)}]
    shared = passage(rng, v, 300)
    for lab in ("span_a", "span_b"):
        ws = words(rng, v, 350)
        docs.append({"label": lab, "text": prose(ws[:175]) + "\n\n" + shared + "\n\n" + prose(ws[175:])})
    short = passage(rng, v, 150)
    for lab in ("short_a", "short_b"):
        ws = words(rng, v, 350)
        docs.append({"label": lab, "text": prose(ws[:175]) + "\n\n" + short + "\n\n" + prose(ws[175:])})
    base = words(rng, v, 400)  # miss_b keeps runs of 7 shared words then 3 new ones: about 30% of 5-gram shingles shared,
    alt = list(base)           # and no shared run longer than 7 words (about 60 bytes, far below the 200-byte span threshold)
    for i in range(400):
        if i % 10 >= 7:
            alt[i] = rng.choice(v)
    docs += [{"label": "miss_a", "text": prose(base)}, {"label": "miss_b", "text": prose(alt)}]
    docs.append({"label": "distinct", "text": prose(words(rng, v, 400))})
    for i in range(200):
        docs.append({"label": f"filler_{i:03d}", "text": prose(words(rng, v, rng.randint(60, 600)))})
    docs.append({"label": "empty", "text": " \n\n  \r\n"})
    out = Path(args.root) / "raw" / "plant-v1"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "plant.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for d in docs:
            f.write(json.dumps(d, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"wrote {len(docs)} records to {out / 'plant.jsonl'} ({(out / 'plant.jsonl').stat().st_size} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
