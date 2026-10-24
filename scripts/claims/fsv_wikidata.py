"""Full-state verification of the Wikidata5m claim checker on real data (no synthetic triples).

Assertions are built from the A2 calibration rows (<out>/lenses/p0-v1/cal/a2-wikidata-<p>.jsonl: `cluster` is the
entity, `y_name` the object of the summary's `property`). The builder reads the canonical shards a different way from
the checker: `iter_canon_docs` + json.loads of the `f` fields (the checker's index parses the `text` field with a byte
regex), so a row the builder cites and an expectation it derives do not come from the index under test. Classes:
  a  claimed, asserted, true triple, cited at its real row                          -> supported
  b  hypothesis, asserted, same entity and property, an object absent from the source -> unsupported_missing
  c  claimed, negated, true triple, cited at its real row                           -> contradicted
  d  claimed, asserted, true triple, cited at a wrong row / past the shard end /
     an unknown shard                                                              -> evidence_mismatch
  e  hypothesis, asserted: absent subject / absent relation / a real relation the
     entity has no statement of                                                    -> unsupported_missing
A negative (b..e) reported `supported` is a defect and fails the run. Ten verdicts are re-checked by zstdcat | grep -F
over the shard bytes. Writes <diag>/fsv.json, assertions.jsonl and verdicts.jsonl."""
from __future__ import annotations

FUNCTIONAL_DECLARED = set()  # this source supplies no proof of exclusivity

import collections
import json
import subprocess
import sys
import time
from pathlib import Path

import blake3

sys.path.insert(0, str(Path(__file__).resolve().parent))
from assertion import OUTCOMES, ClaimError, parse_assertion, parse_verdict  # noqa: E402
from check_wikidata import SHARDS, SOURCE, build_index, check, default_paths  # noqa: E402
from lenscommon import iter_canon_docs  # noqa: E402

PROPS = ["P19", "P27", "P54", "P106"]
DIAG = Path.home() / "diag" / "claims"
ABSENT_REL = "P99999"


def _h(*parts) -> int:
    return int.from_bytes(blake3.blake3("\x00".join(map(str, parts)).encode()).digest()[:8], "big")


def load_cal(paths) -> dict:
    cal = {}
    for p in PROPS:
        fam = f"a2-wikidata-{p.lower()}"
        d = paths.out / "lenses" / paths.version / "cal"
        summ = json.loads((d / f"{fam}.summary.json").read_text(encoding="utf-8"))
        if summ["property"] != p:
            raise ClaimError("MNLM_CLAIM_FSV_CAL", f"{fam} summary property {summ['property']} != {p}")
        rows = [json.loads(x) for x in (d / f"{fam}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        cal[p] = {"rows": rows, "classes": sorted(set(summ["class_names"].values())), "file_blake3": summ["file_blake3"]}
    return cal


def builder_pass(paths, want: set[str]) -> dict:
    """One json pass over the shards: every statement of the wanted entities, all entity and relation ids, counts."""
    stm = collections.defaultdict(list)  # s -> [(p, o, shard, row)]
    ents, rels, counts, shard_len = set(), set(), {}, {}
    for sh in SHARDS:
        n = last = 0
        for rec in iter_canon_docs(paths.canon(SOURCE), sh):
            f = rec["f"]
            s, p, o = f["s"], f["p"], f["o"]
            ents.add(s)
            ents.add(o)
            rels.add(p)
            if s in want:
                stm[s].append((p, o, sh, rec["row"]))
            n += 1
            last = rec["row"]
        counts[sh], shard_len[sh] = n, last + 1
    return {"stm": stm, "ents": ents, "rels": rels, "counts": counts, "shard_end": shard_len}


def build_assertions(cal: dict, b: dict) -> tuple[list[dict], dict]:
    out, skipped = [], collections.Counter()
    ev = lambda sh, row: [{"source": SOURCE, "shard": sh, "row": row}]  # noqa: E731
    max_q = max(int(e[1:]) for e in b["ents"])
    for p in PROPS:
        for i, r in enumerate(cal[p]["rows"]):
            e, y = r["cluster"], r["y_name"]
            st = b["stm"].get(e, [])
            true_rows = [(sh, row) for (pp, o, sh, row) in st if pp == p and o == y]
            if not true_rows:
                skipped["a_true_triple_not_in_source"] += 1
                continue
            sh, row = true_rows[0]
            held = {o for (pp, o, _, _) in st if pp == p}
            base = {"subject": e, "relation": p, "object": y, "qualifiers": {}}
            out.append({"cls": "a", "expect": "supported", "a": {**base, "polarity": "asserted", "epistemic_status": "claimed", "evidence": ev(sh, row)}})
            others = [c for c in cal[p]["classes"] if c not in held]
            if others:
                y2 = others[_h("b", p, e) % len(others)]
                exp = "contradicted" if (len(held) == 1 and p in FUNCTIONAL_DECLARED) else "unsupported_missing"  # agent decision 2026-10-17: open world except declared functional properties
                out.append({"cls": "b", "expect": exp, "held_objects": len(held),
                            "a": {**base, "object": y2, "polarity": "asserted", "epistemic_status": "hypothesis", "evidence": []}})
            else:
                skipped["b_entity_holds_every_class"] += 1
            out.append({"cls": "c", "expect": "contradicted", "a": {**base, "polarity": "negated", "epistemic_status": "claimed", "evidence": ev(sh, row)}})
            kind = i % 3
            wrong = [(s2, r2) for (pp, o, s2, r2) in st if (pp, o) != (p, y)]
            if kind == 0 and wrong:
                ref, how = ev(*wrong[_h("d", e) % len(wrong)]), "other_statement_row"
            elif kind == 1 or (kind == 0 and not wrong):
                ref, how = ev(sh, b["shard_end"][sh] + _h("d", e) % 1000), "past_shard_end"
            else:
                ref, how = [{"source": SOURCE, "shard": "wikidata5m_transductive_nosuch", "row": row}], "unknown_shard"
            out.append({"cls": "d", "expect": "evidence_mismatch", "how": how, "a": {**base, "polarity": "asserted", "epistemic_status": "claimed", "evidence": ref}})
            kind = i % 3
            if kind == 0:
                q = max_q + 1 + _h("e", p, e) % 10**6
                while f"Q{q}" in b["ents"]:
                    q += 1
                ea, how = {**base, "subject": f"Q{q}"}, "absent_subject"
            elif kind == 1:
                ea, how = {**base, "relation": ABSENT_REL}, "absent_relation"
            else:
                props_e = {pp for (pp, _, _, _) in st}
                cand = [x for x in ["P17", "P19", "P20", "P27", "P54", "P106", "P136", "P495", "P641", "P1412"] if x not in props_e and x in b["rels"]]
                if not cand:
                    raise ClaimError("MNLM_CLAIM_FSV_CONTROL", f"{e} holds every candidate relation; no unheld relation to assert")
                ea, how = {**base, "relation": cand[0]}, "relation_not_held_by_subject"
            out.append({"cls": "e", "expect": "unsupported_missing", "how": how, "a": {**ea, "polarity": "asserted", "epistemic_status": "hypothesis", "evidence": []}})
    if ABSENT_REL in b["rels"]:
        raise ClaimError("MNLM_CLAIM_FSV_CONTROL", f"{ABSENT_REL} occurs in the source; pick another absent relation")
    return out, dict(skipped)


def schema_controls() -> dict:
    """Malformed assertions must refuse with the named code (fail closed), never default."""
    good = {"subject": "Q1", "relation": "P31", "object": "Q5", "polarity": "asserted", "epistemic_status": "hypothesis"}
    cases = [({k: v for k, v in good.items() if k != "polarity"}, "MNLM_CLAIM_MISSING_FIELD"), ({**good, "confidence": 1}, "MNLM_CLAIM_UNKNOWN_FIELD"),
             ({**good, "polarity": "maybe"}, "MNLM_CLAIM_BAD_POLARITY"), ({**good, "epistemic_status": "claimed"}, "MNLM_CLAIM_NO_EVIDENCE"),
             ({**good, "subject": "Alice"}, "MNLM_CLAIM_BAD_SUBJECT"), ({**good, "evidence": [{"source": SOURCE, "shard": "x"}]}, "MNLM_CLAIM_MISSING_FIELD"),
             ({**good, "evidence": [{"source": SOURCE, "shard": "x", "row": "7"}]}, "MNLM_CLAIM_BAD_TYPE"), ({**good, "time": {"start": "2001"}}, "MNLM_CLAIM_MISSING_FIELD")]
    res = []
    for d, code in cases:
        try:
            parse_assertion(d)
            got = "ACCEPTED"
        except ClaimError as e:
            got = e.code
        res.append({"expect": code, "got": got, "ok": got == code})
    return {"cases": len(res), "passed": sum(r["ok"] for r in res), "detail": res}


def grep_recheck(paths, picks: list[dict]) -> list[dict]:
    """Re-derive 10 verdicts from the shard bytes by zstdcat | grep -F (no index, no json parse of the whole shard)."""
    pats_by_shard = {sh: set() for sh in SHARDS}
    for v in picks:
        a = v["assertion"]
        for sh in SHARDS:
            pats_by_shard[sh].add(f'"text": "{a["subject"]}\\t{a["relation"]}\\t')
        for ref in a["evidence"]:
            if ref["shard"] in pats_by_shard:
                pats_by_shard[ref["shard"]].add(f'"row": {ref["row"]}, "text"')
    hits = {}
    for sh, pats in pats_by_shard.items():
        pf = DIAG / f"grep-{sh}.pat"
        pf.write_text("\n".join(sorted(pats)) + "\n", encoding="utf-8")
        cmd = f"zstdcat {paths.canon(SOURCE) / (sh + '.jsonl.zst')} | grep -F -f {pf}"
        hits[sh] = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout.splitlines()
    out = []
    for v in picks:
        a = v["assertion"]
        pre = f'"text": "{a["subject"]}\\t{a["relation"]}\\t'
        objs = {ln.split(pre, 1)[1].split('"', 1)[0] for sh in SHARDS for ln in hits[sh] if pre in ln}
        cited = []
        for ref in a["evidence"]:
            key = f'"row": {ref["row"]}, "text": "'
            line = next((ln for ln in hits.get(ref["shard"], []) if key in ln), None)
            cited.append(None if line is None else line.split(key, 1)[1].split('"', 1)[0].replace("\\t", "\t"))
        triple = f'{a["subject"]}\t{a["relation"]}\t{a["object"]}'
        present = a["object"] in objs
        if a["polarity"] == "negated":
            exp = "contradicted" if present else ("evidence_mismatch" if None in cited else "unsupported_missing")
        elif cited and all(c == triple for c in cited):
            exp = "supported"
        elif not present and len(objs) == 1 and a["relation"] in FUNCTIONAL_DECLARED:
            exp = "contradicted"
        elif cited:
            exp = "evidence_mismatch"
        else:
            exp = "supported" if present else "unsupported_missing"
        out.append({"assertion": {k: a[k] for k in ("subject", "relation", "object", "polarity", "evidence")}, "grep_objects_for_sp": sorted(objs),
                    "grep_cited_rows": cited, "grep_outcome": exp, "checker_outcome": v["outcome"], "agree": exp == v["outcome"]})
    return out


def main() -> int:
    t0 = time.time()
    DIAG.mkdir(parents=True, exist_ok=True)
    paths = default_paths()
    cal = load_cal(paths)
    idx = build_index(paths, DIAG / "cache")
    t_idx = time.time() - t0
    want = {r["cluster"] for p in PROPS for r in cal[p]["rows"]}
    b = builder_pass(paths, want)
    if sum(b["counts"].values()) != idx.meta["triples"]:
        raise ClaimError("MNLM_CLAIM_FSV_COUNT", f"builder read {sum(b['counts'].values())} triples, index holds {idx.meta['triples']}")
    items, skipped = build_assertions(cal, b)
    t1 = time.time()
    verdicts, conf = [], collections.defaultdict(collections.Counter)
    with (DIAG / "assertions.jsonl").open("w", encoding="utf-8") as fa, (DIAG / "verdicts.jsonl").open("w", encoding="utf-8") as fv:
        for it in items:
            v = check(parse_assertion(it["a"]), idx).to_json()
            parse_verdict(v)
            v["cls"], v["expect"] = it["cls"], it["expect"]
            conf[it["cls"]][v["outcome"]] += 1
            verdicts.append(v)
            fa.write(json.dumps(it, sort_keys=True) + "\n")
            fv.write(json.dumps(v, sort_keys=True) + "\n")
    t_check = time.time() - t1
    wrong = [v for v in verdicts if v["outcome"] != v["expect"]]
    false_support = [v for v in verdicts if v["cls"] != "a" and v["outcome"] == "supported"]
    picks = []
    for c in "abcde":
        cv = [v for v in verdicts if v["cls"] == c]
        picks += sorted(cv, key=lambda v: _h("pick", json.dumps(v["assertion"], sort_keys=True)))[:2]
    rech = grep_recheck(paths, picks)
    per_cls = {c: {"n": sum(conf[c].values()), "expect": sorted({v["expect"] for v in verdicts if v["cls"] == c}), "got": {o: conf[c][o] for o in OUTCOMES}} for c in "abcde"}
    b_split = collections.Counter((v["expect"], v["outcome"]) for v in verdicts if v["cls"] == "b")
    per_prop = collections.defaultdict(collections.Counter)
    for v in verdicts:
        per_prop[v["assertion"]["relation"]][f'{v["cls"]}:{v["outcome"]}'] += 1
    res = {"ok": not wrong and not false_support and all(r["agree"] for r in rech), "tag": "measured", "host": "aiwonder",
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "properties": PROPS, "cal_file_blake3": {p: cal[p]["file_blake3"] for p in PROPS},
           "index": idx.meta, "builder_counts": b["counts"], "skipped": skipped, "assertions": len(items), "confusion": per_cls,
           "b_expect_vs_got": {f"{k[0]}->{k[1]}": n for k, n in b_split.items()},
           "property_multi_valued_share": {p: idx.multi_share.get(int(p[1:])) for p in PROPS}, "per_property": {k: dict(v) for k, v in per_prop.items()},
           "mismatches": len(wrong), "mismatch_examples": wrong[:5], "false_supported": len(false_support), "schema_controls": schema_controls(),
           "grep_recheck": rech, "secs": {"index": round(t_idx, 1), "check": round(t_check, 2), "total": round(time.time() - t0, 1)},
           "files": {n: str(DIAG / n) for n in ("assertions.jsonl", "verdicts.jsonl")}}
    (DIAG / "fsv.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("ok", "assertions", "confusion", "b_expect_vs_got", "mismatches", "false_supported", "secs")}, indent=1))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ClaimError as e:
        print(json.dumps({"ok": False, "code": e.code, "message": str(e)}), file=sys.stderr)
        sys.exit(2)
