"""Per-atom sense census of the Stage 3 calibration populations (issue #12; operator review recommendation 2: choose the proof
corpus by the senses available around the same referent). Read-only: it never loads a lens's vectors, only the `absent` member
of each slot .npz (zip central directory + one .npy member) and the sealed registry, assay records and frozen panel.

Cost against real N (p0-v1, 2026-10-16): Q = 27 declared questions, L_q <= 91 live lenses registered per question, n_q <= 2 000
calibration atoms. I/O = registry (~0.8 MB, once) + panel (~0.07 MB, once) + Q assay records (once each) + sum_q L_q x (zip
directory + n_q bytes) ~ 27 x 70 x 2 KB ~ 4 MB, independent of slot size (the 35 GB multi-vector slots cost one directory read).
Plus one streaming pass over each calibration file (~80 MB in all) to count distinct
`source` values. Compute O(sum_q L_q x n_q) ~ 4e6 boolean ops. Invariant across the question loop: the registry, catalogue, panel and the static
channel declarations, each read once. The run records the measured L_q and n_q per question so the bound is checked, not guessed.

Usage: python -I census_senses.py --lens-root /zfs/hot/mnlm/lenses/p0-v1 --ledger /zfs/archive/mnlm/ledger/ledger.jsonl --out ~/diag/census
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import blake3
from numpy.lib import format as npf

# Channels the paper's sense taxonomy (spec §13 Def. 13.3, §19) names for each anchor family that the P0 corpus does not hold at
# all. `inference`: would the channel exist for a new, unseen atom at serving time. `lineage`: does it share a source with the oracle.
NOT_IN_CORPUS = {
    "A1": [("repository history (commits, blame, issues, review)", "history-clock", "yes for repo code; none for OCI (synthetic tasks have no repository)", "independent of the sandboxed test oracle"),
           ("author / generator identity beyond the generation_algorithm field", "provenance", "yes", "independent"),
           ("documentation and call sites of the solution in a wider codebase", "graph-neighbourhood", "yes for repo code; none for OCI", "independent"),
           ("type-checker / linter verdicts", "execution-trace", "yes (computable on any new solution)", "partial: same code, a different checker than the tests")],
    "A2": [("full Wikidata statement set beyond wikidata5m triples (qualifiers, references, ranks)", "explicit-fields", "yes", "SAME source as the oracle (Wikidata statements): oracle lineage"),
           ("Wikipedia revision / edit history and page views", "history-clock", "yes for an existing article", "shared: Wikidata statements are often harvested from the article"),
           ("other-language Wikipedia editions and interlanguage links", "cross-source-editions", "yes", "partial: same Wikimedia ecosystem"),
           ("images / infobox media", "surface (visual)", "yes", "independent"),
           ("external identifiers / authority files (VIAF, GND, ISNI)", "cross-source-article", "yes", "partial: many Wikidata statements are imported from them")],
    "A3": [("source article text GDELT codes from (only source_domain is held; article bodies were never fetched)", "cross-source-article", "yes at coding time", "SAME source as the oracle's coder input: the later coding reads such articles"),
           ("GDELT Mentions table (per-article mention rows, confidence)", "cross-source-mentions", "yes", "same GDELT pipeline as the oracle"),
           ("GDELT GKG themes, persons, organisations", "explicit-fields", "yes", "same GDELT pipeline as the oracle"),
           ("non-GDELT event datasets (ACLED, ICEWS) for the same dyad", "cross-source-article", "yes (with lag)", "independent coder")],
    "A5": [("source article text GDELT codes from", "cross-source-article", "yes at coding time", "SAME source as the oracle (re-coding from a different domain's article)"),
           ("GDELT Mentions table / per-source confidence", "cross-source-mentions", "yes", "same GDELT pipeline as the oracle"),
           ("source-outlet metadata (country, ownership, language) beyond the domain string", "provenance", "yes", "independent")],
    "A4": [("commentary (patristic, rabbinic, modern study notes)", "cross-source-article", "yes", "partial: commentaries discuss book genre, i.e. the oracle's book_classes"),
           ("original-language text with morphology / Strong's numbers", "explicit-fields", "yes", "independent of the edition-agreement oracle"),
           ("manuscript apparatus / textual variants", "history-clock", "yes", "partial: the oracle counts edition agreement"),
           ("lectionary / liturgical usage", "graph-neighbourhood", "yes", "independent")],
    "A6": [("docket history (motions, briefs, trial-court record)", "history-clock", "yes for a new opinion", "independent of later citation"),
           ("judicial subsequent history (Ohio Supreme Court review, reversal)", "history-clock", "NO: occurs after filing", "SAME kind as the oracle (later independent documents)"),
           ("citations received from outside the corpus (other districts, secondary sources)", "citation-graph", "NO: occurs after filing", "SAME as the oracle (later citation)"),
           ("judge biographies / panel voting history", "provenance", "yes", "independent"),
           ("party / counsel identity resolution across cases", "graph-neighbourhood", "yes", "independent")],
}
# In-corpus senses whose inference-time availability or oracle lineage is not obvious (the census counts them; this flags them).
IN_CORPUS_FLAGS = {
    "A2": {"cross-source-article": ("yes", "shared: Wikipedia prose is a source Wikidata statements are drawn from (masked lenses delete the answer aliases)"),
           "cross-source-mentions": ("yes", "shared: Wikipedia prose"), "document-context": ("yes", "shared: the abstract is the same article")},
    "A6": {"citation-graph": ("out-edges yes; citations RECEIVED no (they are the oracle)", "out-edges independent; cite-authority two-hop reads received counts of cited cases")},
    "A3": {"history-clock": ("yes (prior 90 days)", "same GDELT coder as the oracle")},
    "A5": {"history-clock": ("yes (prior 90 days)", "same GDELT coder as the oracle")},
    "A1": {"execution-trace": ("yes (runs the solution)", "code-exec-trace retired: its ran feature was the oracle's load-failure clause")},
}


# Strict count (census finding, 2026-10-17): code-static-sensors clears `absent` for any non-empty text (encoders_ctx._code_sensors),
# so outside A1 it reads a line count of prose, not static structure; provenance-source-class is constant on a single-source
# population. Neither is counted as a sense in the strict histogram.
NOT_A_SENSE_OUTSIDE_A1 = {"code-static-sensors"}


def n_sources(cal: Path) -> int:
    seen = set()
    with cal.open(encoding="utf-8") as fh:
        for line in fh:
            seen.add(json.loads(line).get("source"))
    return len(seen)


def fail(code: str, msg: str) -> None:
    raise SystemExit(f"{code}: {msg}")


def read_absent(slot: Path) -> np.ndarray:
    """Only the `absent` member: the zip central directory plus one uncompressed or deflated .npy, never the vectors."""
    with zipfile.ZipFile(slot) as z:
        if "absent.npy" not in z.namelist():
            fail("MNLM_CENSUS_NO_ABSENT", f"{slot}: members {z.namelist()}")
        with z.open("absent.npy") as fh:
            ver = npf.read_magic(fh)
            shape, _fortran, dtype = (npf.read_array_header_1_0 if ver == (1, 0) else npf.read_array_header_2_0)(fh)
            if dtype != np.bool_ or len(shape) != 1:
                fail("MNLM_CENSUS_ABSENT_TYPE", f"{slot}: absent {dtype} {shape}")
            return np.frombuffer(fh.read(), dtype=np.bool_, count=shape[0]).copy()


def load_registry(root: Path, lc) -> tuple[dict, str]:
    return lc.sealed_registry(root), "registry.json (content digest checked; no snapshot fallback)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--code-dir", type=Path, default=Path(__file__).resolve().parent)
    a = ap.parse_args()
    sys.dont_write_bytecode = True  # never write __pycache__ into the code tree we import from
    sys.path.insert(0, str(a.code_dir))
    import catalogue as cat  # noqa: E402
    import lenscommon as lc  # noqa: E402
    root = a.lens_root
    reg, reg_used = load_registry(root, lc)
    from admission_state import load_reports, load_panel, validate_ledger

    admission = json.loads((root / "assay" / "admission.json").read_text(encoding="utf-8"))
    validate_ledger(a.ledger, reg, admission)
    checked_reports = load_reports(root, reg, admission)
    panel = load_panel(root, reg, admission, a.ledger)
    if panel.get("invariants", {}).get("registry_blake3") != reg["registry_blake3_before_ledger"]:
        fail("MNLM_CENSUS_STALE_PANEL", "panel and registry describe different generations")
    defects: list[str] = []
    lenses = reg["lenses"]
    live = {}
    for name, ent in lenses.items():
        spec = ent.get("spec", {})
        if ent.get("error") or spec.get("retired") or (name in cat.NAMES and cat.by_name(name).get("retired")):
            continue
        if name in cat.NAMES and cat.by_name(name)["perspective"] != spec.get("perspective"):
            defects.append(f"{name}: registry perspective {spec.get('perspective')} != catalogue {cat.by_name(name)['perspective']}")
        live[name] = ent
    out = {"lens_root": str(root), "registry_used": reg_used, "registry_ledger_seq": reg.get("ledger_seq"), "panel_version": panel.get("panel_version"),
           "panel_ledger_seq": panel.get("ledger_seq"), "questions": {}, "families": {}, "not_in_corpus": NOT_IN_CORPUS, "in_corpus_flags": IN_CORPUS_FLAGS}
    for q in lc.families_present(root):
        fam = lc.spec_family(q)
        _, summ = lc.read_family(root / "cal", q)
        n = int(summ["n"])
        ap_path = root / "assay" / f"{q}.json"
        assay = checked_reports.get(q)
        admitted = set((assay or {}).get("admission", {}).get("panel", []))
        senses: dict[str, dict] = {}
        present = {}  # sense -> bool[n]
        adm_present = {}
        strict_present = {}
        nsrc = n_sources(root / "cal" / f"{q}.jsonl")
        for name, ent in sorted(live.items()):
            f = ent.get("families", {}).get(q)
            if not f:
                continue
            sense = ent["spec"]["perspective"]
            absent = read_absent(Path(f["slot"]))
            if blake3.blake3(absent.tobytes()).hexdigest() != f.get("absent_blake3"):
                fail("MNLM_CENSUS_SLOT_UNBOUND", f"{q}/{name}: presence bytes are not sealed in this registry; remeasure")
            if absent.shape[0] != n:
                fail("MNLM_CENSUS_LEN", f"{q}/{name}: absent {absent.shape[0]} != cal n {n}")
            if int(absent.sum()) != int(f.get("absent", -1)):
                defects.append(f"{q}/{name}: slot absent {int(absent.sum())} != registry absent {f.get('absent')}")
            s = senses.setdefault(sense, {"lenses": [], "lens_coverage": {}, "best_bits_mean": None, "best_lcb_min": None, "best_lens": None, "admitted_lenses": []})
            s["lenses"].append(name)
            s["lens_coverage"][name] = round(float(1 - absent.mean()), 6)
            present[sense] = present.get(sense, np.zeros(n, bool)) | ~absent
            if not ((fam != "A1" and name in NOT_A_SENSE_OUTSIDE_A1) or (sense == "provenance" and nsrc < 2)):
                strict_present[sense] = strict_present.get(sense, np.zeros(n, bool)) | ~absent
            if name in admitted:
                s["admitted_lenses"].append(name)
                adm_present[sense] = adm_present.get(sense, np.zeros(n, bool)) | ~absent
            al = (assay or {}).get("lenses", {}).get(name)
            if al is None and assay is not None:
                s.setdefault("not_assayed", []).append(name)
            elif al is not None and al.get("bits_mean") is not None:
                if al.get("coverage") is not None and abs(al["coverage"] - s["lens_coverage"][name]) > 1e-6:
                    defects.append(f"{q}/{name}: assay coverage {al['coverage']:.6f} != slot coverage {s['lens_coverage'][name]:.6f}")
                if s["best_lcb_min"] is None or (al.get("lcb_min") or -1e9) > s["best_lcb_min"]:
                    s.update(best_bits_mean=al["bits_mean"], best_lcb_min=al.get("lcb_min"), best_lens=name)
        for sense, s in senses.items():
            s["coverage"] = round(float(present[sense].mean()), 6)
            s["admitted"] = bool(s["admitted_lenses"])
        cnt = np.sum(np.stack(list(present.values())), axis=0) if present else np.zeros(n, int)
        scnt = np.sum(np.stack(list(strict_present.values())), axis=0) if strict_present else np.zeros(n, int)
        acnt = np.sum(np.stack(list(adm_present.values())), axis=0) if adm_present else np.zeros(n, int)
        out["questions"][q] = {"family": fam, "retired": lc.RETIRED_QUESTIONS.get(q), "n": n, "lenses_live": sum(len(s["lenses"]) for s in senses.values()),
                               "assay_ts": (assay or {}).get("ts"), "assay_panel": sorted(admitted), "senses": senses,
                               "sense_hist": {str(k): int(v) for k, v in sorted(Counter(cnt.tolist()).items())},
                               "atoms_ge5_senses": int((cnt >= 5).sum()), "n_sources": nsrc,
                               "strict_sense_hist": {str(k): int(v) for k, v in sorted(Counter(scnt.tolist()).items())}, "atoms_ge5_strict_senses": int((scnt >= 5).sum()),
                               "strict_senses": sorted(s for s, v in strict_present.items() if v.any()), "admitted_sense_hist": {str(k): int(v) for k, v in sorted(Counter(acnt.tolist()).items())},
                               "atoms_ge5_admitted_senses": int((acnt >= 5).sum())}
    for fam, pf in panel.get("anchor_families", {}).items():
        qs = [q for q, r in out["questions"].items() if r["family"] == fam and not r["retired"]]
        out["families"][fam] = {"live_questions": qs, "panel_perspectives_admitted": pf.get("perspectives_admitted", []), "panel_served": pf.get("served"),
                                "senses_union": sorted({s for q in qs for s, v in out["questions"][q]["senses"].items() if v["coverage"] > 0}),
                                "strict_senses_union": sorted({s for q in qs for s in out["questions"][q]["strict_senses"]})}
    out["defects"] = defects
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "census.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (a.out / "census.md").write_text(render_md(out), encoding="utf-8")
    print(f"questions {len(out['questions'])} defects {len(defects)} registry {reg_used}")
    if defects:
        fail("MNLM_CENSUS_DEFECTS", f"{len(defects)} persisted consistency defects; inspect census.json")


def render_md(o: dict) -> str:
    L = [f"# Sense census ({o['lens_root']})", "", f"Registry: {o['registry_used']} (ledger seq {o['registry_ledger_seq']}); panel {o['panel_version']} (seq {o['panel_ledger_seq']}). All numbers measured from persisted slots.", ""]
    L += ["| family | question | n | live lenses | senses present (coverage) | atoms >=5 senses | atoms >=5 strict senses | senses admitted (assay panel) |", "|---|---|---|---|---|---|---|---|"]
    for q, r in sorted(o["questions"].items(), key=lambda kv: (kv[1]["family"], kv[0])):
        sp = ", ".join(f"{s} {v['coverage']:.3f}" for s, v in sorted(r["senses"].items()) if v["coverage"] > 0)
        ad = ", ".join(sorted(s for s, v in r["senses"].items() if v["admitted"])) or "none"
        tag = " (retired)" if r["retired"] else ""
        L.append(f"| {r['family']} | {q}{tag} | {r['n']} | {r['lenses_live']} | {sp} | {r['atoms_ge5_senses']} ({r['atoms_ge5_senses'] / max(r['n'], 1):.3f}) | {r['atoms_ge5_strict_senses']} | {ad} |")
    L += ["", "## Per-atom sense-count histograms (k senses present: atoms)", ""]
    for q, r in sorted(o["questions"].items()):
        L.append(f"- {q}: {r['sense_hist']}; strict {r['strict_sense_hist']}; admitted-only {r['admitted_sense_hist']}")
    L += ["", "## Best standalone bits per sense (assay record)", "", "| question | sense | lenses | best lens | bits_mean | lcb_min |", "|---|---|---|---|---|---|"]
    for q, r in sorted(o["questions"].items()):
        for s, v in sorted(r["senses"].items()):
            bm = "n/a" if v["best_bits_mean"] is None else f"{v['best_bits_mean']:.3f}"
            lb = "n/a" if v["best_lcb_min"] is None else f"{v['best_lcb_min']:.3f}"
            L.append(f"| {q} | {s} | {len(v['lenses'])} | {v['best_lens'] or 'not assayed'} | {bm} | {lb} |")
    L += ["", "## Channels not in corpus (static declaration)", "", "| family | channel | sense | available at inference | oracle lineage |", "|---|---|---|---|---|"]
    for fam, rows in o["not_in_corpus"].items():
        for ch, s, inf, lin in rows:
            L.append(f"| {fam} | {ch} — not in corpus | {s} | {inf} | {lin} |")
    L += ["", "## In-corpus senses flagged for availability or lineage", ""]
    for fam, d in o["in_corpus_flags"].items():
        for s, (inf, lin) in d.items():
            L.append(f"- {fam} {s}: inference {inf}; lineage {lin}")
    L += ["", f"## Defects ({len(o['defects'])})", ""] + [f"- {d}" for d in o["defects"]]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
