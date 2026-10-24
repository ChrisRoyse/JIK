"""Persisted source/population/complete-byte controls on real selected targets."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.renderer_population_readback import read, sha, check
from model.strict_surface import encode_complete, decode_complete
from model.surface_renderer import BOS, EOS
from senses.trajectory_population import publish


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--population", type=Path, required=True)
    p.add_argument("--report-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); check(not a.output.exists(), "FSV_OUTPUT_EXISTS")
    a.output.mkdir(parents=True)
    rep = json.loads((a.population/"report.json").read_bytes())
    before = {n: sha(a.population/n) for n in ["report.json", *rep["files"]]}
    publish(a.output/"before.json", before)
    positive = read(a.population, a.report_sha256); checks = [dict(name="population_positive", passed=True)]

    def refusal(name, code, fn):
        error = None
        try:
            fn()
        except ValueError as exc:
            error = str(exc)
        checks.append(dict(name=name, expected=code, observed=error, passed=error == code))

    def mutant(name, change, expected):
        root = a.output/name; root.mkdir()
        for filename in before:
            os.link(a.population/filename, root/filename)
        r = copy.deepcopy(rep)
        changed_file, value = change(root, r)
        if changed_file:
            path = root/changed_file; path.unlink()  # unlink this hardlink only
            if isinstance(value, np.ndarray):
                np.save(path, value)
            else:
                path.write_text(value, encoding="utf-8")
            r["files"][changed_file] = dict(sha256=sha(path), bytes=path.stat().st_size)
        (root/"report.json").unlink()
        publish(root/"report.json", r)
        refusal(name, expected, lambda: read(root, sha(root/"report.json")))

    def rows_change(fn):
        def change(root, report):
            rows = [json.loads(s) for s in (root/"rows.jsonl").read_text(encoding="utf-8").splitlines()]
            fn(rows)
            return "rows.jsonl", "".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows)
        return change

    refusal("stale_report_pin", "MNLM_RP_READ_REPORT_PIN", lambda: read(a.population, "0"*64))
    mutant("false_admission", lambda root, r: (r.update(scientific_admission=True) or (None, None)), "MNLM_RP_READ_FALSE_CLAIM")
    mutant("oversize_target", rows_change(lambda rows: rows[0]["atoms"][-1].update(text=rows[0]["atoms"][-1]["text"]+"x"*513)), "MNLM_RP_READ_COMPLETE_TARGET")
    mutant("future_history_gap", rows_change(lambda rows: rows[0]["atoms"][3].update(pos=999999)), "MNLM_RP_READ_HISTORY_POSITIONS")
    mutant("foreign_history", rows_change(lambda rows: rows[0]["atoms"][0].update(doc="changed-source-document")), "MNLM_RP_READ_HISTORY_DOCUMENT")
    mutant("reordered_rows", rows_change(lambda rows: rows.reverse()), "MNLM_RP_READ_ROW_ORDER")
    def bad_values(root, r):
        name = r["lenses"][0]+".npy"; values = np.load(root/name); values[0, 0, 0] = np.nan
        return name, values
    mutant("nonfinite_slot", bad_values, "MNLM_RP_READ_SLOT_NORM")
    rows = [json.loads(s) for s in (a.population/"rows.jsonl").read_text(encoding="utf-8").splitlines()]
    texts = [r["atoms"][-1]["text"] for r in rows]
    # Every selected real target, not a toy corpus, survives byte round-trip.
    exact = 0
    for start in range(0, len(texts), 32):
        ids, pad = encode_complete(texts[start:start+32], "cpu")
        for text, row, mask in zip(texts[start:start+32], ids.tolist(), pad.tolist()):
            tokens = [x for x, hidden in zip(row[1:], mask[1:]) if not hidden]
            result = decode_complete(tokens)
            exact += result["status"] == "complete_unasserted" and result["text"] == text
    checks.append(dict(name="all_real_targets_lossless", passed=exact == len(texts), rows=exact))
    refusal("over_capacity_training_refusal", "MNLM_SURFACE_TARGET_CAPACITY", lambda: encode_complete([texts[0]+"x"*513], "cpu"))
    refusal("empty_training_refusal", "MNLM_SURFACE_EMPTY_TARGET", lambda: encode_complete([""], "cpu"))
    controls = {
        "missing_eos": (list(texts[0].encode()), "MNLM_SURFACE_NO_EOS"),
        "invalid_utf8": ([0xC3, EOS], "MNLM_SURFACE_INVALID_UTF8"),
        "illegal_bos": ([BOS, EOS], "MNLM_SURFACE_INVALID_SYMBOL"),
        "empty_output": ([EOS], "MNLM_SURFACE_EMPTY_OUTPUT"),
        "over_capacity_output": ([120]*513+[EOS], "MNLM_SURFACE_OUTPUT_CAPACITY"),
        "capacity_with_eos": ([120]*512+[EOS], "complete_unasserted"),
        "unicode_boundary": (list(("é"*256).encode())+[EOS], "complete_unasserted"),
    }
    for name, (tokens, expected) in controls.items():
        result = decode_complete(tokens); publish(a.output/(name+".json"), result)
        checks.append(dict(name=name, expected=expected, observed=result["status"], passed=result["status"] == expected))
    after = {n: sha(a.population/n) for n in before}; publish(a.output/"after.json", after)
    checks.append(dict(name="source_unchanged", passed=before == after))
    publish(a.output/"fsv.json", dict(acceptance="author-checked", positive=positive, checks=checks,
            passed=sum(c["passed"] for c in checks), total=len(checks), scientific_admission=False,
            boundaries=["pinned source and typed population", "history/target alignment", "complete byte output/refusal"]))
    check(all(c["passed"] for c in checks), "FSV_FAILED")
    print(json.dumps(dict(passed=len(checks), total=len(checks))))


if __name__ == "__main__":
    main()
