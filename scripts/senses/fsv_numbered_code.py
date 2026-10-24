"""Source -> numbered excerpt -> native slots -> decoder boundary evidence.

One count-only corpus pass O(input bytes + syntax nodes); a fixed, outcome-blind
boundary subset alone receives dense nine-channel publication. Controls are not
training data. Decoder exercise is explicitly CPU, engineering-only, no fallback.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
                str(Path(__file__).resolve().parents[1] / "lens")]
from senses.contract import CHANNELS, SenseError, canonical, digest, validate_record
from senses.numbered_code import parse, enrich, CONTRACT, MAX_BYTES
from senses.structured_prefix import PrefixInputs as StructuredInputs
from senses.trajectory_prefix import PrefixInputs
from senses.fsv_fences import audit_ast
from senses.instruments import DIM, contracts, groups, measure
from senses.pipeline import file_hash
from senses.trajectory_population import publish as put
from prefix_panel import publish, load, census


def audit_view(text, view):
    """Separate physical line scanner and AST walk; never uses producer spans."""
    physical = list(re.finditer(r"[^\r\n]*(?:\r\n|\r|\n|$)", text))
    physical = [m for m in physical if m.end() > m.start()]
    header = re.fullmatch(r"\[File: (.+) \((\d+) lines total\)\]", physical[0][0].rstrip("\r\n"))
    assert header and view["path"] == header[1] and view["declared_total"] == int(header[2])
    assert view["response_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    if view["rendered_text"] is None:
        assert not view["line_map"] and not any(f["name"] == "syntax.node" for f in view["fields"])
        return 0
    actual, rendered = [], []
    for m in physical[1:]:
        body = m[0].rstrip("\r\n")
        if not actual and re.fullmatch(r"\(\d+ more lines above\)", body):
            continue
        number = re.match(r"^(\d+):", body)
        if number is None:
            break
        start = m.start() + number.end()
        actual.append({"line": int(number[1]), "response_chars": [start, m.end()]})
        rendered.append(text[start:m.end()])
    assert actual == view["line_map"] and "".join(rendered) == view["rendered_text"]
    assert [x["line"] for x in actual] == list(range(view["above"] + 1, view["declared_total"] - view["below"] + 1))
    assert view["full_display"] == (view["above"] == view["below"] == 0)
    assert view["whole_repository_file_attested"] is False
    return audit_ast(view["rendered_text"], view["fields"])


def run(a):
    import numpy as np
    started = time.perf_counter()
    a.out.mkdir(parents=True, exist_ok=False)
    checks = []
    def check(name, ok):
        assert ok, name
        checks.append(name)
    def refused(name, fn, code):
        try:
            fn()
        except SenseError as e:
            check(name, e.code == code)
        else:
            check(name, False)

    # B1: explicit interpretation boundaries, before reading source records.
    for ending, name in [("\n", "LF"), ("\r\n", "CRLF"), ("\r", "CR")]:
        text = ending.join(["[File: /never-open/Ã©.py (2 lines total)]", "1:Ã© = 1", "2:x = Ã© + 2", ""])
        view = parse(text)
        check(name + "_exact_spans_AST", view["status"] == "parsed" and audit_view(text, view) > 0)
    check("no_language_guess", parse("1:x=1") is None)
    check("quoted_header_not_source", parse("quoted:\n[File: x.py (1 lines total)]\n1:x=1") is None)
    for name, text, code in [
        ("unsupported", "[File: x.js (1 lines total)]\n1:x=1", "MNLM_SYNTAX_LANGUAGE_UNSUPPORTED"),
        ("gap", "[File: x.py (3 lines total)]\n1:x=1\n3:y=2", "MNLM_FILE_VIEW_LINE_RANGE"),
        ("duplicate", "[File: x.py (2 lines total)]\n1:x=1\n1:y=2", "MNLM_FILE_VIEW_LINE_RANGE"),
        ("missing_omission", "[File: x.py (3 lines total)]\n1:x=1", "MNLM_FILE_VIEW_OMISSION"),
        ("wrong_omission", "[File: x.py (3 lines total)]\n1:x=1\n(1 more lines below)", "MNLM_FILE_VIEW_OMISSION"),
        ("above_mismatch", "[File: x.py (3 lines total)]\n(1 more lines above)\n1:x=1", "MNLM_FILE_VIEW_LINE_RANGE"),
        ("huge_range", "[File: x.py (" + "9" * 100 + " lines total)]\n1:x=1", "MNLM_FILE_VIEW_LINE_RANGE"),
        ("byte_budget", "[File: x.py (1 lines total)]\n1:" + "x" * MAX_BYTES, "MNLM_FILE_VIEW_BYTE_BUDGET"),
        ("no_dedent", "[File: x.py (1 lines total)]\n1:    x=1", "MNLM_SYNTAX_UNPARSED"),
    ]:
        view = parse(text)
        check(name, view["status"] == code and not any(f["name"] == "syntax.node" for f in view["fields"]))
    text = "[File: x.py (3 lines total)]\n(1 more lines above)\n2:x=1\n(1 more lines below)\nfooter"
    view = parse(text)
    check("valid_partial_is_not_full", view["status"] == "parsed" and not view["full_display"] and audit_view(text, view) > 0)
    other = parse(text.replace("footer", "unrelated future-like words"))
    check("footer_not_syntax", [f for f in view["fields"] if f["name"].startswith("syntax.")] ==
          [f for f in other["fields"] if f["name"].startswith("syntax.")])
    bad = copy.deepcopy(view)
    bad["line_map"][0]["response_chars"][0] += 1
    try:
        audit_view(text, bad)
    except AssertionError:
        checks.append("independent_mapping_corruption_detected")
    else:
        check("independent_mapping_corruption_detected", False)

    source = PrefixInputs(a.prefix, a.sha256)
    original_sha = file_hash(a.prefix / "inputs.ucr.jsonl")
    reader = StructuredInputs(source)
    reader.write_descriptor(a.out / "descriptor.json")
    put(a.out / "views.json", reader.receipts)
    put(a.out / "index.json", reader.examples)
    # Read persisted receipts back, checking their spans against original bytes.
    views = json.loads((a.out / "views.json").read_bytes())
    nodes, statuses, current_statuses, full = 0, Counter(), Counter(), 0
    for parent, receipts in views.items():
        record = source.records[parent]
        for view in receipts:
            text = record["channels"]["behaviour"]["items"][view["event_index"]]["measurements"][view["field_index"]]["value"]
            nodes += audit_view(text, view)
            statuses[view["status"]] += 1
            if parent in source.examples:
                current_statuses[view["status"]] += 1
                full += view["full_display"]
    checks += ["cold_source_to_excerpt_all_records", "cold_AST_paths_and_utf8_offsets", "partial_extent_retained"]
    chosen = {}
    for atom, ex in reader.examples.items():
        receipts = views[ex["parent"]]
        category = tuple((v["status"], v["full_display"]) for v in receipts)
        # First observation in original sealed order per format boundary; no label opened.
        chosen.setdefault(category, ex["parent"])
    selected = list(chosen.values())
    put(a.out / "selection.json", {"rule": "first-current-per-status-and-extent-in-sealed-order", "parents": selected})
    full_plan = census(reader)
    put(a.out / "full-count-plan.json", full_plan)
    subset = StructuredInputs(source, selected)
    subset.write_descriptor(a.out / "subset-descriptor.json")
    put(a.out / "subset-records.json", subset.records)
    retained = json.loads((a.out / "subset-records.json").read_bytes())
    check("derived_records_cold_readback", all(validate_record(r)["atom_id"] == k for k, r in retained.items())
          and retained == subset.records)
    plan = census(subset)
    put(a.out / "subset-plan.json", plan)  # exact byte preflight, before dense writes
    result = publish(subset, plan, a.out / "panel")

    # B2: replay each persisted native slot, and verify lineage is not a feature.
    from production.native_rows import mapped_rows
    manifest, rows = load(a.out / "panel", result["manifest_sha256"])
    for c in CHANNELS:
        with mapped_rows(a.out / "panel" / (c + ".npz")) as arrays:
            for i, row in enumerate(rows):
                current, past, binding = subset.read(row["atom"])
                assert binding == row["binding"]
                expected = measure(current, c, past if c == "history" else None)
                lo, hi = arrays["offsets"][i:i+2]
                assert np.array_equal(arrays["tokens"][lo:hi], expected["tokens"])
        checks.append("cold_native_" + c)
    atom = next(k for k, e in subset.examples.items()
                if any(v["status"] == "parsed" for v in views[e["parent"]]))
    current, past, binding = subset.read(atom)
    expected = copy.deepcopy((current, past, binding))
    current["unit"] = "caller-mutated"
    past.reverse()
    binding.clear()
    check("reader_copies_isolated", subset.read(atom) == expected)
    refused("derive_twice_refused", lambda: enrich(expected[0]), "MNLM_FILE_VIEW_ALREADY_DERIVED")
    before = copy.deepcopy(subset.examples[atom])
    subset.examples[atom]["past"].reverse()
    refused("cached_order_mutation_refused", lambda: subset.read(atom), "MNLM_PREFIX_BINDING")
    subset.examples[atom] = before
    refused("unselected_parent_refused", lambda: subset.read(before["parent"]), "MNLM_PREFIX_NOT_FOUND")
    base = source.records[before["parent"]]
    derived, _ = enrich(base)
    for c in CHANNELS:
        if c not in ("structure", "provenance"):
            assert derived["channels"][c] == base["channels"][c]
    check("all_original_observations_retained", derived["atom_id"] != base["atom_id"])
    check("provenance_refs_not_numeric_features", np.array_equal(measure(base, "provenance")["tokens"], measure(derived, "provenance")["tokens"]))

    class BadSource:
        identity, reader_code_sha256, examples = source.identity, source.reader_code_sha256, source.examples
        def read(self, key):
            current, past, binding = source.read(key)
            binding["future_visible"] = True
            return current, past, binding
    refused("future_exposure_refused", lambda: StructuredInputs(BadSource(), selected[:1]), "MNLM_PREFIX_VISIBILITY")

    # B3: actual native-bank consumer and decoder, explicitly CPU on this small
    # engineering subset. No training quality, MI or hardware claim is made.
    import torch
    from model.contracts import ModelConfig, parameter_count
    from model.network import SensoryLanguageModel
    from model.prefix_bank import read_bank
    from model.checkpoint import save as save_checkpoint, load as load_checkpoint
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    config = ModelConfig()
    channels = ["surface", "fields", "structure", "provenance", "behaviour"]
    model_ids = [atom]  # one real structurally observed record; unchanged native budget
    specs, inputs, receipt = read_bank(a.out / "panel", result["manifest_sha256"], model_ids, channels, config)
    put(a.out / "model-plan.json", {"runtime": "explicit-cpu-boundary-proof", "fallback": False,
        "rows": model_ids, "parameters_projected": parameter_count(config, specs), "input": receipt,
        "input_elements_projected": sum(x.values.numel() for x in inputs.values()), "training": False})
    model = SensoryLanguageModel(config, specs).eval()
    prefix = torch.tensor([[0]], dtype=torch.int64)
    with torch.no_grad():
        prediction = model(prefix, inputs)
        baseline = prediction["logits"].clone()
        structure = next(s.name for s in specs if s.channel == "structure")
        model.blocks[0].branches[structure].gain.zero_()
        ablated = model(prefix, inputs)["logits"].clone()
        model.blocks[0].branches[structure].gain.fill_(1)
    check("model_consumes_structure", not torch.equal(baseline, ablated))
    check("five_separate_instruments", len(specs) == 5 and int(prediction["instrument_count"][0]) == 5)
    check("no_meaning_claim", prediction["meaning_asserted"] is False)
    refused("uncommissioned_meaning_refused", lambda: model.assert_meaning(inputs), "MNLM_MODEL_UNCOMMISSIONED")
    checkpoint = save_checkpoint(model, a.out / "checkpoint", step=0, input_sha256=digest(receipt))
    np.save(a.out / "logits.npy", baseline.numpy(), allow_pickle=False)
    cold, saved = load_checkpoint(a.out / "checkpoint", "cpu")
    cold.eval()
    check("checkpoint_input_bound", saved["input_sha256"] == digest(receipt) and digest(saved) == checkpoint)
    with torch.no_grad():
        replay = cold(prefix, inputs)["logits"].numpy()
    check("cold_checkpoint_exact_replay", np.array_equal(replay, np.load(a.out / "logits.npy", allow_pickle=False)))
    check("source_bytes_unchanged", file_hash(a.prefix / "inputs.ucr.jsonl") == original_sha)
    report = {"schema": "numbered-code-fsv-v1", "acceptance": "author-checked", "checks": checks,
        "source_manifest": a.sha256, "source_bytes_sha256": original_sha, "adapter": CONTRACT,
        "records": len(source.records), "current": len(source.examples), "views": dict(statuses),
        "current_views": dict(current_statuses), "current_full_displays": full, "ast_nodes": nodes,
        "selected": len(selected), "native_bytes_measured": result["native_bytes"],
        "current_structure_value_bytes_projected": full_plan["channels"]["structure"]["projected_values_bytes"],
        "panel_manifest": result["manifest_sha256"], "model_parameters_measured": sum(p.numel() for p in model.parameters()),
        "model_checkpoint": checkpoint,
        "structure_ablation_max_logit_delta_measured": float((baseline - ablated).abs().max()),
        "files": {p.name: file_hash(p) for p in a.out.iterdir() if p.is_file()},
        "stage3_admitted": False, "stage4_accepted": False, "wall_seconds": time.perf_counter() - started,
        "audit_code_sha256": file_hash(Path(__file__))}
    sha = put(a.out / "report.json", report)
    print(json.dumps({"report_sha256": sha, "checks": len(checks), "views": dict(current_statuses),
                      "selected": len(selected), "native_bytes": result["native_bytes"], "wall_seconds": report["wall_seconds"]}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", type=Path, required=True)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--out", type=Path, required=True)
    run(ap.parse_args())
