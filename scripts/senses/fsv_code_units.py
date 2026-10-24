"""Cold source/AST/UCR/native boundaries for complete code-unit publications."""
import time
ENTRY = time.perf_counter()

import argparse
import ast
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "lens")]
from prefix_panel import load as load_panel
from production.native_rows import mapped_rows, one
from senses.code_snapshot import SnapshotInputs
from senses.code_unit_reader import CodeUnitInputs
from senses.code_unit_panels import load, plan
from senses.code_units import partition, validate_partition
from senses.contract import CHANNELS, SenseError, digest, require, validate_record
from senses.instruments import contracts, measure
from senses.fsv_code_snapshot import configure, model_input
from senses.trajectory_population import publish

CODE = "MNLM_CODE_UNIT_FSV"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def independent_graph(raw):
    starts = [0, *[m.end() for m in re.finditer(rb"\r\n|\r|\n", raw)]]
    result = []

    def visit(node, parent, edge, index):
        ident = len(result)
        span = None
        if getattr(node, "end_lineno", None) is not None:
            span = [starts[node.lineno-1]+node.col_offset, starts[node.end_lineno-1]+node.end_col_offset]
        result.append({"id": ident, "parent": parent, "field": edge, "index": index,
                       "kind": type(node).__name__, "span": span})
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                visit(value, ident, field, -1)
            elif isinstance(value, list):
                for i, child in enumerate(value):
                    if isinstance(child, ast.AST):
                        visit(child, ident, field, i)
    visit(ast.parse(raw.decode("utf-8")), -1, "root", -1)
    return result


def replay(root, output):
    import torch
    from model.checkpoint import load as checkpoint_load, state_hash
    configure()
    binding = json.loads((root / "model-input.json").read_bytes())
    _, specs, native = model_input(Path(binding["panel_path"]), binding["panel_sha256"], binding["atom"])
    model, _ = checkpoint_load(root / "checkpoint", "cpu")
    require(model.specs == specs, CODE, "checkpoint instrument identities changed")
    with torch.no_grad():
        r = model(torch.tensor([[256]], dtype=torch.int64), native)
    result = {"logits_sha256": hashlib.sha256(r["logits"].numpy().tobytes()).hexdigest(),
              "state_sha256": state_hash(model), "channels": r["channel_count"].tolist(),
              "meaning_asserted": r["meaning_asserted"]}
    publish(output, result)


def run(a):
    started = time.perf_counter()
    output = a.output
    output.mkdir(parents=True, exist_ok=False)
    checks = []

    def check(name, ok):
        require(bool(ok), CODE, name)
        checks.append(name)

    def refuses(name, action, expected):
        try:
            action()
        except SenseError as exc:
            check(name, exc.code == expected)
        else:
            raise SenseError(CODE, name + " accepted broken state")

    m = load(a.panel, a.panel_sha256)
    reader = CodeUnitInputs(a.snapshot, a.snapshot_sha256)
    check("snapshot_identity", reader.identity == m["reader_identity"] and m["snapshot_sha256"] == a.snapshot_sha256)
    records = json.loads((a.panel / "inputs/records.json").read_bytes())
    check("records_exact", records == reader.records and len(records) == m["counts"]["records"])
    nodes, source_bytes, edges = 0, 0, 0
    for f in reader.files:
        raw = (a.snapshot / f["phase"] / f["path"]).read_bytes()
        context = reader.context(f["context_sha256"])
        units = [records[atom] for atom in f["atoms"]]
        file_sha = hashlib.sha256(raw).hexdigest()
        check("bytes_"+f["phase"]+"/"+f["path"], b"".join(r["channels"]["surface"]["items"][0]["text"].encode() for r in units) == raw)
        cursor, seen, emitted = 0, [], []
        for r in units:
            validate_record(r)
            lo, hi = r["locator"]["span"]
            check("span_"+r["atom_id"], lo == cursor < hi and r["locator"]["record_digest"] == file_sha)
            cursor = hi
            seen.extend(q["value"] for q in r["channels"]["structure"]["items"] if q["name"] == "syntax.node")
            group = None
            for q in r["channels"]["structure"]["items"]:
                if q["name"] == "syntax.node":
                    group = {}
                    emitted.append(group)
                if group is not None:
                    group[q["name"]] = q["value"]
        check("nodes_once_"+f["phase"]+"/"+f["path"], sorted(seen) == list(range(len(context["nodes"]))))
        if context["parsed"]:
            graph = independent_graph(raw)
            check("ast_"+f["phase"]+"/"+f["path"], graph == [{k:v for k,v in n.items() if k != "owner"} for n in context["nodes"]])
            measured_graph = [{"id": n["syntax.node"], "parent": n["syntax.parent"], "field": n["syntax.edge"],
                "index": n["syntax.index"], "kind": n["syntax.kind"],
                "span": [n["syntax.start"], n["syntax.end"]] if n["syntax.has_span"] else None} for n in emitted]
            check("typed_ast_"+f["phase"]+"/"+f["path"], sorted(measured_graph, key=lambda n:n["id"]) == graph)
            nodes += len(graph)
            edges += sum(n["parent"] >= 0 for n in graph)
        source_bytes += len(raw)
    check("one_task_not_fragment_samples", {r["unit"] for r in records.values()} == {reader.manifest["unit"]}
          and m["counts"]["task_grouping_units"] == 1)
    for atom in reader.examples:
        current, past, binding = reader.read(atom)
        check("history_"+atom, binding["future_visible"] is False and binding["outcomes_visible"] is False
              and all(r["unit"] == current["unit"] and r["locator"]["row"] == 0
                      and r["locator"]["shard"].removeprefix("base/") == current["locator"]["shard"].removeprefix("submitted/") for r in past))
    selected = next(iter(reader.examples))
    binding = deepcopy(reader.bindings[selected])
    reader.bindings[selected]["future_visible"] = True
    refuses("loaded_cutoff_mutation", lambda: reader.read(selected), "MNLM_CODE_UNIT_READER")
    reader.bindings[selected] = binding
    context = next(iter(reader.contexts))
    returned = reader.context(context)
    returned["file_sha256"] = "0"*64
    check("context_return_is_detached", reader.context(context)["file_sha256"] != "0"*64)
    source_done = time.perf_counter()

    control = '# π\r\n@decorate\r\nclass Café:\r\n    @other\r\n    def f(self):\r\n        return "😀"\r\n\r\ndata = (\r\n    (1, "x"),\r\n    (2, "y"),\r\n)\r\n'.encode()
    u, c = partition(control, "python")
    validate_partition(control, u, c)
    check("unicode_crlf_decorators", b"".join(control[x["span"][0]:x["span"][1]] for x in u) == control
          and independent_graph(control) == [{k:v for k,v in n.items() if k != "owner"} for n in c["nodes"]])
    publish(output / "boundary-control.json", {"control_only": True, "source": control.decode(), "context": c})
    du, _ = partition(b'def f():\n    return 1\ndef f():\n    return 2\n', "python")
    check("duplicate_definitions_have_no_identity", all(x["key"] is None for x in du if "FunctionDef" in x["kind"]))
    nested, _ = partition(b'def f():\n    class C:\n        def g(self):\n            return 1\n    return C\n', "python")
    check("nested_class_does_not_cut_function", len(nested) == 1)
    wrong_codec = '# coding: latin-1\nx = "é"\n'.encode("utf-8")
    coded, coded_context = partition(wrong_codec, "python")
    check("declared_encoding_not_reinterpreted", coded_context["parsed"] is False
          and any(f["name"] == "syntax.status" and f["value"] == "MNLM_CODE_UNIT_ENCODING" for f in coded[0]["syntax"]))
    check("encoding_control_has_different_actual_meaning",
          ast.parse(wrong_codec).body[0].value.value != ast.parse(wrong_codec.decode()).body[0].value.value)
    for name, edit in (("span_gap", lambda us: us[0]["span"].__setitem__(1, us[0]["span"][1]-1)),
                       ("node_missing", lambda us: next(x for x in us if x["nodes"])["nodes"].pop()),
                       ("parent_edge_missing", lambda us: next(x for x in us if x["parents"])["parents"].pop())):
        bad = deepcopy(u)
        edit(bad)
        refuses(name, lambda: validate_partition(control, bad, c), "MNLM_CODE_UNITS")
    refuses("changed_source", lambda: validate_partition(control+b"# changed", u, c), "MNLM_CODE_UNITS")
    if a.require_old_refusal:
        refuses("whole_file_still_too_large", lambda: plan(SnapshotInputs(a.snapshot, a.snapshot_sha256)), "MNLM_CODE_UNIT_TOO_LARGE")

    # Cold native arrays are recounted against freshly prepared real observations.
    native_rows, native_comparisons = 0, 0
    chosen = None
    for part in m["parts"]:
        pm, rows = load_panel(a.panel / part["path"], part["manifest_sha256"])
        check("same_instruments_"+part["path"], {k:v["contract"] for k,v in pm["slots"].items()} == {c["channel"]:c for c in contracts()})
        with ExitStack() as stack:
            arrays = {c: stack.enter_context(mapped_rows(a.panel/part["path"]/(c+".npz"))) for c in CHANNELS}
            for i, row in enumerate(rows):
                current, past, _ = reader.read(row["atom"])
                available = 0
                for channel in CHANNELS:
                    values = measure(current, channel, past if channel == "history" else None)["tokens"]
                    actual = one(arrays[channel], i)["tokens"]
                    check("native_"+channel+"_"+row["atom"], np.array_equal(values, actual))
                    native_rows += len(values)
                    native_comparisons += 1
                    available += bool(len(values))
                if chosen is None and available >= 5:
                    chosen = (part, row["atom"])
    native_done = time.perf_counter()

    # Mutations occur only in an owned evidence copy, never authoritative inputs.
    copied = output / "publication-control"
    shutil.copytree(a.panel, copied)
    marker = (copied / "COMMITTED").read_bytes()
    (copied / "COMMITTED").unlink()
    refuses("incomplete_publication", lambda: load(copied, a.panel_sha256), "MNLM_CODE_UNIT_PANELS")
    (copied / "COMMITTED").write_bytes(marker)
    context_path = next((copied / "inputs/contexts").glob("*.json"))
    original = context_path.read_bytes()
    context_path.write_bytes(original+b" ")
    refuses("changed_context", lambda: load(copied, a.panel_sha256), "MNLM_CODE_UNIT_PANELS")
    context_path.write_bytes(original)
    slot = copied / m["parts"][0]["path"] / "structure.npz"
    with slot.open("r+b") as f:
        f.seek(-1, 2); last = f.read(1); f.seek(-1, 2); f.write(bytes([last[0]^1]))
    refuses("changed_native_bytes", lambda: load(copied, a.panel_sha256), "MNLM_PREFIX_PANEL_DIGEST")
    with slot.open("r+b") as f:
        f.seek(-1, 2); f.write(last)
    load(copied, a.panel_sha256)
    manifest_bytes = (copied / "manifest.json").read_bytes()
    partial = deepcopy(m)
    partial["parts"] = partial["parts"][:-1]
    partial_atoms = [atom for p in partial["parts"] for atom in p["atoms"]]
    partial["atoms_sha256"] = digest(partial_atoms)
    partial["counts"]["current_units"] = len(partial_atoms)
    (copied / "manifest.json").write_text(json.dumps(partial), encoding="utf-8")
    partial_sha = sha(copied / "manifest.json")
    (copied / "COMMITTED").write_text(json.dumps({"manifest_sha256": partial_sha}), encoding="utf-8")
    refuses("resigned_partial_scope", lambda: load(copied, partial_sha), "MNLM_CODE_UNIT_PANELS")
    (copied / "manifest.json").write_bytes(manifest_bytes)
    (copied / "COMMITTED").write_bytes(marker)
    incomplete = deepcopy(m)
    del incomplete["inputs"][next(k for k in incomplete["inputs"] if k.startswith("inputs/contexts/"))]
    (copied / "manifest.json").write_text(json.dumps(incomplete), encoding="utf-8")
    incomplete_sha = sha(copied / "manifest.json")
    (copied / "COMMITTED").write_text(json.dumps({"manifest_sha256": incomplete_sha}), encoding="utf-8")
    refuses("unbound_context_file", lambda: load(copied, incomplete_sha), "MNLM_CODE_UNIT_PANELS")
    (copied / "manifest.json").write_bytes(manifest_bytes)
    (copied / "COMMITTED").write_bytes(marker)
    controls_done = time.perf_counter()

    import torch
    from model.network import SensoryLanguageModel
    from model.checkpoint import save, state_hash
    require(chosen is not None, CODE, "no actual unit with five observed channels")
    configure()
    part, atom = chosen
    config, specs, native = model_input(a.panel/part["path"], part["manifest_sha256"], atom)
    model = SensoryLanguageModel(config, specs).eval()
    prefix = torch.tensor([[256]], dtype=torch.int64)
    with torch.no_grad():
        result = model(prefix, native)
        ablations = {}
        for spec in specs:
            if bool(native[spec.name].absent.all()):
                continue
            branch = model.blocks[0].branches[spec.name]
            gain = branch.gain.clone(); branch.gain.zero_()
            changed = model(prefix, native)["logits"]
            branch.gain.copy_(gain)
            ablations[spec.channel] = float((changed-result["logits"]).abs().max())
    check("five_effective_branches", len(ablations) >= 5 and all(x > 0 for x in ablations.values()))
    refuses("meaning_not_admitted", lambda: model.assert_meaning(native), "MNLM_MODEL_UNCOMMISSIONED")
    binding = {"panel_path": str((a.panel/part["path"]).resolve()), "panel_sha256": part["manifest_sha256"], "atom": atom}
    publish(output / "model-input.json", binding)
    save(model, output / "checkpoint", step=0, input_sha256=digest(binding))
    expected = {"logits_sha256": hashlib.sha256(result["logits"].numpy().tobytes()).hexdigest(),
                "state_sha256": state_hash(model), "channels": result["channel_count"].tolist(), "meaning_asserted": False}
    publish(output / "model-before.json", expected)
    child = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--replay", str(output),
                            "--output", str(output / "model-replay.json")], capture_output=True, timeout=30)
    check("fresh_process_replay", child.returncode == 0 and json.loads((output / "model-replay.json").read_bytes()) == expected)
    finished = time.perf_counter()
    report = {"checks": checks, "check_count": len(checks), "source_bytes": source_bytes, "syntax_nodes": nodes,
              "syntax_edges": edges, "native_rows": native_rows, "native_comparisons": native_comparisons,
              "counts": reader.counts, "shards": len(m["parts"]), "native_value_bytes": m["native_value_bytes"],
              "ablations": ablations, "trained": False, "admitted": False, "stage4_accepted": False,
              "author_checked": True}
    publish(output / "report.json", report)
    end = time.perf_counter()
    publish(output / "receipt.json", {"wall_seconds": end-ENTRY, "scope": "cli_including_imports", "phases": {
        "imports": started-ENTRY, "source_ast_ucr": source_done-started, "controls_and_native": native_done-source_done,
        "publication_controls": controls_done-native_done, "model": finished-controls_done, "publication": end-finished}})
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path)
    for name in ("snapshot", "panel", "output"):
        p.add_argument("--"+name, type=Path, required=name == "output")
    for name in ("snapshot-sha256", "panel-sha256"):
        p.add_argument("--"+name)
    p.add_argument("--require-old-refusal", action="store_true")
    a = p.parse_args()
    if a.replay:
        replay(a.replay, a.output)
    else:
        result = run(a)
        print(json.dumps({k:v for k,v in result.items() if k not in ("checks", "ablations")}))
