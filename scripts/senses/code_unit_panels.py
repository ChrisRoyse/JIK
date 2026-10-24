"""Complete code-unit measurements in bounded immutable native shards."""
import time
ENTRY = time.perf_counter()

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "lens")]
from prefix_panel import observation_counts, census_from_counts, publish as publish_panel, load as load_panel
from senses.code_unit_reader import CodeUnitInputs
from senses.contract import CHANNELS, SenseError, digest, require
from senses.instruments import DIM
from senses.pipeline import file_hash
from senses.trajectory_population import publish, read_pinned

CODE = "MNLM_CODE_UNIT_PANELS"
MAX_VALUES = 32 << 20
MAX_ROWS = 8192


class View:
    def __init__(self, reader, atoms):
        self.reader = reader
        require(atoms and len(set(atoms)) == len(atoms) and set(atoms) <= reader.examples.keys(),
                CODE, "unique current units required")
        self.examples = dict.fromkeys(atoms)
        self.identity = digest([reader.identity, atoms])
        self.reader_code_sha256 = {**reader.reader_code_sha256, "code_unit_panels.py": file_hash(Path(__file__))}

    def read(self, atom):
        require(atom in self.examples, CODE, "unit outside this shard")
        return self.reader.read(atom)


def plan(reader):
    batches, atoms, counts, size = [], [], [], 0
    for atom in reader.examples:
        current, past, _ = reader.read(atom)
        row = observation_counts(current, past)
        nbytes = sum(row.values()) * DIM * 4
        require(max(row.values()) <= MAX_ROWS and nbytes <= MAX_VALUES, "MNLM_CODE_UNIT_TOO_LARGE",
                f"unit {current['locator']['shard']} {current['locator']['span']} requires {max(row.values())} "
                f"rows / {nbytes} value bytes; retain source and define a smaller structural unit")
        if atoms and size + nbytes > MAX_VALUES:
            view = View(reader, atoms)
            batches.append((view, census_from_counts(view, counts)))
            atoms, counts, size = [], [], 0
        atoms.append(atom)
        counts.append(row)
        size += nbytes
    if atoms:
        view = View(reader, atoms)
        batches.append((view, census_from_counts(view, counts)))
    require(batches and sum(len(v.examples) for v, _ in batches) == len(reader.examples), CODE,
            "every current unit must be measured exactly once")
    return batches


def run(snapshot, snapshot_sha256, output, *, entry_started=None):
    started = time.perf_counter()
    beginning = started if entry_started is None else entry_started
    reader = CodeUnitInputs(snapshot, snapshot_sha256)
    prepared = time.perf_counter()
    batches = plan(reader)
    planned = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    reader.export(output / "inputs")
    exported = time.perf_counter()
    parts = []
    for i, (view, census) in enumerate(batches):
        name = "part-" + str(i).zfill(4)
        publish(output / (name + "-plan.json"), census)
        measured = publish_panel(view, census, output / name)
        parts.append({"path": name, "manifest_sha256": measured["manifest_sha256"],
                      "atoms": list(view.examples), "native_value_bytes": census["projected_values_bytes"]})
    measured = time.perf_counter()
    inputs = {p.relative_to(output).as_posix(): {"bytes": p.stat().st_size, "sha256": file_hash(p)}
              for p in sorted((output / "inputs").rglob("*")) if p.is_file()}
    manifest = {"schema": "code-unit-panel-set-v1", "reader_identity": reader.identity,
        "snapshot_sha256": snapshot_sha256, "counts": reader.counts, "parts": parts, "inputs": inputs,
        "atoms_sha256": digest(list(reader.examples)), "native_value_bytes": sum(p["native_value_bytes"] for p in parts),
        "max_values_per_shard": MAX_VALUES, "max_rows_per_slot": MAX_ROWS,
        "admitted": False, "stage4_accepted": False}
    sha = publish(output / "manifest.json", manifest)
    publish(output / "COMMITTED", {"manifest_sha256": sha})
    load(output, sha)
    finished = time.perf_counter()
    publish(output / "receipt.json", {"manifest_sha256": sha, "wall_seconds": finished-beginning,
        "scope": "function" if entry_started is None else "cli_including_imports",
        "phases": {"imports": started-beginning, "source_and_units": prepared-started, "count_plan": planned-prepared,
                   "input_export": exported-planned, "measurement_and_shard_readback": measured-exported,
                   "set_publication_and_readback": finished-measured}})
    return manifest


def load(root, sha):
    try:
        return _load(root, sha)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SenseError(CODE, "incomplete or malformed unit publication: " + str(exc)) from exc


def _load(root, sha):
    m = read_pinned(root / "manifest.json", sha)
    require(m["schema"] == "code-unit-panel-set-v1" and m["admitted"] is False
            and m["stage4_accepted"] is False and m["max_values_per_shard"] == MAX_VALUES
            and m["max_rows_per_slot"] == MAX_ROWS, CODE, "diagnostic unit panel set with fixed bounds required")
    require(json.loads((root / "COMMITTED").read_bytes()) == {"manifest_sha256": sha}, CODE,
            "complete publication marker required")
    input_paths = list((root / "inputs").rglob("*"))
    require(input_paths and all(not p.is_symlink() and not p.is_junction()
            and p.resolve().is_relative_to((root / "inputs").resolve()) for p in input_paths),
            CODE, "input links or escaped paths refused")
    actual_inputs = {p.relative_to(root).as_posix() for p in input_paths if p.is_file()}
    require(actual_inputs == m["inputs"].keys() and {"inputs/index.json", "inputs/records.json",
            "inputs/inputs.ucr.jsonl", "inputs/inputs.ucr.manifest.json"} <= actual_inputs,
            CODE, "complete input/context inventory must be hash-bound")
    for name, b in m["inputs"].items():
        p = root / name
        require(p.resolve().is_relative_to(root.resolve()) and not p.is_symlink()
                and p.stat().st_size == b["bytes"] and file_hash(p) == b["sha256"], CODE, "source/context bytes changed")
    index = json.loads((root / "inputs/index.json").read_bytes())
    require(index["counts"] == m["counts"] and index["admitted"] is False, CODE, "input census differs")
    atoms, total = [], 0
    for i, part in enumerate(m["parts"]):
        require(part["path"] == "part-" + str(i).zfill(4), CODE, "shard identity/order changed")
        manifest, rows = load_panel(root / part["path"], part["manifest_sha256"])
        require([r["atom"] for r in rows] == part["atoms"]
                and manifest["input_manifest_sha256"] == digest([m["reader_identity"], part["atoms"]]),
                CODE, "shard unit inventory/input binding changed")
        size = 0
        for row in rows:
            require(row["atom"] in index["bindings"] and row["binding"] == index["bindings"][row["atom"]],
                    CODE, "native row lost its source/cutoff binding")
            for c in CHANNELS:
                shape = row["slots"][c]["shape"]
                require(isinstance(shape, list) and len(shape) == 2 and all(type(v) is int for v in shape)
                        and shape[1] == DIM and 0 <= shape[0] <= MAX_ROWS, CODE, "native unit exceeds its fixed bound")
                size += shape[0] * DIM * 4
        require(size == part["native_value_bytes"] <= MAX_VALUES, CODE, "shard value-byte bound changed")
        total += size
        atoms.extend(part["atoms"])
    require(len(atoms) == len(set(atoms)) == m["counts"]["current_units"]
            and set(atoms) == index["bindings"].keys() and digest(atoms) == m["atoms_sha256"]
            and total == m["native_value_bytes"], CODE,
            "incomplete/duplicated units or total native bytes")
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--snapshot-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    try:
        result = run(a.snapshot, a.snapshot_sha256, a.output, entry_started=ENTRY)
        print(json.dumps({k: result[k] for k in ("counts", "native_value_bytes", "admitted")}))
    except (SenseError, OSError, ValueError, KeyError, TypeError) as exc:
        err = exc if isinstance(exc, SenseError) else SenseError(CODE, str(exc))
        print(json.dumps(err.envelope()), file=sys.stderr)
        sys.exit(2)
