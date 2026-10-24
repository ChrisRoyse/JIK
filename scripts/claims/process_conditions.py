"""Outcome-blind execution inputs committed before a new process occurrence.

The public input contains the actual program and runtime inventory. Archived
target references and experiment labels are private routing, never model input.
These are provenance-bound conditions, not an admitted sense or prediction.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest


FILES = ("condition-inputs.json", "condition-commitment.json", "conditions-COMMITTED")
HEX = re.compile(r"[0-9a-f]{64}")


def condition_input(spec, name, program, harness_pin):
    """Explicit projection; no target, condition label, output or return code."""
    wheels = spec["conditions"][name]["wheels"]
    check(isinstance(program, bytes) and program
          and sha(program) == spec["program_sha256"], "CONDITION_PROGRAM")
    check(isinstance(wheels, dict) and wheels and all(
        isinstance(n, str) and n == Path(n).name and "\\" not in n and ":" not in n
        and n.endswith(".whl") and isinstance(p, str) and HEX.fullmatch(p)
        for n, p in wheels.items()), "CONDITION_WHEELS")
    check(isinstance(spec["image"], str) and spec["image"].startswith("sha256:")
          and HEX.fullmatch(spec["image"][7:]) and HEX.fullmatch(harness_pin),
          "CONDITION_RUNTIME")
    return dict(schema="process-condition-input-v1", program=program.decode("utf-8"),
        program_sha256=sha(program), image=spec["image"], wheels=dict(wheels),
        harness_sha256=harness_pin, mode="observe-script",
        scope="declared_program_and_available_runtime_before_execution")


def write_committed(path, value):
    raw = canonical(value)
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return sha(raw)


def publish(root, spec, prepared_pin, harness_pin):
    """Caller has authenticated the prepared inventory; recheck exact inputs."""
    root = Path(root)
    check(not any((root/n).exists() for n in FILES), "CONDITION_ALREADY_COMMITTED")
    inputs, routing = {}, {}
    for name, item in sorted(spec["conditions"].items()):
        check(name in ("with_dependency", "without_dependency"), "CONDITION_ROUTE")
        program = (root/name/"work/program.py").read_bytes()
        actual = {p.name: sha(p.read_bytes()) for p in sorted((root/name/"runtime").iterdir())
                  if p.is_file()}
        check(actual == item["wheels"], "CONDITION_RUNTIME_BYTES")
        value = condition_input(spec, name, program, harness_pin)
        key = digest(value)
        inputs[key] = value
        routing[name] = key
    check(len(inputs) == len(routing) == 2, "CONDITION_DISTINCT_ROSTER")
    public = dict(schema="process-condition-inputs-v1", inputs=inputs)
    public_pin = write_committed(root/FILES[0], public)
    commitment = dict(schema="process-condition-commitment-v1", inputs=public_pin,
        prepared=prepared_pin, source=spec["source"], program_source=spec["program_source"],
        program_span=spec["program_span"], routing=routing,
        committed_at=datetime.now(timezone.utc).isoformat(),
        clock="host_utc_before_execute_call", prediction_present=False,
        scientific_admission=False, model_consumption_established=False)
    pin = write_committed(root/FILES[1], commitment)
    with (root/FILES[2]).open("x", encoding="ascii") as f:
        f.write(pin)
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return pin


class ConditionInputs:
    """Loads only input and commitment files; no outcome artifacts are opened."""
    def for_condition(self, key):
        check(key in self.inputs, "CONDITION_UNKNOWN")
        return self

    def __init__(self, root, expected):
        self.root, self.pin = Path(root), expected
        raw = (self.root/FILES[1]).read_bytes()
        check(sha(raw) == expected == (self.root/FILES[2]).read_text(), "CONDITION_COMMITMENT_PIN")
        self.commitment = json.loads(raw)
        c = self.commitment
        check(canonical(c) == raw and c["schema"] == "process-condition-commitment-v1"
              and c["prediction_present"] is False and c["scientific_admission"] is False
              and c["model_consumption_established"] is False, "CONDITION_COMMITMENT_SCOPE")
        public = (self.root/FILES[0]).read_bytes()
        check(sha(public) == c["inputs"], "CONDITION_INPUT_PIN")
        value = json.loads(public)
        check(canonical(value) == public and set(value) == {"schema", "inputs"}
              and value["schema"] == "process-condition-inputs-v1", "CONDITION_INPUT_SCHEMA")
        self.inputs = value["inputs"]
        check(len(self.inputs) == 2 and set(c["routing"].values()) == set(self.inputs),
              "CONDITION_INPUT_ROSTER")
        for key, row in self.inputs.items():
            check(key == digest(row) and set(row) == {"schema", "program", "program_sha256",
                "image", "wheels", "harness_sha256", "mode", "scope"}, "CONDITION_INPUT_FIELDS")
            rebuilt = condition_input(dict(image=row["image"], program_sha256=row["program_sha256"],
                conditions={"input":dict(wheels=row["wheels"])}), "input", row["program"].encode(),
                row["harness_sha256"])
            check(rebuilt == row, "CONDITION_INPUT_CANONICAL")

    def read(self, key):
        check(key in self.inputs, "CONDITION_UNKNOWN")
        check(digest(self.inputs[key]) == key, "CONDITION_MEMORY_DRIFT")
        # Return a copy; metadata such as routing and prior target references is excluded.
        return json.loads(canonical(self.inputs[key]))

    def episode(self, key, evidence=()):
        body = dict(claim_kind="execution_precondition", content=self.read(key),
            source=dict(condition=key, commitment=self.pin), evidence=list(evidence),
            event_time=self.commitment["committed_at"],
            who=dict(actor="execution_plan_publisher"), what="declared_execution_conditions",
            where=dict(image=self.inputs[key]["image"]),
            when=dict(clock=self.commitment["clock"], at=self.commitment["committed_at"]),
            why=None, why_status="unresolved", independent_occurrence=False,
            model_consumption_established=False, native_measured=False)
        return dict(id=digest(body), **body)

    def validate_episode(self, episode):
        key = episode.get("source", {}).get("condition")
        check(key in self.inputs and episode == self.episode(key, episode.get("evidence", ()))
              and len(set(episode["evidence"])) == len(episode["evidence"]), "CONDITION_EPISODE")

    def assess(self, episode):
        self.validate_episode(episode)
        body = dict(checker="precommitted-execution-conditions-v1", episode=episode["id"],
            claim_kind="execution_precondition", scope="precommitted_execution_conditions",
            source=episode["source"], world_validity="declared_configuration_only",
            prediction_present=False, model_consumption_established=False, native_measured=False)
        return dict(id=digest(body), **body)

    def match(self, key, entry, observed):
        row = self.read(key)
        result = observed["result"]
        check(entry["program_sha256"] == result["program_sha256"] == row["program_sha256"]
              and entry["wheels"] == result["wheels"] == row["wheels"]
              and observed["image"] == row["image"]
              and result["harness_sha256"] == row["harness_sha256"], "CONDITION_OUTCOME_MISMATCH")
        start = datetime.fromisoformat(observed["started"].replace("Z", "+00:00"))
        committed = datetime.fromisoformat(self.commitment["committed_at"])
        check(start.tzinfo is not None and committed.tzinfo is not None and start > committed,
              "CONDITION_NOT_PREEXECUTION")
        return dict(condition=key, commitment=self.pin, inputs=self.commitment["inputs"],
            committed_at=self.commitment["committed_at"], started=observed["started"],
            scope="conditions_committed_before_new_occurrence", prediction_present=False)
