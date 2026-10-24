"""Externally captured process facts, scoped to an exact isolated execution."""
import base64
import json
from pathlib import Path

from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.process_conditions import ConditionInputs, FILES as CONDITION_FILES
from claims import forecast_commitment
from oracle.container_run import validate_config

SCOPE = "observed_process_in_pinned_runtime"


class ProcessObservations:
    def for_observation(self, key):
        check(key in self.entries, "PROCESS_OBSERVATION_UNKNOWN")
        return self

    def __init__(self, root, pin, source):
        self.root, self.pin, self.source = Path(root), pin, source
        self.manifest = self._manifest()
        check(self.manifest["schema"] == "tool-process-observations-v1"
              and self.manifest["source"] == source.pin, "PROCESS_SOURCE")
        self.entries = self.manifest["observations"]
        self.forecast = None
        if "forecast" in self.manifest:
            check(self.manifest["forecast"]["root"] == "forecast", "PROCESS_FORECAST_PATH")
            for name in ("forecast.json", "journal.sqlite3", "COMMITTED"):
                self.file("forecast/"+name)
            self.forecast = forecast_commitment.read(self.root/"forecast", self.manifest["forecast"]["pin"])
        check(self.entries and all(k == digest(v) for k, v in self.entries.items()),
              "PROCESS_ENTRY_ID")
        for key in self.entries:
            self.read(key)

    def _manifest(self):
        raw = (self.root/"manifest.json").read_bytes()
        check(sha(raw) == self.pin == (self.root/"COMMITTED").read_text(), "PROCESS_MANIFEST_PIN")
        return json.loads(raw)

    def file(self, name):
        check(isinstance(name, str) and name and "\\" not in name and ":" not in name
              and not Path(name).is_absolute() and ".." not in Path(name).parts,
              "PROCESS_PATH")
        value = (self.root/name).read_bytes()
        expected = self.manifest["files"][name]
        check(len(value) == expected["bytes"] and sha(value) == expected["sha256"],
              "PROCESS_ARTIFACT_PIN")
        return value

    def read(self, key):
        check(self._manifest() == self.manifest and key in self.entries, "PROCESS_ENTRY")
        entry = self.entries[key]
        target, program_ref = entry["target"], entry["program_source"]
        check(target == self.source.reference(target["task"], target["turn"])
              and target["role"] == "recorded_response"
              and program_ref == self.source.reference(program_ref["task"], program_ref["turn"])
              and program_ref["task"] == target["task"]
              and program_ref["turn"] < target["turn"], "PROCESS_SOURCE_JOIN")
        start, end = entry["program_span"]
        program = self.file(entry["program"])
        source_bytes = self.source.texts[program_ref["task"], program_ref["turn"]].encode()
        check(source_bytes[start:end] == program and sha(program) == entry["program_sha256"],
              "PROCESS_PROGRAM_SOURCE")
        path = entry["execution"]
        execution = json.loads(self.file(path+"/execution.json"))
        before, after = (json.loads(self.file(path+"/"+s+".json")) for s in ("before", "after"))
        check(execution["mode"] == "observe-script" and execution["status"] == "observed"
              and execution["exitcode"] == 0 and not execution["oom_killed"]
              and execution["container_id"] == before["Id"] == after["Id"]
              and not after["State"]["Running"] and after["State"]["ExitCode"] == 0
              and after["State"]["FinishedAt"] > after["State"]["StartedAt"],
              "PROCESS_EXECUTION")
        mounts = {m["Destination"]: m["Source"] for m in before["Mounts"]}
        check(set(mounts) == {"/work", "/runtime", "/harness.py"}, "PROCESS_MOUNTS")
        label = before["Config"]["Labels"]["mnlm.oracle.run"]
        for state in (before, after):
            validate_config(state, execution["image"], label, mounts)
            check(state["Config"]["Entrypoint"] == ["python"]
                  and state["Config"]["Cmd"] == ["-I", "-B", "/harness.py", "observe-script"],
                  "PROCESS_COMMAND")
        result = execution["result"]
        check(result["schema"] == "observed-python-process-v1"
              and result["program_sha256"] == entry["program_sha256"]
              and result["wheels"] == entry["wheels"]
              and result["harness_sha256"] == sha(self.file("process_inside.py")),
              "PROCESS_RESULT")
        captured = {}
        for name in ("stdout", "stderr"):
            value = result[name]
            raw = base64.b64decode(value["base64"], validate=True)
            check(len(raw) == value["bytes"] and sha(raw) == value["sha256"], "PROCESS_CAPTURE")
            captured[name] = raw.decode("utf-8", errors="strict")
        outer = self.file(path+"/stdout.bin").decode("utf-8")
        markers = [line.split("=", 1)[1] for line in outer.splitlines()
                   if line.startswith("MNLM_EXECUTION_RESULT=")]
        check(len(markers) == 1 and json.loads(markers[0]) == result, "PROCESS_ENVELOPE")
        for name, pin in entry["wheels"].items():
            check(sha(self.file(entry["runtime"]+"/"+name)) == pin, "PROCESS_WHEEL")
        observed = dict(result=result, captured=captured, image=execution["image"],
                    container=execution["container_id"],
                    started=after["State"]["StartedAt"], finished=after["State"]["FinishedAt"])
        if "condition_commitment" in self.manifest:
            for name in CONDITION_FILES:
                self.file(name)
            conditions = ConditionInputs(self.root, self.manifest["condition_commitment"])
            check(conditions.commitment["prepared"] == self.manifest["prepared"]
                  and conditions.commitment["source"] == self.source.pin
                  and conditions.commitment["program_source"] == program_ref
                  and conditions.commitment["program_span"] == entry["program_span"],
                  "PROCESS_CONDITION_SOURCE")
            observed["conditions"] = conditions.match(entry["condition"], entry, observed)
        else:
            check("condition" not in entry, "PROCESS_UNCOMMITTED_CONDITION")
        if self.forecast is not None:
            matches = [r for r in self.forecast["forecasts"].values()
                       if r["request"]["task"] == target["task"] and r["request"]["turn"]+1 == target["turn"]]
            check(len(matches) == 1, "PROCESS_FORECAST_TARGET")
            forecast_commitment.match(self.forecast, matches[0], entry, observed)
        return observed

    def execution_conditions(self, key):
        """Historical receipts remain readable; no missing commitment is inferred."""
        check("condition_commitment" in self.manifest, "PROCESS_PRECONDITION_UNAVAILABLE")
        return self.read(key)["conditions"]

    def condition_inputs(self):
        check("condition_commitment" in self.manifest, "PROCESS_PRECONDITION_UNAVAILABLE")
        for name in CONDITION_FILES:
            self.file(name)
        conditions = ConditionInputs(self.root, self.manifest["condition_commitment"])
        c = conditions.commitment
        ref = c["program_source"]
        check(c["source"] == self.source.pin and c["prepared"] == self.manifest["prepared"]
              and ref == self.source.reference(ref["task"], ref["turn"]), "PROCESS_CONDITION_SOURCE")
        start, end = c["program_span"]
        program = self.source.texts[ref["task"], ref["turn"]].encode()[start:end]
        check(all(row["program"].encode() == program for row in conditions.inputs.values()),
              "PROCESS_CONDITION_PROGRAM")
        return conditions

    def episode(self, key, evidence=()):
        observed = self.read(key)
        entry = self.entries[key]
        body = dict(claim_kind="event_occurrence", content=observed["captured"],
            source=dict(observation_set=self.pin, observation=key, reexecution_of=entry["target"]),
            evidence=list(evidence), event_time=observed["finished"],
            who=dict(actor="isolated_python_process", python=observed["result"]["python"]),
            what=dict(program_sha256=entry["program_sha256"],
                      returncode=observed["result"]["returncode"]),
            where=dict(image=observed["image"], container=observed["container"]),
            when=dict(clock="container_runtime_utc", started=observed["started"],
                      finished=observed["finished"]),
            why=None, why_status="unresolved", independent_occurrence=True,
            scope=SCOPE, native_measured=False)
        if "conditions" in observed:
            body["execution_conditions"] = observed["conditions"]
        return dict(id=digest(body), **body)

    def validate_episode(self, episode):
        key = episode.get("source", {}).get("observation")
        check(key in self.entries and episode == self.episode(key, episode.get("evidence", ()))
              and len(set(episode["evidence"])) == len(episode["evidence"]), "PROCESS_EPISODE")

    def assess(self, episode):
        self.validate_episode(episode)
        body = dict(checker="isolated-process-observation-v1", episode=episode["id"],
                    claim_kind="event_occurrence", scope=SCOPE, source=episode["source"],
                    world_validity="scoped_observed_event",
                    historical_occurrence_established=False, native_measured=False)
        return dict(id=digest(body), **body)

    def compare_report(self, key):
        observed = self.read(key)
        target = self.entries[key]["target"]
        text = self.source.texts[target["task"], target["turn"]]
        marker = "\n\n(Open file:"
        check(text.count(marker) == 1, "PROCESS_REPORT_FORMAT")
        report = text.split(marker)[0]
        stderr, stdout = observed["captured"]["stderr"], observed["captured"]["stdout"]
        if report.startswith("Traceback (most recent call last):\n"):
            terminal = report.rstrip().splitlines()[-1]
            check(terminal.startswith("ModuleNotFoundError:"), "PROCESS_REPORT_UNSUPPORTED")
            concordant = (observed["result"]["returncode"] == 1 and stdout == ""
                          and stderr.rstrip().splitlines()[-1:] == [terminal])
            comparison = dict(kind="terminal_exception_only", terminal=terminal)
        else:
            concordant = observed["result"]["returncode"] == 0 and stdout == report+"\n" and stderr == ""
            comparison = dict(kind="exact_stdout", stdout=report+"\n")
        return dict(concordant=concordant, comparison=comparison,
                    scope="new_execution_consistent_with_reported_result",
                    historical_occurrence_established=False)
