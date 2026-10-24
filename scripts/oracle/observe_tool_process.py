"""Run a reviewed source program in two pinned, isolated runtime conditions."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims.process_conditions import ConditionInputs, FILES, condition_input, publish as publish_conditions
from claims import forecast_commitment
from oracle.container_run import execute
from oracle.prepare_swe import inventory


def write(path, value):
    with path.open("xb") as f:
        f.write(canonical(value))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--pin", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--forecast-root", type=Path)
    p.add_argument("--forecast-pin")
    p.add_argument("--condition-source", type=Path)
    p.add_argument("--condition-pin")
    p.add_argument("--only-condition", choices=("without_dependency", "with_dependency"))
    args = p.parse_args()
    started = time.perf_counter()
    raw = (args.prepared/"prepared.json").read_bytes()
    check(sha(raw) == args.pin, "PROCESS_PREPARED_PIN")
    spec = json.loads(raw)
    check(spec["schema"] == "tool-process-prepared-v1" and len(spec["conditions"]) == 2,
          "PROCESS_PREPARED_SCHEMA")
    for name, item in spec["conditions"].items():
        check(name in ("without_dependency", "with_dependency")
              and inventory(args.prepared/name) == item["files"], "PROCESS_PREPARED_BYTES")
    prospective = (args.forecast_root, args.forecast_pin, args.condition_source,
                   args.condition_pin, args.only_condition)
    check(all(v is None for v in prospective) or all(v is not None for v in prospective),
          "PROCESS_FORECAST_ARGUMENTS")
    seal, selected = None, list(spec["conditions"])
    if args.forecast_root is not None:
        seal = forecast_commitment.read(args.forecast_root, args.forecast_pin)
        conditions = ConditionInputs(args.condition_source, args.condition_pin)
        selected = [args.only_condition]
        check(conditions.commitment["prepared"] == args.pin
              and len(seal["forecasts"]) == 1, "PROCESS_FORECAST_SOURCE")
        record = next(iter(seal["forecasts"].values()))
        native = record["request"]["execution_conditions"]["native"]
        declared = seal["conditions"]
        providers = declared["providers"] if declared.get("schema") == "condition-native-catalog-v1" else [declared]
        check(native["commitment"] == conditions.pin
              and sum(all(native.get(k) == v for k,v in provider.items()) for provider in providers) == 1,
              "PROCESS_FORECAST_SOURCE")
        target = spec["conditions"][args.only_condition]["target"]
        check(record["request"]["execution_conditions"]["native"]["condition"] ==
                  conditions.commitment["routing"][args.only_condition]
              and record["request"]["task"] == target["task"]
              and record["request"]["turn"]+1 == target["turn"], "PROCESS_FORECAST_CONDITION")
    check(not args.output.exists(), "PROCESS_OUTPUT_EXISTS")
    shutil.copytree(args.prepared, args.output)
    harness = Path(__file__).with_name("process_inside.py")
    shutil.copyfile(harness, args.output/"process_inside.py")
    if seal is None:
        condition_pin = publish_conditions(args.output, spec, args.pin, sha(harness.read_bytes()))
    else:
        for name in FILES:
            shutil.copyfile(args.condition_source/name, args.output/name)
        for name in spec["conditions"]:
            value = condition_input(spec, name, (args.output/name/"work/program.py").read_bytes(),
                                    sha(harness.read_bytes()))
            check(value == conditions.read(conditions.commitment["routing"][name]), "PROCESS_FORECAST_INPUT")
        condition_pin = args.condition_pin
        shutil.copytree(args.forecast_root, args.output/"forecast")
    conditions = ConditionInputs(args.output, condition_pin)
    validated = time.perf_counter()
    probe = execute(spec["image"], args.output/"without_dependency/work",
                    args.output/"without_dependency/runtime", args.output/"probe", "probe")
    check(probe["status"] == "observed" and probe["exitcode"] == 0
          and probe["result"]["probe"]["uid"] == 65534
          and all(v is True for k, v in probe["result"]["probe"].items() if k != "uid"),
          "PROCESS_ISOLATION")
    probed = time.perf_counter()
    observations = {}
    for name in selected:
        item = spec["conditions"][name]
        execute(spec["image"], args.output/name/"work", args.output/name/"runtime",
                args.output/name/"execution", "observe-script")
        stored = json.loads((args.output/name/"execution/execution.json").read_bytes())
        check(stored["status"] == "observed" and stored["exitcode"] == 0, "PROCESS_CAPTURE_FAILED")
        check(inventory(args.output/name/"work") ==
              {k.removeprefix("work/"): v for k, v in item["files"].items() if k.startswith("work/")}
              and inventory(args.output/name/"runtime") ==
              {k.removeprefix("runtime/"): v for k, v in item["files"].items() if k.startswith("runtime/")},
              "PROCESS_RUNTIME_CHANGED")
        entry = dict(target=item["target"], program_source=spec["program_source"],
            program_span=spec["program_span"], program_sha256=spec["program_sha256"],
            program=f"{name}/work/program.py", runtime=f"{name}/runtime",
            wheels=item["wheels"], execution=f"{name}/execution",
            condition=conditions.commitment["routing"][name])
        observations[digest(entry)] = entry
        if seal is not None:
            after = json.loads((args.output/name/"execution/after.json").read_bytes())
            observed = dict(result=stored["result"], image=stored["image"], started=after["State"]["StartedAt"])
            observed["conditions"] = conditions.match(entry["condition"], entry, observed)
            forecast_commitment.match(seal, record, entry, observed)
    done = time.perf_counter()
    files = {f.relative_to(args.output).as_posix(): dict(bytes=f.stat().st_size,
             sha256=sha(f.read_bytes())) for f in sorted(args.output.rglob("*")) if f.is_file()}
    hashed = time.perf_counter()
    result = dict(schema="tool-process-observations-v1", source=spec["source"],
        prepared=args.pin, files=files, observations=observations, condition_commitment=condition_pin,
        phases_seconds=dict(preparation=validated-started, probe=probed-validated,
                            execution=done-probed, inventory=hashed-done),
        wall_seconds=hashed-started, timing_scope="main_through_artifact_inventory",
        historical_occurrence_established=False,
        code={n: sha(Path(__file__).with_name(n).read_bytes())
              for n in ("observe_tool_process.py", "container_run.py", "process_inside.py")})
    result["code"]["process_conditions.py"] = sha(
        Path(__file__).parents[1].joinpath("claims/process_conditions.py").read_bytes())
    if seal is not None:
        result["forecast"] = dict(root="forecast", pin=args.forecast_pin,
                                  scope="new_occurrence_after_exposed_program_forecast")
        result["code"]["forecast_commitment.py"] = sha(
            Path(__file__).parents[1].joinpath("claims/forecast_commitment.py").read_bytes())
    write(args.output/"manifest.json", result)
    pin = sha((args.output/"manifest.json").read_bytes())
    with (args.output/"COMMITTED").open("x") as f:
        f.write(pin)
    print(json.dumps(dict(pin=pin, observations=len(observations), wall_seconds=result["wall_seconds"])))


if __name__ == "__main__":
    main()
