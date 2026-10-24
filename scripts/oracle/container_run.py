"""One bounded, inspected, network-isolated CPU container for an oracle condition."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import time
import uuid

from senses.contract import SenseError, require
from senses.trajectory_population import publish

CODE = "MNLM_EXECUTION_SANDBOX"
MAX_WALL = 30
MAX_OUTPUT = 4 * 1024 * 1024
MEMORY = 512 * 1024 * 1024
TMPFS = "rw,noexec,nosuid,size=33554432"
WORK_TMPFS = "rw,noexec,nosuid,size=134217728"  # site-v1 writable checkout copy; counts toward MEMORY
PREFIX = "MNLM_EXECUTION_RESULT="


def docker(*args):
    try:
        p = subprocess.run(["docker", *args], capture_output=True, timeout=15)
    except subprocess.TimeoutExpired as exc:
        raise SenseError("MNLM_EXECUTION_DOCKER_TIMEOUT", "Docker operation timed out; inspect this run's container ID") from exc
    require(p.returncode == 0, CODE, p.stderr.decode("utf-8", "replace")[:1000])
    return p.stdout


def inspect(cid):
    raw = json.loads(docker("inspect", cid))[0]
    # Only our created object's fields; never host credentials or unrelated images.
    return {"Id": raw["Id"], "Image": raw["Image"], "State": raw["State"],
            "HostConfig": {k: raw["HostConfig"][k] for k in (
                "NetworkMode", "ReadonlyRootfs", "CapDrop", "Privileged", "SecurityOpt",
                "PidsLimit", "Memory", "MemorySwap", "NanoCpus", "Tmpfs", "LogConfig")},
            "Config": {k: raw["Config"][k] for k in ("User", "Labels", "Entrypoint", "Cmd")},
            "Mounts": [{k: m[k] for k in ("Source", "Destination", "RW", "Type")} for m in raw["Mounts"]]}


def tmpfs(work):
    return {"/tmp": TMPFS, **({"/work-rw": WORK_TMPFS} if work else {})}


def validate_config(info, image, label, mounts, work_tmpfs=False):
    h, c = info["HostConfig"], info["Config"]
    require(info["Image"] == image and c["User"] == "65534:65534"
            and c["Labels"].get("mnlm.oracle.run") == label, CODE, "runtime ownership or identity mismatch")
    require(h["NetworkMode"] == "none" and h["ReadonlyRootfs"] is True and h["Privileged"] is False
            and h["CapDrop"] == ["ALL"] and "no-new-privileges" in h["SecurityOpt"]
            and h["PidsLimit"] == 64 and h["Memory"] == h["MemorySwap"] == MEMORY
            and h["NanoCpus"] == 1_000_000_000 and h["Tmpfs"] == tmpfs(work_tmpfs)
            and h["LogConfig"]["Type"] == "none", CODE, "container confinement differs")
    actual = {m["Destination"]: m["Source"] for m in info["Mounts"] if m["Type"] == "bind" and not m["RW"]}
    require(actual == mounts and len(info["Mounts"]) == len(mounts), CODE, "unexpected or writable host mount")


def execute(image, condition, runtime, output, mode="tests", modules=(), wall=MAX_WALL):
    require(os.name == "posix" and type(wall) is int and 0 < wall <= MAX_WALL, CODE,
            "Linux host and bounded wall required")
    require(image.startswith("sha256:") and len(image) == 71
            and all(c in "0123456789abcdef" for c in image[7:]), CODE, "immutable image ID required")
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    image_info = json.loads(docker("image", "inspect", image))[0]
    require(image_info["Id"] == image and image_info["Os"] == "linux", CODE, "image missing or changed")
    label = str(uuid.uuid4())
    work_tmpfs = (runtime / "mnlm-runtime.json").exists()  # site-v1 layout
    harness = "process_inside.py" if mode == "observe-script" else "pytest_inside.py"
    mounts = {"/work": str(condition.resolve()), "/runtime": str(runtime.resolve()),
              "/harness.py": str(Path(__file__).with_name(harness).resolve())}
    args = ["create", "--pull=never", "--name", "mnlm-oracle-" + label, "--label", "mnlm.oracle.run=" + label,
            "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--user=65534:65534", "--pids-limit=64", "--memory=" + str(MEMORY), "--memory-swap=" + str(MEMORY),
            "--cpus=1", "--log-driver=none", "--tmpfs=/tmp:" + TMPFS,
            *(["--tmpfs=/work-rw:" + WORK_TMPFS] if work_tmpfs else []), "--workdir=/work", "--entrypoint=python"]
    for destination, source in mounts.items():
        require(Path(source).exists() and "," not in source, CODE, "invalid bind source")
        args += ["--mount", f"type=bind,source={source},destination={destination},readonly"]
    args += [image, "-I", "-B", "/harness.py", mode, *modules]
    cid = docker(*args).decode().strip()
    try:
        before = inspect(cid)
        validate_config(before, image, label, mounts, work_tmpfs)
        publish(output / "before.json", before)
        configured = time.perf_counter()
        p = subprocess.Popen(["docker", "start", "--attach", cid], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        selected = selectors.DefaultSelector()
        selected.register(p.stdout, selectors.EVENT_READ, "stdout")
        selected.register(p.stderr, selectors.EVENT_READ, "stderr")
        data = {"stdout": bytearray(), "stderr": bytearray()}
        failure, count, killer, stop_started = None, 0, None, None
        try:
            while selected.get_map():
                if not failure and time.perf_counter() - configured > wall:
                    failure = "MNLM_EXECUTION_TIMEOUT"
                if failure and killer is None:
                    # Keep draining/discarding bounded-output pipes while Docker
                    # delivers the stop. Blocking kill with a full attached pipe
                    # can stall the daemon's stream/termination handshake.
                    killer = subprocess.Popen(["docker", "kill", cid], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stop_started = time.perf_counter()
                if killer is not None and time.perf_counter() - stop_started > 15:
                    killer.kill()
                    killer.wait(timeout=5)
                    raise SenseError("MNLM_EXECUTION_DOCKER_TIMEOUT", "stop did not finish while draining; inspect run ID")
                for key, _ in selected.select(timeout=0.1):
                    chunk = os.read(key.fileobj.fileno(), 16384)
                    if not chunk:
                        selected.unregister(key.fileobj)
                    else:
                        room = MAX_OUTPUT - count
                        data[key.data] += chunk[:room]
                        count += min(len(chunk), room)
                        if len(chunk) > room:
                            failure = failure or "MNLM_EXECUTION_OUTPUT_BUDGET"
            if killer is not None:
                killer.communicate(timeout=5)
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)
                failure = failure or "MNLM_EXECUTION_ATTACH_FAILURE"
        finally:
            selected.close()
            p.stdout.close()
            p.stderr.close()
            for name, value in data.items():
                (output / (name + ".bin")).write_bytes(value)
            if killer is not None and killer.poll() is None:
                killer.kill()
                killer.wait(timeout=5)
        after = inspect(cid)
        if after["State"]["Running"]:
            docker("kill", cid)
            after = inspect(cid)
            failure = failure or "MNLM_EXECUTION_ATTACH_FAILURE"
        validate_config(after, image, label, mounts, work_tmpfs)
        publish(output / "after.json", after)
        execution_done = time.perf_counter()
        markers = [line[len(PREFIX):] for line in data["stdout"].decode("utf-8", "replace").splitlines()
                   if line.startswith(PREFIX)]
        result = None
        if not failure:
            try:
                require(len(markers) == 1, CODE, "missing or repeated result envelope")
                result = json.loads(markers[0])
            except (SenseError, ValueError):
                failure = "MNLM_EXECUTION_RESULT_ENVELOPE"
        report = {"schema": "isolated-execution-v1", "image": image, "container_id": cid,
                  "mode": mode, "status": failure or "observed", "result": result,
                  "exitcode": after["State"]["ExitCode"], "oom_killed": after["State"]["OOMKilled"],
                  "output_bytes": count, "timings": {"configuration": configured - started,
                      "execution_capture": execution_done - configured}, "admitted": False}
        publish(output / "execution.json", report)
        return report
    finally:
        # Clean up only the exact container created here, after checking its label.
        state = inspect(cid)
        require(state["Config"]["Labels"].get("mnlm.oracle.run") == label, CODE, "container ownership changed")
        if state["State"]["Running"]:
            docker("kill", cid)
        docker("rm", cid)
