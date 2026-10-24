"""Capture one pinned Python program inside the existing confined container."""
import base64
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

PREFIX = "MNLM_EXECUTION_RESULT="


def main():
    spec = json.loads(Path("/work/observation.json").read_bytes())
    program = Path("/work/program.py")
    raw = program.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["program_sha256"]:
        raise ValueError("MNLM_PROCESS_PROGRAM_PIN")
    wheels = sorted(Path("/runtime").glob("*.whl"))
    inventory = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in wheels}
    if inventory != spec["wheels"]:
        raise ValueError("MNLM_PROCESS_RUNTIME_PIN")
    if sys.argv[1] == "child":
        sys.path[:0] = [str(p) for p in wheels]
        runpy.run_path(str(program), run_name="__main__")
        return
    if sys.argv[1:] != ["observe-script"]:
        raise ValueError("MNLM_PROCESS_MODE")
    child = subprocess.run([sys.executable, "-I", "-B", __file__, "child"],
                           capture_output=True, timeout=25, cwd="/work")
    if len(child.stdout)+len(child.stderr) > 2*1024*1024:
        raise ValueError("MNLM_PROCESS_CAPTURE_BOUND")
    result = dict(schema="observed-python-process-v1", program_sha256=spec["program_sha256"],
                  wheels=inventory, python=sys.version, returncode=child.returncode,
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for name in ("stdout", "stderr"):
        value = getattr(child, name)
        result[name] = dict(base64=base64.b64encode(value).decode(), bytes=len(value),
                           sha256=hashlib.sha256(value).hexdigest())
    print(PREFIX+json.dumps(result))


if __name__ == "__main__":
    main()
