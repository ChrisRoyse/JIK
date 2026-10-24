"""Exact pytest identities; parameters are opaque, never command options."""
from pathlib import PurePosixPath

from senses.contract import require

CODE = "MNLM_EXECUTION_TEST_CONTRACT"


def selectors(required):
    require(isinstance(required, dict) and required
            and set(required.values()) <= {"FAIL_TO_PASS", "PASS_TO_PASS"}
            and "FAIL_TO_PASS" in required.values(), CODE, "required test roles missing")
    for node in required:
        require(isinstance(node, str) and all(ord(c) >= 32 for c in node), CODE,
                "test ID must be a printable string")
        address, bracket, params = node.partition("[")
        path, *names = address.split("::")
        p = PurePosixPath(path)
        require(path and not path.startswith("-") and "\\" not in path and ":" not in path
                and not p.is_absolute() and ".." not in p.parts and str(p) == path
                and p.suffix == ".py" and names and all(n.isidentifier() for n in names)
                and (not bracket or params.endswith("]")), CODE,
                "exact relative Python test selector required; do not infer aliases")
    return sorted(required)


def junit_identity(node):
    # Mirrors the pinned pytest JUnit address contract, preserving :: in params.
    address, bracket, params = node.partition("[")
    path, *names = address.split("::")
    names[-1] += bracket + params
    return (".".join([path.removesuffix(".py").replace("/", "."), *names[:-1]]), names[-1])
