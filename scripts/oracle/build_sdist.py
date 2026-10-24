"""Build a pure-Python sdist-only dependency inside the bounded container (decision #12 R2 addendum).

Never runs third-party code on the host. Version rule: the highest non-pre-release
version uploaded no later than the decision date that has an sdist and satisfies
every declared specifier naming the package. The backend is pinned setuptools +
wheel; compiled sources fail (the image has no compiler) and stay runtime gaps.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import zipfile

from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from oracle.container_run import execute
from oracle.resolve_runtime import HARNESS_DATE, META_CAP
from oracle.runtime_site import rank, unpack
from senses.contract import SenseError, require
from senses.trajectory_population import publish

CODE = "MNLM_SDIST_BUILD"
BACKEND = (("setuptools", "75.8.0"), ("wheel", "0.45.1"))
BACKEND_38 = (("setuptools", "75.3.0"), ("wheel", "0.45.1"))  # setuptools >=75.4 needs Python >=3.9
BACKEND_37 = (("setuptools", "68.0.0"), ("wheel", "0.42.0"))  # last releases supporting Python 3.7


def backend(interpreter):
    version = interpreter["python"] if interpreter else "3.12"
    return BACKEND_37 if version.startswith("3.7.") else BACKEND_38 if version.startswith("3.8.") else BACKEND
SDIST_CAP = 16 << 20


def choose(project, specifier):
    best = None
    for version, files in project["releases"].items():
        try:
            v = Version(version)
        except InvalidVersion:
            continue
        if v.is_prerelease or v not in specifier:
            continue
        sdists = [f for f in files if f["packagetype"] == "sdist" and not f.get("yanked")
                  and f["upload_time_iso_8601"] <= HARNESS_DATE and f["filename"].endswith((".tar.gz", ".zip"))]
        if sdists and (best is None or v > best[0]):
            best = (v, sorted(sdists, key=lambda f: f["filename"])[0])
    require(best is not None, CODE, "no sdist satisfies the declared specifiers before the decision date")
    return str(best[0]), best[1]


def extract(data, filename, dest):
    dest.mkdir(parents=True)
    members = []
    if filename.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            members = [(i.filename, None if i.is_dir() else z.read(i)) for i in z.infolist()]
    else:
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            for m in t:
                require(m.isdir() or m.isfile(), CODE, "sdist links/devices refused")
                members.append((m.name, t.extractfile(m).read() if m.isfile() else None))
    tops = {PurePosixPath(n).parts[0] for n, _ in members}
    require(len(tops) == 1, CODE, "sdist must have one top-level directory")
    for name, payload in members:
        p = PurePosixPath(name)
        require(not p.is_absolute() and ".." not in p.parts and "\\" not in name, CODE, "sdist entry escapes")
        if payload is None or len(p.parts) == 1:
            continue
        target = dest / PurePosixPath(*p.parts[1:])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)


def build(fetcher, image, name, specifier, work, interpreter=None, built_dir="built", build_image=None):
    """build_image: a compiler-equipped image of the same CPython minor and glibc (F4); compiled
    outputs are accepted only from it, and only with a tag compatible with the runtime interpreter."""
    """Returns (wheel binding, receipt binding). Raises SenseError with a named reason."""
    name = canonicalize_name(name)
    meta_path = fetcher.raw / "pypi" / f"{name}.project.json"
    if not meta_path.exists():
        data = fetcher.get(f"https://pypi.org/pypi/{name}/json", "metadata", META_CAP)
        with meta_path.open("xb") as f:
            f.write(data)
    version, sdist = choose(json.loads(meta_path.read_bytes()), SpecifierSet(specifier))
    sdist_path = fetcher.raw / "sdists" / sdist["filename"]
    sdist_path.parent.mkdir(exist_ok=True)
    if not sdist_path.exists():
        require(sdist["size"] <= SDIST_CAP, CODE, "sdist above cap")
        data = fetcher.get(sdist["url"], "wheels", SDIST_CAP)
        require(hashlib.sha256(data).hexdigest() == sdist["digests"]["sha256"], CODE, "sdist digest differs from PyPI")
        with sdist_path.open("xb") as f:
            f.write(data)
    data = sdist_path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == sdist["digests"]["sha256"], CODE, "stored sdist differs")
    case = work / f"{name}-{version}"
    n = 0
    while (case / str(n)).exists():  # each attempt keeps its own scratch; a failed attempt is never reused
        n += 1
    case = case / str(n)
    extract(data, sdist["filename"], case / "src")
    site, owners, backend_wheels = case / "runtime" / "site", {}, []
    site.mkdir(parents=True)
    for bname, bversion in backend(interpreter):
        w, m = fetcher.wheel({"name": bname, "version": bversion, "hashes": None}, None, interpreter)
        unpack(fetcher.root.joinpath(w["path"]).read_bytes(), Path(w["path"]).name, site, owners, [], interpreter)
        backend_wheels.append(w)
    record = execute(build_image or image, case / "src", case / "runtime", case / "build", "build-sdist", wall=30)
    require(record["status"] == "observed" and record["result"] and record["result"]["exitcode"] == 0
            and len(record["result"]["wheels"]) == 1, CODE,
            "sandboxed build failed: " + (record["result"] or {}).get("log_tail", record["status"])[-300:])
    built = record["result"]["wheels"][0]
    wname, wversion, _, _ = parse_wheel_filename(built["name"])
    require(wname == name and str(wversion) == version and rank(built["name"], interpreter) is not None, CODE,
            "built wheel name/version/tag differs from the chosen sdist")
    payload = base64.b64decode(built["b64"])
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        compiled = any(n.endswith((".so", ".pyd")) for n in z.namelist())
        require(not compiled or build_image is not None, CODE, "compiled output from a builder without a declared compiler image")
    out = fetcher.raw / built_dir / built["name"]
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        require(out.read_bytes() == payload, CODE, "rebuild differs from the stored built wheel")
    else:
        with out.open("xb") as f:
            f.write(payload)
    receipt = case / "receipt.json"
    publish(receipt, {"schema": "sandboxed-sdist-build-v1", "package": name, "version": version,
                      "sdist": fetcher.binding(sdist_path), "pypi_sdist_sha256": sdist["digests"]["sha256"],
                      "backend": backend_wheels, "wheel": fetcher.binding(out), "image": build_image or image,
                      "execution": str((case / "build" / "execution.json").relative_to(fetcher.root)),
                      "pypi_artifact": False, "admitted": False})
    return fetcher.binding(out), fetcher.binding(receipt)
