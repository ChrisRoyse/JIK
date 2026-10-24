"""Runtime layout site-v1: hash-pinned wheels unpacked exactly into one import directory.

The container runs CPython 3.12.13 on glibc 2.41 (measured from the pinned
image). A wheel is accepted only when one of its tags is in that interpreter's
compatible set. Unpacking runs no code: wheel installation is a zip copy.
The harness adds the directory with site.addsitedir inside the sandbox, so .pth
files run there exactly as in a venv (e.g. setuptools' distutils shim). Nothing outside <name>.data/{purelib,platlib} of a
wheel's .data tree is installed; such entries are recorded, not hidden.
"""
from __future__ import annotations

import configparser
import io
from pathlib import Path, PurePosixPath
import zipfile

from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import parse_wheel_filename

from senses.contract import SenseError, require

CODE = "MNLM_EXECUTION_RUNTIME"
LAYOUT = "site-v1"
DEFAULT = {"python": "3.12.13", "glibc": "2.41"}  # the original pinned image; plans without "interpreter"
_PRIORITY = {}


def priority(interpreter=None):
    """Tag priority for a measured interpreter {python: "3.x.y", glibc: "2.n"}."""
    it = interpreter or DEFAULT
    key = (it["python"], it["glibc"])
    if key not in _PRIORITY:
        py = tuple(int(x) for x in it["python"].split(".")[:2]); libc = int(it["glibc"].split(".")[1])
        abi = f"cp{py[0]}{py[1]}" + ("m" if py < (3, 8) else "")  # CPython <3.8 wheels carry the pymalloc "m" ABI
        platforms = ([f"manylinux_2_{m}_x86_64" for m in range(libc, 16, -1)]
                     + ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64"])
        tags = [str(t) for t in cpython_tags(py, [abi, "abi3", "none"], platforms)] + \
               [str(t) for t in compatible_tags(py, abi, platforms)]
        _PRIORITY[key] = {t: i for i, t in enumerate(dict.fromkeys(tags))}
    return _PRIORITY[key]


def rank(filename, interpreter=None):
    """Best (lowest) tag priority of a wheel for this interpreter, or None if incompatible."""
    try:
        _, _, _, tags = parse_wheel_filename(filename)
    except ValueError:
        return None
    table = priority(interpreter)
    ranks = [table[str(t)] for t in tags if str(t) in table]
    return min(ranks) if ranks else None


def importable(rel):
    """Paths that can change what Python imports or runs; collisions there always refuse."""
    return "/" in rel or rel.endswith((".py", ".pth", ".so", ".pyd")) or rel.endswith(".dist-info")


def unpack(data, filename, site, owners, skipped, interpreter=None):
    it = interpreter or DEFAULT
    require(rank(filename, it) is not None, CODE,
            f"wheel {filename} is not compatible with CPython {it['python']}/glibc {it['glibc']}")
    with zipfile.ZipFile(io.BytesIO(data)) as wheel:
        for info in wheel.infolist():
            if info.is_dir():
                continue
            name = info.filename
            p = PurePosixPath(name)
            require("\\" not in name and not p.is_absolute() and ".." not in p.parts and ":" not in name,
                    CODE, f"wheel entry escapes the runtime: {filename}:{name}")
            if p.parts[0].endswith(".data"):
                if len(p.parts) > 2 and p.parts[1] in ("purelib", "platlib"):
                    p = PurePosixPath(*p.parts[2:])
                else:
                    skipped.append(f"{filename}:{name}")
                    continue
            rel = p.as_posix()
            payload = wheel.read(info)
            target = site / rel
            if rel in owners:
                if target.read_bytes() != payload:
                    # pip overwrites silently; only stray top-level documents may collide (first kept).
                    require(not importable(rel), CODE,
                            f"wheels {owners[rel]} and {filename} install different bytes at {rel}")
                    skipped.append(f"collision-first-kept:{rel}:{owners[rel]}|{filename}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            owners[rel] = filename


def plugins(site):
    """pytest11 entry points declared by the unpacked distributions, sorted."""
    found = set()
    for ep in sorted(site.glob("*.dist-info/entry_points.txt")):
        cfg = configparser.ConfigParser(interpolation=None, delimiters=("=",))
        cfg.optionxform = str
        cfg.read_string(ep.read_text(encoding="utf-8"))
        if cfg.has_section("pytest11"):
            for _, value in cfg.items("pytest11"):
                module = value.split(":", 1)[0].strip()
                require(module.replace(".", "").replace("_", "").isalnum(), CODE, f"plugin entry {value!r}")
                found.add(module)
    return sorted(found)
