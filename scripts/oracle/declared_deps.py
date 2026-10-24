"""Static, non-executing read of a snapshot's declared Python dependencies.

Never imports or runs repository code. setup.py is read as an AST and only
literal values (or module-level names bound to literals) count. Anything the
reader cannot evaluate is recorded in `unevaluated`, never guessed.
"""
from __future__ import annotations

import ast
import configparser
import posixpath
from pathlib import PurePosixPath
import re
import tomllib

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

TEST_EXTRAS = {"test", "tests", "testing"}
REQ_FILE = re.compile(r"^(?:requirements|reqs)?[\w.-]*(?:test|dev|ci)[\w.-]*\.txt$|^requirements\.txt$", re.I)
SCHEMA = "declared-python-deps-v1"


class Literal:
    def __init__(self, tree):
        self.names, self.unevaluated = {}, []
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    self.names[node.targets[0].id] = self.value(node.value)
                except ValueError:
                    pass

    def value(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.value(e) for e in node.elts]
        if isinstance(node, ast.Dict):
            return {self.value(k): self.value(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.Name) and node.id in self.names:
            return self.names[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self.value(node.left) + self.value(node.right)
        raise ValueError(ast.dump(node)[:120])


def from_setup_py(text, out):
    tree = ast.parse(text)
    lit = Literal(tree)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", None)) == "setup"]
    if len(calls) != 1:
        out["unevaluated"].append("setup.py: setup() call count %d" % len(calls))
        return
    for kw in calls[0].keywords:
        if kw.arg == "name" and isinstance(kw.value, ast.Constant):
            out["names"].append(str(kw.value.value))
        if kw.arg not in ("install_requires", "tests_require", "extras_require", "package_dir"):
            continue
        try:
            value = lit.value(kw.value)
        except ValueError as exc:
            out["unevaluated"].append(f"setup.py:{kw.arg}: {exc}")
            continue
        if kw.arg == "install_requires":
            add(out, "runtime", value, "setup.py:install_requires")
        elif kw.arg == "tests_require":
            add(out, "test", value, "setup.py:tests_require")
        elif kw.arg == "extras_require":
            out["extras"].update({k: v for k, v in value.items() if isinstance(v, list)})
        elif kw.arg == "package_dir" and value.get("") not in (None, "", "."):
            out["source_roots"].append({"path": value[""], "declared_by": "setup.py:package_dir"})
    mutations = sorted({n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                        and getattr(n.func, "attr", None) in ("append", "extend")
                        and getattr(n.func.value, "id", None) in lit.names})
    if mutations:
        out["unevaluated"].append("setup.py: list mutations at lines %s not applied" % mutations)


def literal_slices(text, out):
    """Python-2 setup.py (not parseable as Python 3): read only literal `kw=[...]`/`{...}` slices, statically."""
    for kw, kind in (("install_requires", "runtime"), ("tests_require", "test"), ("extras_require", "extras")):
        m = re.search(r"\b" + kw + r"\s*=\s*([\[{])", text)
        if not m:
            continue
        open_, close = m[1], "]" if m[1] == "[" else "}"
        depth, i = 0, m.start(1)
        for j in range(i, len(text)):
            depth += text[j] == open_
            depth -= text[j] == close
            if depth == 0:
                break
        try:
            value = ast.literal_eval(text[i:j + 1])
        except (ValueError, SyntaxError):
            out["unevaluated"].append(f"setup.py(py2):{kw}: not a literal")
            continue
        if kind == "extras":
            out["extras"].update({k: v for k, v in value.items() if isinstance(v, list)})
        else:
            add(out, kind, value, f"setup.py(py2 literal):{kw}")
    out["unevaluated"].append("setup.py: Python-2 syntax; only literal slices read")


def from_setup_cfg(text, out):
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read_string(text)
    if cfg.has_option("metadata", "name"):
        out["names"].append(cfg.get("metadata", "name").strip())
    if cfg.has_option("options", "install_requires"):
        add(out, "runtime", cfg.get("options", "install_requires").splitlines(), "setup.cfg:install_requires")
    if cfg.has_option("options", "tests_require"):
        add(out, "test", cfg.get("options", "tests_require").splitlines(), "setup.cfg:tests_require")
    if cfg.has_option("options", "package_dir"):
        for line in cfg.get("options", "package_dir").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "" and value.strip() not in ("", "."):
                out["source_roots"].append({"path": value.strip(), "declared_by": "setup.cfg:package_dir"})
    if cfg.has_section("options.extras_require"):
        out["extras"].update({k: cfg.get("options.extras_require", k).splitlines()
                              for k in cfg.options("options.extras_require")})


def from_pyproject(text, out):
    data = tomllib.loads(text)
    project = data.get("project", {})
    if isinstance(project.get("name"), str):
        out["names"].append(project["name"])
    add(out, "runtime", project.get("dependencies", []), "pyproject:project.dependencies")
    out["extras"].update(project.get("optional-dependencies", {}))
    for group, values in data.get("dependency-groups", {}).items():
        if group in TEST_EXTRAS | {"dev"}:
            add(out, "test", [v for v in values if isinstance(v, str)], "pyproject:dependency-groups." + group)
    pkgdir = data.get("tool", {}).get("setuptools", {}).get("package-dir", {})
    if pkgdir.get("") not in (None, "", "."):
        out["source_roots"].append({"path": pkgdir[""], "declared_by": "pyproject:tool.setuptools.package-dir"})
    where = data.get("tool", {}).get("setuptools", {}).get("packages", {})
    if isinstance(where, dict) and where.get("find", {}).get("where") not in (None, ["."]):
        for w in where["find"]["where"]:
            out["source_roots"].append({"path": w, "declared_by": "pyproject:tool.setuptools.packages.find.where"})
    for pkg in data.get("tool", {}).get("hatch", {}).get("build", {}).get("targets", {}).get("wheel", {}).get("packages", []):
        parent = str(PurePosixPath(pkg).parent)
        if parent != ".":
            out["source_roots"].append({"path": parent, "declared_by": "pyproject:tool.hatch...wheel.packages"})
    poetry = data.get("tool", {}).get("poetry", {})
    if poetry.get("dependencies"):
        out["unevaluated"].append("pyproject: poetry dependency syntax not translated")
    for p in poetry.get("packages", []):
        if isinstance(p, dict) and p.get("from"):
            out["source_roots"].append({"path": p["from"], "declared_by": "pyproject:tool.poetry.packages.from"})


def from_requirements(files, name, out, kind, seen):
    if name in seen or name not in files:
        if name not in files:
            out["unevaluated"].append("missing requirements include " + name)
        return
    seen.add(name)
    lines = []
    for raw in files[name].decode("utf-8", "replace").splitlines():
        line = raw.split(" #")[0].strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r ", "--requirement ")):
            from_requirements(files, posixpath.normpath(posixpath.join(posixpath.dirname(name), line.split(None, 1)[1])),
                              out, kind, seen)
        elif line.startswith("-c "):
            continue
        elif line.startswith("-") or "://" in line or line.startswith((".", "/")):
            out["unevaluated"].append(f"{name}: unsupported line {line[:80]!r}")
        else:
            lines.append(line)
    add(out, kind, lines, name)


def from_tox(files, out):
    cfg = configparser.ConfigParser(interpolation=None, strict=False)
    cfg.read_string(files["tox.ini"].decode("utf-8", "replace"))
    if not cfg.has_section("testenv"):
        return

    def expand(text, depth=0):
        # tox {[section]key} references; anything else in braces stays and is refused below.
        def sub(m):
            if depth < 4 and cfg.has_option(m[1], m[2]):
                return expand(cfg.get(m[1], m[2]), depth + 1)
            return m[0]
        return re.sub(r"\{\[([^\]]+)\]([\w-]+)\}", sub, text).replace("{toxinidir}/", "")

    if cfg.has_option("testenv", "deps"):
        plain = []
        for line in expand(cfg.get("testenv", "deps")).splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if re.match(r"^[\w{},.!-]+:\s", line):
                out["unevaluated"].append("tox.ini: factor-conditional dep skipped " + repr(line[:80]))
            elif line.startswith("-r"):
                from_requirements(files, posixpath.normpath(line[2:].strip()), out, "test", out["_seen"])
            else:
                plain.append(line)
        add(out, "test", plain, "tox.ini:testenv.deps")
    if cfg.has_option("testenv", "extras"):
        out["tox_extras"] = [e.strip() for e in re.split(r"[,\s]+", expand(cfg.get("testenv", "extras"))) if e.strip()]


def add(out, kind, values, source):
    if isinstance(values, str):
        values = [values]
    for v in values:
        v = re.split(r"\s+#", v)[0].strip() if isinstance(v, str) else v  # inline comments (setup.cfg)
        if not v:
            continue
        try:
            req = Requirement(v)
        except (InvalidRequirement, TypeError):
            out["unevaluated"].append(f"{source}: invalid requirement {str(v)[:80]!r}")
            continue
        out[kind].append({"requirement": str(req), "source": source})


def read(files):
    """files: snapshot path -> bytes. Returns the declared-dependency record."""
    out = {"schema": SCHEMA, "runtime": [], "test": [], "extras": {}, "source_roots": [], "unevaluated": [],
           "names": [], "_seen": set()}
    for name, fn in (("setup.py", from_setup_py), ("setup.cfg", from_setup_cfg), ("pyproject.toml", from_pyproject)):
        if name in files:
            try:
                fn(files[name].decode("utf-8"), out)
            except SyntaxError:
                if name == "setup.py":
                    literal_slices(files[name].decode("utf-8"), out)
                else:
                    out["unevaluated"].append(f"{name}: SyntaxError")
            except (ValueError, configparser.Error, tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
                out["unevaluated"].append(f"{name}: {type(exc).__name__}")
    if "tox.ini" in files:
        try:
            from_tox(files, out)
        except configparser.Error as exc:
            out["unevaluated"].append(f"tox.ini: {type(exc).__name__}")
    for name in sorted(files):
        if REQ_FILE.match(PurePosixPath(name).name) and PurePosixPath(name).parent.name in ("", "requirements", "tests", "test"):
            kind = "runtime" if PurePosixPath(name).name.lower() == "requirements.txt" and "/" not in name else "test"
            from_requirements(files, name, out, kind, out["_seen"])
    for extra in sorted(set(out["extras"]) & (TEST_EXTRAS | set(out.get("tox_extras", [])))):
        add(out, "test", out["extras"][extra], "extras:" + extra)
    # setuptools automatic discovery: src-layout when src/<pkg>/__init__.py exists and no root package does.
    if not out["source_roots"]:
        src = {PurePosixPath(n).parts[1] for n in files if n.startswith("src/") and n.count("/") == 2
               and n.endswith("/__init__.py")}
        root_pkgs = {PurePosixPath(n).parts[0] for n in files if n.count("/") == 1 and n.endswith("/__init__.py")
                     and not n.startswith(("tests/", "test/", "src/", "docs/"))}
        if src and not root_pkgs:
            out["source_roots"].append({"path": "src", "declared_by": "setuptools-automatic-discovery:src-layout"})
    for r in out["source_roots"]:
        r["path"] = r["path"].strip("/")
    first = {}
    for r in out["source_roots"]:
        first.setdefault(r["path"], r)  # first declaration wins; later duplicates dropped
    out["source_roots"] = list(first.values())
    own = {canonicalize_name(n) for n in out["names"]}
    for kind in ("runtime", "test"):
        out[kind] = [r for r in out[kind] if canonicalize_name(Requirement(r["requirement"]).name) not in own]
    del out["_seen"]
    out["extras"] = {k: out["extras"][k] for k in sorted(out["extras"])}
    return out
