"""Stage 0 — census of the P0 sources (spec §16). Reads manifests and listings only; never a payload.

    python -I scripts/census/census.py --config config/sources.p0.json --root D:/MNLMData [--actor <id>] [--only <source-id>]

For each source: resolve the provider's own manifest (HF tree listing with LFS sha256 and sizes; Common Crawl path list
plus HEAD per selected file; GDELT master file list with md5 and size; HTTP HEAD for plain URLs; a git tree at a pinned
tag or commit on GitHub or a Gitea host with per-blob sha1 and size; a GCS bucket listing with per-object md5; a
bounded directory listing for host-local corpora), write `<root>/census/<version>/<source>/manifest.jsonl`, seal its BLAKE3 as the first
ledger entry for that source, and write `census.json` sealed as the census artifact. A source whose manifest cannot be
read is recorded `discovery_complete=false` with the error; it is never recorded as zero bytes.

Cost: O(sources + manifest bytes + selected entries); one listing per source; one HEAD per selected Common Crawl file.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import io
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ledger import Ledger, digest, digest_file  # noqa: E402

UA = "MNLM-census/0.1 (+https://github.com/ChrisRoyse/MNLM)"
TIMEOUT = 60


class SourceError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code, self.message = code, message


def http(url: str, method: str = "GET", stream: bool = False):
    req = urllib.request.Request(url, method=method, headers={"User-Agent": UA})
    try:
        return urllib.request.urlopen(req, timeout=TIMEOUT)
    except urllib.error.HTTPError as exc:
        raise SourceError("MNLM_CENSUS_HTTP", f"{method} {url} -> {exc.code}") from exc
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        raise SourceError("MNLM_CENSUS_UNREACHABLE", f"{method} {url}: {exc}") from exc


def head(url: str) -> dict:
    with http(url, "HEAD") as r:
        return {"status": r.status, "content_length": int(r.headers.get("Content-Length") or -1),
                "etag": (r.headers.get("ETag") or "").strip('"'), "final_url": r.geturl()}


def hf_tree(repo: str, revision: str, path: str) -> tuple[dict, list[dict]]:
    meta_url = f"https://huggingface.co/api/datasets/{repo}/revision/{urllib.parse.quote(revision, safe='')}"
    with http(meta_url) as r:
        meta = json.load(r)
    info = {"revision_sha": meta.get("sha"), "last_modified": meta.get("lastModified"),
            "card_license": (meta.get("cardData") or {}).get("license"), "gated": meta.get("gated"), "private": meta.get("private")}
    base = f"https://huggingface.co/api/datasets/{repo}/tree/{info['revision_sha'] or revision}/{path}".rstrip("/")
    url, entries = base + "?recursive=true&expand=true", []
    while url:
        with http(url) as r:
            page = json.load(r)
            link = r.headers.get("Link") or ""
        for it in page:
            if it.get("type") != "file":
                continue
            lfs = it.get("lfs") or {}
            entries.append({"path": it["path"], "bytes": int(lfs.get("size") or it.get("size") or 0),
                            "digest": lfs.get("oid") or it.get("oid"), "digest_kind": "sha256" if lfs else "git-blob-sha1"})
        url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part[part.find("<") + 1:part.find(">")]
    return info, entries


def cc_pathlist(spec: dict) -> tuple[dict, list[dict]]:
    with http(spec["pathlist"]) as r:
        raw = r.read()
    lines = gzip.GzipFile(fileobj=io.BytesIO(raw)).read().decode("utf-8").split("\n")
    sel = spec["select"]
    bucket = [ln for ln in lines if f"/{sel['bucket']}/" in ln]
    chosen = sorted(bucket)[: sel["first_n"]]
    info = {"pathlist_blake3": digest(raw), "pathlist_lines": sum(1 for ln in lines if ln), "bucket": sel["bucket"],
            "files_in_bucket": len(bucket), "selected": len(chosen)}
    entries = []
    for p in chosen:
        h = head(spec["base"] + p)
        entries.append({"path": p, "bytes": h["content_length"], "digest": h["etag"] or None, "digest_kind": "s3-etag"})
    return info, entries


def gdelt_master(spec: dict, window_end: str) -> tuple[dict, list[dict]]:
    end = _dt.date.fromisoformat(window_end)
    start = end - _dt.timedelta(days=spec["days"] - 1)
    days = {(start + _dt.timedelta(days=i)).strftime("%Y%m%d") for i in range(spec["days"])}
    entries, scanned = [], 0
    with http(spec["url"]) as r:
        for raw in r:
            scanned += 1
            parts = raw.decode("utf-8", "replace").split()
            if len(parts) != 3 or f".{spec['base_select']}.CSV.zip" not in parts[2]:
                continue
            stamp = parts[2].rsplit("/", 1)[-1][:8]
            if stamp in days:
                entries.append({"path": parts[2], "bytes": int(parts[0]), "digest": parts[1], "digest_kind": "md5"})
    info = {"masterfilelist_lines": scanned, "window_start": start.isoformat(), "window_end": end.isoformat(),
            "quarter_hours_expected": spec["days"] * 96, "quarter_hours_listed": len(entries)}
    return info, entries


def http_head_set(spec: dict) -> tuple[dict, list[dict]]:
    entries, errors = [], []
    for u in spec["urls"]:
        if spec.get("pause_s"):
            time.sleep(spec["pause_s"])  # a host that answers 403 to rapid listing (ebible.org) gets one HEAD per pause
        try:
            h = head(u)
            entries.append({"path": u, "bytes": h["content_length"], "digest": h["etag"] or None, "digest_kind": "etag",
                            "status": h["status"], "final_url": h["final_url"]})
        except SourceError as exc:
            errors.append({"url": u, "code": exc.code, "message": exc.message})
    if errors:
        raise SourceError("MNLM_CENSUS_PARTIAL", json.dumps(errors))
    return {"urls": len(spec["urls"])}, entries


def _json(url: str):
    with http(url) as r:
        return json.load(r), r.headers


def archive_item(spec: dict) -> tuple[dict, list[dict]]:
    """archive.org items: every file the metadata API lists, with size, sha1 (md5 when absent), format and runtime;
    `suffixes` selects file names. The item's licenseurl and runtime ride along in `info`."""
    entries, info = [], {"items": []}
    for item in spec["items"]:
        d, _ = _json(f"https://archive.org/metadata/{item}")
        files = d.get("files") or []
        if not files:
            raise SourceError("MNLM_CENSUS_HTTP", f"archive.org item {item}: no files listed")
        kept = 0
        for f in files:
            if spec.get("suffixes") and not any(f["name"].endswith(s) for s in spec["suffixes"]):
                continue
            entries.append({"path": f"https://archive.org/download/{item}/{f['name']}", "bytes": int(f.get("size") or -1),
                            "digest": f.get("sha1") or f.get("md5"), "digest_kind": "sha1" if f.get("sha1") else "md5",
                            "item": item, "format": f.get("format"), "length": f.get("length")})
            kept += 1
        md = d.get("metadata") or {}
        info["items"].append({"item": item, "files_listed": len(files), "selected": kept, "licenseurl": md.get("licenseurl"), "runtime": md.get("runtime")})
    return info, entries


def html_links(spec: dict) -> tuple[dict, list[dict]]:
    """Links on the listed HTML pages whose target ends with one of `suffixes`, resolved against the page, then one
    HEAD each (http-head semantics). Zero links refuses: an empty page is not an empty source."""
    import re

    urls: list[str] = []
    for page in spec["pages"]:
        with http(page) as r:
            html = r.read().decode("utf-8", "replace")
        for m in re.finditer(r"""href=["']([^"']+)["']""", html, re.I):
            u = urllib.parse.urljoin(page, m.group(1))
            if any(u.lower().endswith(s) for s in spec["suffixes"]) and u not in urls:
                urls.append(u)
    if not urls:
        raise SourceError("MNLM_CENSUS_HTTP", f"no link ending in {spec['suffixes']} on {spec['pages']}")
    info, entries = http_head_set({"urls": urls, "pause_s": spec.get("pause_s", 0)})
    info["pages"] = spec["pages"]
    return info, entries


def git_tree(spec: dict) -> tuple[dict, list[dict]]:
    """Blobs of one git tree at a pinned ref (tag, branch or commit) on GitHub or a Gitea host, each with its git blob
    sha1 and size; `prefixes` selects paths, `exclude_suffixes` drops some. The commit sha is the pin; a truncated tree
    listing refuses (the manifest would be incomplete, PC-13)."""
    host, repo, ref = spec["host"], spec["repo"], spec["ref"]
    if host == "github":
        api = f"https://api.github.com/repos/{repo}"
        if len(ref) == 40 and all(ch in "0123456789abcdef" for ch in ref):
            commit = ref
        else:
            try:
                d, _ = _json(f"{api}/git/ref/tags/{urllib.parse.quote(ref)}")
                commit = d["object"]["sha"]
                if d["object"]["type"] == "tag":
                    commit = _json(f"{api}/git/tags/{commit}")[0]["object"]["sha"]
            except SourceError:
                commit = _json(f"{api}/branches/{urllib.parse.quote(ref)}")[0]["commit"]["sha"]
        tree, _ = _json(f"{api}/git/trees/{commit}?recursive=1")
        if tree.get("truncated"):
            raise SourceError("MNLM_CENSUS_TREE_TRUNCATED", f"{repo}@{commit[:10]}: GitHub truncated the recursive listing")
        blobs = [e for e in tree["tree"] if e["type"] == "blob"]
        raw_base = f"https://raw.githubusercontent.com/{repo}/{commit}/"
    elif host == "gitea":
        api = f"{spec['api'].rstrip('/')}/repos/{repo}"
        try:
            commit = _json(f"{api}/tags/{urllib.parse.quote(ref)}")[0]["commit"]["sha"]
        except SourceError:
            commit = _json(f"{api}/branches/{urllib.parse.quote(ref)}")[0]["commit"]["id"]
        blobs, page = [], 1
        while True:
            tree, _ = _json(f"{api}/git/trees/{commit}?recursive=true&per_page=1000&page={page}")
            if tree.get("truncated"):
                raise SourceError("MNLM_CENSUS_TREE_TRUNCATED", f"{repo}@{commit[:10]}: Gitea truncated the recursive listing")
            blobs += [e for e in tree.get("tree") or [] if e["type"] == "blob"]
            if not tree.get("tree") or len(blobs) >= int(tree.get("total_count") or 0):
                break
            page += 1
        raw_base = f"{spec['api'].split('/api/')[0]}/{repo}/raw/commit/{commit}/"
    else:
        raise SourceError("MNLM_CENSUS_ACCESS_UNKNOWN", f"git host {host}")
    prefixes, excl = spec.get("prefixes") or [""], tuple(spec.get("exclude_suffixes") or ())
    chosen = [b for b in blobs if any(b["path"].startswith(p) for p in prefixes) and not b["path"].endswith(excl)]
    entries = [{"path": b["path"], "bytes": int(b["size"]), "digest": b["sha"], "digest_kind": "git-blob"} for b in sorted(chosen, key=lambda b: b["path"])]
    info = {"host": host, "repo": repo, "ref": ref, "commit": commit, "raw_base": raw_base, "blobs_in_tree": len(blobs),
            "selected": len(entries), "bytes": sum(e["bytes"] for e in entries)}
    return info, entries


def gcs_prefix(spec: dict) -> tuple[dict, list[dict]]:
    """Objects under `prefixes` of a public GCS bucket via the JSON listing API: name, size, md5 (hex) and generation per
    object. `exclude_suffixes` drops names (e.g. Sefaria's merged.json, which carries no licence field)."""
    import base64
    bucket, excl = spec["bucket"], tuple(spec.get("exclude_suffixes") or ())
    objs, listed = [], 0
    for prefix in spec["prefixes"]:
        token = None
        while True:
            q = {"prefix": prefix, "maxResults": "5000", "fields": "nextPageToken,items(name,size,md5Hash,generation,updated)"}
            if token:
                q["pageToken"] = token
            d, _ = _json(f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?" + urllib.parse.urlencode(q))
            items = d.get("items") or []
            listed += len(items)
            objs += [o for o in items if not o["name"].endswith(excl) and not o["name"].endswith("/")]
            token = d.get("nextPageToken")
            if not token:
                break
    entries = [{"path": o["name"], "bytes": int(o["size"]), "digest": base64.b64decode(o["md5Hash"]).hex(), "digest_kind": "md5",
                "generation": o["generation"], "updated": o.get("updated")} for o in sorted(objs, key=lambda o: o["name"])]
    info = {"bucket": bucket, "prefixes": spec["prefixes"], "objects_listed": listed, "selected": len(entries),
            "bytes": sum(e["bytes"] for e in entries), "base_url": f"https://storage.googleapis.com/{bucket}/"}
    return info, entries


def host_local(spec: dict) -> tuple[dict, list[dict]]:
    p = Path(spec["path"])
    if not p.is_dir():
        raise SourceError("MNLM_CENSUS_NOT_ON_HOST", f"{p} is not present on {socket.gethostname()}")
    entries = []
    with os.scandir(p) as it:  # top level only: a bounded listing, never a recursive walk (PC-13)
        for d in it:
            if d.is_file(follow_symlinks=False):
                st = d.stat()
                entries.append({"path": d.name, "bytes": st.st_size, "digest": None, "digest_kind": None})
            elif d.is_dir(follow_symlinks=False):
                entries.append({"path": d.name + "/", "bytes": None, "digest": None, "digest_kind": "directory"})
    sha = p / "OUTPUT.sha256"
    info = {"host": socket.gethostname(), "top_level_entries": len(entries),
            "output_sha256_present": sha.is_file(), "output_sha256_blake3": digest_file(sha)[0] if sha.is_file() else None}
    return info, entries


def poly_sealed_store(spec: dict) -> tuple[dict, list[dict]]:
    """Poly's production source store: sources/<source_id>/generations/gen-<digest>/manifest.json (Poly #1356).
    One bounded listing per source and one manifest read per generation; canonical/raw payloads are never opened."""
    root = Path(spec["path"])
    if not (root / "sources").is_dir():
        raise SourceError("MNLM_CENSUS_NOT_ON_HOST", f"{root} is not present on {socket.gethostname()}")
    entries, per_source = [], {}
    for sdir in sorted(p for p in (root / "sources").iterdir() if p.is_dir()):
        gdir = sdir / "generations"
        if not gdir.is_dir():
            per_source[sdir.name] = {"generations": 0, "note": "no generations directory"}
            continue
        agg = {"generations": 0, "manifests_missing": 0, "canonical_bytes": 0, "raw_bytes": 0, "records": 0, "completeness": {}}
        for g in sorted(p for p in gdir.iterdir() if p.is_dir() and p.name.startswith("gen-")):
            mp = g / "manifest.json"
            if not mp.is_file():
                agg["manifests_missing"] += 1
                continue
            mbytes = mp.read_bytes()
            m = json.loads(mbytes)
            canon = [a for a in m.get("artifacts", []) if a.get("role") == "canonical"]
            raw = [a for a in m.get("artifacts", []) if a.get("role") == "raw"]
            cb, rb = sum(int(a.get("byte_count") or 0) for a in canon), sum(int(a.get("byte_count") or 0) for a in raw)
            rec = int(m.get("canonical_record_count") or 0)
            entries.append({"path": f"{sdir.name}/{g.name}", "bytes": cb, "digest": m.get("generation_digest"),
                            "digest_kind": "poly-generation-digest", "manifest_blake3": digest(mbytes), "raw_bytes": rb,
                            "records": rec, "completeness": m.get("completeness"), "captured_at_unix_ms": m.get("captured_at_unix_ms"),
                            "canonical_sha256": canon[0].get("sha256") if canon else None})
            agg["generations"] += 1
            agg["canonical_bytes"] += cb
            agg["raw_bytes"] += rb
            agg["records"] += rec
            agg["completeness"][m.get("completeness")] = agg["completeness"].get(m.get("completeness"), 0) + 1
        per_source[sdir.name] = agg
    info = {"host": socket.gethostname(), "sources": len(per_source), "generations": sum(v.get("generations", 0) for v in per_source.values()),
            "canonical_bytes": sum(v.get("canonical_bytes", 0) for v in per_source.values()),
            "raw_bytes": sum(v.get("raw_bytes", 0) for v in per_source.values()),
            "records": sum(v.get("records", 0) for v in per_source.values()), "per_source": per_source}
    return info, entries


def local_authored(spec: dict) -> tuple[dict, list[dict]]:
    """Files authored on this host by a generator (identity-v1, issue #10): a bounded top-level listing with sha256 and
    BLAKE3 of every file except receipts and partials; BUILD.json (the generator's own manifest) rides along in `info` so
    the census records which inputs and commit produced the bytes. Zero files refuses: an empty directory is not a source."""
    import hashlib

    p = Path(spec["path"])
    if not p.is_dir():
        raise SourceError("MNLM_CENSUS_NOT_ON_HOST", f"{p} is not present on {socket.gethostname()}")
    entries = []
    with os.scandir(p) as it:
        for d in sorted(it, key=lambda d: d.name):
            if not d.is_file(follow_symlinks=False) or d.name == "receipts.jsonl" or d.name.endswith((".part", ".REJECTED")):
                continue
            h, n = hashlib.sha256(), 0
            with open(d.path, "rb") as fh:
                while b := fh.read(1 << 20):
                    h.update(b)
                    n += len(b)
            entries.append({"path": d.name, "bytes": n, "digest": h.hexdigest(), "digest_kind": "sha256", "blake3": digest_file(Path(d.path))[0]})
    if not entries:
        raise SourceError("MNLM_CENSUS_EMPTY_AUTHORED", f"{p} holds no files")
    build = p / "BUILD.json"
    info = {"host": socket.gethostname(), "files": len(entries), "bytes": sum(e["bytes"] for e in entries),
            "build": json.loads(build.read_text(encoding="utf-8")) if build.is_file() else None}
    declared = ((info["build"] or {}).get("outputs") or {})  # the generator's own digests must be the bytes on disk
    bad = [e["path"] for e in entries if e["path"] in declared and declared[e["path"]].get("sha256") != e["digest"]]
    if bad:
        raise SourceError("MNLM_CENSUS_AUTHORED_MISMATCH", f"{p}: BUILD.json names other bytes for {bad}")
    return info, entries


def manifest_copy(spec: dict) -> tuple[dict, list[dict]]:
    """A manifest.jsonl produced by the census on another host and copied here, carrying that host's seal."""
    p = Path(spec["path"])
    if not p.is_file():
        raise SourceError("MNLM_CENSUS_MANIFEST_COPY_MISSING", f"{p} not present on {socket.gethostname()}; copy it from {spec.get('origin', {}).get('host')}")
    entries = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {"copied_from": spec.get("origin"), "copied_manifest_blake3": digest_file(p)[0], "host": socket.gethostname()}, entries


def resolve(src: dict, window_end: str) -> tuple[dict, list[dict]]:
    a = src["access"]
    if a.get("type") == "by-host":
        a = a.get(socket.gethostname()) or a.get("default")
        if not a:
            raise SourceError("MNLM_CENSUS_NOT_ON_HOST", f"no access rule for host {socket.gethostname()}")
    t = a["type"]
    if t == "poly-sealed-store":
        return poly_sealed_store(a)
    if t == "manifest-copy":
        return manifest_copy(a)
    if t == "hf-tree":
        return hf_tree(a["repo"], a["revision"], a["path"])
    if t == "cc-pathlist":
        return cc_pathlist(a)
    if t == "gdelt-masterfilelist":
        return gdelt_master(a, window_end)
    if t == "http-head":
        return http_head_set(a)
    if t == "git-tree":
        return git_tree(a)
    if t == "gcs-prefix":
        return gcs_prefix(a)
    if t == "archive-item":
        return archive_item(a)
    if t == "html-links":
        return html_links(a)
    if t == "aiwonder-local":
        return host_local(a)
    if t == "local-authored":
        return local_authored(a)
    if t == "local-manifest-pending":
        raise SourceError("MNLM_CENSUS_MANIFEST_PENDING", a.get("note", "manifest not located"))
    raise SourceError("MNLM_CENSUS_ACCESS_UNKNOWN", t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--actor", default=f"census@{socket.gethostname()}")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "canon"))
    from training_scope import POLICY_SHA256, allowed, require_training_source
    if args.only:
        require_training_source(args.only)

    cfg_path = Path(args.config)
    cfg_bytes = cfg_path.read_bytes()
    cfg = json.loads(cfg_bytes)
    from census_scope import effective_sources
    sources = effective_sources(cfg, args.only)
    root = Path(args.root)
    out = root / "census" / cfg["census_version"]
    out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(root / "ledger" / "ledger.jsonl", actor=args.actor)
    ledger.append("census.config", cfg_path.name, {"config_blake3": digest(cfg_bytes), "census_version": cfg["census_version"]})

    report = {"census_version": cfg["census_version"], "training_scope_sha256": POLICY_SHA256,
              "window_end_utc": cfg["window_end_utc"], "host": socket.gethostname(),
              "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "config_blake3": digest(cfg_bytes),
              "sources": {}}
    if args.only:  # one source re-censused: the others keep their rows (--only used to write a one-source census.json and seal it)
        if not (out / "census.json").exists():
            raise SystemExit(f"MNLM_CENSUS_ONLY_WITHOUT_CENSUS: --only {args.only} needs an existing {out / 'census.json'}")
        prior = json.loads((out / "census.json").read_text(encoding="utf-8"))
        report["sources"] = {k: v for k, v in prior["sources"].items() if k != args.only and allowed(k)}
        report["prior_census_blake3"] = digest_file(out / "census.json")[0]
    for src in sources:
        sid = src["id"]
        if args.only and sid != args.only:
            continue
        row = {"kind": src["kind"], "licence": src["licence"], "consent": src["consent"], "declared": src["declared"],
               "fields": src["fields"], "clocks": src["clocks"], "anchors": src["anchors"], "role": src["role"],
               "access": src["access"], "discovery_complete": False, "entries": None, "bytes_total": None,
               "manifest_blake3": None, "info": None, "error": None}
        try:
            for attempt in range(1, 4):  # a reset mid-stream is a transient refusal; three bounded attempts, then refuse
                try:
                    info, entries = resolve(src, cfg["window_end_utc"])
                    break
                except (OSError, ConnectionResetError, TimeoutError) as exc:
                    if attempt == 3:
                        raise SourceError("MNLM_CENSUS_UNREACHABLE", f"{type(exc).__name__}: {exc} after {attempt} attempts") from exc
                    print(f"retry   {sid}: attempt {attempt} failed with {type(exc).__name__}: {exc}", flush=True)
                except SourceError as exc:
                    if attempt == 3 or exc.code not in ("MNLM_CENSUS_UNREACHABLE", "MNLM_CENSUS_HTTP"):
                        raise
                    print(f"retry   {sid}: attempt {attempt} refused {exc.code}", flush=True)
            sdir = out / sid
            sdir.mkdir(exist_ok=True)
            mpath = sdir / "manifest.jsonl"
            with mpath.open("w", encoding="utf-8", newline="\n") as f:
                for e in entries:
                    f.write(json.dumps(e, sort_keys=True, ensure_ascii=False) + "\n")
            mdig, _ = digest_file(mpath)
            sized = [e["bytes"] for e in entries if isinstance(e.get("bytes"), int) and e["bytes"] >= 0]
            row.update(discovery_complete=True, entries=len(entries), bytes_total=sum(sized) if sized else None,
                       bytes_known_for=len(sized), manifest_blake3=mdig, info=info)
            ledger.append("census.source", sid, {"manifest_blake3": mdig, "entries": len(entries), "bytes_total": row["bytes_total"]})
            print(f"ok      {sid}: {len(entries)} entries, {row['bytes_total']} bytes", flush=True)
        except SourceError as exc:
            row["error"] = {"code": exc.code, "message": exc.message}
            ledger.append("census.source.refused", sid, {"code": exc.code})
            print(f"refused {sid}: {exc.code}: {exc.message[:200]}", flush=True)
        report["sources"][sid] = row

    report["discovery_complete"] = all(r["discovery_complete"] for r in report["sources"].values())
    report["finished_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    cpath = out / "census.json"
    if set(report["sources"]) != {s["id"] for s in sources}:
        raise SystemExit(f"MNLM_CENSUS_INCOMPLETE: census names {sorted(report['sources'])}, effective scope {sorted(s['id'] for s in sources)}")
    tmp = out / "census.json.tmp"
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, cpath)
    cdig, cbytes = digest_file(cpath)
    sealed = ledger.append("census.artifact", cfg["census_version"], {"census_blake3": cdig, "census_bytes": cbytes,
                                                                        "discovery_complete": report["discovery_complete"]})
    (out / "census.json.blake3").write_text(f"{cdig}  census.json\n", encoding="utf-8")
    (out / "SEAL.json").write_text(json.dumps({"ledger_head": sealed["hash"], "ledger_seq": sealed["seq"], "census_blake3": cdig},
                                              indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "census": str(cpath), "census_blake3": cdig, "ledger_head": sealed["hash"],
                      "ledger_seq": sealed["seq"], "discovery_complete": report["discovery_complete"]}, indent=2))
    return 0 if report["discovery_complete"] else 3


if __name__ == "__main__":
    sys.exit(main())
