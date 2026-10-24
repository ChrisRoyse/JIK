"""Read native substring coordinates against original typed payload bytes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3

from native_population import file_hash, POLICY_SHA256
from senses.contract import canonical, require


def audit(population, db, threshold):
    require(threshold == 200, "MNLM_NATIVE_SUBSTRING_CONTRACT", "fixed 200-byte contract required")
    require(db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            and not db.execute("PRAGMA foreign_key_check").fetchall(),
            "MNLM_NATIVE_SUBSTRING_DATABASE", "annotation integrity/foreign keys failed")
    expected, texts, payloads, pos = [], [], {}, 0
    for ph, body in population.db.execute("SELECT digest,body FROM payloads WHERE channel='surface' ORDER BY digest"):
        require(hashlib.sha256(body).hexdigest() == ph, "MNLM_NATIVE_PAYLOAD_DIGEST", ph)
        items = json.loads(body)["items"]
        payloads[ph] = len(items)
        for i, item in enumerate(items):
            raw = item["text"].encode("utf-8")
            expected.append((len(texts), ph, i, pos, len(raw), hashlib.sha256(raw).hexdigest()))
            texts.append(raw)
            pos += len(raw) + 1
    require(dict(db.execute("SELECT digest,items FROM payloads")) == payloads
            and list(db.execute("SELECT id,payload,item,start,n,sha256 FROM segments ORDER BY id")) == expected,
            "MNLM_NATIVE_SUBSTRING_SCOPE", "surface items, order or source byte coordinates differ")
    h, count, total = hashlib.sha256(), 0, 0
    for row in db.execute("SELECT origin,start,end,target,dup_start,sha256 FROM spans ORDER BY origin,start,end,target,dup_start"):
        a, start, end, b, dup, wanted = row
        require(all(type(x) is int for x in (a, start, end, b, dup))
                and 0 <= a < len(texts) and 0 <= b < len(texts)
                and 0 <= start < end <= len(texts[a]) and end - start >= threshold
                and 0 <= dup <= len(texts[b]) - (end - start)
                and expected[b][3] + dup < expected[a][3] + start,
                "MNLM_NATIVE_SUBSTRING_COORDINATES", "invalid, reversed or boundary-crossing span")
        x, y = texts[a][start:end], texts[b][dup:dup + end - start]
        require(x == y and hashlib.sha256(x).hexdigest() == wanted,
                "MNLM_NATIVE_SUBSTRING_BYTES", "duplicate bytes differ from retained source")
        try:
            # Both coordinates, not just the equal interior, must be UTF-8 boundaries.
            texts[a][:start].decode(); x.decode(); texts[b][:dup].decode(); y.decode()
        except UnicodeDecodeError:
            require(False, "MNLM_NATIVE_SUBSTRING_UTF8", "coordinate splits a UTF-8 character")
        h.update(canonical(list(row)) + b"\n")
        count += 1
        total += end - start
    return {"payloads": len(payloads), "items": len(texts), "surface_bytes": pos - len(texts),
            "corpus_bytes": pos, "spans": count, "annotated_bytes": total, "spans_sha256": h.hexdigest()}


def _inspect(population, source, manifest_sha256, parent_manifest):
    source = Path(source)
    mp, dp = source / "manifest.json", source / "substrings.sqlite"
    require(mp.is_file() and file_hash(mp) == manifest_sha256,
            "MNLM_NATIVE_SUBSTRING_MANIFEST", "pinned completion manifest required")
    m = json.loads(mp.read_bytes())
    require(m["schema"] == "native-substrings-v1" and m["training_scope_sha256"] == POLICY_SHA256
            and m["parent_manifest_sha256"] == parent_manifest and m["observations_removed"] == 0,
            "MNLM_NATIVE_SUBSTRING_SCOPE", "substring publication does not bind this unchanged population")
    require(dp.is_file() and dp.stat().st_size == m["files"]["substrings.sqlite"]["bytes"]
            and file_hash(dp) == m["files"]["substrings.sqlite"]["sha256"],
            "MNLM_NATIVE_SUBSTRING_DIGEST", "annotation database bytes changed")
    db = sqlite3.connect(dp.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        result = audit(population, db, m["threshold_bytes"])
        require(result == m["census"], "MNLM_NATIVE_SUBSTRING_CENSUS", "span counts/digest differ")
    finally:
        db.close()
    return m


def inspect(population, source, manifest_sha256, parent_manifest):
    try:
        return _inspect(population, source, manifest_sha256, parent_manifest)
    except (OSError, ValueError, KeyError, TypeError, sqlite3.DatabaseError) as exc:
        require(False, "MNLM_NATIVE_SUBSTRING_INVALID", f"malformed annotation publication: {type(exc).__name__}")


def retain(population, source, manifest_sha256, output, parent_manifest):
    source, output = Path(source), Path(output)
    m = inspect(population, source, manifest_sha256, parent_manifest)
    for name, dest in (("substrings.sqlite", "substrings.sqlite"), ("manifest.json", "substrings.manifest.json")):
        with (source / name).open("rb") as reader, (output / dest).open("xb") as writer:
            shutil.copyfileobj(reader, writer); writer.flush(); os.fsync(writer.fileno())
    require(file_hash(output / "substrings.sqlite") == m["files"]["substrings.sqlite"]["sha256"]
            and file_hash(output / "substrings.manifest.json") == manifest_sha256,
            "MNLM_NATIVE_SUBSTRING_DIGEST", "annotation changed during retention")
    return {"manifest_sha256": manifest_sha256, **m["census"], "observations_removed": 0}
