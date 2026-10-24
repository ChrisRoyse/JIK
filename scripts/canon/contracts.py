"""Stage 1 canonical extraction contracts (spec §17 op 1; issue #8). Contract id `canon-v1` is frozen: changing any rule
below is a new contract id, never an edit.

A contract turns one raw shard into an ordered stream of canonical records (row, text, fields, kind): `row` is the raw
row ordinal (identity, stable across re-runs), `text` the canonical UTF-8 string, `fields` the explicit columns kept
verbatim beside it (never embedded; spec §18), `kind` one of PROSE (joins the near-dup and exact-substring layers), ROW
(structured: exact-row layer only) or ALIAS (the alias layer, spec §17 op 4). Empty text after normalisation is yielded
as text=None and counted, never written as a doc. Schema drift refuses the shard with MNLM_CANON_SCHEMA_MISMATCH; a raw
digest that disagrees with its receipt refuses with MNLM_CANON_RAW_DIGEST_MISMATCH (checked by the driver).
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
import unicodedata
import zipfile
from collections.abc import Iterator
from pathlib import Path

import identity  # noqa: E402  (identity-v1 readers, issue #10)
import scripture  # noqa: E402  (scripture-v1 readers, issue #9)

CONTRACT_ID = "canon-v1"
PROSE, ROW, ALIAS = "prose", "row", "alias"

# Keep-first order for every dedup layer (issue #8). Lower wins. Sources absent here never enter a dedup population.
SOURCE_PRIORITY = {
    "plant-v1": 0,
    "cuyahoga-eighth-district": 1,
    "wikipedia-en-20231101": 2,
    "wikidata5m": 3,
    "opencodeinstruct": 4,
    "nemotron-cc-high-actual-slice": 5,
    "fineweb-edu-sample-10BT": 6,
    "gdelt-events-90d": 7,
    # scripture partition (issue #9): original-language and ancient versions keep first, then the Jewish corpus, then English
    "unfoldingword-uhb": 10, "unfoldingword-ugnt": 11, "tanach-us-uxlc": 12, "sblgnt": 13, "nestle1904": 14, "byztxt-rp2018": 15,
    "crosswire-sword-pd": 16, "sefaria-export": 17, "sefaria-export-nc": 18, "ebible-english": 19, "berean-bsb": 20, "open-english-bible": 21,
    "unfoldingword-ult": 22, "unfoldingword-ust": 23, "openbible-crossrefs": 24, "ubsicap-versification": 25, "stepbible-data": 26,
    "identity-v1": 30,  # identity partition (issue #10): its own population; the verses it quotes live in scripture-v1
}
DEDUP_EXCLUDED = {"nemotron-cc-high-synthetic-twins"}  # canonicalised, never a base atom (spec: Stage 8 experiment)
SOURCES_CONFIG = Path(__file__).resolve().parent.parent.parent / "config" / "sources.p0.json"
SOURCES_CONFIGS = (SOURCES_CONFIG, SOURCES_CONFIG.with_name("sources.scripture.json"), SOURCES_CONFIG.with_name("sources.identity.json"))


def near_removable(source_id: str) -> bool:
    """Near-duplicate *removal* applies only to bulk web prose whose only anchor family is A5 (recurrence): FineWeb-Edu and
    Nemotron-CC. A source that carries A1/A2/A3/A4/A6 identities (tests, entities, events, opinions) keeps every record and
    carries the near link as a tag instead (spec §22: redundant atoms are weighted, never deleted). The planted corpus is
    removable so the negative controls exercise the removal path."""
    if source_id == "plant-v1":
        return True
    lookup = scripture.RAW_DIR_ALIAS.get(source_id, source_id)
    for cfg_path in SOURCES_CONFIGS:
        for src in json.loads(cfg_path.read_text(encoding="utf-8"))["sources"]:
            if src["id"] == lookup:
                return set(src.get("anchors") or []) <= {"A5"}
    raise ContractError("MNLM_CANON_UNKNOWN_SOURCE", f"{source_id} is not in {[c.name for c in SOURCES_CONFIGS]}")

PARQUET_SCHEMAS = {
    "fineweb-edu-sample-10BT": {"text": "text", "explicit": ["id", "dump", "url", "file_path", "language", "language_score", "token_count", "score", "int_score"]},
    "wikipedia-en-20231101": {"text": "text", "explicit": ["id", "url", "title"]},
    "opencodeinstruct": {"text": None, "text_parts": ["input", "output"],
                         "explicit": ["id", "domain", "generation_algorithm", "llm_judgement", "unit_tests", "tests_execution_status", "average_test_score"]},
}
NEMOTRON_FIELDS = {"text": "text", "explicit": ["warc_record_id", "url", "language"]}
GDELT_COLUMNS = (
    "GlobalEventID Day MonthYear Year FractionDate Actor1Code Actor1Name Actor1CountryCode Actor1KnownGroupCode Actor1EthnicCode "
    "Actor1Religion1Code Actor1Religion2Code Actor1Type1Code Actor1Type2Code Actor1Type3Code Actor2Code Actor2Name Actor2CountryCode "
    "Actor2KnownGroupCode Actor2EthnicCode Actor2Religion1Code Actor2Religion2Code Actor2Type1Code Actor2Type2Code Actor2Type3Code "
    "IsRootEvent EventCode EventBaseCode EventRootCode QuadClass GoldsteinScale NumMentions NumSources NumArticles AvgTone "
    "Actor1Geo_Type Actor1Geo_Fullname Actor1Geo_CountryCode Actor1Geo_ADM1Code Actor1Geo_ADM2Code Actor1Geo_Lat Actor1Geo_Long Actor1Geo_FeatureID "
    "Actor2Geo_Type Actor2Geo_Fullname Actor2Geo_CountryCode Actor2Geo_ADM1Code Actor2Geo_ADM2Code Actor2Geo_Lat Actor2Geo_Long Actor2Geo_FeatureID "
    "ActionGeo_Type ActionGeo_Fullname ActionGeo_CountryCode ActionGeo_ADM1Code ActionGeo_ADM2Code ActionGeo_Lat ActionGeo_Long ActionGeo_FeatureID "
    "DATEADDED SOURCEURL"
).split()
GDELT_EXPLICIT = ["GlobalEventID", "Day", "Actor1Code", "Actor2Code", "EventCode", "QuadClass", "GoldsteinScale", "NumMentions", "NumSources", "AvgTone",
                  "ActionGeo_CountryCode", "ActionGeo_Lat", "ActionGeo_Long", "DATEADDED", "SOURCEURL"]
GDELT_IDX = {c: i for i, c in enumerate(GDELT_COLUMNS)}
CUYAHOGA_INGEST = ["ohctapp8-modern.jsonl", "cap-cuyahoga.jsonl"]


class ContractError(Exception):
    def __init__(self, code: str, message: str, **detail):
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.detail = code, message, detail


def canon_text(s: str | None) -> str | None:
    """canon-v1 normalisation: NFC; CRLF/CR -> LF; trailing whitespace stripped per line; outer blank lines stripped."""
    if s is None:
        return None
    s = unicodedata.normalize("NFC", s).replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in s.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) if lines else None


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------- parquet
def _parquet_docs(path: Path, spec: dict) -> Iterator[tuple[int, str | None, dict, str]]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    need = set(spec["explicit"]) | ({spec["text"]} if spec["text"] else set(spec["text_parts"]))
    if not need <= names:
        raise ContractError("MNLM_CANON_SCHEMA_MISMATCH", f"{path.name} lacks {sorted(need - names)}", have=sorted(names))
    cols = sorted(need)
    row = 0
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=cols).to_pydict()
        n = len(tbl[cols[0]])
        for i in range(n):
            if spec["text"]:
                raw = tbl[spec["text"]][i]
            else:
                raw = "\n\n".join(str(tbl[c][i]) for c in spec["text_parts"] if tbl[c][i] is not None)
            fields = {c: tbl[c][i] for c in spec["explicit"]}
            yield row, canon_text(raw), fields, PROSE
            row += 1


# ---------------------------------------------------------------------------------------------------------------- jsonl
def _nemotron_docs(path: Path) -> Iterator[tuple[int, str | None, dict, str]]:
    import zstandard

    with path.open("rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as zr:
        tr = io.TextIOWrapper(zr, encoding="utf-8")
        for row, line in enumerate(tr):
            if not line.strip():
                yield row, None, {}, PROSE
                continue
            d = json.loads(line)
            if row == 0 and NEMOTRON_FIELDS["text"] not in d:
                raise ContractError("MNLM_CANON_SCHEMA_MISMATCH", f"{path.name} record lacks 'text'", have=sorted(d))
            yield row, canon_text(d.get("text")), {k: d.get(k) for k in NEMOTRON_FIELDS["explicit"]}, PROSE


def _cuyahoga_docs(path: Path) -> Iterator[tuple[int, str | None, dict, str]]:
    with path.open("r", encoding="utf-8") as f:
        for row, line in enumerate(f):
            d = json.loads(line)
            if row == 0 and "text" not in d:
                raise ContractError("MNLM_CANON_SCHEMA_MISMATCH", f"{path.name} record lacks 'text'", have=sorted(d))
            meta = d.get("metadata")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    meta = {"metadata_raw": meta}
            fields = {k: (meta or {}).get(k) for k in ("opinion_id", "cluster_id", "docket_id", "court_id", "date_filed")}
            fields["anchors"] = d.get("anchors")
            yield row, canon_text(d.get("text")), fields, PROSE


def _plant_docs(path: Path) -> Iterator[tuple[int, str | None, dict, str]]:
    with path.open("r", encoding="utf-8") as f:
        for row, line in enumerate(f):
            d = json.loads(line)
            yield row, canon_text(d.get("text")), {"label": d.get("label")}, d.get("kind", PROSE)


# ---------------------------------------------------------------------------------------------------------------- structured
def _gdelt_docs(paths: list[Path]) -> Iterator[tuple[int, str | None, dict, str]]:
    """One day of 15-minute export zips; row ordinal is global within the day. Every field is explicit (spec §18)."""
    row = 0
    for p in paths:
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
            if len(names) != 1:
                raise ContractError("MNLM_CANON_SCHEMA_MISMATCH", f"{p.name} holds {len(names)} members", have=names)
            text = z.read(names[0]).decode("utf-8", "surrogateescape")
        for line in text.split("\n"):
            line = line.rstrip("\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != len(GDELT_COLUMNS):
                raise ContractError("MNLM_CANON_SCHEMA_MISMATCH", f"{p.name} row {row} has {len(parts)} fields, not {len(GDELT_COLUMNS)}")
            fields = {c: parts[GDELT_IDX[c]] for c in GDELT_EXPLICIT}
            fields["file"] = p.name
            yield row, line, fields, ROW
            row += 1


def _wikidata5m_members(path: Path) -> Iterator[tuple[str, str, Iterator]]:
    """Yields (shard_name, kind, docs) per member of a Wikidata5M archive."""
    if path.name.endswith("text.txt.gz"):
        def docs():
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for row, line in enumerate(f):
                    ent, _, txt = line.rstrip("\n").partition("\t")
                    yield row, canon_text(txt), {"entity_id": ent}, PROSE
        yield "wikidata5m_text", PROSE, docs()
        return
    with tarfile.open(path, "r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = Path(m.name).name
            data = tf.extractfile(m).read().decode("utf-8")
            if "alias" in path.name:
                def adocs(data=data, name=name):
                    for row, line in enumerate(data.split("\n")):
                        if not line:
                            continue
                        parts = line.split("\t")
                        yield row, line, {"id": parts[0], "aliases": parts[1:], "member": name}, ALIAS
                yield name.rsplit(".", 1)[0], ALIAS, adocs()
            else:
                def tdocs(data=data, name=name):
                    for row, line in enumerate(data.split("\n")):
                        if not line:
                            continue
                        s, p, o = line.split("\t")
                        yield row, line, {"s": s, "p": p, "o": o, "split": name}, ROW
                yield name.rsplit(".", 1)[0], ROW, tdocs()


# ---------------------------------------------------------------------------------------------------------------- dispatch
def list_shards(source_id: str, raw_dir: Path, receipts: dict, cuyahoga_root: Path | None = None) -> list[dict]:
    """Shard plan for a source: [{shard, files:[Path], sha256:[...], kind}] in deterministic order. Only files with a
    verified receipt (or a manifest-attested digest for Cuyahoga) are planned."""
    plan: list[dict] = []
    if source_id in scripture.SOURCES:
        return scripture.list_shards(source_id, raw_dir, receipts)
    if source_id in identity.SOURCES:
        return identity.list_shards(source_id, raw_dir, receipts)
    if source_id == "cuyahoga-eighth-district":
        root = cuyahoga_root or raw_dir
        man = json.loads((root / "ingest" / "ingest_manifest.json").read_text(encoding="utf-8"))
        for name in CUYAHOGA_INGEST:
            plan.append({"shard": name.rsplit(".", 1)[0], "files": [root / "ingest" / name], "sha256": [man["files"][name]["sha256"]],
                         "kind": PROSE, "attested_rows": man["files"][name]["rows"]})
        return plan
    if source_id == "plant-v1":
        for p in sorted(raw_dir.glob("*.jsonl")):
            plan.append({"shard": p.stem, "files": [p], "sha256": [sha256_file(p)], "kind": PROSE})
        return plan
    verified = {k: v for k, v in receipts.items() if v.get("verified") is True or (v.get("verified") is None and v.get("attestation") == "content-length" and v.get("size_match") is True)}
    if source_id == "gdelt-events-90d":
        by_day: dict[str, list] = {}
        for k in sorted(verified):
            by_day.setdefault(Path(k).name[:8], []).append(k)
        for day, keys in sorted(by_day.items()):
            plan.append({"shard": f"gdelt-{day}", "files": [raw_dir / Path(k).name for k in keys],
                         "sha256": [verified[k]["sha256"] for k in keys], "kind": ROW})
        return plan
    for k in sorted(verified):
        if source_id in PARQUET_SCHEMAS and not k.endswith(".parquet"):
            continue  # dataset cards and licence files ride along in the HF tree; they are census items, not shards
        local = raw_dir / (Path(k).name.split("?")[0] if k.startswith("http") else k)
        if not local.exists():
            local = raw_dir / Path(k).name  # cc-pathlist entries are stored by basename (acquire.local_name)
        kind = PROSE if source_id != "wikidata5m" else "archive"
        plan.append({"shard": local.name.split(".")[0], "files": [local], "sha256": [verified[k]["sha256"]], "kind": kind})
    return plan


def open_shard(source_id: str, shard: dict) -> Iterator[tuple[str, str, Iterator]]:
    """Yields (shard_name, kind, docs) streams for a planned shard. Digest checks happen in the driver (once per file)."""
    files = shard["files"]
    if source_id in scripture.SOURCES or source_id in identity.SOURCES:
        mod = scripture if source_id in scripture.SOURCES else identity
        for name, kind, docs in mod.open_shard(source_id, shard):  # canon-v1 normalisation applies to every source
            yield name, kind, ((row, canon_text(text), f, k) for row, text, f, k in docs)
    elif source_id in PARQUET_SCHEMAS:
        yield shard["shard"], PROSE, _parquet_docs(files[0], PARQUET_SCHEMAS[source_id])
    elif source_id.startswith("nemotron-cc"):
        yield shard["shard"], PROSE, _nemotron_docs(files[0])
    elif source_id == "cuyahoga-eighth-district":
        yield shard["shard"], PROSE, _cuyahoga_docs(files[0])
    elif source_id == "plant-v1":
        yield shard["shard"], PROSE, _plant_docs(files[0])
    elif source_id == "gdelt-events-90d":
        yield shard["shard"], ROW, _gdelt_docs(files)
    elif source_id == "wikidata5m":
        yield from _wikidata5m_members(files[0])
    else:
        raise ContractError("MNLM_CANON_NO_CONTRACT", f"no canon-v1 contract for {source_id}")
