"""Bounded CAP originals for already-captured evaluation cases (#14/#27).

Download named JSON/HTML pairs only. Raw acquisition is not layout acceptance,
court eligibility, a grounded correctness judgement, or a sealed evaluation.
"""
import argparse
from contextlib import closing
import hashlib
import http.client
import json
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.trajectory_population import publish

MAX_RESPONSE = 8*1024*1024
MAX_TOTAL = 256*1024*1024
TIMEOUT = 30
IDENTIFIER = re.compile(r"f2d_([0-9]+)/html/([0-9]+-[0-9]+)\.html\Z")


def check(ok, code):
    if not ok:
        raise ValueError("MNLM_CAP_CAPTURE_"+code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def acquire(session, url, output, totals):
    started = time.perf_counter()
    for attempt in range(2):
        try:
            session.request("GET", urlsplit(url).path, headers={"User-Agent": "MNLM source-provenance research"})
            with session.getresponse() as response:
                if response.status in (429, 500, 502, 503, 504) and attempt == 0:
                    delay = response.getheader("Retry-After", "1")
                    check(delay.isdigit() and int(delay) <= TIMEOUT, "RETRY_AFTER")
                    session.close(); time.sleep(int(delay)); continue
                check(response.status == 200, "HTTP_"+str(response.status))
                declared = response.getheader("Content-Length")
                check(declared is None or int(declared) <= MAX_RESPONSE, "RESPONSE_BUDGET")
                data = bytearray()
                while block := response.read(65536):
                    totals["received_bytes"] += len(block)
                    check(totals["received_bytes"]+totals["reused_bytes"] <= MAX_TOTAL
                          and len(data)+len(block) <= MAX_RESPONSE, "BYTE_BUDGET")
                    data.extend(block)
                raw = bytes(data)
                check(bool(raw), "EMPTY_RESPONSE")
                if output.suffix == ".json":
                    check(isinstance(json.loads(raw), dict), "JSON_OBJECT")
                else:
                    check("casebody" in raw.decode("utf-8"), "HTML_CASEBODY")
                with output.open("xb") as f:
                    f.write(raw)
                return dict(url=url, file=output.name, bytes=len(raw), sha256=sha(raw), status=200,
                            headers={k: response.getheader(k) for k in ("Content-Type", "Content-Length", "ETag", "Last-Modified", "Date") if response.getheader(k) is not None},
                            seconds=time.perf_counter()-started, attempts=attempt+1, reused=False)
        except (OSError, http.client.HTTPException):
            session.close()
            if attempt == 1:
                raise
    raise ValueError("MNLM_CAP_CAPTURE_ATTEMPTS")


def run(source, pin, output, limit=512, reuse=None):
    started = time.perf_counter(); check(not output.exists(), "EXISTS")
    raw = (source/"report.json").read_bytes(); report = json.loads(raw)
    check(sha(raw) == pin and report["complete"] is True and report["errors"] == [], "SOURCE_PIN")
    path = source/"legal.jsonl"; legal_hash = file_sha(path)
    check(legal_hash == report["sources"]["legal"]["sha256"], "LEGAL_HASH")
    check(type(limit) is int and 1 <= limit <= 512, "LIMIT")
    with path.open("rb") as f:
        selected = [json.loads(line) for line in f][:limit]
    check(len(selected) == limit, "SOURCE_COUNT")
    reusable = {}
    if reuse:
        prev = json.loads((reuse/"report.json").read_bytes())
        check(prev["source_report_sha256"] == pin and prev["complete"] is True and not prev["errors"], "REUSE_PARENT")
        reusable = {r["document_id"]: r for r in prev["cases"]}
    output.mkdir(parents=True); (output/"raw").mkdir(); (output/"checkpoints").mkdir()
    totals = dict(received_bytes=0, reused_bytes=0); cases = []; errors = []
    with closing(http.client.HTTPSConnection("static.case.law", timeout=TIMEOUT)) as session, (output/"cases.jsonl").open("xb") as ledger:
        for row in selected:
            item = dict(document_id=row["id"], captured_source=row["source"], artifacts={})
            try:
                m = IDENTIFIER.fullmatch(row["source"]["document_id"])
                check(m is not None, "DOCUMENT_ID")
                volume, name = m.groups()
                for kind, folder in (("json", "cases"), ("html", "html")):
                    url = f"https://static.case.law/f2d/{volume}/{folder}/{name}.{kind}"
                    dest = output/"raw"/(row["id"]+"."+kind)
                    old = reusable.get(row["id"])
                    if old and kind in old["artifacts"]:
                        rec = old["artifacts"][kind]; origin = reuse/"raw"/dest.name
                        check(rec["url"] == url and rec["file"] == dest.name and file_sha(origin) == rec["sha256"]
                              and origin.stat().st_size == rec["bytes"], "REUSE_BYTES")
                        check(totals["received_bytes"]+totals["reused_bytes"]+rec["bytes"] <= MAX_TOTAL, "BYTE_BUDGET")
                        shutil.copyfile(origin, dest); totals["reused_bytes"] += rec["bytes"]
                        item["artifacts"][kind] = {**rec, "reused": True}
                    else:
                        check(totals["received_bytes"]+totals["reused_bytes"] < MAX_TOTAL, "BYTE_BUDGET")
                        item["artifacts"][kind] = acquire(session, url, dest, totals)
            except (http.client.HTTPException, OSError, ValueError, KeyError, TypeError) as exc:
                session.close()
                errors.append(dict(document_id=row["id"], error=str(exc)))
            cases.append(item)
            ledger.write(json.dumps(item, sort_keys=True).encode()+b"\n"); ledger.flush()
            publish(output/"checkpoints"/f"{len(cases):04d}.json", dict(source_report_sha256=pin,
                    completed_cases=len(cases), errors=len(errors), totals=totals, last_document_id=row["id"]))
            print(json.dumps(dict(completed_cases=len(cases), errors=len(errors), **totals)), flush=True)
    result = dict(schema="cap-evaluation-originals-v1", source_report_sha256=pin, legal_sha256=legal_hash,
                  selected_document_ids=[r["id"] for r in selected], cases=cases, errors=errors, totals=totals,
                  complete=not errors and len(cases) == limit and all(set(r["artifacts"]) == {"json", "html"} for r in cases),
                  requested_cases=limit, layout_acceptance=False, eligibility_acceptance=False,
                  sealed_evaluation=False, served=False, scientific_admission=False,
                  budgets=dict(per_response_bytes=MAX_RESPONSE, total_bytes=MAX_TOTAL, timeout_seconds=TIMEOUT, attempts=2),
                  code_sha256=file_sha(Path(__file__)), wall_seconds=time.perf_counter()-started)
    publish(output/"report.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-report-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=512)
    p.add_argument("--reuse", type=Path)
    a = p.parse_args(); result = run(a.source, a.source_report_sha256, a.output, a.limit, a.reuse)
    print(json.dumps({k: result[k] for k in ("complete", "errors", "totals", "wall_seconds")}))
    if not result["complete"]:
        raise SystemExit(2)
