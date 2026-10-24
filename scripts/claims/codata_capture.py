"""Small original-provider scientific reference snapshot; no world-truth certificate."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text.evaluation_multiscale import file_sha
from senses.trajectory_population import publish

SOURCES = {
    "2018.txt": "https://physics.nist.gov/cuu/Constants/ArchiveASCII/allascii_2018.txt",
    "2022.txt": "https://physics.nist.gov/cuu/Constants/Table/allascii.txt",
    "archive2018.html": "https://physics.nist.gov/cuu/Constants/archive2018.html",
    "overview.html": "https://physics.nist.gov/cuu/Constants/index.html",
    "history.html": "https://pml.nist.gov/cuu/Reference/versioncon.shtml",
    "methodology.html": "https://physics.nist.gov/cuu/Constants/article2022.html",
    "rights.html": "https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications",
}
LIMIT = 512 * 1024


def capture_file(url, path):
    # The public endpoint returned 403 to urllib on both hosts; curl GET returns
    # the published bytes. No credentials, browser cookies or TLS bypass.
    if path.exists():
        raise ValueError("MNLM_CODATA_CAPTURE_EXISTS")
    result = subprocess.run([
        "curl.exe" if os.name == "nt" else "curl", "--fail", "--silent", "--show-error",
        "--location", "--proto", "=https", "--proto-redir", "=https",
        "--max-redirs", "3", "--max-time", "30", "--max-filesize", str(LIMIT),
        "--output", str(path), "--write-out", "%{json}", url],
        capture_output=True, text=True, timeout=35)
    if result.returncode:
        raise ValueError("MNLM_CODATA_CAPTURE_HTTP_"+str(result.returncode))
    meta = json.loads(result.stdout)
    host = urlsplit(meta["url_effective"]).hostname
    if meta["http_code"] != 200 or not host or not host.endswith(".nist.gov"):
        raise ValueError("MNLM_CODATA_CAPTURE_ORIGIN")
    size = path.stat().st_size
    if not 0 < size <= LIMIT or size != meta["size_download"]:
        raise ValueError("MNLM_CODATA_CAPTURE_SIZE")
    return dict(url=url, final_url=meta["url_effective"], bytes=size,
                content_type=meta["content_type"], sha256=file_sha(path),
                transport="curl-https", credentials_used=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    receipts, errors, phases = {}, [], {}
    for name, url in SOURCES.items():
        tick = time.perf_counter()
        try:
            receipts[name] = dict(capture_file(url, args.output/name),
                                  retrieved_at=datetime.now(timezone.utc).isoformat())
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            errors.append(dict(file=name, error=str(exc)))
        phases[name] = time.perf_counter()-tick
    tick = time.perf_counter()
    report = dict(schema="codata-source-snapshot-v1", files=receipts, errors=errors,
                  complete=not errors, bounds=dict(files=7, file_bytes=LIMIT, total_bytes=4*1024**2),
                  attribution="NIST Standard Reference Database 121 / CODATA",
                  rights="SRD policy retained; no public-domain or redistribution license asserted",
                  usage="internal research/reference; raw-source republication not enabled",
                  world_truth_established=False, production_admission=False,
                  code_sha256=file_sha(Path(__file__)))
    phases["manifest_preparation"] = time.perf_counter()-tick
    report.update(phase_seconds=phases, wall_seconds=time.perf_counter()-start)
    publish(args.output/"manifest.json", report)
    pin = file_sha(args.output/"manifest.json")
    if not errors:
        (args.output/"COMMITTED").write_text(pin, encoding="ascii")
    print(json.dumps(dict(complete=not errors, errors=errors, manifest_sha256=pin,
                          bytes=sum(r["bytes"] for r in receipts.values()))))
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
