"""Bounded single-article recovery from NLM's native, versioned asset manifest.

This retains originals and provider OCR as separate views. No OCR is generated,
no sense is admitted, and neither original capture nor evaluation groups change.
PC-07/13/28: named assets only, one process, create-only generation owned by #3.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit, parse_qs
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text.evaluation_multiscale import file_sha
from text.source_material import IMAGE, material
from senses.trajectory_population import publish
from senses.contract import canonical

HOST = "pmc-oa-opendata.s3.amazonaws.com"
MAX_OBJECT, MAX_TOTAL, TIMEOUT = 16 << 20, 128 << 20, 30


def check(ok, code):
    if not ok: raise ValueError("MNLM_PMC_RECOVERY_"+code)


def source_context(root, pin):
    """Hash/parse the named source once per batch; never once per asset."""
    check(file_sha(root/"report.json") == pin, "SOURCE_REPORT")
    capture = json.loads((root/"report.json").read_bytes()); meta = capture["sources"]["biomedical"]
    check(file_sha(root/meta["file"]) == meta["sha256"], "SOURCE_FILE")
    with (root/meta["file"]).open("rb") as f:
        rows = [json.loads(line) for line in f]
    records = {r["id"]:r for r in rows}
    check(len(records) == len(rows), "DUPLICATE_DOCUMENT")
    return pin, records


def inputs(config, context=None):
    cfg = json.loads(config.read_bytes()); root = Path(cfg["source_root"])
    context = context or source_context(root, cfg["source_report_sha256"])
    check(context[0] == cfg["source_report_sha256"], "SOURCE_REPORT")
    check(cfg["document_id"] in context[1], "DOCUMENT")
    source = context[1][cfg["document_id"]]; field, = source["fields"]
    check(material(field)["status"] == "IMAGE_ONLY_SOURCE", "NOT_IMAGE_ONLY")
    discovery = Path(cfg["discovery_root"]); pmcid = source["source"]["document_id"]
    check(re.fullmatch(r"PMC[0-9]+", pmcid) is not None, "PMCID")
    listing = discovery/(pmcid+".versions.xml")
    check(file_sha(listing) == cfg["versions_sha256"], "VERSIONS_PIN")
    tree = ET.fromstring(listing.read_bytes()); ns = {"s":"http://s3.amazonaws.com/doc/2006-03-01/"}
    check(tree.findtext("s:IsTruncated", namespaces=ns) == "false", "TRUNCATED_DISCOVERY")
    versions = [n.text for n in tree.findall("s:CommonPrefixes/s:Prefix", ns)]
    check(len(versions) == 1, "VERSION_SELECTION_REQUIRED")
    prefix = versions[0]; check(re.fullmatch(pmcid+r"\.[0-9]+/", prefix) is not None, "VERSION_PATH")
    path = discovery/(prefix[:-1]+".json")
    check(file_sha(path) == cfg["metadata_sha256"], "METADATA_PIN")
    metadata = json.loads(path.read_bytes())
    check(metadata["pmcid"] == pmcid and prefix == f"{pmcid}.{metadata['version']}/"
          and metadata["is_historical_ocr"] is True and metadata["license_code"] == "CC BY"
          and metadata["is_retracted"] is False, "METADATA_IDENTITY")
    inventory = asset_inventory(metadata, field, cfg.get("allow_provider_additional_media", False))
    if "inventory_sha256" in cfg:
        check(hashlib.sha256(canonical(inventory)).hexdigest() == cfg["inventory_sha256"]
              and len(inventory) == cfg["asset_count"], "INVENTORY_PIN")
    else:
        check(len(inventory) == 7 and sum(r["kind"] == "image" for r in inventory) == 4,
              "DECLARED_SEVEN_ASSET_SCOPE")
    return cfg, source, metadata, inventory


def asset_inventory(metadata, field, allow_extra=False):
    prefix = f"{metadata['pmcid']}.{metadata['version']}/"
    inventory = []
    for kind, urls in (("xml", [metadata["xml_url"]]), ("provider_ocr_text", [metadata["text_url"]]),
                       ("pdf", [metadata["pdf_url"]]), ("image", metadata["media_urls"])):
        for url in urls:
            parsed = urlsplit(url); name = parsed.path.removeprefix("/"+prefix)
            md5 = parse_qs(parsed.query).get("md5", [])
            check(parsed.scheme == "s3" and parsed.netloc == "pmc-oa-opendata"
                  and parsed.path == "/"+prefix+name and Path(name).name == name and "\\" not in name
                  and len(md5) == 1 and re.fullmatch(r"[0-9a-f]{32}", md5[0]), "ASSET_PATH")
            inventory.append(dict(kind=kind, file=name, md5=md5[0], url="https://"+HOST+parsed.path))
    check(len({r["file"] for r in inventory}) == len(inventory), "DUPLICATE_ASSET")
    stems = {Path(m.group(1)).stem for m in re.finditer(r'!\[\]\(([^ )]+)', field["text"])}
    media = {Path(r["file"]).stem for r in inventory if r["kind"] == "image"}
    check(bool(stems) and (stems <= media if allow_extra else stems == media), "IMAGE_CORRESPONDENCE")
    return inventory


def inspect(path, kind, pmcid):
    if kind == "image":
        with Image.open(path) as image:
            image.load()
            return dict(format=image.format, mode=image.mode, width=image.width, height=image.height,
                        pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest())
    if kind == "provider_ocr_text":
        text = path.read_text(encoding="utf-8"); check(bool(text.strip()), "EMPTY_OCR")
        return dict(characters=len(text), utf8_bytes=len(text.encode()), human_correctness_audited=False)
    if kind == "pdf":
        with path.open("rb") as f: check(f.read(5) == b"%PDF-", "PDF_HEADER")
        return dict(pdf_header=True)
    root = ET.fromstring(path.read_bytes())
    identifiers = [(n.get("pub-id-type"), n.text) for n in root.findall(".//article-id")]
    check(any(k in ("pmc", "pmcid") and str(v).removeprefix("PMC") == pmcid.removeprefix("PMC")
              for k,v in identifiers), "XML_PM CID".replace(" ", ""))
    return dict(article_ids=identifiers, body_paragraphs=len(root.findall("./body//p")))


def run(config, output, context=None):
    started = time.perf_counter(); check(not output.exists(), "EXISTS")
    cfg, source, metadata, inventory = inputs(config, context)
    output.mkdir(parents=True); (output/"assets").mkdir()
    publish(output/"source.json", source); publish(output/"metadata.json", metadata)
    files, errors, total, phases = [], [], 0, {}
    phases["input_seconds"] = time.perf_counter()-started; fetch_time = decode_time = 0
    for asset in inventory:
        path = output/"assets"/asset["file"]
        try:
            start = time.perf_counter(); n = 0
            request = Request(asset["url"], headers={"User-Agent":"MNLM source-provenance research"})
            with urlopen(request, timeout=TIMEOUT) as response, path.open("xb") as out:
                check(response.status == 200 and urlsplit(response.geturl()).netloc == HOST, "HTTP")
                declared = response.headers.get("Content-Length")
                check(declared is None or 0 < int(declared) <= MAX_OBJECT, "OBJECT_BUDGET")
                while block := response.read(65536):
                    n += len(block); total += len(block)
                    check(n <= MAX_OBJECT and total <= MAX_TOTAL, "BYTE_BUDGET"); out.write(block)
                check(declared is None or int(declared) == n, "HTTP_LENGTH")
                headers = {k:response.headers[k] for k in ("Content-Type","Content-Length","ETag","Last-Modified") if k in response.headers}
            fetch_time += time.perf_counter()-start; start = time.perf_counter()
            raw = path.read_bytes(); check(hashlib.md5(raw).hexdigest() == asset["md5"], "PROVIDER_DIGEST")
            details = inspect(path, asset["kind"], metadata["pmcid"])
            files.append(dict(**asset, bytes=n, sha256=hashlib.sha256(raw).hexdigest(), headers=headers, inspection=details))
            decode_time += time.perf_counter()-start
        except (OSError, ValueError, ET.ParseError) as exc:
            errors.append(dict(file=asset["file"], error=str(exc)))
    phases.update(fetch_seconds=fetch_time, validation_seconds=decode_time)
    wall = time.perf_counter()-started; phases["other_seconds"] = wall-sum(phases.values()); phases["wall_seconds"] = wall
    report = dict(schema="pmc-native-recovery-v1", source_report_sha256=cfg["source_report_sha256"],
                  document_id=source["id"], source_record_sha256=source["source"]["record_sha256"],
                  pmcid=metadata["pmcid"], metadata_input_sha256=cfg["metadata_sha256"],
                  versions_sha256=cfg["versions_sha256"], metadata_sha256=file_sha(output/"metadata.json"),
                  source_sha256=file_sha(output/"source.json"), files=files, errors=errors, received_bytes=total,
                  complete=not errors and len(files) == len(inventory), scientific_admission=False, sealed_evaluation=False,
                  native_panel_requirements=dict(text=5,image=5), text_origin="provider historical OCR, unadjudicated correctness",
                  original_source_modified=False, scope="single-article-development-recovery",
                  additional_media_alignment="article-only;no-page-position-inferred",
                  budgets=dict(object_bytes=MAX_OBJECT,total_bytes=MAX_TOTAL,timeout_seconds=TIMEOUT),
                  code_sha256=file_sha(Path(__file__)), timing=phases)
    publish(output/"report.json", report); return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    a = p.parse_args(); r = run(a.config,a.output)
    print(json.dumps({k:r[k] for k in ("complete","errors","received_bytes","timing")}))
    if not r["complete"]: raise SystemExit(2)
