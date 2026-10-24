"""Discover only the eighteen remaining source-bound image-only PMC records.

One capture read, <=36 bounded metadata requests, no asset or bucket crawl.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text.pmc_recovery import HOST, TIMEOUT, check, source_context, inputs
from text.evaluation_multiscale import file_sha
from text.source_material import material
from senses.contract import canonical
from senses.trajectory_population import publish

MAX_METADATA = 1 << 20


def fetch(url, path):
    with urlopen(Request(url, headers={"User-Agent":"MNLM source-provenance research"}), timeout=TIMEOUT) as r:
        check(r.status == 200 and urlsplit(r.geturl()).netloc == HOST, "HTTP")
        data = r.read(MAX_METADATA+1); check(len(data) <= MAX_METADATA, "METADATA_BUDGET")
        declared = r.headers.get("Content-Length")
        check(declared is None or int(declared) == len(data), "HTTP_LENGTH")
        with path.open("xb") as f: f.write(data)
        return dict(url=url,bytes=len(data),sha256=file_sha(path))


def run(cfg, out):
    start=time.perf_counter(); out.mkdir(parents=True,exist_ok=False); phases={}
    quality=Path(cfg["quality_root"]); capture=Path(cfg["source_root"])
    check(file_sha(quality/"report.json") == cfg["quality_sha256"], "QUALITY_PIN")
    q=json.loads((quality/"report.json").read_bytes())
    check(q["complete"] and q["parents"]["source"]["sha256"] == cfg["source_report_sha256"], "QUALITY_SOURCE")
    check(file_sha(quality/"recovery.jsonl") == q["files"]["recovery.jsonl"]["sha256"], "QUEUE_PIN")
    queue=[json.loads(l) for l in (quality/"recovery.jsonl").read_bytes().splitlines()]
    context=source_context(capture,cfg["source_report_sha256"])
    rows=[r for r in queue if r["domain"] == "biomedical" and r["status"] == "IMAGE_ONLY_SOURCE"]
    check(len(rows) == 19 and len({r["document_id"] for r in rows}) == 19, "QUEUE_COUNT")
    prior=Path(cfg["prior_root"]); check(file_sha(prior/"report.json") == cfg["prior_report_sha256"], "PRIOR_PIN")
    p=json.loads((prior/"report.json").read_bytes()); check(p["complete"] and p["pmcid"] == "PMC1976754", "PRIOR_COMPLETE")
    for f in p["files"]:
        check(file_sha(prior/"assets"/f["file"]) == f["sha256"], "PRIOR_ASSET")
    rows=sorted((r for r in rows if r["document_id"] != p["document_id"]),key=lambda r:r["source"]["document_id"])
    check(len(rows) == 18, "REMAINING_COUNT")
    for r in rows:
        s=context[1][r["document_id"]]; field,=s["fields"]
        check(material(field)["status"] == "IMAGE_ONLY_SOURCE" and field["sha256"] == r["text_sha256"]
              and s["source"] == r["source"], "QUEUE_SOURCE")
    publish(out/"config.json",cfg); publish(out/"selected.json",rows)
    reused={}
    if "metadata_reuse" in cfg:
        src=Path(cfg["metadata_reuse"]["root"])
        check(file_sha(src/"report.json") == cfg["metadata_reuse"]["report_sha256"], "REUSE_REPORT")
        old=json.loads((src/"report.json").read_bytes())
        check(file_sha(src/"entries.json") == old["entries_sha256"], "REUSE_ENTRIES")
        for e in json.loads((src/"entries.json").read_bytes()):
            for r in e["requests"]:
                name=e["pmcid"]+".versions.xml" if "?list-type=" in r["url"] else Path(urlsplit(r["url"]).path).name
                check(file_sha(src/name) == r["sha256"], "REUSE_FILE")
                reused[r["url"]]=(src/name,r)

    def acquire(url,path):
        if url not in reused:
            return fetch(url,path)
        source,receipt=reused[url];shutil.copyfile(source,path)
        check(file_sha(path) == receipt["sha256"], "REUSE_COPY")
        return dict(**receipt,reused=True)
    phases["selection_and_source_pins"]=time.perf_counter()-start
    tick=time.perf_counter(); entries=[]
    ns={"s":"http://s3.amazonaws.com/doc/2006-03-01/"}
    for row in rows:
        pmcid=row["source"]["document_id"]; item=dict(pmcid=pmcid,document_id=row["document_id"],requests=[])
        try:
            check(re.fullmatch(r"PMC[0-9]+",pmcid) is not None,"PMCID")
            listing=out/(pmcid+".versions.xml")
            url="https://"+HOST+"/?"+urlencode({"list-type":2,"prefix":pmcid+".","delimiter":"/"})
            item["requests"].append(acquire(url,listing)); tree=ET.fromstring(listing.read_bytes())
            check(tree.findtext("s:IsTruncated",namespaces=ns) == "false", "TRUNCATED_DISCOVERY")
            versions=[n.text for n in tree.findall("s:CommonPrefixes/s:Prefix",ns)]
            check(len(versions) == 1, "VERSION_SELECTION_REQUIRED")
            prefix=versions[0]; check(re.fullmatch(pmcid+r"\.[0-9]+/",prefix) is not None, "VERSION_PATH")
            meta=out/(prefix[:-1]+".json")
            item["requests"].append(acquire("https://"+HOST+"/"+prefix+prefix[:-1]+".json",meta))
            m=json.loads(meta.read_bytes()); field,=context[1][row["document_id"]]["fields"]
            from text.pmc_recovery import asset_inventory
            assets=asset_inventory(m,field,allow_extra=True)
            config=dict(source_root=str(capture),source_report_sha256=cfg["source_report_sha256"],
                        document_id=row["document_id"],discovery_root=str(out),versions_sha256=file_sha(listing),
                        metadata_sha256=file_sha(meta),inventory_sha256=hashlib.sha256(canonical(assets)).hexdigest(),
                        asset_count=len(assets),allow_provider_additional_media=True)
            config_path=out/(pmcid+".config.json"); publish(config_path,config)
            inputs(config_path,context)
            item.update(status="inventoried",config=config_path.name,config_sha256=file_sha(config_path),assets=assets,
                        images=sum(a["kind"]=="image" for a in assets),license=m["license_code"],historical_ocr=m["is_historical_ocr"])
        except (OSError,ValueError,KeyError,ET.ParseError) as exc:
            item.update(status="unavailable",error=str(exc))
        entries.append(item); print(json.dumps({k:item[k] for k in ("pmcid","status")}),flush=True)
    phases["metadata_requests_and_inspection"]=time.perf_counter()-tick
    tick=time.perf_counter()
    check(source_context(capture,cfg["source_report_sha256"])[0] == context[0], "SOURCE_DRIFT")
    publish(out/"entries.json",entries)
    phases["final_source_pin_and_publication"]=time.perf_counter()-tick; wall=time.perf_counter()-start
    report=dict(schema="pmc-remaining-inventory-v1",entries_sha256=file_sha(out/"entries.json"),selected_sha256=file_sha(out/"selected.json"),
                config_sha256=file_sha(out/"config.json"),articles=len(entries),available=sum(e["status"]=="inventoried" for e in entries),
                assets=sum(len(e.get("assets",[])) for e in entries),images=sum(e.get("images",0) for e in entries),
                metadata_bytes=sum(r["bytes"] for e in entries for r in e["requests"]),code_sha256=file_sha(Path(__file__)),
                asset_bytes_known=False,scientific_admission=False,sealed_evaluation=False,phases_seconds=phases,
                wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(out/"report.json",report); print(json.dumps(report))


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True,type=Path);p.add_argument("--output",required=True,type=Path)
    a=p.parse_args();run(json.loads(a.config.read_bytes()),a.output)
