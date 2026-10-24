"""Cold digest, source identity and native-image readback of actual PMC recovery."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text.pmc_recovery import inputs, check, inspect
from text.evaluation_multiscale import file_sha
from senses.trajectory_population import publish


def audit(config, publication, context=None):
    cfg, source, metadata, expected = inputs(config, context)
    report = json.loads((publication/"report.json").read_bytes())
    check(report["complete"] and not report["errors"] and report["scientific_admission"] is False
          and report["sealed_evaluation"] is False and report["original_source_modified"] is False
          and report["native_panel_requirements"] == dict(text=5,image=5), "READBACK_CLAIMS")
    check(report["document_id"] == source["id"] and report["source_record_sha256"] == source["source"]["record_sha256"]
          and report["source_report_sha256"] == cfg["source_report_sha256"]
          and report["metadata_input_sha256"] == cfg["metadata_sha256"]
          and report["versions_sha256"] == cfg["versions_sha256"], "READBACK_PARENT")
    check(json.loads((publication/"source.json").read_bytes()) == source
          and file_sha(publication/"source.json") == report["source_sha256"], "READBACK_SOURCE")
    check(json.loads((publication/"metadata.json").read_bytes()) == metadata
          and file_sha(publication/"metadata.json") == report["metadata_sha256"], "READBACK_METADATA")
    check(len(expected) == len(report["files"]), "READBACK_COUNT")
    total, pixels, characters = 0, 0, 0
    for contract, actual in zip(expected, report["files"]):
        check(all(actual[k] == contract[k] for k in contract), "READBACK_INVENTORY")
        path = publication/"assets"/actual["file"]
        check(path.is_file(), "READBACK_MISSING_ASSET")
        raw = path.read_bytes(); total += len(raw)
        check(len(raw) == actual["bytes"] and hashlib.md5(raw).hexdigest() == contract["md5"]
              and hashlib.sha256(raw).hexdigest() == actual["sha256"], "READBACK_ASSET_DIGEST")
        detail = actual["inspection"]
        if actual["kind"] == "image":
            with Image.open(path) as image: image.verify()
            with Image.open(path) as image:
                pixels += image.width*image.height
                check(detail == dict(format=image.format,mode=image.mode,width=image.width,height=image.height,
                      pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest()), "READBACK_PIXELS")
        elif actual["kind"] == "provider_ocr_text":
            text = raw.decode("utf-8"); characters = len(text)
            check(detail == dict(characters=len(text),utf8_bytes=len(raw),human_correctness_audited=False), "READBACK_OCR")
        elif actual["kind"] == "xml":
            root = ET.fromstring(raw)
            ids = [[n.get("pub-id-type"),n.text] for n in root.iter("article-id")]
            check(["pmcid",metadata["pmcid"]] in ids and ids == detail["article_ids"], "READBACK_XML_ID")
        else:
            check(actual["kind"] == "pdf" and raw.startswith(b"%PDF-"), "READBACK_PDF")
    check(total == report["received_bytes"], "READBACK_BYTES")
    return dict(author_checked=True,pmcid=metadata["pmcid"],assets=len(expected),
                images=sum(a["kind"] == "image" for a in expected),decoded_pixels=pixels,
                provider_ocr_characters=characters,source_characters=source["fields"][0]["characters"],
                bytes=total,report_sha256=file_sha(publication/"report.json"),original_capture_unchanged=True,
                native_measurements_pending=True,scientific_admission=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("config","publication","output"): p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    before={str(p):file_sha(p) for p in a.publication.rglob('*') if p.is_file()}
    publish(a.output/"before.json",before); cold=audit(a.config,a.publication)
    publish(a.output/"readback.json",cold); checks=[]
    def check_result(name,ok,**details): checks.append(dict(name=name,passed=bool(ok),**details))
    def refuses(name,code,fn):
        observed=None
        try: fn()
        except ValueError as exc: observed=str(exc)
        check_result(name,observed is not None and code in observed,expected=code,observed=observed)
    check_result("B1 actual article source and provider manifest agree",True,**cold)
    shadow=a.output/"owned-shadow"; shutil.copytree(a.publication,shadow)
    report=json.loads((shadow/"report.json").read_bytes())
    image=next(r for r in report["files"] if r["kind"]=="image")
    path=shadow/"assets"/image["file"]; raw=path.read_bytes(); path.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
    refuses("B2 modified native image refuses","READBACK_ASSET_DIGEST",lambda:audit(a.config,shadow))
    path.write_bytes(raw)
    ocr=next(r for r in report["files"] if r["kind"]=="provider_ocr_text")
    path=shadow/"assets"/ocr["file"]; raw=path.read_bytes(); path.write_bytes(raw+b' changed')
    refuses("B2 modified provider OCR refuses","READBACK_ASSET_DIGEST",lambda:audit(a.config,shadow))
    path.write_bytes(raw)
    bad=copy.deepcopy(report); bad["native_panel_requirements"]["image"]=0
    (shadow/"report.json").write_text(json.dumps(bad))
    refuses("B3 OCR cannot substitute for image panel","READBACK_CLAIMS",lambda:audit(a.config,shadow))
    bad=copy.deepcopy(report); bad["scientific_admission"]=True
    (shadow/"report.json").write_text(json.dumps(bad))
    refuses("B3 asset recovery cannot admit meaning","READBACK_CLAIMS",lambda:audit(a.config,shadow))
    (shadow/"report.json").write_bytes((a.publication/"report.json").read_bytes())
    xml=next(r for r in report["files"] if r["kind"]=="xml")
    path=shadow/"assets"/xml["file"]; raw=path.read_bytes()
    path.write_bytes(raw.replace(report["pmcid"].encode(),b'PMC0000000'))
    refuses("B1 wrong article XML refuses","XML_PMCID",lambda:inspect(path,"xml",report["pmcid"]))
    path.write_bytes(raw)
    check_result("B2 restored shadow replays",audit(a.config,shadow)==cold)
    # A source-linked text view is now possible; native images stay separate.
    from text.multiscale_spans import build
    text=(a.publication/"assets"/ocr["file"]).read_text(encoding="utf-8")
    graph=build(text,ocr["sha256"],"document:"+report["document_id"])
    publish(a.output/"provider-ocr-multiscale.json",graph)
    counts={level:sum(n["level"]==level for n in graph["nodes"]) for level in ("sentence","sentence_pair","paragraph","page")}
    check_result("B3 recovered OCR views retain separate native image requirement",counts["page"]==0,levels=counts)
    after={str(p):file_sha(p) for p in a.publication.rglob('*') if p.is_file()}
    publish(a.output/"after.json",after); check_result("B1 recovered originals unchanged",before==after)
    publish(a.output/"fsv.json",dict(checks=checks,passed=all(r["passed"] for r in checks),author_checked=True))
    print(json.dumps(dict(passed=all(r["passed"] for r in checks),checks=len(checks),readback=cold,levels=counts)))
    if not all(r["passed"] for r in checks): raise SystemExit(2)


if __name__ == "__main__": main()
