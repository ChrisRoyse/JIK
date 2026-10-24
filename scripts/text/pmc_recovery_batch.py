"""Recover a pinned finite PMC inventory and publish linked native/text views."""
import argparse
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from text.pmc_recovery import run as article_run, source_context, check, inputs
from text.evaluation_multiscale import file_sha
from text.multiscale_spans import build
from senses.contract import canonical
from senses.trajectory_population import publish


def parents(inventory,pin,source=None):
    check(file_sha(inventory/"report.json") == pin,"BATCH_INVENTORY_PIN")
    rep=json.loads((inventory/"report.json").read_bytes())
    for name in ("entries","selected","config"):
        check(file_sha(inventory/(name+".json")) == rep[name+"_sha256"],"BATCH_PARENT_PIN")
    cfg=json.loads((inventory/"config.json").read_bytes())
    ctx=source_context(source or Path(cfg["source_root"]),cfg["source_report_sha256"])
    entries=json.loads((inventory/"entries.json").read_bytes())
    selected=json.loads((inventory/"selected.json").read_bytes())
    check(len(entries) == len(selected) == rep["articles"] == 18,"BATCH_COUNT")
    check([(e["document_id"],e["pmcid"]) for e in entries] ==
          [(r["document_id"],r["source"]["document_id"]) for r in selected],"BATCH_SELECTION")
    for e in entries:
        if e["status"] == "inventoried":
            check(file_sha(inventory/e["config"]) == e["config_sha256"],"BATCH_CONFIG_PIN")
    return rep,cfg,ctx,entries


def article_config(inventory,entry,destination,source=None):
    """Relocate physical parents without changing pinned metadata or source identity."""
    cfg=json.loads((inventory/entry["config"]).read_bytes())
    cfg["discovery_root"]=str(inventory)
    if source: cfg["source_root"]=str(source)
    publish(destination,cfg)
    return destination


def views(source,root):
    report=json.loads((root/"report.json").read_bytes());check(report["complete"],"BATCH_INCOMPLETE_ARTICLE")
    ocr=next(f for f in report["files"] if f["kind"] == "provider_ocr_text")
    text=(root/"assets"/ocr["file"]).read_text(encoding="utf8")
    group=source["group_keys"][-1]
    graph=build(text,ocr["sha256"],group)
    stems={Path(m.group(1)).stem for m in re.finditer(r'!\[\]\(([^ )]+)',source["fields"][0]["text"])}
    images=[]
    for f in report["files"]:
        if f["kind"] != "image":continue
        stem=Path(f["file"]).stem
        images.append(dict(document_id=source["id"],pmcid=report["pmcid"],file=f["file"],sha256=f["sha256"],
                           **f["inspection"],group_keys=source["group_keys"],split_group=group,
                           source_reference_stem=stem if stem in stems else None,
                           relation="source-page-reference" if stem in stems else "additional-provider-media",
                           medium="image",required_native_senses=5,admitted_native_senses=0,
                           text_page_alignment=None,scientific_admission=False))
    return dict(document_id=source["id"],pmcid=report["pmcid"],ocr_file=ocr["file"],ocr_sha256=ocr["sha256"],
                text_origin="provider-historical-ocr",ocr_correctness_adjudicated=False,
                source_group_keys=source["group_keys"],graph=graph,images=images,
                native_panel_requirements=dict(text=5,image=5),scientific_admission=False,sealed_evaluation=False)


def run(inventory,pin,out):
    started=time.perf_counter();out.mkdir(parents=True,exist_ok=False);phases={}
    rep,cfg,ctx,entries=parents(inventory,pin)
    (out/"configs").mkdir();(out/"views").mkdir()
    phases["parent_pin_and_source_read"]=time.perf_counter()-started
    tick=time.perf_counter();results=[]
    for entry in entries:
        row=dict(pmcid=entry["pmcid"],document_id=entry["document_id"])
        try:
            check(entry["status"] == "inventoried","BATCH_UNAVAILABLE_INVENTORY")
            path=article_config(inventory,entry,out/"configs"/(entry["pmcid"]+".json"))
            root=out/entry["pmcid"];a=article_run(path,root,ctx)
            check(a["complete"],"BATCH_INCOMPLETE_ARTICLE")
            v=views(ctx[1][entry["document_id"]],root);vpath=out/"views"/(entry["pmcid"]+".json")
            publish(vpath,v)
            row.update(status="recovered",report_sha256=file_sha(root/"report.json"),views_sha256=file_sha(vpath),
                       bytes=a["received_bytes"],assets=len(a["files"]),images=len(v["images"]),
                       provider_ocr_characters=next(f["inspection"]["characters"] for f in a["files"] if f["kind"]=="provider_ocr_text"))
        except (OSError,ValueError,KeyError) as exc:
            row.update(status="unavailable",error=str(exc))
        results.append(row);print(json.dumps(row),flush=True)
    phases["article_fetch_validation_and_multiscale"]=time.perf_counter()-tick
    tick=time.perf_counter();source_context(Path(cfg["source_root"]),ctx[0]);publish(out/"catalog.json",results)
    check(file_sha(inventory/"report.json") == pin,"BATCH_INVENTORY_DRIFT")
    phases["final_source_pin_and_catalog"]=time.perf_counter()-tick;wall=time.perf_counter()-started
    report=dict(schema="pmc-recovery-batch-v1",inventory_report_sha256=pin,source_report_sha256=ctx[0],
                catalog_sha256=file_sha(out/"catalog.json"),articles=len(results),recovered=sum(r["status"]=="recovered" for r in results),
                complete=all(r["status"]=="recovered" for r in results),bytes=sum(r.get("bytes",0) for r in results),
                assets=sum(r.get("assets",0) for r in results),images=sum(r.get("images",0) for r in results),
                provider_ocr_characters=sum(r.get("provider_ocr_characters",0) for r in results),
                code_sha256=file_sha(Path(__file__)),phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,
                scientific_admission=False,sealed_evaluation=False,ocr_correctness_adjudicated=False)
    publish(out/"report.json",report);print(json.dumps(report));return report


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    for n in ("inventory","output"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--inventory-sha256",required=True)
    a=p.parse_args();r=run(a.inventory,a.inventory_sha256,a.output)
    if not r["complete"]:sys.exit(2)
