"""Cold asset/pixel replay and independent multiscale relation reconstruction."""
from collections import Counter
import json
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.fsv_pmc_recovery import audit as audit_article
from audit.multiscale_readback import audit_spans
from text.pmc_recovery import check, inputs
from text.evaluation_multiscale import file_sha


def read(inventory,inventory_pin,root,pin,ctx,entries,configs):
    check(file_sha(root/"report.json") == pin,"BATCH_REPORT_PIN")
    report=json.loads((root/"report.json").read_bytes())
    check(report["schema"] == "pmc-recovery-batch-v1" and report["complete"] is True
          and all(report[k] is False for k in ("scientific_admission","sealed_evaluation","ocr_correctness_adjudicated")),"BATCH_CLAIM")
    check(report["inventory_report_sha256"] == inventory_pin and report["source_report_sha256"] == ctx[0],"BATCH_LINEAGE")
    check(file_sha(root/"catalog.json") == report["catalog_sha256"],"BATCH_CATALOG_PIN")
    catalog=json.loads((root/"catalog.json").read_bytes())
    check(len(catalog)==len(entries)==18 and [r["pmcid"] for r in catalog]==[e["pmcid"] for e in entries],"BATCH_POPULATION")
    counts=Counter();articles=[]
    for row,entry in zip(catalog,entries):
        check(row["status"] == "recovered" and row["document_id"] == entry["document_id"],"BATCH_ARTICLE_STATUS")
        article=root/entry["pmcid"];check(file_sha(article/"report.json") == row["report_sha256"],"BATCH_ARTICLE_PIN")
        config=configs/(entry["pmcid"]+".json")
        cfg,source,metadata,assets=inputs(config,ctx)
        check(assets == entry["assets"],"BATCH_ARTICLE_INVENTORY")
        cold=audit_article(config,article,ctx);articles.append(cold)
        for k in ("bytes","assets","images","provider_ocr_characters"):
            check(cold[k]==row[k],"BATCH_ARTICLE_SUMMARY");counts[k]+=cold[k]
        counts["decoded_pixels"]+=cold["decoded_pixels"]
        path=root/"views"/(entry["pmcid"]+".json");check(file_sha(path)==row["views_sha256"],"BATCH_VIEW_PIN")
        v=json.loads(path.read_bytes());a=json.loads((article/"report.json").read_bytes())
        ocr=next(f for f in a["files"] if f["kind"]=="provider_ocr_text")
        check(v["document_id"]==source["id"] and v["pmcid"]==metadata["pmcid"]
              and v["ocr_file"]==ocr["file"] and v["ocr_sha256"]==ocr["sha256"],"BATCH_VIEW_LINEAGE")
        check(v["ocr_correctness_adjudicated"] is False and v["scientific_admission"] is False
              and v["sealed_evaluation"] is False and v["native_panel_requirements"]==dict(text=5,image=5)
              and v["text_origin"]=="provider-historical-ocr","BATCH_VIEW_CLAIM")
        check(v["source_group_keys"]==source["group_keys"],"BATCH_VIEW_GROUP")
        text=(article/"assets"/ocr["file"]).read_bytes().decode("utf8")
        c=audit_spans(v["graph"],text,{"text_sha256":ocr["sha256"]},source["group_keys"][-1],source_id=ocr["sha256"])
        counts.update({k:c[k] for k in ("sentence","sentence_pair","paragraph","page","nodes","edges")})
        media=[f for f in a["files"] if f["kind"]=="image"]
        stems={Path(m.group(1)).stem for m in re.finditer(r'!\[\]\(([^ )]+)',source["fields"][0]["text"])}
        check(len(v["images"])==len(media),"BATCH_IMAGE_COUNT");matched=set()
        for obs,asset in zip(v["images"],media):
            stem=Path(asset["file"]).stem;known=stem in stems
            expected=dict(document_id=source["id"],pmcid=metadata["pmcid"],file=asset["file"],sha256=asset["sha256"],
                          **asset["inspection"],group_keys=source["group_keys"],split_group=source["group_keys"][-1],
                          source_reference_stem=stem if known else None,
                          relation="source-page-reference" if known else "additional-provider-media",medium="image",
                          required_native_senses=5,admitted_native_senses=0,text_page_alignment=None,scientific_admission=False)
            check(obs==expected,"BATCH_IMAGE_BINDING")
            if known:matched.add(stem)
            counts["source_page_references" if known else "additional_images"]+=1
        check(matched==stems,"BATCH_PAGE_REFERENCE_COVERAGE")
    for k in ("bytes","assets","images","provider_ocr_characters"):
        check(counts[k]==report[k],"BATCH_TOTAL")
    check(report["articles"]==report["recovered"]==18,"BATCH_TOTAL")
    check(.9<=report["phase_coverage"]<=1.001 and abs(sum(report["phases_seconds"].values())/report["wall_seconds"]-report["phase_coverage"])<1e-10,"BATCH_TIMING")
    return dict(author_checked=True,report_sha256=pin,inventory_report_sha256=inventory_pin,articles=articles,totals=dict(counts),
                native_measurements_pending=True,scientific_admission=False,ocr_correctness_adjudicated=False)
