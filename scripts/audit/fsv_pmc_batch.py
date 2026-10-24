"""Three physical boundaries for the finite original-media recovery publication."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.pmc_batch_readback import read
from text.pmc_recovery_batch import parents, article_config
from text.pmc_recovery import asset_inventory, source_context
from text.evaluation_multiscale import file_sha
from senses.contract import canonical
from senses.trajectory_population import publish


def main():
    p=argparse.ArgumentParser()
    for n in ("inventory","publication","source","output"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--inventory-sha256",required=True);p.add_argument("--publication-sha256",required=True)
    a=p.parse_args();started=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False);phases={}
    rep,cfg,ctx,entries=parents(a.inventory,a.inventory_sha256,a.source)
    configs=a.output/"configs";configs.mkdir()
    for e in entries:article_config(a.inventory,e,configs/(e["pmcid"]+".json"),a.source)
    phases["parent_selection_and_pins"]=time.perf_counter()-started
    tick=time.perf_counter()
    before={str(f.relative_to(a.publication)):file_sha(f) for f in a.publication.rglob('*') if f.is_file()}
    publish(a.output/"before.json",before)
    cold=read(a.inventory,a.inventory_sha256,a.publication,a.publication_sha256,ctx,entries,configs)
    publish(a.output/"cold-readback.json",cold)
    phases["full_asset_pixel_and_span_readback"]=time.perf_counter()-tick
    checks=[dict(name="B1/B2/B3 all original assets and native/text views cold-read",passed=True)]
    tick=time.perf_counter();shadow=a.output/"owned-shadow";shutil.copytree(a.publication,shadow)
    controls=a.output/"controls";controls.mkdir()
    report=json.loads((shadow/"report.json").read_bytes());catalog=json.loads((shadow/"catalog.json").read_bytes())
    first=entries[0]["pmcid"]

    def trial(name,code,operation):
        try:operation();observed="accepted"
        except (ValueError,FileNotFoundError) as exc:observed=str(exc)
        checks.append(dict(name=name,expected=code,observed=observed,passed=observed==code))

    def fault(name,relative,mutation,code,kind="view"):
        path=shadow/relative;original=path.read_bytes();value=json.loads(original)
        mutation(value);changed=canonical(value);path.write_bytes(changed)
        cat=copy.deepcopy(catalog);r=copy.deepcopy(report)
        if kind=="view":
            next(v for v in cat if v["pmcid"]==path.stem)["views_sha256"]=file_sha(path)
            (shadow/"catalog.json").write_bytes(canonical(cat));r["catalog_sha256"]=file_sha(shadow/"catalog.json")
            (shadow/"report.json").write_bytes(canonical(r))
        elif kind=="catalog":
            r["catalog_sha256"]=file_sha(path);(shadow/"report.json").write_bytes(canonical(r))
        publish(controls/(name+".json"),dict(relative=relative,mutated=value,mutant_sha256=file_sha(path)))
        trial(name,code,lambda:read(a.inventory,a.inventory_sha256,shadow,file_sha(shadow/"report.json"),ctx,entries,configs))
        path.write_bytes(original)
        for f in ("catalog.json","report.json"):(shadow/f).write_bytes((a.publication/f).read_bytes())

    fault("B1_false_source_binding","report.json",lambda r:r.update(source_report_sha256="0"*64),"MNLM_PMC_RECOVERY_BATCH_LINEAGE","report")
    fault("B1_missing_article","catalog.json",lambda r:r.pop(),"MNLM_PMC_RECOVERY_BATCH_POPULATION","catalog")
    config=configs/(first+".json");saved=config.read_bytes();bad=json.loads(saved);bad["metadata_sha256"]="0"*64;config.write_bytes(canonical(bad))
    publish(controls/"B1_wrong_metadata_pin.json",bad)
    trial("B1_wrong_metadata_pin","MNLM_PMC_RECOVERY_METADATA_PIN",lambda:read(a.inventory,a.inventory_sha256,shadow,file_sha(shadow/"report.json"),ctx,entries,configs))
    config.write_bytes(saved)
    metadata=json.loads((shadow/first/"metadata.json").read_bytes());source=ctx[1][entries[0]["document_id"]]
    metadata["media_urls"].pop(0);publish(controls/"B1_missing_source_page.json",metadata)
    trial("B1_missing_source_page","MNLM_PMC_RECOVERY_IMAGE_CORRESPONDENCE",lambda:asset_inventory(metadata,source["fields"][0],allow_extra=True))
    ar=json.loads((shadow/first/"report.json").read_bytes())
    for kind in ("image","provider_ocr_text"):
        asset=next(f for f in ar["files"] if f["kind"]==kind);path=shadow/first/"assets"/asset["file"]
        data=path.read_bytes();path.write_bytes(data[:-1]+bytes([data[-1]^1]))
        publish(controls/("B2_changed_"+kind+".json"),dict(file=asset["file"],mutant_sha256=file_sha(path)))
        trial("B2_changed_"+kind,"MNLM_PMC_RECOVERY_READBACK_ASSET_DIGEST",lambda:read(a.inventory,a.inventory_sha256,shadow,file_sha(shadow/"report.json"),ctx,entries,configs))
        path.write_bytes(data)
    view="views/"+first+".json"
    fault("B3_false_OCR_truth",view,lambda r:r.update(ocr_correctness_adjudicated=True),"MNLM_PMC_RECOVERY_BATCH_VIEW_CLAIM")
    fault("B3_image_floor_cannot_be_text",view,lambda r:r["native_panel_requirements"].update(image=0),"MNLM_PMC_RECOVERY_BATCH_VIEW_CLAIM")
    fault("B3_fabricated_image_senses",view,lambda r:r["images"][0].update(admitted_native_senses=5),"MNLM_PMC_RECOVERY_BATCH_IMAGE_BINDING")
    fault("B3_false_page_alignment",view,lambda r:r["images"][0].update(text_page_alignment=[0,100]),"MNLM_PMC_RECOVERY_BATCH_IMAGE_BINDING")
    fault("B3_missing_association",view,lambda r:r["graph"]["edges"].pop(),"MNLM_MULTI_READBACK_RELATIONS")
    fault("B3_changed_span",view,lambda r:r["graph"]["nodes"][0].update(end=1),"MNLM_MULTI_READBACK_ID")
    fault("B3_false_group",view,lambda r:r.update(source_group_keys=["other-document"]),"MNLM_PMC_RECOVERY_BATCH_VIEW_GROUP")
    fault("B3_inflated_image_total","report.json",lambda r:r.update(images=131),"MNLM_PMC_RECOVERY_BATCH_TOTAL","report")
    fault("B3_false_batch_admission","report.json",lambda r:r.update(scientific_admission=True),"MNLM_PMC_RECOVERY_BATCH_CLAIM","report")
    phases["persisted_negative_controls"]=time.perf_counter()-tick
    tick=time.perf_counter()
    after={str(f.relative_to(a.publication)):file_sha(f) for f in a.publication.rglob('*') if f.is_file()}
    publish(a.output/"after.json",after)
    checks.append(dict(name="B1 originals unchanged and restored shadow reads identically",passed=before==after and
                       read(a.inventory,a.inventory_sha256,shadow,a.publication_sha256,ctx,entries,configs)==cold))
    source_context(a.source,ctx[0])
    phases["final_drift_and_restoration_readback"]=time.perf_counter()-tick;wall=time.perf_counter()-started
    result=dict(passed=all(r["passed"] for r in checks),checks=checks,author_checked=True,scientific_admission=False,
                source_report_sha256=ctx[0],publication_sha256=a.publication_sha256,
                phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(a.output/"fsv.json",result)
    print(json.dumps(dict(passed=result["passed"],checks=len(checks),failed=[r for r in checks if not r["passed"]],
                         totals=cold["totals"],wall_seconds=wall,phase_coverage=result["phase_coverage"])))
    return 0 if result["passed"] else 2


if __name__=="__main__":sys.exit(main())
