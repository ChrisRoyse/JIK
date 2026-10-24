"""Physical source/split/window boundaries for the source-backed readout population."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.anchor_population_readback import read,lines
from text.evaluation_multiscale import file_sha
from senses.contract import canonical
from senses.trajectory_population import publish


def manifest(root):
    return {p.relative_to(root).as_posix():file_sha(p) for p in root.rglob('*') if p.is_file()}


def main():
    p=argparse.ArgumentParser()
    for name in ("population","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--report-sha256",required=True);a=p.parse_args()
    started=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False);phases={}
    before=manifest(a.population);publish(a.output/"before.json",before)
    cold=read(a.population,a.report_sha256);publish(a.output/"cold-readback.json",cold)
    shadow=a.output/"owned-shadow";shutil.copytree(a.population,shadow)
    controls=a.output/"controls";controls.mkdir()
    phases["source_cold_read_and_owned_copy"]=time.perf_counter()-started
    tick=time.perf_counter();checks=[dict(name="B1/B2/B3 source split and all native windows cold read",passed=True)]
    report=json.loads((shadow/"report.json").read_bytes())

    def fault(name,relative,change,code,jsonl=False):
        path=shadow/relative;original=path.read_bytes();rep=copy.deepcopy(report)
        data=lines(path) if jsonl else json.loads(original)
        change(data);raw=b''.join(canonical(r)+b'\n' for r in data) if jsonl else canonical(data)
        path.write_bytes(raw)
        if relative!="report.json":
            rep["files"][relative]=dict(bytes=len(raw),sha256=file_sha(path))
            (shadow/"report.json").write_bytes(canonical(rep))
        publish(controls/(name+".json"),dict(relative=relative,mutated_sha256=file_sha(path),mutated=data))
        try:read(shadow,file_sha(shadow/"report.json"));observed="accepted"
        except ValueError as exc:observed=str(exc)
        finally:
            path.write_bytes(original);(shadow/"report.json").write_bytes((a.population/"report.json").read_bytes())
        checks.append(dict(name=name,expected=code,observed=observed,passed=observed==code))

    prefix="MNLM_ANCHOR_POP_READ_"
    fault("B1_changed_source_case","rows.jsonl",lambda r:r[0].update(entity="Q0"),prefix+"CASE",True)
    fault("B1_label_sibling_substitution","rows.jsonl",lambda r:r[0].update(positive_classes=[5]),prefix+"TARGET_SET",True)
    fault("B1_source_size_inflation","report.json",lambda r:r.update(source_characters=r["source_characters"]+1),prefix+"SOURCE_SIZE")
    fault("B2_dropped_chunk","rows.jsonl",lambda r:r[0]["chunks"].pop(),prefix+"CHUNK_OWNERSHIP",True)
    fault("B2_changed_split","rows.jsonl",lambda r:r[0].update(split="other"),prefix+"SPLIT_LEAKAGE",True)
    fault("B2_false_overlap_group","rows.jsonl",lambda r:r[0].update(group="0"*64),prefix+"SPLIT_LEAKAGE",True)
    fault("B3_false_native_count","report.json",lambda r:r["lanes"][0].update(windows=1),prefix+"NATIVE_COUNT")
    fault("B3_wrong_instrument","report.json",lambda r:r["lanes"][0].update(lens_id="0"*64),prefix+"LENS_ID")
    for key in ("scientific_admission","sealed_evaluation","source_time_composition_separated","encoder_training_exposure_certified"):
        fault("B3_false_"+key,"report.json",lambda r,k=key:r.update({k:True}),prefix+"CLAIM")
    relative="lanes/"+report["lanes"][-1]["name"]+"/vectors.npy";path=shadow/relative
    raw=path.read_bytes();path.write_bytes(raw[:-1]+bytes([raw[-1]^1]));rep=copy.deepcopy(report)
    rep["files"][relative]["sha256"]=file_sha(path);(shadow/"report.json").write_bytes(canonical(rep))
    publish(controls/"B3_changed_native_bytes.json",dict(relative=relative,mutated_sha256=file_sha(path)))
    try:read(shadow,file_sha(shadow/"report.json"));observed="accepted"
    except ValueError as exc:observed=str(exc)
    finally:path.write_bytes(raw);(shadow/"report.json").write_bytes((a.population/"report.json").read_bytes())
    checks.append(dict(name="B3 changed native bytes despite repinned outer manifest",expected=prefix+"NATIVE_PIN",observed=observed,passed=observed==prefix+"NATIVE_PIN"))
    absent=shadow/relative;raw=absent.read_bytes();absent.unlink()
    try:read(shadow,a.report_sha256);observed="accepted"
    except ValueError as exc:observed=str(exc)
    finally:absent.write_bytes(raw)
    publish(controls/"B3_missing_native_file.json",dict(relative=relative,observed=observed))
    checks.append(dict(name="B3 missing native file",expected=prefix+"FILE_SET",observed=observed,passed=observed==prefix+"FILE_SET"))
    phases["persisted_negative_controls"]=time.perf_counter()-tick;tick=time.perf_counter()
    after=manifest(a.population);publish(a.output/"after.json",after)
    checks.append(dict(name="B1 originals and exact restored shadow unchanged",passed=before==after==manifest(shadow) and read(shadow,a.report_sha256)==cold))
    phases["drift_and_restoration_readback"]=time.perf_counter()-tick;wall=time.perf_counter()-started
    result=dict(author_checked=True,passed=all(c["passed"] for c in checks),checks=checks,population_sha256=a.report_sha256,
                scientific_admission=False,phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(a.output/"fsv.json",result)
    print(json.dumps(dict(passed=result["passed"],checks=len(checks),failed=[c for c in checks if not c["passed"]],wall_seconds=wall)))
    return 0 if result["passed"] else 2


if __name__=="__main__":sys.exit(main())
