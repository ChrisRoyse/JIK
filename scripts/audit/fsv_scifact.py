"""Real-source FSV: archive, atomic/rationale bindings and protected consumption."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.scifact_readback import read, raw_source, jsonl
from senses.contract import canonical
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha
from text.scifact_relations import Inputs


def inventory(root):
    return {f.name:file_sha(f) for f in sorted(root.iterdir()) if f.is_file()}


def main():
    p = argparse.ArgumentParser()
    for k in ("source","publication","output"):p.add_argument("--"+k,type=Path,required=True)
    for k in ("source-sha256","publication-sha256"):p.add_argument("--"+k,required=True)
    a = p.parse_args(); started = time.perf_counter(); a.output.mkdir(parents=True,exist_ok=False); phases = {}
    before = dict(source=inventory(a.source),publication=inventory(a.publication)); publish(a.output/"before.json",before)
    context = raw_source(a.source,a.source_sha256)
    cold = read(a.source,a.source_sha256,a.publication,a.publication_sha256,context)
    publish(a.output/"cold-readback.json",cold)
    checks = [dict(name="B1/B2 all source archive, views, groups and rationale sets independently recounted",passed=True)]
    phases["cold_source_and_projection"] = time.perf_counter()-started; tick = time.perf_counter()
    shadow = a.output/"owned-shadow"; shutil.copytree(a.publication,shadow)
    controls = a.output/"controls"; controls.mkdir()
    report_bytes = (shadow/"report.json").read_bytes(); report = json.loads(report_bytes)

    def trial(name,code,fn):
        try:fn(); observed="accepted"
        except (ValueError,FileNotFoundError) as exc:observed=str(exc)
        checks.append(dict(name=name,expected=code,observed=observed,passed=observed==code))

    def fault(name,filename,mutation,code,lines=True):
        path = shadow/filename; original = path.read_bytes()
        value = jsonl(path) if lines else json.loads(original); mutation(value)
        changed = b"".join(canonical(v)+b"\n" for v in value) if lines else canonical(value)
        path.write_bytes(changed); r = copy.deepcopy(report)
        if filename != "report.json":
            r["files"][filename] = dict(bytes=len(changed),sha256=file_sha(path))
            (shadow/"report.json").write_bytes(canonical(r))
        publish(controls/(name+".json"),dict(file=filename,mutant=value,sha256=file_sha(path)))
        trial(name,code,lambda:read(a.source,a.source_sha256,shadow,file_sha(shadow/"report.json"),context))
        path.write_bytes(original); (shadow/"report.json").write_bytes(report_bytes)

    trial("B1 wrong source pin","MNLM_SCIFACT_READBACK_SOURCE_PIN",
          lambda:raw_source(a.source,"0"*64))
    fault("B1_false_parent","report.json",lambda r:r.update(source_sha256="0"*64),"MNLM_SCIFACT_READBACK_LINEAGE",False)
    fault("B2_dropped_document","documents.jsonl",lambda r:r.pop(),"MNLM_SCIFACT_READBACK_DOCUMENT_POPULATION")
    fault("B2_changed_sentence","documents.jsonl",lambda r:r[0]["views"]["source_sentences"].__setitem__(0,"corrupted"),
          "MNLM_SCIFACT_READBACK_SOURCE_SENTENCES")
    fault("B2_changed_pair","documents.jsonl",lambda r:next(n for n in r[0]["views"]["nodes"] if n["level"]=="sentence_pair")["sentence_range"].__setitem__(1,1),
          "MNLM_SCIFACT_READBACK_NATIVE_VIEWS")
    fault("B2_missing_edge","documents.jsonl",lambda r:r[0]["views"]["edges"].pop(),"MNLM_SCIFACT_READBACK_NATIVE_EDGES")
    def break_set(values):
        next(s for v in values for s in v["rationales"] if len(s["sentences"])>1)["sentences"].pop()
    fault("B2_partial_rationale","labels.jsonl",break_set,"MNLM_SCIFACT_READBACK_RATIONALE_BINDING")
    fault("B2_missing_alternative","labels.jsonl",lambda r:next(v for v in r if len(v["rationales"])>1)["rationales"].pop(),
          "MNLM_SCIFACT_READBACK_RATIONALE_BINDING")
    fault("B2_absence_is_not_contradiction","labels.jsonl",lambda r:next(v for v in r if not v["rationales"]).update(label="CONTRADICT"),
          "MNLM_SCIFACT_READBACK_RATIONALE_BINDING")
    fault("B2_fabricated_context","claims.jsonl",lambda r:next(v for v in r if not v["context_links"]).update(context_links=[dict(row=0,variant=0,match="exact")]),
          "MNLM_SCIFACT_READBACK_CLAIM_BINDING")
    fault("B2_changed_citance","contexts.jsonl",lambda r:r[0]["citance_spans"][0].__setitem__(0,0),
          "MNLM_SCIFACT_READBACK_CITANCE_SPANS")
    fault("B3_broken_component","groups.json",lambda r:next(iter(r["groups"].values()))["members"].pop(),
          "MNLM_SCIFACT_READBACK_COMPONENTS",False)
    fault("B3_wrong_partition","cases.jsonl",lambda r:r[0].update(partition="other"),"MNLM_SCIFACT_READBACK_CASE_BINDING")
    fault("B3_label_input","claims.jsonl",lambda r:r[0].update(label="SUPPORT"),"MNLM_SCIFACT_READBACK_CLAIM_BINDING")
    for flag in ("scientific_admission","meaning_asserted","world_truth_adjudicated","sealed_evaluation","gold_in_model_input"):
        fault("B3_false_"+flag,"report.json",lambda r,k=flag:r.update({k:True}),"MNLM_SCIFACT_READBACK_CLAIMS",False)
    fault("B3_lower_text_floor","report.json",lambda r:r.update(native_text_floor=4),"MNLM_SCIFACT_READBACK_CLAIMS",False)
    phases["persisted_corruption_controls"] = time.perf_counter()-tick; tick = time.perf_counter()
    inputs = Inputs(shadow,a.publication_sha256); cases = jsonl(shadow/"cases.jsonl")
    originals = {n:(shadow/n).read_bytes() for n in ("labels.jsonl","contexts.jsonl")}
    for n in originals:(shadow/n).write_bytes(b"unavailable gold or provenance; must not be loaded")
    isolated = Inputs(shadow,a.publication_sha256); decisions = []
    keys = {"claim_text","title","source_sentences","views","structural_edges","decision","code","native_text_floor"}
    for case in cases:
        value = isolated.read(case["id"],partition=case["partition"])
        equal = value == inputs.read(case["id"],partition=case["partition"])
        decisions.append(dict(case=case["id"],partition=case["partition"],input_keys=sorted(value),
                              gold_independent=equal and set(value)==keys,decision=value["decision"],code=value["code"]))
    for n,data in originals.items():(shadow/n).write_bytes(data)
    with (a.output/"consumer-verdicts.jsonl").open("xb") as out:
        for row in decisions:out.write(canonical(row)+b"\n")
    checks.append(dict(name="B3 all real consumers independent of unavailable gold/context and refuse unadmitted meaning",
                       passed=all(r["gold_independent"] and r["decision"]=="refuse"
                                  and r["code"]=="MNLM_SCIFACT_UNADMITTED_PANEL" for r in decisions)))
    trial("B3 consumer wrong partition","MNLM_SCIFACT_INPUT_PARTITION",
          lambda:inputs.read(cases[0]["id"],partition="other"))
    trial("B3 consumer missing case","MNLM_SCIFACT_INPUT_CASE",lambda:inputs.read("absent",partition="train"))
    case = cases[0]; baseline = inputs.read(case["id"],partition=case["partition"])
    changed = inputs.read(case["id"],partition=case["partition"]); changed["source_sentences"][0] = "caller mutation"
    changed["views"][0]["sentence_range"][0] = -1
    checks.append(dict(name="B3 returned mutable views cannot alter later observations",
                       passed=inputs.read(case["id"],partition=case["partition"])==baseline))
    phases["actual_consumer_isolation"] = time.perf_counter()-tick; tick = time.perf_counter()
    after = dict(source=inventory(a.source),publication=inventory(a.publication)); publish(a.output/"after.json",after)
    restored = read(a.source,a.source_sha256,shadow,a.publication_sha256)
    checks.append(dict(name="B1/B2/B3 final full source read, exact shadow restoration and unchanged originals",
                       passed=before==after and inventory(shadow)==before["publication"] and restored==cold))
    phases["final_full_read_and_drift"] = time.perf_counter()-tick; wall = time.perf_counter()-started
    result = dict(passed=all(c["passed"] for c in checks),checks=checks,counts=cold["counts"],
                  author_checked=True,scientific_admission=False,phases_seconds=phases,wall_seconds=wall,
                  phase_coverage=sum(phases.values())/wall)
    publish(a.output/"fsv.json",result)
    print(json.dumps(dict(passed=result["passed"],checks=len(checks),failed=[c for c in checks if not c["passed"]],
                         wall_seconds=wall,phase_coverage=result["phase_coverage"])))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
