"""Three-boundary persisted controls for paired reference-root diagnostics."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.paired_roots_readback import readback, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("publication", "diagnostic", "output"):
        p.add_argument("--"+arg,type=Path,required=True)
    p.add_argument("--diagnostic-sha256",required=True)
    a = p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    before = {p.name:sha(p.read_bytes()) for p in a.publication.iterdir() if p.is_file()}
    cold = readback(a.publication,a.diagnostic,a.diagnostic_sha256)
    checks = [{"boundary":"B1 source/plans", "name":"all source-bound index plans replay", "pass":True},
              {"boundary":"B2 statistics/bounds", "name":"all interval arithmetic and fixed raw samples replay", "pass":True},
              {"boundary":"B3 claims", "name":"publication grants no admission", "pass":True}]
    question = "biomedical-P1057"; metric = "standalone:e5-base-v2"
    def put(path,value):
        path.write_text(json.dumps(value,sort_keys=True),encoding="utf-8")
    def control(name,boundary,code,edit):
        dest = a.output/name; shutil.copytree(a.publication,dest)
        root = json.loads((dest/"report.json").read_bytes()); edit(dest,root)
        put(dest/(question+".json"),root["findings"][question]); put(dest/"report.json",root)
        try:
            readback(dest,a.diagnostic,a.diagnostic_sha256); observed="no refusal"
        except ValueError as exc:
            observed=str(exc)
        checks.append({"name":name,"boundary":boundary,"observed":observed,"expected":code,
                       "pass":observed=="MNLM_DIAGNOSTIC_READBACK_"+code,
                       "mutated_report_sha256":sha((dest/"report.json").read_bytes())})
    def artifact(edit):
        def mutate(dest,root):
            path=dest/(question+".npz")
            with np.load(path,allow_pickle=False) as z:
                values={k:z[k].copy() for k in z.files}
            edit(values,root["findings"][question]); np.savez(path,**values)
            root["findings"][question]["artifact_sha256"]=sha(path.read_bytes())
        return mutate
    control("wrong-parent","B1","PAIRED_BINDING",lambda d,r:r.update(source_report_sha256="0"*64))
    control("changed-label","B1","PAIRED_LABELS",artifact(lambda z,r:z["authentic__labels"].__setitem__(0,99)))
    control("replacement-sample","B1","PAIRED_PLAN",artifact(lambda z,r:z["authentic__indices"].__setitem__((0,0,0),
            int(z["authentic__indices"][0,0,1]))))
    control("changed-raw-statistic","B2","PAIRED_RAW_SAMPLE",artifact(lambda z,r:
            z["authentic__samples__"+r["aliases"]["single:e5-base-v2"]].__setitem__((0,0),9.0)))
    control("inflated-lower-bound","B2","PAIRED_BOUNDS",lambda d,r:
            r["findings"][question]["cases"]["authentic"]["statistics"][metric]["per_seed"]["7"].update(low=9.0))
    control("inflated-envelope","B2","PAIRED_ENVELOPE",lambda d,r:
            r["findings"][question]["cases"]["authentic"]["statistics"][metric].update(outer_low=9.0))
    control("incomplete-publication","B3","PAIRED_COMPLETE",lambda d,r:r.update(complete=False))
    control("false-admission","B3","PAIRED_CLAIM",lambda d,r:r.update(conditional_information_certified=True))
    control("changed-seeds","B3","PAIRED_RULES",lambda d,r:r["rules"].update(seeds=[0]))
    checks.append({"boundary":"B3", "name":"original publication unchanged", "pass":before==
                   {p.name:sha(p.read_bytes()) for p in a.publication.iterdir() if p.is_file()}})
    result={"passed":sum(c["pass"] for c in checks),"total":len(checks),"checks":checks,
            "cold_readback":cold,"acceptance":"author-checked"}
    put(a.output/"fsv.json",result);print(json.dumps({k:result[k] for k in ("passed","total")}))
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
