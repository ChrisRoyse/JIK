"""Known-law commissioning of a full-Z, variation-corrected CMI bound (#12).

Artificial controls are mathematical boundary tests, never source/training data.
No threshold tuning, new outcomes, GPU work or production publication.
"""
import argparse
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/"lens"),str(Path(__file__).resolve().parent)]
import bounded_cmi as cmi
from finite_cmi import BoundError, canonical, digest, require
from finite_cmi_native import native_states, file_sha
from bounded_cmi_readback import replay, probability_inequalities, analytical_law_check

NAMES=("unique-one-bit","bounded-positive","xor","noise","osc80","osc512","full-conditioning")
REAL_SHA="ce0976085308c49592802f46929c043d833c10e8c5d46cad6feb76d82634c245"


def write(path, obj):
    with path.open("xb") as f:
        f.write(canonical(obj));f.flush();os.fsync(f.fileno())


def laws():
    zero=dict(u=0.,v=0.,both=0.);wide=dict(u=1.,v=1.,both=1.)
    return {
        "unique-one-bit":dict(partition={"kind":"one-cell"},variation={"all":zero},truth=1.,
            argument="U fair Bernoulli independent of float32-recorded uniform Z; V=U; conditional probabilities constant everywhere"),
        "bounded-positive":dict(partition={"kind":"one-cell"},variation={"all":dict(u=0.,v=0.,both=.01)},
            truth_min=1-(-.12*np.log2(.12)-.88*np.log2(.88)),
            argument="Z float32-recorded uniform[-1,1], U independent fair bit, V=U XOR Bernoulli(.1+.02*Z); marginals 1/2, p11=.45-.01*Z, symmetric mean .45"),
        "xor":dict(partition=dict(kind="dense-coordinate-cut",lens="native-panel",coordinate=0,cut=.5),
            variation={"below":zero,"at_or_above":zero},truth=1.,
            argument="Z=(independent fair B, independent continuous W); U fair independent bit, V=U XOR B; each B cell has constant conditional law for all W"),
        "noise":dict(partition={"kind":"one-cell"},variation={"all":zero},truth=0.,
            argument="U,V,Z mutually independent; both fair bits; conditional law constant"),
        "osc80":dict(partition={"kind":"one-cell"},variation={"all":wide},truth=0.,
            argument="U=V=1[sin(80*Z)>0]; both deterministic given complete Z, so CMI zero; probability variation at most one"),
        "osc512":dict(partition={"kind":"one-cell"},variation={"all":wide},truth=0.,
            argument="U=V=1[sin(512*Z)>0]; both deterministic given complete Z, so CMI zero; probability variation at most one"),
        "full-conditioning":dict(partition={"kind":"one-cell"},variation={"all":wide},truth=0.,
            argument="U=V=fair bit; Z=(independent continuous W,V), so V is known given full Z; probability variation at most one")}


def plan_for(law):
    basis=digest(law)
    return dict(schema=cmi.SCHEMA,alpha=.05,seeds=list(cmi.SEEDS),comparisons={
        f"{name}-{n}":dict(candidate_event="retained binary U",outcome_event="retained binary V",
            panel_ids=["native-panel"],evaluation_units=n,partition=law[name]["partition"],variation=law[name]["variation"],
            assumptions=dict(sampling="iid-independent-units",sampling_basis="independent control draws; artificial mathematical controls only",
                regularity_basis=law[name]["argument"],basis_sha256=basis,scope="full-native-Z-population-almost-sure",evaluation_used=False))
        for name in NAMES for n in (600,2000)})


def generate(name,n,seed):
    rng=np.random.default_rng(seed)
    z=rng.uniform(-1,1,(n,1)).astype(np.float32)
    u=rng.integers(0,2,n).astype(bool);v=rng.integers(0,2,n).astype(bool)
    if name=="unique-one-bit":v=u.copy()
    if name=="bounded-positive":v=u^(rng.random(n)<(.1+.02*z[:,0].astype(float)))
    if name=="xor":
        b=rng.integers(0,2,n).astype(bool);z=np.column_stack((b,z)).astype(np.float32);v=u^b
    if name.startswith("osc"):u=np.sin(int(name[3:])*z[:,0].astype(float))>0;v=u.copy()
    if name=="full-conditioning":v=u.copy();z=np.column_stack((z,v)).astype(np.float32)
    return z,u,v


def case(root,plan,name,n,seed):
    path=root/f"{name}-{n}-{seed}";path.mkdir()
    z,u,v=generate(name,n,seed)
    with (path/"panel.npz").open("xb") as f:np.savez(f,X=z,absent=np.zeros(n,bool))
    with (path/"witnesses.npz").open("xb") as f:np.savez(f,u=u,v=v)
    descriptor=dict(id="native-panel",path=str(path/"panel.npz"),sha256=file_sha(path/"panel.npz"))
    states=native_states([descriptor],["native-panel"],n)
    payload=dict(units=[f"{name}/{n}/{seed}/{i}" for i in range(n)],states=[base64.b64encode(x).decode() for x in states],
        u=u.tolist(),v=v.tolist(),panel=[{**descriptor,"path":"panel.npz"}],panel_ids=["native-panel"],
        witnesses_sha256=file_sha(path/"witnesses.npz"),artificial_boundary_only=True)
    write(path/"inputs.json",payload)
    cold=json.loads((path/"inputs.json").read_bytes())
    rows=list(zip(cold["units"],[base64.b64decode(x) for x in cold["states"]],cold["u"],cold["v"]))
    result=cmi.estimate(plan,f"{name}-{n}",seed,rows)
    write(path/"result.json",result)
    checked=replay(path,plan)
    return checked,rows


def boundaries(plan,rows):
    checks=[]
    def reject(name,action,code):
        before=digest(plan)
        try:action()
        except BoundError as exc:
            assert exc.code==code,(name,exc)
            assert before==digest(plan)
            checks.append(dict(name=name,code=exc.code,passed=True))
        else:raise AssertionError(name+": invalid call succeeded")
    key="unique-one-bit-600"
    def call(p=plan,r=rows,seed=42):return cmi.estimate(p,key,seed,r)
    for name,change,code in (
        ("unknown_variation",lambda c:c.update(variation=None),"MNLM_CMI_REGULARITY_UNESTABLISHED"),
        ("missing_cell",lambda c:c.update(variation={}),"MNLM_CMI_REGULARITY_UNESTABLISHED"),
        ("fitted_on_evaluation",lambda c:c["assumptions"].update(evaluation_used=True),"MNLM_CMI_REGULARITY_UNESTABLISHED"),
        ("only_coarsened_scope",lambda c:c["assumptions"].update(scope="binned-Z-only"),"MNLM_CMI_REGULARITY_UNESTABLISHED"),
        ("missing_external_argument",lambda c:c["assumptions"].update(regularity_basis=""),"MNLM_CMI_REGULARITY_UNESTABLISHED"),
        ("unknown_sampling",lambda c:c["assumptions"].update(sampling="unknown"),"MNLM_CMI_ASSUMPTIONS"),
        ("negative_allowance",lambda c:c["variation"]["all"].update(u=-.1),"MNLM_CMI_REGULARITY_UNESTABLISHED"),
        ("nonfinite_allowance",lambda c:c["variation"]["all"].update(u=float("nan")),"MNLM_CMI_REGULARITY_UNESTABLISHED")):
        p=copy.deepcopy(plan);change(p["comparisons"][key]);reject(name,lambda:call(p=p),code)
    reject("repeated_unit",lambda:call(r=[rows[1],*rows[1:]]),"MNLM_CMI_UNIT_REUSE")
    reject("changed_extent",lambda:call(r=rows[:-1]),"MNLM_CMI_SAMPLE_PLAN")
    reject("unbudgeted_seed",lambda:call(seed=47),"MNLM_CMI_COMPARISON")
    reject("missing_full_conditioning",lambda:call(r=[(rows[0][0],b"[]",*rows[0][2:]),*rows[1:]]),"MNLM_CMI_PANEL_INCOMPLETE")
    for name,change in (
        ("missing_native_values",lambda s:s[0]["members"].pop("X")),
        ("nonfinite_unpartitioned_value",lambda s:s[0]["members"]["X"].update(bytes=base64.b64encode(np.array([float("nan")],np.float32).tobytes()).decode())),
        ("changed_shape_extent",lambda s:s[0]["members"]["X"].update(shape=[2])),
        ("noncanonical_absence",lambda s:s[0]["members"]["absent"].update(bytes=base64.b64encode(b"\x02").decode()))):
        s=json.loads(rows[0][1]);change(s)
        r=[(rows[0][0],canonical(s),*rows[0][2:]),*rows[1:]]
        reject(name,lambda:call(r=r),"MNLM_CMI_NATIVE_STATE")
    a=call()
    altered=list(rows);s=json.loads(altered[0][1]);s[0]["members"]["X"]["bytes"]=base64.b64encode(np.array([123.],np.float32).tobytes()).decode()
    altered[0]=(altered[0][0],canonical(s),*altered[0][2:])
    b=call(r=altered)
    assert a["bits_lcb"]==b["bits_lcb"] and a["input_sha256"]!=b["input_sha256"]
    checks.append(dict(name="unpartitioned_native_coordinate_remains_bound",passed=True))
    return checks


def layout_boundaries(root,plan):
    """Published Multi/sparse native states remain typed through a one-cell count."""
    checks=[]
    for kind,arrays in (
        ("multi",dict(tokens=np.asarray([[1.,2.],[2.,1.]],np.float32),offsets=np.asarray([0,1,2],np.int64),
                      absent=np.zeros(2,bool),windows=np.asarray([1,2],np.int64))),
        ("sparse",dict(data=np.asarray([1.,1.],np.float32),indices=np.asarray([0,2],np.int32),
                       indptr=np.asarray([0,1,2],np.int64),shape=np.asarray([2,4],np.int64),absent=np.zeros(2,bool)))):
        p=root/(kind+"-native.npz")
        with p.open("xb") as f:np.savez(f,**arrays)
        states=native_states([dict(id="native-panel",path=str(p),sha256=file_sha(p))],["native-panel"],2)
        assert states[0]!=states[1]
        assert all(cmi.cell_of(s,plan["comparisons"]["unique-one-bit-600"])=="all" for s in states)
        checks.append(dict(name=kind+"_full_native_partition",passed=True,source_sha256=file_sha(p)))
        try:cmi.cell_of(states[0],plan["comparisons"]["xor-600"])
        except BoundError as exc:assert exc.code=="MNLM_CMI_PARTITION_NATIVE"
        else:raise AssertionError("silently flattened "+kind)
        checks.append(dict(name=kind+"_cannot_be_dense_coordinate",passed=True))
    return checks


def run(args):
    start=time.perf_counter()
    source=args.input/"record-vector.npz";before=file_sha(source)
    require(before==REAL_SHA,"MNLM_CMI_NATIVE_CHANGED","retained real input differs from historical evidence; do not replace the expected digest")
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    law=laws();plan=plan_for(law)
    write(root/"laws.json",law);write(root/"plan.json",plan)
    cmi.validate_plan(plan)
    write(root/"runtime.json",dict(python=platform.python_version(),numpy=np.__version__,platform=platform.system()))
    scripts=Path(__file__).resolve().parents[1]
    names=("audit/fsv_bounded_cmi.py","audit/bounded_cmi_readback.py","audit/finite_cmi_readback.py",
           "lens/bounded_cmi.py","lens/finite_cmi.py","lens/finite_cmi_native.py","lens/slot_layout.py")
    write(root/"code.json",{n:file_sha(scripts/n) for n in names})
    prepared=time.perf_counter();results=[];sample=None
    for name in NAMES:
        for n in (600,2000):
            for seed in cmi.SEEDS:
                result,rows=case(root,plan,name,n,seed);results.append(result)
                truth=law[name].get("truth",law[name].get("truth_min"))
                assert result["bits_lcb"] <= truth+1e-9,(name,n,seed,result)
                if name=="unique-one-bit" and n==600 and seed==42:sample=rows
        print(name,"cold-read complete",flush=True)
    evaluated=time.perf_counter()
    checks=boundaries(plan,sample)+layout_boundaries(root,plan)
    algebra=probability_inequalities();law_check=analytical_law_check(law)
    positive=[r for r in results if any(r["case"].startswith(n+"-2000-") for n in ("unique-one-bit","bounded-positive","xor"))]
    assert len(positive)==15 and all(r["bits_lcb"]>=.05 for r in positive)
    # A deliberately FALSE smoothness assumption must be visible as such, not certified.
    bad=copy.deepcopy(plan);key="osc512-2000"
    bad["comparisons"][key]["variation"]["all"]={"u":0.,"v":0.,"both":0.}
    p=json.loads((root/"osc512-2000-42/inputs.json").read_bytes())
    rows=list(zip(p["units"],[base64.b64decode(x) for x in p["states"]],p["u"],p["v"]))
    unsafe=cmi.estimate(bad,key,42,rows)
    assert unsafe["bits_lcb"]>.05 and unsafe["assumptions_proven_by_this_estimator"] is False
    write(root/"false-assumption-counterexample.json",dict(declared_assumption_is_false=True,true_cmi=0.,result=unsafe))
    # Real physical slot: no outcome labels read, no unsupported assumption accepted.
    with np.load(source,allow_pickle=False) as ar:n=len(ar["X"])
    real_states=native_states([dict(id="native-panel",path=str(source),sha256=before)],["native-panel"],n)
    assert n==1992
    p=copy.deepcopy(plan);p["comparisons"]["unique-one-bit-600"]["variation"]=None
    try:cmi.estimate(p,"unique-one-bit-600",42,[(f"real/{i}",s,False,False) for i,s in enumerate(real_states)])
    except BoundError as exc:assert exc.code=="MNLM_CMI_REGULARITY_UNESTABLISHED"
    else:raise AssertionError("real missing regularity accepted")
    assert file_sha(source)==before
    checked=time.perf_counter()
    report=dict(acceptance="author-checked only",cases=results,boundaries=checks,algebra=algebra,law_check=law_check,
        real_native_rows=n,real_native_sha256=before,real_outcomes_read=False,
        real_status="MNLM_CMI_REGULARITY_UNESTABLISHED",artificial_controls_are_training_data=False,
        full_native_states_preserved=True,positive_2000_all_five_seeds=len(positive),
        assumptions_proven_for_real_corpus=False,production_admission=False,stage4_accepted=False,
        cost_seconds=dict(preflight=prepared-start,controls_and_readback=evaluated-prepared,boundaries=checked-evaluated),
        wall_seconds=checked-start)
    write(root/"report.json",report)
    files={str(p.relative_to(root)):file_sha(p) for p in sorted(root.rglob("*")) if p.is_file()}
    write(root/"manifest.json",dict(files=files,owner="ChrisRoyse/MNLM#12"))
    for name,h in files.items():assert file_sha(root/name)==h
    finished=time.perf_counter()
    write(root/"timing.json",dict(total_seconds=finished-start,report_manifest_readback_seconds=finished-checked))
    print(json.dumps(dict(cases=len(results),boundaries=len(checks),positive=len(positive),wall=finished-start)))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ("input","output"):ap.add_argument("--"+n,type=Path,required=True)
    run(ap.parse_args())
