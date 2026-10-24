"""Independent cold readback and fixed scaling/interval comparison (#12).

Does not import either new producer, index, or reconstruction implementation.
Rebuilds predictions from frozen coefficients and counts from direct distances.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import digamma
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text.calyx_bits import subsample_m, RESAMPLES
from audit.calyx_diagnostic_readback import ross


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError("MNLM_CLOSURE_READBACK_"+reason)


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def load_report(root):
    require(all((root/name).is_file() for name in ("manifest.json", "report.json", "intent.json")),
            "INCOMPLETE_PUBLICATION")
    manifest = json.loads((root/"manifest.json").read_bytes())
    require(sha(root/"report.json") == manifest["report_sha256"] and
            sha(root/"intent.json") == manifest["intent_sha256"], "PUBLICATION_PIN")
    report = json.loads((root/"report.json").read_bytes())
    require(report["complete"] and all(report[k] is False for k in
            ("scientific_admission", "conditional_information_certified", "served")), "CLAIMS")
    return report


def scan(blocks, y, rows):
    arrays = [np.asarray(b, np.float32) for b in blocks]
    values, radii = [], []
    for row in rows:
        distance = np.zeros(len(y), np.float32)
        for block in arrays:
            distance = np.maximum(distance, np.abs(block-block[row]).max(1))
        distance[row] = np.inf
        same = y == y[row]
        radius = sorted(distance[same])[2]
        values.append((sum(distance[same] <= radius), int(same.sum()), sum(distance <= radius)))
        radii.append(radius)
    return np.asarray(values), np.asarray(radii)


def bits(c):
    return float((digamma(len(c))+digamma(c[:, 0])-digamma(c[:, 1])-digamma(c[:, 2])).mean()/np.log(2))


def resolution(root):
    report = load_report(root); checked = 0
    for item in report["controls"]+report["real"]:
        path = root/item["file"]
        require(sha(path) == item["sha256"], "ARRAY_PIN")
        with np.load(path, allow_pickle=False) as f:
            y = f["labels"]; blocks = [f[f"block_{i}"] for i in range(len(item["dimensions"]))]
            variants = {"panel": blocks[:-1], "joint": blocks} if item["mode"] == "gain" else {"single": blocks}
            for key, selected in variants.items():
                c = f[key+"_counts"]; r = f[key+"_radii"]; rows = f[key+"_readback_rows"]
                independent, radius = scan(selected, y, rows)
                require(np.array_equal(c[rows], independent) and np.array_equal(r[rows], radius), "RESOLUTION_COUNTS")
                require(abs(bits(c)-item["bits"][key]) < 1e-12, "RESOLUTION_FORMULA")
                checked += len(rows)
    return dict(report_sha256=sha(root/"report.json"), cases=len(report["controls"])+len(report["real"]),
                independently_recounted_rows=checked, wall_seconds_measured=report["wall_seconds"])


def prediction(model, panel):
    require(panel.shape[1] == model["panel_dimensions"] and model["labels_used"] is False
            and model["preserves_panel"] == "append_prediction_to_complete_panel", "COMPLETE_PANEL")
    standardized = (panel-np.asarray(model["mean"]))/np.asarray(model["scale"])
    base = np.column_stack((np.ones(len(panel)), standardized))
    out = np.empty((len(panel), len(model["models"])))
    for column, entry in enumerate(model["models"]):
        matrix = base
        if entry["kind"] == "periodic":
            angle = panel[:, entry["column"]]*entry["frequency"]
            matrix = np.column_stack((base, np.sin(angle), np.cos(angle)))
        else:
            require(entry["kind"] == "affine", "MODEL_KIND")
        out[:, column] = np.einsum("ij,j->i", matrix, np.asarray(entry["coefficients"]))
    return out


def distance(blocks):
    n = len(blocks[0]); result = np.zeros((n,n), np.float32)
    for block in blocks:
        v = np.asarray(block, np.float32)
        for i in range(n):
            result[i] = np.maximum(result[i], np.abs(v-v[i]).max(1))
    return result


def standardize(design, evaluation):
    mu, sd = design.mean(0), design.std(0)
    sd[sd == 0] = 1
    return (evaluation-mu)/sd


def intervals(y, distances, definitions, seed):
    """Reference paired m-out-of-n roots; fixed independent design fit."""
    n = len(y); m = subsample_m(n, np.unique(y, return_counts=True)[1])
    unique = {}; aliases = {}
    for key, d in distances.items():
        pin = hashlib.sha256(d.tobytes()).hexdigest()
        aliases[key] = pin; unique.setdefault(pin, d)
    points = {pin: ross(d,y) for pin,d in unique.items()}
    samples = {pin: np.empty(RESAMPLES) for pin in unique}
    indices = np.empty((RESAMPLES,m), dtype=np.int32)
    rng = np.random.default_rng(seed)
    for b in range(RESAMPLES):
        for _ in range(8):
            ix = rng.choice(n,m,replace=False)
            classes, count = np.unique(y[ix],return_counts=True)
            if len(classes) == len(np.unique(y)) and min(count) > 3:
                break
        else:
            raise ValueError("MNLM_CLOSURE_READBACK_SUBSAMPLE_SUPPORT")
        indices[b] = ix
        for pin,d in unique.items():
            samples[pin][b] = ross(d[np.ix_(ix,ix)],y[ix])
    summaries = {}; arrays = dict(indices=indices)
    for name, (joint,panel) in definitions.items():
        a,b = aliases[joint],aliases[panel]
        estimate = points[a]-points[b]; values = samples[a]-samples[b]
        roots = np.sort(np.sqrt(m/(1-m/n))*(values-estimate))
        at = lambda p: int(round((RESAMPLES-1)*p))
        summaries[name] = dict(point=estimate, low=min(estimate,estimate-roots[at(.975)]/np.sqrt(n)),
                               high=max(estimate,estimate-roots[at(.025)]/np.sqrt(n)), n=n,m=m)
        arrays[name+"_subsamples"] = values
    return summaries, arrays


def closure(root, output, selected_names=None):
    report = load_report(root); rows = []; timing = defaultdict(float); tamper = {}; sources = {}
    if selected_names is not None:
        chosen = set(selected_names)
        require(len(chosen) == len(selected_names) and chosen <= {r['name'] for r in report['rows']}, 'SELECTED_ROWS')
        report['rows'] = [r for r in report['rows'] if r['name'] in chosen]
    previous_geometry = None
    geometry_builds = 0
    fixed_sources = set()
    for i,item in enumerate(report["rows"]):
        tick = time.perf_counter()
        path, model_path = root/item["array_file"], root/item["model_file"]
        require(sha(path) == item["array_sha256"] and sha(model_path) == item["model_sha256"], "CLOSURE_PIN")
        model = json.loads(model_path.read_bytes())
        with np.load(path, allow_pickle=False) as f:
            data = {key: f[key].copy() for key in f.files}
        provenance = item.get("source")
        if provenance and "fixed_profile" in provenance:
            original_path = Path(provenance['fixed_profile'])
            require(sha(original_path) == provenance['sha256'], 'FIXED_PROFILE_PIN')
            fixed_sources.add((str(original_path), provenance['sha256']))
            with np.load(original_path, allow_pickle=False) as original:
                require(np.array_equal(original['labels'], data['labels']) and
                        np.array_equal(original['design'], data['design']) and
                        np.array_equal(original['evaluation'], data['evaluation']), 'FIXED_PROFILE_SPLIT_LABELS')
                require(np.array_equal(original[provenance['candidate']+'__profile'], data['candidate']), 'FIXED_PROFILE_CANDIDATE')
                require(len(provenance['conditioning']) == len(item['blocks']) and
                        all(np.array_equal(original[name+'__profile'], data[f'block_{k}'])
                            for k, name in enumerate(provenance['conditioning'])), 'FIXED_PROFILE_COMPLETE_PANEL')
        if provenance and "pins" in provenance:
            original = {}
            for name, record in provenance["pins"].items():
                key = (record["file"], record["sha256"])
                if key not in sources:
                    require(sha(key[0]) == key[1], "ORIGINAL_SOURCE_PIN")
                    with np.load(key[0], allow_pickle=False) as f:
                        sources[key] = {k:f[k].copy() for k in f.files if
                                        k in ("profile", "labels", "pairs") or k.endswith("__profile")}
                original[name] = sources[key]
            require(np.array_equal(original["public"]["labels"], data["labels"]), "ORIGINAL_LABELS")
            def profile(name):
                if name in original:
                    require(np.array_equal(original[name]["labels"], data["labels"]) and
                            np.array_equal(original[name]["pairs"], original["public"]["pairs"]),
                            "ORIGINAL_ALIGNMENT")
                    return original[name]["profile"]
                return original["profiles"][name+"__profile"]
            require(np.array_equal(profile(provenance["candidate"]), data["candidate"]), "ORIGINAL_CANDIDATE")
            require(len(provenance["conditioning"]) == len(item["blocks"]) and
                    all(np.array_equal(profile(name), data[f"block_{k}"])
                        for k,name in enumerate(provenance["conditioning"])), "ORIGINAL_COMPLETE_PANEL")
        design,evaluation = data["design"],data["evaluation"]
        require(not set(design)&set(evaluation) and sorted(np.r_[design,evaluation]) == list(range(item["n"])), "SPLIT")
        blocks = [data[f"block_{k}"] for k in range(len(item["blocks"]))]
        panel = np.concatenate(blocks,axis=1); x = data["candidate"]
        expected = prediction(model,panel[evaluation])
        require(np.allclose(expected,data["predictions"],rtol=0,atol=1e-10), "PREDICTION")
        # Use byte-identical persisted predictions after independent arithmetic replay.
        pred = data["predictions"]; yy = data["labels"][evaluation]; xx = x[evaluation]
        selected = [b[evaluation] for b in blocks]
        sx = standardize(x[design],xx); sp = standardize(x[design],pred)
        sz = [standardize(b[design],b[evaluation]) for b in blocks]
        variants = dict(panel=selected,joint=selected+[xx],closed_panel=selected+[pred],closed_joint=selected+[pred,xx],
                        scaled_panel=sz,scaled_joint=sz+[sx],scaled_closed_panel=sz+[sp],scaled_closed_joint=sz+[sp,sx])
        geometry_key = (item['array_sha256'], item['model_sha256'], json.dumps(item['bits'], sort_keys=True))
        if previous_geometry is not None and previous_geometry[0] == geometry_key:
            distances = previous_geometry[1]
        else:
            distances = {}
            for key, values in variants.items():
                distances[key] = distance(values)
                if key+"_counts" in data:
                    count,radius = scan(values,yy,np.arange(len(yy)))
                    require(np.array_equal(count,data[key+"_counts"]) and np.array_equal(radius,data[key+"_radii"]), "CLOSURE_COUNTS")
                    require(abs(bits(count)-item["bits"][key]) < 1e-12, "CLOSURE_FORMULA")
            if provenance and 'fixed_profile' in provenance:
                require(np.array_equal(distance(selected+[selected[-1]]), distances['panel']), 'FIXED_PROFILE_DUPLICATE')
            previous_geometry = (geometry_key, distances)
            geometry_builds += 1
        timing["readback_and_geometry_seconds"] += time.perf_counter()-tick
        tick=time.perf_counter()
        definitions=dict(raw=("joint","panel"),closed=("closed_joint","closed_panel"),
                         scaled=("scaled_joint","scaled_panel"),scaled_closed=("scaled_closed_joint","scaled_closed_panel"))
        statistics, saved = intervals(yy,distances,definitions,item["seed"])
        control_statistics = None
        if provenance and 'fixed_profile' in provenance:
            # The exact duplicate adds no distance or sigma-algebra information.
            control_y = np.random.default_rng(20261020).permutation(yy)
            control_statistics, control_arrays = intervals(control_y, distances, definitions, item['seed'])
            saved.update({'permuted_'+key: value for key, value in control_arrays.items()})
            saved['permuted_labels'] = control_y
            control_statistics['exact_duplicate_gain'] = 0.0
        timing["interval_seconds"] += time.perf_counter()-tick
        destination = output/(item["name"]+".npz")
        with destination.open("xb") as f:
            np.savez(f, **saved)
        rows.append(dict(name=item["name"],statistics=statistics,truth=item["true_cmi_bits"],
                         controls=control_statistics,
                         source_sha256=item["array_sha256"],artifact_sha256=sha(destination)))
        if i == 0:
            # Actual boundary violations must be caught, not return success.
            fault_source=output/"fault-source.npz"
            fault_model=output/"fault-model.json"
            fault_counts=output/"fault-counts.npy"
            fault_source.write_bytes(path.read_bytes()+b" ")
            fault_model.write_bytes(model_path.read_bytes()+b" ")
            with fault_counts.open("xb") as f:
                np.save(f,data["joint_counts"]+1)
            for label, action in {
                "dropped_conditioning_coordinate": lambda: prediction(model,panel[evaluation,:-1]),
                "tampered_count": lambda: require(np.array_equal(np.load(fault_counts,allow_pickle=False),scan(selected+[xx],yy,np.arange(len(yy)))[0]),"CLOSURE_COUNTS"),
                "tampered_source": lambda: require(sha(fault_source)==item["array_sha256"],"CLOSURE_PIN"),
                "tampered_model": lambda: require(sha(fault_model)==item["model_sha256"],"CLOSURE_PIN"),
            }.items():
                try:
                    action()
                except ValueError as exc:
                    tamper[label]=str(exc)
                else:
                    raise ValueError("MNLM_CLOSURE_READBACK_MISSING_NEGATIVE")
            require(set(model)=={"schema","mean","scale","panel_dimensions","candidate_dimensions",
                                 "design_rows","ridge","models","labels_used","preserves_panel"},"MODEL_SCHEMA")
            tamper["fault_files"]={p.name:sha(p) for p in (fault_source,fault_model,fault_counts)}
        if (i+1)%10 == 0:
            print(json.dumps(dict(cases_read_back=i+1,total=len(report["rows"]))),flush=True)
    groups=defaultdict(list)
    for row in rows:
        groups[row["name"].rsplit("-",1)[0]].append(row)
    summary={}
    for name,items in groups.items():
        require(len(items)==5,"FIVE_SEEDS")
        summary[name]={method:dict(min_point=min(r["statistics"][method]["point"] for r in items),
                                   max_point=max(r["statistics"][method]["point"] for r in items),
                                   outer_low=min(r["statistics"][method]["low"] for r in items),
                                   outer_high=max(r["statistics"][method]["high"] for r in items))
                       for method in ("raw","closed","scaled","scaled_closed")}
    return dict(report_sha256=sha(root/"report.json"),rows=rows,summary=summary,negative_controls=tamper,
                original_source_files_rebound=len(sources)+len(fixed_sources),geometry_builds=geometry_builds,timing=dict(timing))


def main(a):
    started=time.perf_counter(); a.output.mkdir(parents=True,exist_ok=False)
    with threadpool_limits(limits=1):
        tick=time.perf_counter(); first=resolution(a.resolution) if a.resolution else None
        resolution_seconds=time.perf_counter()-tick
        second=closure(a.closure,a.output)
    result=dict(schema="calyx-resolution-closure-cold-v1",resolution=first,closure=second,
                author_checked=True,independently_accepted=False,scientific_admission=False,
                conditional_information_certified=False,served=False,
                resolution_readback_seconds=resolution_seconds,wall_seconds=time.perf_counter()-started,
                code_sha256=sha(__file__))
    write(a.output/"report.json",result)
    write(a.output/"manifest.json",dict(report_sha256=sha(a.output/"report.json")))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resolution",type=Path)
    for name in ("closure","output"):
        p.add_argument("--"+name,type=Path,required=True)
    main(p.parse_args())
