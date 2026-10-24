"""Independent byte/SQL/clock/native-score readback; no model or oracle imports."""
import argparse
import base64
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha(v):
    return hashlib.sha256(v).hexdigest()


def journal(path):
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        assert db.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        contract = db.execute("SELECT contract FROM metadata").fetchone()[0]
        payloads = {}
        for pin, raw in db.execute("SELECT hash,data FROM payloads"):
            assert sha(raw) == pin and canonical(json.loads(raw)) == raw
            payloads[pin] = json.loads(raw)
        events, head = [], "0"*64
        for seq, previous, raw, pin in db.execute("SELECT seq,previous,body,hash FROM events ORDER BY seq"):
            body = json.loads(raw)
            assert seq == len(events) and previous == head and canonical(body) == raw
            assert sha(canonical(dict(seq=seq,previous=previous,body=body))) == pin
            events.append(body)
            head = pin
    return json.loads(contract), events, payloads, head


def distance(a,b):
    a=a/np.linalg.norm(a,axis=1,keepdims=True)
    b=b/np.linalg.norm(b,axis=1,keepdims=True)
    cos=np.clip(a@b.T,-1,1)
    costs=np.empty((len(a)+1,len(b)+1))
    costs[:,0],costs[0,:]=np.arange(len(a)+1),np.arange(len(b)+1)
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            costs[i,j]=min(costs[i-1,j]+1,costs[i,j-1]+1,costs[i-1,j-1]+1-cos[i-1,j-1])
    return float(costs[-1,-1]/max(len(a),len(b)))


def run(config, *, forecast_root=None, snapshot=None, output=None):
    start=time.perf_counter()
    c=json.loads(config.read_bytes());root=Path(c["output"]);obs=Path(c["observations"])
    sealed_root=forecast_root or root/"forecast"
    sealraw=(sealed_root/"forecast.json").read_bytes();pin=sha(sealraw);seal=json.loads(sealraw)
    assert pin==(sealed_root/"COMMITTED").read_text()
    assert sha((sealed_root/"journal.sqlite3").read_bytes())==seal["journal"]
    contract,events,payloads,head=journal(sealed_root/"journal.sqlite3")
    assert head==seal["head"] and len(events)==seal["events"] and sha(canonical(contract))==seal["contract"]
    assert "process_observations" not in contract and "execution_inputs" not in contract
    prior_enrolled={k for e in events if e["kind"]=="execution_enrollment" for k in e["links"]}
    assert not prior_enrolled.intersection(seal["forecasts"])
    forecasts={}
    for event in events:
        payload=payloads.get(event.get("payload"),{})
        if "forecast" in payload:
            r=payload["forecast"];forecasts[sha(canonical(r))]=r
    for k,r in seal["forecasts"].items():assert forecasts[k]==r
    c2,e2,p2,h2=journal(root/"experience.sqlite3")
    assert c2==contract and e2[:len(events)]==events
    enrolled=e2[len(events)]
    assert enrolled["kind"]=="execution_enrollment" and enrolled["prior_head"]==head
    assert enrolled["spec"]["forecast_pin"]==pin
    raw=(obs/"manifest.json").read_bytes();manifest=json.loads(raw)
    assert sha(raw)==enrolled["spec"]["observation_pin"]==(obs/"COMMITTED").read_text()
    assert manifest["forecast"]==dict(root="forecast",pin=pin,scope="new_occurrence_after_exposed_program_forecast")
    for name,v in manifest["files"].items():
        raw=(obs/name).read_bytes();assert len(raw)==v["bytes"] and sha(raw)==v["sha256"]
    joins=[]
    for key,observation in enrolled["links"].items():
        record=seal["forecasts"][key];entry=manifest["observations"][observation]
        execution=json.loads((obs/entry["execution"]/"execution.json").read_bytes())
        after=json.loads((obs/entry["execution"]/"after.json").read_bytes())
        assert entry["condition"]==record["request"]["execution_conditions"]["native"]["condition"]
        assert entry["target"]["turn"]==record["request"]["turn"]+1
        assert entry["target"]["task"]==record["request"]["task"]
        sealed=datetime.fromisoformat(seal["sealed_at"])
        began=datetime.fromisoformat(after["State"]["StartedAt"].replace("Z","+00:00"))
        assert sealed<began and execution["container_id"]==after["Id"] and not after["State"]["Running"]
        captured={}
        for stream in ("stdout","stderr"):
            item=execution["result"][stream];raw=base64.b64decode(item["base64"],validate=True)
            assert len(raw)==item["bytes"] and sha(raw)==item["sha256"]
            captured[stream]=raw.decode()
        joins.append(dict(forecast=key,observation=observation,sealed_at=seal["sealed_at"],
            started=after["State"]["StartedAt"],lead_seconds=(began-sealed).total_seconds(),
            returncode=execution["result"]["returncode"],captured=captured))
    middle=time.perf_counter()
    spec=enrolled["spec"];native=Path(spec["native"]);population=Path(spec["population"])
    refs=[json.loads(r) for r in (population/"references.jsonl").read_bytes().splitlines()]
    nm=json.loads((native/"report.json").read_bytes())
    assert sha((native/"report.json").read_bytes())==spec["native_pin"]
    base_population=Path(c["population"])
    base_report=json.loads((base_population/"report.json").read_bytes())
    assert sha((base_population/"report.json").read_bytes())==c["population_pin"]
    span_raw=(base_population/"reading_spans.jsonl").read_bytes()
    # Baselines address a source message, which may contain several measured atoms.
    span_pin=base_report["files"]["reading_spans.jsonl"]
    assert sha(span_raw)==(span_pin["sha256"] if isinstance(span_pin,dict) else span_pin)
    spans=[json.loads(r) for r in span_raw.splitlines()]
    base_manifest=json.loads((Path(c["native"])/"report.json").read_bytes())
    assert sha((Path(c["native"])/"report.json").read_bytes())==c["native_pin"]
    scores={}
    for event in e2[len(events)+1:]:
        payload=p2.get(event.get("payload"),{})
        if "execution" not in payload:continue
        ob=payload["execution"]["observation"]
        if ob not in enrolled["links"].values():continue
        key=next(k for k,v in enrolled["links"].items() if v==ob)
        record=forecasts[key];scores[key]={}
        for name,lane in nm["lenses"].items():
            lane_raw=(native/name/"report.json").read_bytes();assert sha(lane_raw)==lane["report_sha256"]
            lane_report=json.loads(lane_raw)
            assert sha((native/name/"slots.npz").read_bytes())==lane_report["files"]["slots.npz"]["sha256"]
            with np.load(native/name/"slots.npz",allow_pickle=False) as z:
                target=np.concatenate([z["vectors"][z["offsets"][r["unit"]]:z["offsets"][r["unit"]+1]]
                    for r in refs if r["observation"]==ob and r["unit"] is not None]).astype(np.float64)
            v=record["predictions"][name];raw=base64.b64decode(v["base64"],validate=True);assert sha(raw)==v["sha256"]
            prediction=np.frombuffer(raw,dtype="<f4").reshape(v["shape"]).astype(np.float64)
            base_lane_raw=(Path(c["native"])/name/"report.json").read_bytes()
            assert sha(base_lane_raw)==base_manifest["lenses"][name]["report_sha256"]
            assert sha((Path(c["native"])/name/"slots.npz").read_bytes())==json.loads(base_lane_raw)["files"]["slots.npz"]["sha256"]
            with np.load(Path(c["native"])/name/"slots.npz",allow_pickle=False) as z:
                baseline=np.concatenate([z["vectors"][z["offsets"][r["atom"]]:z["offsets"][r["atom"]+1]]
                    for r in spans if r["unit"]==record["baseline_unit"]]).astype(np.float64)
            score=dict(prediction_loss=distance(target,prediction),persistence_loss=distance(target,baseline))
            assert all(abs(v-payload["assessment"]["instruments"][name][k])<1e-12 for k,v in score.items())
            scores[key][name]=score
    assert len(scores)==len(joins)==len(enrolled["links"])
    final=json.loads((snapshot or root/"snapshot.json").read_bytes())
    assert h2==final["head"] and len(e2)==final["events"]
    end=time.perf_counter();phases=dict(chronology=middle-start,native_scores=end-middle)
    result=dict(complete=True,joins=joins,scores=scores,forecast_head=head,final_head=h2,
        sealed_events=len(events),final_events=len(e2),phases=phases,wall_seconds=end-start,
        phase_coverage=sum(phases.values())/(end-start),model_imported=False,acceptance="author-checked only")
    with (output or root/"independent.json").open("xb") as f:f.write(canonical(result))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True)
    p.add_argument("--forecast-root",type=Path)
    p.add_argument("--snapshot",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    r=run(a.config,forecast_root=a.forecast_root,snapshot=a.snapshot,output=a.output)
    print(json.dumps({k:r[k] for k in ("complete","joins","sealed_events","final_events","wall_seconds")}))
