"""Count exact target references from the actual committed source-successor tape."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.text_learning_readback import journal
from senses.trajectory_population import publish
from text.constellation_corpus import file_sha, need


def run(a):
    need(file_sha(a.run/"report.json") == a.pin, "REGION_TARGET_AUDIT_PIN")
    report = json.loads((a.run/"report.json").read_bytes())
    base = report["data"]
    while "groups" not in base:
        base = base["base"]
    groups = {i: g for g in base["groups"] for i in g["indices"]}
    expected = report["tapes"]["baseline"]
    checked = journal(a.run/"baseline.db", expected["retained"], expected)
    counts = {split: dict(transitions=0, target_available=0, target_selected_as_label=0,
                          own_unobserved_references=0) for split in ("train", "development")}
    with sqlite3.connect(f"file:{a.run/'baseline.db'}?mode=ro", uri=True) as db:
        for raw, in db.execute("SELECT body FROM events ORDER BY seq"):
            body = json.loads(raw)
            if body["kind"] != "source_reveal":
                continue
            payload = json.loads(db.execute("SELECT data FROM payloads WHERE hash=?", (body["payload"],)).fetchone()[0])
            group = groups[body["left"]]
            available = payload["region_vocabulary"]
            row = counts[group["split"]]
            row["transitions"] += 1
            row["target_available"] += body["right"] in available
            row["target_selected_as_label"] += available[payload["label"]] == body["right"]
            future = group["indices"][group["indices"].index(body["left"])+1:]
            row["own_unobserved_references"] += len(set(available) & set(future))
    need(counts["train"]["transitions"] == 104 and counts["development"]["transitions"] == 93,
         "REGION_TARGET_AUDIT_COVERAGE")
    result = dict(schema="text-region-target-availability-audit-v1", source_report=a.pin,
        counts=counts, journal=checked, acceptance="author-checked only",
        training_development_exact_target_mismatch=bool(counts["train"]["target_available"]
            and not counts["development"]["target_available"]), scientific_admission=False)
    publish(a.output, result)
    print(json.dumps(dict(report_sha256=file_sha(a.output), counts=counts)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--pin", required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
