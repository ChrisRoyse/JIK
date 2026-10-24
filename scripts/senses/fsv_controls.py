"""Known-truth boundary controls. These are not corpus/training measurements."""
from __future__ import annotations

import copy
import json
import sqlite3

from .contract import CHANNELS, canonical, digest, field, present, seal_record
from .instruments import contracts, measure, slot_digest
from .store import Store
from .views import view
from .referents import ReferentMap


def run_controls(dest, template, check, refuses):
    controls = []
    for i, (source, day) in enumerate((("control-a", 1), ("control-a", 2), ("control-b", 3), ("control-b", 1))):
        r = copy.deepcopy(template)
        r["locator"].update(source=source, shard="known-truth-control", row=i)
        r["unit"] = f"control-unit-{i}"
        r["referents"] = [{"key": "control:shared", "method": "source_identifier", "confidence": 1, "evidence": "known-truth control only"}]
        r["channels"]["surface"] = present([{"text": "target-label and independently observed context"}])
        r["channels"]["fields"] = present([field("observed", f"2026-01-0{day}T00:00:00Z", "clock", "UTC", "observed"),
                                           field("hidden_label", "target-label", "category")])
        r["channels"]["behaviour"] = present([{"event": "completed", "mode": "observed", "evidence": "control-event", "measurements": [field("duration", i + 1, "number", "second")]}])
        r["channels"]["perception"] = present([{"modality": "sensor", "sha256": digest([i]), "media_type": "application/x-control",
                                                "measurements": [field("amplitude", i + 1, "number", "volt")]}])
        controls.append(seal_record(r))
    (dest / "known-truth-controls.jsonl").write_bytes(b"".join(canonical(r) + b"\n" for r in controls))
    s = Store(dest / "controls.sqlite", create=True)
    s.ingest(controls, {"input_sha256": digest(controls), "count": len(controls), "kind": "known-truth controls only"}, None)
    target = controls[1]
    hist = s.context(target, "history", s.head)
    check("history_strict_past", {r["atom_id"] for r in hist} == {controls[0]["atom_id"], controls[3]["atom_id"]})
    cross = s.context(target, "co_reference", s.head)
    check("cross_source_no_future", [r["atom_id"] for r in cross] == [controls[3]["atom_id"]])
    policy = {"schema": "ucr-view-v1", "question": "control", "allow_channels": list(CHANNELS),
              "deny_fields": ["hidden_label"], "deny_relations": ["P31"], "blocked_units": [controls[3]["unit"]],
              "label_aliases": ["target-label"], "rationale": "known-truth leakage boundary control"}
    v, ctx, bindings = view(s, target, policy, s.head)
    forged = copy.deepcopy(target)
    forged["channels"]["surface"] = present([{"text": "invented answer under a real identifier"}])
    refuses("record_identity_cannot_launder_observations", lambda: view(s, forged, policy, s.head), "MNLM_UCR_VIEW_BINDING")
    refuses("calibration_cannot_replace_source", lambda: view(s, target, policy, s.head, surface="invented answer"),
            "MNLM_UCR_SURFACE_BINDING")
    check("blocked_label_unit", ctx["co_reference"] == [])
    check("aliases_masked_target_and_context", "target-label" not in canonical([v, ctx]).decode())
    check("oracle_field_removed", all(f["name"] != "hidden_label" for f in v["channels"]["fields"]["items"]))
    check("oracle_relation_removed", v["channels"]["relations"]["state"] == "absent")
    check("behaviour_measurement", measure(v, "behaviour")["kind"] == "Multi")
    check("perception_measurement", measure(v, "perception")["kind"] == "Multi")
    # Same-source text copies cannot become a cross-source sense.
    check("co_reference_requires_another_source", all(r["locator"]["source"] != target["locator"]["source"] for r in cross))
    bad = copy.deepcopy(target)
    bad["referents"][0]["confidence"] = 0.8
    refuses("guessed_identity_rejected", lambda: seal_record(bad), "MNLM_UCR_REFERENT")
    bad = copy.deepcopy(target)
    bad["channels"]["new-name-for-same-text"] = bad["channels"]["surface"]
    refuses("invented_sense_name_rejected", lambda: seal_record(bad), "MNLM_UCR_SCHEMA")
    excluded = copy.deepcopy(target)
    excluded["locator"]["source"] = "cuyahoga-eighth-district"
    seal_record(excluded)
    old = s.head
    refuses("operator_source_exclusion", lambda: s.ingest([excluded], {"input_sha256": digest(excluded), "count": 1}, old), "MNLM_SOURCE_EXCLUDED")
    check("excluded_source_no_mutation", s.head == old)
    # A bad record after a valid new observation must roll the *whole* batch back.
    first = copy.deepcopy(target)
    first["locator"]["row"] = 80
    seal_record(first)
    second = copy.deepcopy(first)
    second["atom_id"] = "0" * 64
    refuses("mid_batch_rollback", lambda: s.ingest([first, second], {"input_sha256": digest([first, second]), "count": 2}, old), "MNLM_UCR_ID")
    check("rolled_back_valid_prefix", s.head == old and s.db.execute("SELECT count(*) FROM records WHERE atom=?", (first["atom_id"],)).fetchone()[0] == 0)
    artifact = {"schema": "ucr-referents-v1", "rows": [{"source": "control-a", "shard": "known-truth-control", "row": 0, "status": "resolved", "key": "control:shared", "evidence": "known-truth"},
                                                    {"source": "control-b", "shard": "known-truth-control", "row": 8, "status": "unresolved", "key": None, "evidence": "no matching identity"}]}
    mapping = ReferentMap(artifact, digest(artifact))
    check("heldout_identity_gold", mapping.evaluate([{"source": "control-a", "shard": "known-truth-control", "row": 0, "key": "control:shared"}])["correct"] == 1)
    check("unresolved_identity_not_guessed", mapping.evaluate([{"source": "control-b", "shard": "known-truth-control", "row": 8, "key": "control:other"}])["unresolved"] == 1)
    # SQLite protections are defense in depth; physical readback also detects tampering.
    s.close()
    conn = sqlite3.connect(dest / "controls.sqlite")
    conn.execute("DROP TRIGGER immutable_slots_UPDATE")
    conn.execute("UPDATE slots SET digest=? WHERE atom=?", ("0" * 64, controls[0]["atom_id"]))
    conn.commit()
    conn.close()
    corrupt = Store(dest / "controls.sqlite")
    refuses("physical_slot_tamper", lambda: corrupt.read_slot(controls[0]["atom_id"], contracts()[0]["lens_id"]), "MNLM_UCR_SLOT_DIGEST")
    corrupt.close()
