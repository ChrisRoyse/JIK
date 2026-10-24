"""Known-truth archive publication/crash controls; never a training corpus."""
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

from canonicalise import process_plan
from common import append_jsonl, sealed_shards
from contracts import ContractError
from plan_seals import make_seal


def controls(root, check):
    root.mkdir()
    raw = root / "known-truth-bundle.tar.gz"
    dest = root / "canon"

    def archive(bad=False, empty=False):
        with tarfile.open(raw, "w:gz") as tar:
            for name, content in ([] if empty else [("one.tsv", b"Q1\tP31\tQ2\n"),
                                                    ("two.tsv", b"invalid\n" if bad else b"Q3\tP31\tQ4\n")]):
                info = tarfile.TarInfo(name)
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
        shutil.copyfile(raw, root / ("snapshot-" + ("empty" if empty else "bad" if bad else "complete") + ".tar.gz"))
        return {"shard": "known-truth-bundle", "files": [raw], "sha256": [hashlib.sha256(raw.read_bytes()).hexdigest()],
                "kind": "archive"}

    def refused(label, path=dest):
        code = None
        try:
            sealed_shards(path)
        except ContractError as exc:
            code = exc.code
        check(label, code == "MNLM_CANON_PLAN_UNSEALED", code=code)

    plan = archive(bad=True)
    rows = process_plan(("wikidata5m", plan, dest, root))
    check("archive_actual_parser_partial_failure", len(rows) == 2 and not rows[0].get("error") and bool(rows[1].get("error")))
    for row in rows:
        append_jsonl(dest / "manifest.jsonl", row)
    refused("partial_archive_never_becomes_smaller_population")
    plan = archive()
    complete = process_plan(("wikidata5m", plan, dest, root))
    check("archive_retry_extracts_both_members", len(complete) == 2 and all(not r.get("error") for r in complete))
    # Simulate death between append/fsync of the first and second member row.
    append_jsonl(dest / "manifest.jsonl", complete[0])
    refused("crash_between_member_manifests_refuses")
    append_jsonl(dest / "manifest.jsonl", complete[1])
    refused("complete_members_without_plan_seal_refuse")
    seal = make_seal(rows + complete, complete)
    append_jsonl(dest / "plans.jsonl", seal)
    accepted = sealed_shards(dest)
    check("complete_archive_recovery_supersedes_old_failure", {r["shard"] for r in accepted} == {"one", "two"}
          and all(r["raw_sha256"] == plan["sha256"] for r in accepted))
    # A new unsealed attempt invalidates a previous valid plan; old receipts
    # cannot accept a subsequently appended member from another attempt.
    changed = copy.deepcopy(complete[0])
    changed["secs"] += 1
    append_jsonl(dest / "manifest.jsonl", changed)
    refused("prior_seal_cannot_accept_later_manifest_append")
    history = rows + complete + [changed]
    new_rows = [changed, complete[1]]
    append_jsonl(dest / "plans.jsonl", make_seal(history, new_rows))
    check("new_complete_plan_accepts_exact_current_members", len(sealed_shards(dest)) == 2)
    seal_file = dest / "plans.jsonl"
    with seal_file.open("a", encoding="utf-8") as f:
        forged = make_seal(history, new_rows)
        forged["sha256"] = "0" * 64
        f.write(json.dumps(forged) + "\n")
    refused("modified_plan_seal_refuses")
    plan = archive(empty=True)
    empty_dest = root / "empty-canon"
    empty = process_plan(("wikidata5m", plan, empty_dest, root))
    check("empty_extraction_is_not_success", len(empty) == 1
          and empty[0].get("error", {}).get("code") == "MNLM_CANON_PLAN_EMPTY" and not empty_dest.exists())
