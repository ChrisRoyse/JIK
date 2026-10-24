"""Authenticate actual recurrent weights by replaying their source prediction tapes."""
import json
from pathlib import Path

from claims.codata_source import check
from model.loop_training_checkpoint import fingerprint
from model.text_context_regions import PrefixSession
from model.text_dynamics_checkpoint import load
from model.text_sequence_data import PURPOSE
from senses.contract import canonical, digest
from text.constellation_corpus import file_sha


class CandidateEvidence:
    def __init__(self, root, expected, model, data):
        from audit.text_learning_support import tape
        self.root, self.model, self.data = Path(root), model, data
        self.source_contract = canonical(data.bind)
        check(file_sha(self.root/"report.json") == expected, "TEXT_SELECTION_REPORT_PIN")
        report = json.loads((self.root/"report.json").read_bytes())
        check(report["schema"] == "text-dynamics-learning-witness-v1" and report["data"] == data.bind,
              "TEXT_SELECTION_SOURCE_BINDING")
        self.report_pin, self.evidence = expected, report["evidence"]
        records = {}
        paths = [self.root/"report.json"]
        for name, tape_name, step in (("initial-training", "baseline", 0), ("cycle-1", "cycle-1", 16), ("cycle-2", "cycle-2", 40)):
            checkpoint = self.root/name
            manifest = json.loads((checkpoint/"manifest.json").read_bytes())
            tape_checkpoint = self.root/"initial-model" if step == 0 else checkpoint/"model"
            tape_pin = report["initial_model"] if step == 0 else manifest["model"]
            measured, examples = tape(self.root/(tape_name+".db"), model, data,
                tape_checkpoint, tape_pin, retained=report["tapes"][tape_name]["retained"],
                session_class=PrefixSession)
            check(measured == report["tapes"][tape_name], "TEXT_SELECTION_NEURAL_REPLAY")
            if step == 0:
                data.bind_examples(examples)
            optimizer, loaded = load(checkpoint, model, data, expected=report["checkpoints"][name],
                                     evidence=self.evidence, purpose=PURPOSE)
            check(loaded == manifest and manifest["step"] == step, "TEXT_SELECTION_CHECKPOINT")
            parent = None if step == 0 else report["checkpoints"]["initial-training" if step == 16 else "cycle-1"]
            check(manifest["parent"] == parent, "TEXT_SELECTION_PARENT")
            records[name] = dict(checkpoint=report["checkpoints"][name], model=manifest["model"],
                fingerprint=fingerprint(model, optimizer), parent=parent, frozen=manifest["frozen"], step=step,
                tape=measured["retained"], outcomes=measured["outcomes"],
                development=measured["summary"]["development"])
            paths.extend([self.root/(tape_name+".db"), checkpoint/"manifest.json", checkpoint/"COMMITTED",
                          checkpoint/"training.pt", checkpoint/"model"/"manifest.json",
                          checkpoint/"model"/"COMMITTED", checkpoint/"model"/"state.pt"])
            paths.extend(tape_checkpoint/n for n in ("manifest.json", "COMMITTED", "state.pt"))
        check(len({r["frozen"] for r in records.values()}) == 1, "TEXT_SELECTION_FROZEN_REFERENCE")
        self._records = canonical(records)
        self.pin = digest(dict(report=expected, records=records))
        self.stamps = {p: (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino) for p in paths}
        self.activate("initial-training")

    def records(self):
        return json.loads(self._records)

    def fresh(self):
        self.data.inputs.fresh()
        check(canonical(self.data.bind) == self.source_contract, "TEXT_SELECTION_SOURCE_BINDING")
        check(all(p.is_file() and (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino) == stamp
                  for p, stamp in self.stamps.items()), "TEXT_SELECTION_EVIDENCE_CHANGED")

    def activate(self, name):
        self.fresh()
        check(name in self.records(), "TEXT_SELECTION_CHECKPOINT_NAME")
        record = self.records()[name]
        self.optimizer, _ = load(self.root/name, self.model, self.data, expected=record["checkpoint"],
                                evidence=self.evidence, purpose=PURPOSE)
        check(self.fingerprint() == record["fingerprint"], "TEXT_SELECTION_ACTIVATION")
        return self.fingerprint()

    def fingerprint(self):
        return fingerprint(self.model, self.optimizer)
