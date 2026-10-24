"""Replayed candidate decisions in the same chain as their grounded outcomes."""
from pathlib import Path

from claims.codata_source import check
from claims.episodic_memory import digest
from model.constellation_loop import ConstellationLoop
from model.constellation_learning import LearningData, code_pins, decision
from model.constellation_training_checkpoint import load


class LearningConstellationLoop(ConstellationLoop):
    def __init__(self, *args, candidate_root, **kwargs):
        self.candidate_root = Path(candidate_root).resolve()
        self.learning_history = []
        self.learning_code = code_pins()
        super().__init__(*args, **kwargs)

    def _candidate_path(self, name):
        check(isinstance(name, str) and name and Path(name).name == name
              and name not in (".", "..") and "/" not in name and "\\" not in name,
              "LEARNING_CANDIDATE_PATH")
        path = (self.candidate_root/name).resolve()
        check(path.parent == self.candidate_root, "LEARNING_CANDIDATE_PATH")
        return path

    def _decision_body(self, name, pin, train, evaluate):
        data = LearningData(self, train=train, evaluate=evaluate)
        model, _, manifest = load(self._candidate_path(name), data, expected=pin)
        check(manifest["step"] == data.plan["steps"], "LEARNING_INCOMPLETE_CANDIDATE")
        check(manifest["parent"] == self.loop_contract["checkpoint"], "LEARNING_CANDIDATE_PARENT")
        baseline, candidate = data.evaluate(self.model), data.evaluate(model)
        result = decision(data, baseline, candidate, model)
        return dict(kind="constellation_learning_decision", candidate=name, pin=pin,
            train=list(train), evaluate=list(evaluate), binding=digest(data.binding),
            baseline=digest(baseline), assessment=digest(candidate),
            result=result, code=self.learning_code, prior_head=self.head)

    def consider(self, name, *, pin, train, evaluate):
        self._fresh()
        check(not any(r["candidate"] == name for r in self.learning_history), "LEARNING_DUPLICATE")
        body = self._decision_body(name, pin, train, evaluate)
        self._append(body)
        self.learning_history.append(body)
        return body

    def require_selection(self, name):
        self._fresh()
        found = [r for r in self.learning_history if r["candidate"] == name]
        check(len(found) == 1 and found[0]["result"]["selected"], "LEARNING_CANDIDATE_NOT_SELECTED")

    def _replay(self, body):
        if body.get("kind") == "constellation_learning_decision":
            check(not any(r["candidate"] == body["candidate"] for r in self.learning_history),
                  "LEARNING_DUPLICATE")
            check(body == self._decision_body(body["candidate"], body["pin"], body["train"], body["evaluate"]),
                  "LEARNING_DECISION_REPLAY")
            self.learning_history.append(body)
        else:
            super()._replay(body)

    def snapshot(self):
        return dict(**super().snapshot(), learning_history=self.learning_history,
                    active_model=self.loop_contract["checkpoint"])
