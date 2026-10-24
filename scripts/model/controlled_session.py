"""Journaled decisions for controlled-feedback learners; original evidence stays fixed."""
from pathlib import Path

from claims.codata_source import check
from claims.episodic_memory import digest
from model.constellation_learning import decision
from model.controlled_learning import ControlledData, code_pins
from model.field_echo import EchoExperimentLoop
from model.tool_rehearsal import termination_gate
from model import field_checkpoint, rehearsal_retention, tool_retention


def candidate_path(session, name):
    check(isinstance(name, str) and name not in ('', '.', '..') and '/' not in name and '\\' not in name,
          'CONTROLLED_CANDIDATE_PATH')
    path = (session.controlled_root/name).resolve()
    check(path.parent == session.controlled_root, 'CONTROLLED_CANDIDATE_PATH')
    return path


class ControlledLoop(EchoExperimentLoop):
    def __init__(self, *args, controlled_root, **kwargs):
        self.controlled_root = Path(controlled_root).resolve()
        self.controlled_candidates = {}
        super().__init__(*args, **kwargs)

    def consider_controlled(self, name, pin, spec, expected=None):
        self._fresh()
        check(name not in self.controlled_candidates and all(r['candidate'] != name for r in self.decisions),
              'CONTROLLED_DUPLICATE_CANDIDATE')
        data = ControlledData(self, spec)
        model, _, manifest = field_checkpoint.load(candidate_path(self, name), data, expected=pin)
        check(manifest['step'] == data.plan['steps'] and manifest['parent'] == data.parent_pin,
              'CONTROLLED_CANDIDATE_INCOMPLETE')
        original, _, _ = field_checkpoint.start(data)
        baseline, candidate = data.evaluate(original), data.evaluate(model)
        result = decision(data, baseline, candidate, model)
        result = tool_retention.assess(data, result, candidate)
        result = termination_gate(data, baseline, candidate, result)
        result = rehearsal_retention.assess(data, candidate, result)
        previous = None if not self.controlled_candidates else next(reversed(self.controlled_candidates.values()))
        retention = {}
        if previous is not None:
            old = self.journal.payload(previous['payload'])['candidate']
            for key in old:
                a, b = rehearsal_retention.measurements(old[key]), rehearsal_retention.measurements(candidate[key])
                retention[key] = {kind:{n:dict(previous=v, candidate=b[kind][n], retained=b[kind][n] <= v)
                                       for n,v in values.items()} for kind,values in a.items()}
        kept = all(v['retained'] for row in retention.values() for group in row.values() for v in group.values())
        result = dict(result, controlled_predecessor_retention=retention,
            metric_gate_passed=result['metric_gate_passed'] and kept,
            reasons=result['reasons']+([] if kept else ['CONTROLLED_PREDECESSOR_REGRESSION']))
        payload = dict(baseline=baseline, candidate=candidate, result=result,
                       controlled_pairs=data.controlled, labels={k:e['region'] for k,e in data.examples.items()})
        body = dict(kind='controlled_feedback_candidate', candidate=name, pin=pin, parent=data.parent_pin,
            spec=spec, binding=digest(data.binding), result=result, payload=digest(payload),
            prior_head=self.head, code=code_pins())
        if expected is None:
            self.journal.append(body, payload)
        else:
            check(body == expected and payload == self.journal.payload(expected['payload']),
                  'CONTROLLED_DECISION_REPLAY')
        self.controlled_candidates[name] = body
        return body

    def require_selection(self, name):
        if name in self.controlled_candidates:
            check(self.controlled_candidates[name]['result']['selected'], 'TOOL_CANDIDATE_NOT_SELECTED')
            check(False, 'CONTROLLED_ACTIVATION_NOT_ADMITTED')
        return super().require_selection(name)

    def _replay(self, body):
        if body.get('kind') == 'controlled_feedback_candidate':
            check(body['code'] == code_pins(), 'CONTROLLED_CODE_PIN')
            self.consider_controlled(body['candidate'], body['pin'], body['spec'], expected=body)
        else:
            super()._replay(body)

    def snapshot(self):
        value = super().snapshot()
        if self.controlled_candidates:
            value['controlled_feedback_candidates'] = dict(self.controlled_candidates)
        return value
