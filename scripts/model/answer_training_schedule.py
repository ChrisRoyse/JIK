"""Private answer supervision over observed request fields, with no prompt labels."""
from copy import deepcopy

from model.capability_answer_targets import AnswerTargets
from text.answer_native_plan import AnswerNativePlan
from text.capability_answer_source import stamp
from text.constellation_corpus import canonical, digest, need


def decisions(contexts, answers):
    need(type(contexts) is int and type(answers) is int and contexts > 0 and answers > 0,
         'ANSWER_SCHEDULE_COUNTS')
    return [dict(position=contexts-1+i, answer_index=i if i < answers else None, end=i == answers)
            for i in range(answers+1)]


class AnswerSchedule:
    def __init__(self, root, pin):
        self.plan = AnswerNativePlan(root, pin)
        ref = self.plan.report['source']; self.targets = AnswerTargets(ref['path'], ref['pin'])
        self.rows = {}; self.seals = {}
        for key, mapping in self.plan.tasks.items():
            original = self.targets.rows[key]; request = self.targets.request(key)
            need(mapping['split'] == original['split'] and mapping['population'] == original['population']
                 and mapping['split_group'] == original['split_group'], 'ANSWER_SCHEDULE_PARTITION')
            expected = [dict(field=f['id'], input=digest(f['text'].encode()), role=f['role'], clock=f['clock'])
                        for f in request['context']]
            need(mapping['context'] == expected and not mapping['field_order_is_event_chronology']
                 and not mapping['native_windows_are_answer_atoms'], 'ANSWER_SCHEDULE_CONTEXT')
            answer = self.targets.target(key, fitting=mapping['split'] == 'train')
            need(mapping['targets'] == [a['measurement'] for a in answer['atoms']], 'ANSWER_SCHEDULE_TARGETS')
            row = dict(task=key, split=mapping['split'], group=mapping['split_group'], family=original['family'],
                request=dict(operation=request['operation'], fields=deepcopy(expected),
                             position_kind='request_field_layout_not_event_chronology'),
                private=dict(targets=list(mapping['targets']), end=answer['end'],
                             decisions=decisions(len(expected), len(mapping['targets']))))
            self.rows[key] = row; self.seals[key] = digest(canonical(row))
        # Authenticate full publications at construction. Per-request checks are
        # file identities and the consumed row, not serialisation of all inputs.
        self.stamps = {str(self.plan.root/n):stamp(self.plan.root/n)
                       for n in (*self.plan.report['files'], 'report.json', 'COMMITTED')}

    def fresh(self, key):
        self.targets.fresh(key)
        need(all(stamp(p) == s for p,s in self.stamps.items()), 'ANSWER_SCHEDULE_PUBLICATION_CHANGED')
        need(key in self.rows and digest(canonical(self.rows[key])) == self.seals[key], 'ANSWER_SCHEDULE_CHANGED')

    def request(self, key):
        self.fresh(key)
        return deepcopy(self.rows[key]['request'])

    def supervision(self, key, *, fitting):
        self.fresh(key)
        need(type(fitting) is bool and self.rows[key]['split'] == ('train' if fitting else 'query'),
             'ANSWER_SCHEDULE_FIT_SPLIT')
        return deepcopy(self.rows[key]['private'])
