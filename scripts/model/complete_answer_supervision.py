"""Purpose-bound complete answers; recorded failures are never successful repairs."""
from copy import deepcopy
from types import SimpleNamespace

from claims.episodic_memory import digest
from model.capability_supervision import CapabilitySupervision, PURPOSES
from text.arithmetic_tasks import equalities
from text.constellation_corpus import digest as byte_digest, need


def arithmetic(text):
    rows = equalities(text)
    coverage = text.count('<<') == text.count('>>') == len(rows)
    return dict(records=rows, delimiter_coverage=coverage,
        all_recognized_equalities_pass=bool(rows) and coverage and all(r['status']=='pass' for r in rows),
        scope='annotated_equalities_only', problem_interpretation_checked=False)


class CompleteAnswerSupervision:
    def __init__(self, view):
        self.view = view
        corpus = SimpleNamespace(train=view.source.train,query=view.source.query,
            sources={k:SimpleNamespace(population=p) for k,p in view.source.populations.items()},
            tasks={k:dict(population=r['population'],binding=r['source_task'],family=r['family'],
                          kind=r['operation'],split_group=r['split_group']) for k,r in view.rows.items()})
        self.single = CapabilitySupervision(corpus); self.rows = {}; self.text = {}; self.seals = {}
        publication = byte_digest((view.root/'report.json').read_bytes())
        for key in view.source.train+view.source.query:
            a = view.target(key,fitting=key in view.source.train); pop,_,_ = view.source.selected(key)
            raw = pop.fields[a['source_field']]['text'].encode(); lo,hi = a['reference_span']; raw = raw[lo:hi]
            need(byte_digest(raw) == a['reference_sha256'], 'COMPLETE_SUPERVISION_REFERENCE')
            original = self.single.evidence(key)
            if original['scope'] == 'retained_execution_test_scope':
                witness = original['observations'][-1]
                receipt = self.single.executions[(witness['population'],witness['condition'])]
                need(receipt['action_sha256'] == a['reference_sha256'], 'COMPLETE_SUPERVISION_EXECUTED_PATCH')
            row = dict(task=key,split=original['split'],group=original['group'],family=original['family'],
                operation=original['operation'],source_publication=publication,
                original_evidence=original,source_field=a['source_field'],answer_sha256=a['reference_sha256'],
                reference_span=a['reference_span'],atoms=[x['measurement'] for x in a['atoms']],end=a['end'],
                arithmetic=arithmetic(raw.decode()),scientific_admission=False,world_correctness_established=False)
            self.rows[key] = row; self.text[key] = raw; self.seals[key] = digest(row)

    def evidence(self,key):
        need(key in self.rows,'COMPLETE_SUPERVISION_TASK'); self.view.fresh(key)
        need(digest(self.rows[key]) == self.seals[key] and byte_digest(self.text[key]) == self.rows[key]['answer_sha256'],
             'COMPLETE_SUPERVISION_MUTATED')
        return deepcopy(self.rows[key])

    def decide(self,key,purpose,*,fitting=False):
        need(purpose in PURPOSES and type(fitting) is bool,'COMPLETE_SUPERVISION_PURPOSE')
        need(not fitting or key in self.view.source.train,'COMPLETE_SUPERVISION_QUERY_FIT')
        e = self.evidence(key); old = self.single.decide(key,purpose,fitting=fitting)
        need(old['evidence'] == digest(e['original_evidence']), 'COMPLETE_SUPERVISION_ORIGINAL_CHANGED')
        allowed,reason = old['allowed'],old['refusal']
        if purpose == 'annotated_arithmetic':
            allowed = e['original_evidence']['scope'] == 'annotated_equalities_only' and e['arithmetic']['all_recognized_equalities_pass']
            reason = None if allowed else 'COMPLETE_SUPERVISION_ARITHMETIC_UNESTABLISHED'
        return dict(task=key,purpose=purpose,fitting=fitting,split=e['split'],allowed=allowed,
            refusal=reason,evidence=self.seals[key],answer_sha256=e['answer_sha256'],atom_count=len(e['atoms']),
            scope=e['original_evidence']['scope'],observed_outcome=e['original_evidence']['outcome'])

    def training_target(self,key,*,purpose):
        decision = self.decide(key,purpose,fitting=True); need(decision['allowed'],decision['refusal'])
        return decision,self.view.target(key,fitting=True)

    def assess(self,key,raw):
        need(isinstance(raw,bytes) and raw,'COMPLETE_SUPERVISION_SURFACE'); e = self.evidence(key)
        try: text = raw.decode('utf-8',errors='strict')
        except UnicodeDecodeError:
            return dict(task=key,answer_sha256=byte_digest(raw),utf8_valid=False,source_agreement=False,
                        accepted=False,serving=False,reason='COMPLETE_SUPERVISION_UTF8')
        match = raw == self.text[key]; original = e['original_evidence']; execution = None
        if match and original['scope'] == 'retained_execution_test_scope':
            execution = dict(outcome=original['outcome'],observations=original['observations'],
                status='matches_recorded_passing_patch' if original['outcome']==1 else 'matches_recorded_failed_patch',
                scope='this_recorded_task_action_and_pinned_runtime_only',new_execution=False)
        return dict(task=key,answer_sha256=byte_digest(raw),utf8_valid=True,source_agreement=match,
            evidence=self.seals[key],reference_sha256=e['answer_sha256'],arithmetic=arithmetic(text),execution=execution,
            requires_new_execution=original['scope']=='retained_execution_test_scope' and not match,
            source_scope=original['scope'],accepted=False,serving=False,scientific_admission=False,
            world_correctness_established=False,problem_interpretation_checked=False,
            response_appropriateness_checked=False,donor_execution_transfers=False)
