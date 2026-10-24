"""Versioned consumers for source-relative revisions and clocked dialogue continuation."""
from fractions import Fraction
import re

from claims.capability_population import CapabilityPopulation, need


class RevisionPopulation(CapabilityPopulation):
    def __init__(self, root, pin):
        super().__init__(root, pin)
        kind = self.report['source'].get('source_kind')
        need(kind in ('condaqa-evidence-revision-v1', 'ami-chronological-continuation-v1'),
             'REVISION_SOURCE_KIND')
        if kind == 'ami-chronological-continuation-v1':
            need(self.family == 'conversation', 'REVISION_SOURCE_FAMILY')
        else:
            need(self.family in ('conversation', 'reasoning'), 'REVISION_SOURCE_FAMILY')

    def conversation(self, t):
        if t['kind'] == 'respond_to_source_question':
            return self.reading(t, revision=False)
        need(t['kind'] == 'continue_attested_dialogue', 'CONTINUATION_KIND')
        need(self.report['source']['source_kind'] == 'ami-chronological-continuation-v1',
             'CONTINUATION_SOURCE')
        c, y = ([self.fields[i] for i in t[k]] for k in ('context', 'targets'))
        p = t['provenance']
        need(1 <= len(c) <= 8 and len(y) == 1 and not t['consequences'], 'CONTINUATION_SHAPE')
        need(all(f['role'] == 'attested_speaker_act' and f['clock'] for f in c + y),
             'CONTINUATION_ROLE')
        need(p['cutoff'] == y[0]['clock']['start']
             and all(Fraction(f['clock']['end']) <= Fraction(p['cutoff']) for f in c)
             and len({f['clock']['id'] for f in c + y}) == 1, 'CONTINUATION_FUTURE')
        need(all(f['clock']['id'] == 'ami-annotation:' + p['meeting'] for f in c + y)
             and [f['source']['element'] for f in c] == p['context_acts']
             and [y[0]['source']['element']] == p['target_acts'], 'CONTINUATION_JOIN')
        need(t['split_group'] == 'ami-series:' + re.sub('[a-d]$', '', p['meeting']),
             'CONTINUATION_SPLIT_GROUP')
        need(p['annotation_response_link_asserted'] is False
             and p['response_appropriateness_checked'] is False and p['world_claims_checked'] is False,
             'CONTINUATION_CLAIM_SCOPE')
        for f in c:
            self.edge(f['id'], y[0]['id'], 'completed_before_utterance')

    def reasoning(self, t):
        need(t['kind'] == 'revise_with_source_evidence', 'REVISION_KIND')
        self.reading(t, revision=True)

    def reading(self, t, *, revision):
        source, p = self.report['source'], t['provenance']
        need(source['source_kind'] == 'condaqa-evidence-revision-v1', 'READING_SOURCE')
        c, y = ([self.fields[i] for i in t[k]] for k in ('context', 'targets'))
        expected = ['source_passage', 'source_question']
        if revision:
            expected += ['source_answer', 'source_passage']
        need([f['role'] for f in c] == expected and len(y) == 1 and y[0]['role'] == 'source_answer'
             and not t['consequences'], 'READING_ROLES')
        need(all(f['clock'] is None and f['source']['artifact_sha256'] == source['files']['train.json']
                 and f['source']['passage_id'] == p['passage_id']
                 and f['source']['question_id'] == p['question_id'] for f in c + y), 'READING_JOIN')
        need(t['split_group'] == 'condaqa:' + source['files']['train.json'] + ':passage:'
             + str(p['passage_id']), 'READING_SPLIT_GROUP')
        need(p['temporal_order_asserted'] is False and p['source_judgment_is_world_fact'] is False
             and p['human_judgment_checked'] is False, 'READING_CLAIM_SCOPE')
        need(p['artifact_sha256'] == source['files']['train.json']
             and y[0]['source']['row'] == p['row'] and y[0]['source']['edit_id'] == p['edit_id']
             and y[0]['source']['field'] == 'label', 'READING_TARGET_SOURCE')
        if revision:
            need(all(f['source']['row'] == p['original_row'] and f['source']['edit_id'] == 0
                     for f in c[:3]) and c[3]['source']['row'] == p['row']
                 and c[3]['source']['edit_id'] == p['edit_id'] and p['edit_id'] in (1, 2, 3),
                 'REVISION_DIRECTION')
            need([f['source']['field'] for f in c] == ['sentence1', 'sentence2', 'label', 'sentence1'],
                 'REVISION_FIELD_KEYS')
            self.edge(c[0]['id'], c[2]['id'], 'source_answer_from_passage')
            self.edge(c[1]['id'], c[2]['id'], 'source_answer_for_question')
            self.edge(c[0]['id'], c[3]['id'], 'provider_passage_edit')
            self.edge(c[2]['id'], y[0]['id'], 'provider_answer_revision')
            self.edge(c[3]['id'], y[0]['id'], 'source_answer_from_passage')
        else:
            need(all(f['source']['row'] == p['row'] and f['source']['edit_id'] == p['edit_id'] for f in c)
                 and [f['source']['field'] for f in c] == ['sentence1', 'sentence2'], 'READING_ROW')
            self.edge(c[0]['id'], y[0]['id'], 'source_answer_from_passage')
            self.edge(c[1]['id'], y[0]['id'], 'source_answer_for_question')
