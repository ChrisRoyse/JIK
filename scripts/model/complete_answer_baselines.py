"""TRAIN-only whole-answer retrieval controls. Lexical features are not senses."""
from collections import Counter, defaultdict
from copy import deepcopy
import math

from model.capability_baselines import terms
from text.constellation_corpus import canonical, digest, need


def identity(public):
    need(isinstance(public, dict) and set(public) == {'operation','fields'}
         and isinstance(public['operation'], str) and isinstance(public['fields'], list)
         and public['fields'] and all(set(f) == {'role','text'} and
             isinstance(f['role'], str) and isinstance(f['text'], str) for f in public['fields']),
         'COMPLETE_BASELINE_PUBLIC_SCHEMA')
    return digest(canonical(public))


def validate(row):
    need(set(row) == {'task','split','group','family','public','answer'} and row['split'] == 'train',
         'COMPLETE_BASELINE_TRAIN_ONLY')
    identity(row['public']); a = row['answer']; raw = a['text'].encode()
    need(set(a) == {'text','sha256','atoms','end','reference_span','source_field'}
         and digest(raw) == a['sha256'] and a['atoms'], 'COMPLETE_BASELINE_ANSWER_PIN')
    offset = a['reference_span'][0]
    need(a['reference_span'][1]-offset == len(raw), 'COMPLETE_BASELINE_SPAN')
    cursor = offset
    for i, atom in enumerate(a['atoms']):
        lo, hi = atom['surface_span']
        need(atom['index'] == i and lo == cursor and lo < hi <= offset+len(raw)
             and digest(raw[lo-offset:hi-offset]) == atom['surface_sha256'], 'COMPLETE_BASELINE_ATOM_ORDER')
        cursor = hi
    need(cursor == offset+len(raw) and a['end']['kind'] == 'end'
         and a['end']['byte_offset'] == cursor and a['end']['source_field'] == a['source_field']
         and a['end']['observed'] is True and a['end']['establishes_correctness'] is False,
         'COMPLETE_BASELINE_END')


class CompleteAnswerBaselines:
    def __init__(self, training):
        self.bank = {}; self.terms = {}; self.indices = {}; self.visits = 0; cache = {}
        for row in training:
            validate(row); key = row['task']; need(key not in self.bank, 'COMPLETE_BASELINE_DUPLICATE')
            self.bank[key] = deepcopy(row); combined = Counter()
            for f in row['public']['fields']:
                pin = digest(f['text'].encode())
                if pin not in cache: cache[pin] = terms(f['text'])
                combined.update(cache[pin])
            self.terms[key] = combined
        need(self.bank, 'COMPLETE_BASELINE_EMPTY')
        self.unique_training_fields = len(cache)
        self._field_terms = cache
        for operation in sorted({r['public']['operation'] for r in self.bank.values()}):
            keys = tuple(sorted(k for k,r in self.bank.items() if r['public']['operation'] == operation))
            df = Counter(); postings = defaultdict(list); exact = defaultdict(list)
            answers = Counter(); lengths = Counter()
            for key in keys:
                counts = self.terms[key]; length = sum(counts.values()); df.update(counts.keys())
                for token, count in counts.items(): postings[token].append((key,count,length))
                row = self.bank[key]; exact[identity(row['public'])].append(key)
                answers[row['answer']['sha256']] += 1; lengths[len(row['answer']['atoms'])] += 1
            winner = min(answers, key=lambda h:(-answers[h], h))
            average = sum(sum(self.terms[k].values()) for k in keys)/len(keys)
            need(average > 0, 'COMPLETE_BASELINE_NO_TERMS')
            self.indices[operation] = dict(keys=keys, df=dict(df), postings=dict(postings), exact=dict(exact),
                average=average, modal=min(k for k in keys if self.bank[k]['answer']['sha256'] == winner),
                length=min(lengths, key=lambda n:(-lengths[n], n)))
        self.seal = digest(canonical(dict(bank=self.bank, terms=self.terms, indices=self.indices)))

    def fresh(self):
        need(digest(canonical(dict(bank=self.bank, terms=self.terms, indices=self.indices))) == self.seal,
             'COMPLETE_BASELINE_INDEX_CHANGED')

    def candidate(self, key, method, *, score=None):
        need(key in self.bank, 'COMPLETE_BASELINE_DONOR')
        row = self.bank[key]; validate(row)
        return dict(method=method, status='candidate', donor=key, answer=deepcopy(row['answer']),
            score=score, endpoint='TRAIN_donor_endpoint', validity='unassessed',
            donor_execution_transfers=False, native_readings_bound=False)

    def select(self, public):
        pin = identity(public); operation = public['operation']
        if operation not in self.indices:
            return {n:dict(method=n, status='unavailable', reason='NO_TRAIN_OPERATION')
                    for n in ('bm25','modal','exact','length_mode')}
        index = self.indices[operation]; scores = defaultdict(list); count = len(index['keys'])
        query_terms = set()
        for f in public['fields']:
            field_pin = digest(f['text'].encode())
            if field_pin not in self._field_terms: self._field_terms[field_pin] = terms(f['text'])
            query_terms.update(self._field_terms[field_pin])
        for token in sorted(query_terms):
            df = index['df'].get(token, 0); weight = math.log1p((count-df+.5)/(df+.5))
            for key, frequency, length in index['postings'].get(token, ()):
                self.visits += 1
                scores[key].append(weight*frequency*2.2/(frequency+1.2*(.25+.75*length/index['average'])))
        ranked = {k:math.fsum(scores[k]) for k in index['keys']}
        best = min(ranked, key=lambda k:(-ranked[k], k))
        result = dict(bm25=dict(self.candidate(best, 'bm25', score=ranked[best]), zero_overlap=ranked[best] == 0),
            modal=self.candidate(index['modal'], 'modal'),
            length_mode=dict(method='length_mode', status='length_only', atoms=index['length'],
                             validity='unassessed', content_generated=False))
        matches = index['exact'].get(pin, ()); answers = {self.bank[k]['answer']['sha256'] for k in matches}
        if len(answers) == 1:
            result['exact'] = dict(self.candidate(min(matches), 'exact'), matches=len(matches))
        else:
            result['exact'] = dict(method='exact', status='clarification_required' if answers else 'unavailable',
                reason='CONFLICTING_TRAIN_ANSWERS' if answers else 'NO_EXACT_TEXT_ROLE_MATCH',
                matches=len(matches), distinct_answers=len(answers))
        return result
