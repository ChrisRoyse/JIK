"""TRAIN-only whole-continuation lookup. Exact source addresses are not senses.

The chooser sees only the observed ordered prefix. Private reference comparison
is a separate operation after the complete selection has been published.
"""
from collections import Counter
import hashlib

import numpy as np
from text.constellation_corpus import need

POLICY = dict(schema='legal-whole-answer-exact-control-v1',
    input='ordered_exact_public_prefix_uids', fit='TRAIN_only',
    prefix='min(2, source_atom_count-1)', selection='modal_TRAIN_continuation',
    tie='answer_sha256_then_lowest_TRAIN_document', terminal='actual_donor_document_END',
    conflicting_prefix='clarification_required_with_unassessed_modal_control',
    missing_prefix='unavailable_retained_in_denominator',
    collection_used_for_selection=False, private_reference_used_for_selection=False,
    source_ids_are_model_features=False, scientific_admission=False,
    scope='exact_recorded_document_continuation_not_legal_QA_or_world_truth')
DTYPE = np.dtype([(n, '<i8') for n in ('ordinal', 'document', 'side', 'component',
    'observed', 'matches', 'distinct_answers', 'modal_count', 'donor',
    'donor_answer_start', 'donor_stop', 'status')] + [('uids', '<i8', (2,)), ('answer', 'V32')])
STATUS = {0:'unavailable', 1:'candidate', 2:'clarification_required'}


def bounds(source, document):
    need(type(document) is int and 0 <= document < len(source.documents), 'EXACT_ANSWER_DOCUMENT')
    lo, hi = map(int, source.documents[document]); n = min(2, hi-lo-1)
    need(n in (1,2), 'EXACT_ANSWER_PREFIX')
    return lo, lo+n, hi


def prefix(source, document):
    lo, at, _ = bounds(source, document)
    return tuple(map(int, source.mapping[lo:at]))


def answer_pin(values):
    need(values.ndim == 1 and values.dtype == np.dtype('<i8') and len(values)>0,
         'EXACT_ANSWER_ADDRESS_TYPE')
    h = hashlib.sha256(b'legal-answer-uid-sequence-v1\0')
    h.update(len(values).to_bytes(8,'little')); h.update(memoryview(values))
    return h.hexdigest()


class ExactControl:
    def __init__(self, source, *, training=None):
        expected = np.flatnonzero(source.sides == 0).tolist()
        training = expected if training is None else training
        need(type(training) is list and all(type(i) is int for i in training)
             and training == expected, 'EXACT_ANSWER_TRAIN_SCOPE')
        self.source = source; self.groups = {}; self.documents = len(training)
        self.answer_occurrences = 0; self.hashed_bytes = 0
        for document in training:
            _, at, hi = bounds(source,document); values = source.mapping[at:hi]
            pin = answer_pin(values); key = prefix(source,document)
            group = self.groups.setdefault(key,{})
            if pin in group:
                other = group[pin]['donor']; _, a, b = bounds(source,other)
                need(np.array_equal(values,source.mapping[a:b]), 'EXACT_ANSWER_HASH_COLLISION')
                group[pin]['count'] += 1
            else: group[pin] = dict(donor=document,count=1)
            self.answer_occurrences += len(values); self.hashed_bytes += values.nbytes
        self.choices = {}
        for key, group in self.groups.items():
            pin = min(group,key=lambda h:(-group[h]['count'],h,group[h]['donor']))
            self.choices[key] = dict(matches=sum(x['count'] for x in group.values()),
                distinct_answers=len(group),modal_count=group[pin]['count'],
                donor=group[pin]['donor'],answer=pin,status=1 if len(group)==1 else 2)
        self.summary = dict(train_documents=self.documents, train_prefix_groups=len(self.groups),
            conflicting_train_prefix_groups=sum(len(g)>1 for g in self.groups.values()),
            train_documents_in_conflicting_groups=sum(sum(v['count'] for v in g.values())
                for g in self.groups.values() if len(g)>1),
            unique_prefix_input_deterministic_exact_ceiling_numerator=sum(
                max(v['count'] for v in g.values()) for g in self.groups.values()),
            ceiling_denominator=self.documents, train_answer_occurrences=self.answer_occurrences,
            answer_bytes_hashed=self.hashed_bytes)
        source.fresh()

    def choose(self, observed):
        need(type(observed) is tuple and len(observed) in (1,2)
             and all(type(u) is int and 0<=u<len(self.source.counts) for u in observed),
             'EXACT_ANSWER_PUBLIC_INPUT')
        return dict(self.choices.get(observed,dict(matches=0,distinct_answers=0,modal_count=0,
            donor=-1,answer='00'*32,status=0)))

    def row(self, ordinal, document):
        need(type(ordinal) is int and ordinal>=0 and int(self.source.sides[document]) in (1,2),
             'EXACT_ANSWER_HELDOUT_ONLY')
        key = prefix(self.source,document); selected = self.choose(key)
        row = np.zeros((),dtype=DTYPE); row['uids'] = -1; row['uids'][:len(key)] = key
        for n,v in dict(ordinal=ordinal,document=document,side=int(self.source.sides[document]),
            component=int(self.source.components[document]),observed=len(key),
            **{k:v for k,v in selected.items() if k!='answer'}).items(): row[n]=v
        row['answer'] = bytes.fromhex(selected['answer'])
        row['donor_answer_start'] = row['donor_stop'] = -1
        if selected['donor'] >= 0:
            d = selected['donor']; _, at, hi = bounds(self.source,d)
            need(self.source.sides[d] == 0 and self.source.components[d] != row['component'],
                 'EXACT_ANSWER_DONOR_SPLIT')
            row['donor_answer_start'],row['donor_stop'] = at,hi
        return row


def validate_selection(source, control, rows):
    ids = np.flatnonzero(source.sides != 0)
    need(rows.dtype == DTYPE and rows.shape == ids.shape
         and np.array_equal(rows['document'],ids), 'EXACT_ANSWER_CASE_COVERAGE')
    for ordinal,document in enumerate(ids):
        need(rows[ordinal].tobytes() == control.row(ordinal,int(document)).tobytes(),
             'EXACT_ANSWER_SELECTION_CHANGED')


def private_census(source, rows):
    """Diagnostic after selection seal. No outcome is returned to the chooser."""
    counts = Counter(); groups = {}; by_collection = {}; cases = []
    for row in rows:
        document = int(row['document']); _, at, hi = bounds(source,document)
        values = source.mapping[at:hi]; pin = answer_pin(values); key = prefix(source,document)
        group = groups.setdefault(key,Counter()); group[pin] += 1
        donor = int(row['donor']); matched = False
        if donor>=0:
            a,b = int(row['donor_answer_start']),int(row['donor_stop'])
            matched = bool(np.array_equal(source.mapping[a:b],values))
        status = STATUS[int(row['status'])]; collection=source.metadata[document]['source_path']
        delta=dict(cases=1, prefix_seen_in_train=int(donor>=0),
            unambiguous_train_prefix=int(status=='candidate'),
            ambiguous_train_prefix=int(status=='clarification_required'),
            unavailable=int(status=='unavailable'), modal_exact_recorded_answer=int(matched),
            unambiguous_exact_recorded_answer=int(matched and status=='candidate'),
            reference_answer_occurrences=len(values))
        counts.update(delta); by_collection.setdefault(collection,Counter()).update(delta)
        cases.append(dict(ordinal=int(row['ordinal']),document=document,reference_answer=pin,
            reference_atoms=len(values),donor=donor,status=status,modal_exact=matched))
    return dict(counts=dict(counts),by_collection={k:dict(v) for k,v in sorted(by_collection.items())},
        heldout_prefix_groups=len(groups),conflicting_heldout_prefix_groups=sum(len(g)>1 for g in groups.values()),
        heldout_deterministic_exact_ceiling_numerator=sum(max(g.values()) for g in groups.values()),
        ceiling_denominator=len(rows), reference_scored_after_selection=True,
        ceiling_scope='observed_exact_uid_prefix_to_recorded_uid_sequence_on_this_finite_population',
        distinct_valid_answers_not_errors=True, semantic_validity_established=False),cases
