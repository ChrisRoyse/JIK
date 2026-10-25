"""Same native position/length metric for retrieved complete TRAIN answers.

All six slots remain separate. A donor's END is known from its TRAIN source;
reference length enters only this scorer. Clarification is a separate policy.
"""
import math

import torch
from model.complete_answer_scores import native_similarity
from model.complete_answer_summary import COUNTS,units,rounded
from text.constellation_corpus import need

POLICY=dict(schema='legal-exact-answer-native-scores-v1',
    metric='complete_answer_scores.native_similarity_fp64',
    missing_similarity=-1.,extra_similarity=-1.,
    arms=['modal_control','clarify_conflicts'],all_cases_in_denominator=True,
    reference_length_used_for_selection=False,donor_end='TRAIN_document_END',
    semantic_validity_established=False,admitted_senses=0)


def native(frames,roster):
    for f in frames:
        need(set(f)==set(roster),'EXACT_SCORE_ROSTER')
        for n,x in f.items():
            need(isinstance(x,torch.Tensor) and x.dtype==torch.float32 and x.device.type=='cpu'
                 and x.ndim==2 and x.shape[1]==768 and len(x)>0 and not x.requires_grad
                 and bool(x.isfinite().all()),'EXACT_SCORE_NATIVE:'+n)


def metrics(donor,reference,*,expected,roster,status):
    need(type(expected) is int and expected>0 and status in (0,1,2),'EXACT_SCORE_PROTOCOL')
    need((status==0)==(len(donor)==0) and len(reference)==min(expected,len(donor)),
         'EXACT_SCORE_COMPLETE_DONOR')
    native(donor,roster);native(reference,roster)
    values={n:[-1.]*expected for n in roster}
    for j,f in enumerate(reference):
        for n in roster:values[n][j]=native_similarity(donor[j][n],f[n])
    generated=len(donor);extra=max(0,generated-expected);ended=status!=0
    return dict(expected_atoms=expected,generated_atoms=generated,ended=ended,
        ended_at_reference_boundary=ended and generated==expected,censored=not ended,
        matched_positions=len(reference),missing=expected-len(reference),extra=extra,
        per_slot=values,full_output_mean={n:(sum(v)-extra)/(expected+extra) for n,v in values.items()},
        missing_similarity=-1.,extra_similarity=-1.,refusal='NO_EXACT_TRAIN_PREFIX' if not ended else None,
        reference_scored_after_generation=True,semantic_validity_established=False)


def arms(modal,status,roster):
    result={'modal_control':modal}
    if status==2:
        strict=metrics((),(),expected=modal['expected_atoms'],roster=roster,status=0)
        strict['refusal']='CONFLICTING_TRAIN_ANSWERS';result['clarify_conflicts']=strict
    else:result['clarify_conflicts']=modal
    return result


def validate_case(case,roster):
    need(set(case['arms'])==set(POLICY['arms']),'EXACT_SCORE_ARMS')
    for m in case['arms'].values():
        e,g=m['expected_atoms'],m['generated_atoms'];covered=min(e,g);extra=max(0,g-e)
        need(e>0 and g>=0 and m['matched_positions']==covered and m['missing']==e-covered
             and m['extra']==extra and set(m['per_slot'])==set(roster),'EXACT_SCORE_COUNTS')
        for n,v in m['per_slot'].items():
            need(len(v)==e and all(type(x) is float and math.isfinite(x) and -1<=x<=1 for x in v)
                 and v[covered:]==[-1.]*(e-covered)
                 and m['full_output_mean'][n]==(sum(v)-extra)/(e+extra),'EXACT_SCORE_VALUES')


class Summary:
    def __init__(self,roster):self.roster=roster;self.seen=set();self.groups={}
    def add(self,case):
        ident=case['identity'];d=ident['document'];need(d not in self.seen,'EXACT_SCORE_DUPLICATE_CASE')
        validate_case(case,self.roster);self.seen.add(d)
        for arm,m in case['arms'].items():
            for collection in ('ALL',ident['collection']):
                key=(collection,ident['side'],arm)
                s=self.groups.setdefault(key,dict(cases=0,counts={n:0 for n in COUNTS},
                    means={n:0 for n in self.roster},sums={n:0 for n in self.roster},
                    selection_status={str(i):0 for i in (0,1,2)}))
                s['cases']+=1;s['selection_status'][str(case['selection_status'])]+=1
                for n in COUNTS:s['counts'][n]+=int(m[n])
                for n in self.roster:
                    s['means'][n]+=units(m['full_output_mean'][n])
                    s['sums'][n]+=units(sum(m['per_slot'][n])-m['extra'])
    def finish(self):
        need(self.seen,'EXACT_SCORE_EMPTY');groups=[]
        for (collection,side,arm),s in sorted(self.groups.items()):
            denominator=s['counts']['expected_atoms']+s['counts']['extra']
            groups.append(dict(collection=collection,side=side,arm=arm,cases=s['cases'],
                counts=s['counts'],selection_status=s['selection_status'],per_slot={n:dict(
                    case_mean=rounded(s['means'][n])/s['cases'],
                    atom_mean=rounded(s['sums'][n])/denominator) for n in self.roster}))
        return dict(cases=len(self.seen),groups=groups,semantic_validity_established=False,
            architecture_winner=False,uncertainty_estimated=False)
