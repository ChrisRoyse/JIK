"""Independent reconstruction of evidence addresses, exclusion closure and census."""
from collections import Counter,defaultdict
from itertools import groupby
import json
from pathlib import Path
import unicodedata
from text.evaluation_multiscale import file_sha

def need(ok,code):
    if not ok:raise ValueError('MNLM_EVIDENCE_READBACK_'+code)

def read(root,data,parent_hashes):
    value=lambda n:json.loads((root/n).read_bytes())
    h=value('report.json')
    need(h['schema']=='multirc-evidence-scope-v1' and h['complete'] is True,'STATE')
    need(h['before']==h['after']==parent_hashes,'PARENTS')
    names={'private-plans.json','contrasts.json','summary.json','parent-readback.json'}
    need(set(h['files'])==names and all(file_sha(root/n)==h['files'][n] for n in names),'FILES')
    need({p.name for p in root.iterdir()}==names|{'report.json'},'INVENTORY')
    expected_claims={n:False for n in ('model_input','answer_specific_sufficiency_proven','evidence_necessity_proven',
        'altered_input_truth_assessed','world_truth_assessed','scientific_admission','served')}
    expected_claims['private_audit_only']=True
    need(h['claims']==expected_claims and h['input_records_removed']==0 and h['input_labels_changed'] is False
        and h['input_partitions_changed'] is False and h['source_evidence_is_world_grounding'] is False,'CLAIMS')
    need(h['code_sha256']==file_sha(Path(__file__).resolve().parents[1]/'text/multirc_evidence_scope.py'),'CODE')
    need(h['wall_seconds']>0 and all(x>=0 for x in h['phases_seconds'].values()) and .9<=h['phase_coverage']<=1
         and abs(h['phase_coverage']-sum(h['phases_seconds'].values())/h['wall_seconds'])<1e-12,'COST')
    records=value('private-plans.json');need(len(records)==len(data['cases'])==5131,'COVERAGE')
    documents={d['id']:d for d in data['documents']};refs=data['references'];summary={};option_keys=[]
    for record,c in zip(records,data['cases']):
        source=data['source'][c['source_row']];q=source['paragraph']['questions'][c['source_question']]
        d=documents[c['document']];a=data['assignments'][c['id']];ev=q['sentences_used']
        expected=dict(case=c['id'],source_row=c['source_row'],source_question=c['source_question'],document=source['id'],
            partition=a['partition'],group=a['group'],eligible=a['eligible'],quality_clean=c['id'] not in data['excluded'],
            source_evidence_indices=ev,evidence_references=[d['sentences'][i] for i in ev],provider_multisent=q['multisent'],
            window_rule='all-native-windows-of-each-reference',withholding_scope='source-occurrence-overlap; semantic absence unproven',
            annotation_scope='question-level selected references; answer-specific sufficiency unproven')
        need(all(record.get(k)==v for k,v in expected.items()),'EVIDENCE_ADDRESS')
        need(set(record)==set(expected)|{'withheld_references','retained_references','shared_native_units','empty_remaining_context'},'RECORD_SCHEMA')
        # Reconstruct overlap directly, independently of the producer incidence index.
        all_views=d['sentences']+d['pairs']+[d['paragraph']];removed=[];kept=[]
        for rid in all_views:
            lo,hi=refs[rid]['source_sentence_range']
            (removed if any(lo<=i<hi for i in ev) else kept).append(rid)
        need(record['withheld_references']==removed and record['retained_references']==kept,'CONTAINMENT_CLOSURE')
        need(record['empty_remaining_context']==(len(kept)==0),'EMPTY_CONTEXT')
        removed_units={refs[x]['unit'] for x in removed}
        alias=sorted({refs[x]['unit'] for x in kept if refs[x]['unit'] in removed_units})
        need(record['shared_native_units']==alias,'NATIVE_ALIASES')
        for split in ('all',a['partition']):
            z=summary.setdefault(split,Counter())
            z.update(questions=1,eligible=int(a['eligible']),clean_eligible=int(a['eligible'] and c['id'] not in data['excluded']),
                annotation_occurrences=len(ev),multi_reference_questions=int(len(set(ev))>1),provider_multisent_questions=int(q['multisent']),
                withheld_view_occurrences=len(removed),retained_view_occurrences=len(kept),empty_remaining_contexts=int(not kept),
                retained_withheld_native_alias_cases=int(bool(alias)))
        for answer in q['answers']:
            option_keys.append((q['question'],answer['text'],source['id'],answer['isAnswer'],a['eligible'],c['id'] not in data['excluded'],a['partition']))
    need(value('summary.json')==summary,'SUMMARY')
    expected_contrasts={}
    for mode in ('exact','normalized'):
        def key(r):
            v=r[:2]
            return tuple(' '.join(unicodedata.normalize('NFKC',x).casefold().split()) for x in v) if mode=='normalized' else v
        grouped=[list(g) for _,g in groupby(sorted(option_keys,key=key),key=key)]
        scopes={}
        for cohort in ('original','eligible','clean'):
            for split in ('all','train','calibration','evaluation'):
                count=Counter(question_answer_groups=0,opposite_label_cross_document_pairs=0,group_members=0)
                for gg in grouped:
                    mm=[v for v in gg if (cohort=='original' or v[4]) and (cohort!='clean' or v[5]) and (split=='all' or v[6]==split)]
                    positives=Counter(v[2] for v in mm if v[3]);negatives=Counter(v[2] for v in mm if not v[3])
                    total_negative=sum(negatives.values());pairs=sum(n*(total_negative-negatives[doc]) for doc,n in positives.items())
                    if pairs:count.update(question_answer_groups=1,opposite_label_cross_document_pairs=pairs,group_members=len(mm))
                scopes[cohort+'/'+split]=dict(count)
        expected_contrasts[mode]=scopes
    need(value('contrasts.json')==expected_contrasts,'CONTRAST_CENSUS')
    return dict(report_sha256=file_sha(root/'report.json'),questions=len(records),source_options=len(option_keys),
        every_annotation_and_view_reconstructed=True,full_denominators_reconstructed=True,
        counterfactual_truth_assessed=False,world_truth_assessed=False,acceptance='author-checked')
