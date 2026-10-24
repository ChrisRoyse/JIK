"""Private, source-addressed evidence withholding plans; no counterfactual labels.

PC-07: document interval incidence is invariant across its questions. Hash-group
counts avoid enumerating option pairs. All native windows of a withheld reference
must be withheld by consumers; this artifact never selects a single pooled vector.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.connected_population_readback import readback as population_readback
from model.multirc_context import QUALITY
from model.qasc_context import POPULATION
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha
from text.multirc_anchor_quality import audit_raw,norm,rows

RAW='9811f0abbcea2aa61ea1943426ca2052d87632a1b08e71c34ef19c300019c560'
SCHEMA='multirc-evidence-scope-v1'
CLAIMS=dict(private_audit_only=True,model_input=False,answer_specific_sufficiency_proven=False,
            evidence_necessity_proven=False,altered_input_truth_assessed=False,
            world_truth_assessed=False,scientific_admission=False,served=False)

def check(ok,code):
    if not ok:raise ValueError('MNLM_EVIDENCE_SCOPE_'+code)

def parent_paths(population,quality,raw):
    return {'raw/train.json':raw/'multirc-training-v1/train.json',
            **{'population/'+n:population/n for n in ('report.json','cases.jsonl','documents.jsonl',
               'references.jsonl','private/targets.jsonl','assignments.jsonl')},
            **{'quality/'+n:quality/n for n in QUALITY}}

def parents(population,quality,raw):
    return {k:file_sha(p) for k,p in parent_paths(population,quality,raw).items()}

def load(population,quality,raw):
    before=parents(population,quality,raw)
    check(before['raw/train.json']==RAW and before['population/report.json']==POPULATION,'PINS')
    check(all(before['quality/'+n]==pin for n,pin in QUALITY.items()),'QUALITY_PINS')
    cold=population_readback(population,raw/'multirc-training-v1',raw/'qasc-hf-training-v1',raw/'multirc-v1')
    values=[json.loads((quality/n).read_bytes()) for n in QUALITY]
    quality_cold=audit_raw(population,raw/'multirc-training-v1/train.json',*values)
    data=dict(source=json.loads((raw/'multirc-training-v1/train.json').read_bytes())['data'],
              cases=[v for v in rows(population/'cases.jsonl') if v['family']=='multirc'],
              documents=rows(population/'documents.jsonl'),references={v['id']:v for v in rows(population/'references.jsonl')},
              targets={v['case']:v for v in rows(population/'private/targets.jsonl') if v['family']=='multirc'},
              assignments={v['case']:v for v in rows(population/'assignments.jsonl') if v['policy']=='multirc'},
              excluded={v['case'] for v in values[1]})
    check(len(data['source'])==len(data['documents'])==456 and len(data['cases'])==5131,'COUNTS')
    return data,before,dict(population=cold,quality=quality_cold)

def plans(data):
    docs={d['id']:d for d in data['documents']};refs=data['references'];incidence={};views={}
    for d in data['documents']:
        ids=[*d['sentences'],*d['pairs'],d['paragraph']];views[d['id']]=ids
        by_sentence=[set() for _ in d['sentences']]
        for rid in ids:
            ref=refs[rid];lo,hi=ref['source_sentence_range']
            check(ref['owner']==d['id'] and 0<=lo<hi<=len(by_sentence),'VIEW_INTERVAL')
            for i in range(lo,hi):by_sentence[i].add(rid)
        incidence[d['id']]=by_sentence
    result=[]
    for c in data['cases']:
        d=docs[c['document']];a=data['assignments'][c['id']];t=data['targets'][c['id']]
        q=data['source'][c['source_row']]['paragraph']['questions'][c['source_question']]
        evidence=q['sentences_used'];check(t['source_evidence_indices']==evidence,'SOURCE_ANNOTATION')
        removed=set().union(*(incidence[d['id']][i] for i in evidence))
        withheld=[rid for rid in views[d['id']] if rid in removed];retained=[rid for rid in views[d['id']] if rid not in removed]
        aliases=sorted({refs[r]['unit'] for r in withheld}&{refs[r]['unit'] for r in retained})
        result.append(dict(case=c['id'],source_row=c['source_row'],source_question=c['source_question'],document=d['id'],
            partition=a['partition'],group=a['group'],eligible=a['eligible'],quality_clean=c['id'] not in data['excluded'],
            source_evidence_indices=evidence,evidence_references=[d['sentences'][i] for i in evidence],
            provider_multisent=q['multisent'],withheld_references=withheld,retained_references=retained,
            shared_native_units=aliases,empty_remaining_context=not retained,
            window_rule='all-native-windows-of-each-reference',
            withholding_scope='source-occurrence-overlap; semantic absence unproven',
            annotation_scope='question-level selected references; answer-specific sufficiency unproven'))
    return result

def contrast_census(data):
    """Count exact and normalized Q/A judgments across different source documents."""
    result={}
    for mode in ('exact','normalized'):
        buckets=defaultdict(list)
        for c in data['cases']:
            q=data['source'][c['source_row']]['paragraph']['questions'][c['source_question']];a=data['assignments'][c['id']]
            for i,answer in enumerate(q['answers']):
                key=(q['question'],answer['text'])
                if mode=='normalized':key=tuple(norm(s) for s in key)
                buckets[key].append((c['document'],int(answer['isAnswer']),a['eligible'],c['id'] not in data['excluded'],a['partition']))
        summary={}
        for cohort in ('original','eligible','clean'):
            for split in ('all','train','calibration','evaluation'):
                groups=pairs=members=0
                for rows_ in buckets.values():
                    vv=[v for v in rows_ if (cohort=='original' or v[2]) and (cohort!='clean' or v[3]) and (split=='all' or v[4]==split)]
                    labels=Counter(v[1] for v in vv);by_doc=defaultdict(Counter)
                    for doc,label,*_ in vv:by_doc[doc][label]+=1
                    count=labels[0]*labels[1]-sum(c[0]*c[1] for c in by_doc.values())
                    if count:groups+=1;pairs+=count;members+=len(vv)
                summary[cohort+'/'+split]=dict(question_answer_groups=groups,opposite_label_cross_document_pairs=pairs,group_members=members)
        result[mode]=summary
    return result

def summarize(records):
    out={}
    for split in ('all','train','calibration','evaluation'):
        rr=[v for v in records if split=='all' or v['partition']==split]
        out[split]=dict(questions=len(rr),eligible=sum(v['eligible'] for v in rr),
            clean_eligible=sum(v['eligible'] and v['quality_clean'] for v in rr),
            annotation_occurrences=sum(len(v['source_evidence_indices']) for v in rr),
            multi_reference_questions=sum(len(set(v['source_evidence_indices']))>1 for v in rr),
            provider_multisent_questions=sum(v['provider_multisent'] for v in rr),
            withheld_view_occurrences=sum(len(v['withheld_references']) for v in rr),
            retained_view_occurrences=sum(len(v['retained_references']) for v in rr),
            empty_remaining_contexts=sum(v['empty_remaining_context'] for v in rr),
            retained_withheld_native_alias_cases=sum(bool(v['shared_native_units']) for v in rr))
    return out

def build(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();phases={}
    data,before,cold=load(a.population,a.quality,a.raw)
    phases['immutable_source_and_population_readback']=time.perf_counter()-start;tick=time.perf_counter()
    records=plans(data);contrasts=contrast_census(data);summary=summarize(records)
    phases['interval_incidence_and_complete_census']=time.perf_counter()-tick;tick=time.perf_counter()
    publish(a.output/'private-plans.json',records);publish(a.output/'contrasts.json',contrasts)
    publish(a.output/'summary.json',summary);publish(a.output/'parent-readback.json',cold)
    after=parents(a.population,a.quality,a.raw);check(before==after,'PARENT_DRIFT')
    files={n:file_sha(a.output/n) for n in ('private-plans.json','contrasts.json','summary.json','parent-readback.json')}
    phases['publication_and_parent_rehash']=time.perf_counter()-tick;wall=time.perf_counter()-start
    report=dict(schema=SCHEMA,complete=True,before=before,after=after,files=files,claims=CLAIMS,
        input_records_removed=0,input_labels_changed=False,input_partitions_changed=False,
        source_evidence_is_world_grounding=False,code_sha256=file_sha(Path(__file__)),
        phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,acceptance='author-checked')
    publish(a.output/'report.json',report)
    return dict(report_sha256=file_sha(a.output/'report.json'),summary=summary,contrasts=contrasts,wall_seconds=wall)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('population','quality','raw','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    try:print(json.dumps(build(a)))
    except Exception as exc:
        if a.output.exists() and not (a.output/'failure.json').exists():publish(a.output/'failure.json',dict(complete=False,error=str(exc)))
        raise
