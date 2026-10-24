"""Training-only source addresses and leakage census; never a training admission.

Cost: O(source bytes + rows + field occurrences), union-find for shared facts.
No all-pairs similarity, no held-out payloads, no inferred page/time boundaries.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import unicodedata
import zipfile
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from text.connected_training_capture import MULTIRC_PARENT,MEMBER,HF_PIN,HF_MEMBER
from text.evaluation_multiscale import file_sha
from senses.trajectory_population import publish

PINS={'multirc':'b55704151e1e7e110f321d36da07b7b210b71d213ced07d5b94c4d607a573d2c',
      'qasc':'b833e3c1a51b95440d83412b5c3f93f2ed3ea5c08dd1300d862c597e2c353616'}
FLAGS=dict(training_ready=False,scientific_admission=False,served=False,heldout_labels_extracted=False)


def check(ok,code):
    if not ok:raise ValueError('MNLM_CONNECTED_CENSUS_'+code)


def sha(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()
def norm(text):return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())
def nonblank(value):return isinstance(value,str) and bool(value.strip())
def histogram(values):return {str(k):v for k,v in sorted(Counter(values).items())}


def bound_capture(root,pin,source):
    check(file_sha(root/'report.json')==pin,'REPORT_PIN')
    r=json.loads((root/'report.json').read_bytes())
    check(r['complete'] is True and r['source']==source and all(r[k] is v for k,v in FLAGS.items()),'CAPTURE_STATE')
    for name,v in r['files'].items():
        check(Path(name).name==name and '\\' not in name and ':' not in name,'SOURCE_NAME')
        check((root/name).stat().st_size==v['bytes'] and file_sha(root/name)==v['sha256'],'SOURCE_FILE')
    return r


def load_sources(multirc,qasc,parent):
    m=bound_capture(multirc,PINS['multirc'],'multirc');q=bound_capture(qasc,PINS['qasc'],'qasc')
    p=bound_capture(parent,MULTIRC_PARENT,'multirc')
    check(json.loads((multirc/'capture.json').read_bytes())==m['provenance'],'MULTIRC_PROVENANCE')
    check(json.loads((qasc/'capture.json').read_bytes())==q['provenance'],'QASC_PROVENANCE')
    check(m['provenance']['parent_sha256']==MULTIRC_PARENT and 'research-only' in m['provenance']['use_scope'],'LICENCE_SCOPE')
    check('https://github.com/CogComp/multirc/blob/master/LICENSE' in json.loads((multirc/'provider-resource.json').read_bytes())['basic']['description'],'LICENCE_BINDING')
    check('Research and Academic Use License' in (multirc/'LICENSE').read_text(encoding='utf-8'),'LICENCE_CONTENT')
    with zipfile.ZipFile(parent/'mutlirc-v2.zip') as z:
        check(z.read(MEMBER)==(multirc/'train.json').read_bytes(),'ARCHIVE_TRAIN_MEMBER')
    tree=json.loads((qasc/'provider-tree.json').read_bytes());entry=[v for v in tree if v['path']==HF_MEMBER]
    check(len(entry)==1 and entry[0]['lfs']['oid']==HF_PIN==file_sha(qasc/'train.parquet'),'QASC_PROVIDER_PIN')
    check('cc-by-4.0' in (qasc/'README.md').read_text(encoding='utf-8'),'QASC_LICENCE')
    check(p['payload_rows_read']==m['payload_rows_read']==q['payload_rows_read']==0,'CAPTURE_PARSE_SCOPE')
    return json.loads((multirc/'train.json').read_bytes()),pq.read_table(qasc/'train.parquet').to_pylist()


def multirc_census(payload):
    check(isinstance(payload,dict) and set(payload)=={'data'} and isinstance(payload['data'],list),'MULTIRC_ROOT')
    docs=[];seen=set();counts=Counter();evlens=[];sentence_counts=[];positive_counts=[];paragraphs=Counter();zero=0
    for d in payload['data']:
        check(set(d)=={'id','paragraph'} and nonblank(d['id']) and d['id'] not in seen,'DOCUMENT_ID');seen.add(d['id'])
        p=d['paragraph'];check(set(p)=={'text','questions'} and nonblank(p['text']) and isinstance(p['questions'],list),'DOCUMENT_FIELDS')
        text=p['text'];matches=list(re.finditer(r'<b>Sent (\d+): </b>(.*?)<br> *',text,re.S))
        check(matches and ''.join(m.group(0) for m in matches)==text,'SENTENCE_COVERAGE')
        check([int(m[1]) for m in matches]==list(range(1,len(matches)+1)),'SENTENCE_IDS')
        atoms=[]
        for i,m in enumerate(matches):
            check(nonblank(m[2]),'SENTENCE_EMPTY')
            atoms.append(dict(index=i,display_id=i+1,source_field='paragraph.text',codepoint_span=list(m.span(2)),envelope_span=list(m.span()),text_sha256=sha(m[2])))
        questions=[];qids=set()
        for v in p['questions']:
            check(set(v)=={'question','sentences_used','answers','idx','multisent'} and nonblank(v['question']),'QUESTION_FIELDS')
            check(nonblank(v['idx']) and v['idx'] not in qids and type(v['multisent']) is bool,'QUESTION_ID');qids.add(v['idx'])
            ev=v['sentences_used'];check(isinstance(ev,list) and ev and all(type(i) is int and 0<=i<len(atoms) for i in ev),'EVIDENCE_INDEX')
            check(len(ev)==len(set(ev)),'EVIDENCE_DUPLICATE');evlens.append(len(ev));zero+=0 in ev
            answers=v['answers'];check(isinstance(answers,list) and answers,'ANSWERS')
            for a in answers:
                check(set(a)=={'text','isAnswer','scores'} and isinstance(a['text'],str) and type(a['isAnswer']) is bool and a['scores']=={},'ANSWER_SCHEMA')
            pos=sum(a['isAnswer'] for a in answers);positive_counts.append(pos)
            blank=sum(not nonblank(a['text']) for a in answers)
            counts.update(questions=1,answers=len(answers),positive_answers=pos,multisent_true=int(v['multisent']),blank_answers=blank,quarantined_questions=int(blank>0))
            questions.append(dict(id=v['idx'],text_sha256=sha(v['question']),source_evidence_indices=ev,
                display_evidence_ids=[i+1 for i in ev],multisent=v['multisent'],quarantine_reasons=['EMPTY_ANSWER'] if blank else [],
                answers=[dict(index=i,text_sha256=sha(a['text']),isAnswer=a['isAnswer']) for i,a in enumerate(answers)]))
        sentence_counts.append(len(atoms));paragraphs[sha(text)]+=1
        docs.append(dict(id=d['id'],text_sha256=sha(text),atoms=atoms,questions=questions))
    summary=dict(documents=len(docs),sentences=sum(sentence_counts),**dict(counts),sentence_count_histogram=histogram(sentence_counts),
        evidence_size_histogram=histogram(evlens),correct_answer_count_histogram=histogram(positive_counts),zero_reference_questions=zero,
        duplicate_paragraph_excess=sum(v-1 for v in paragraphs.values()),evidence_index_basis='zero-based positions; display IDs are position+1',
        evidence_scope='question-level selected references; not certified answer-specific sufficient sets',licence_scope='research-only',
        original_source_entity_time_seals=False,page_boundaries_available=False)
    return docs,summary


def qasc_census(rows):
    check(isinstance(rows,list) and rows,'QASC_ROWS');ids=set();cases=[];owners={};parent=list(range(len(rows)));freq=Counter();pairs=Counter();questions=Counter();counts=Counter()
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,r in enumerate(rows):
        check(set(r)=={'id','question','choices','answerKey','fact1','fact2','combinedfact','formatted_question'},'QASC_FIELDS')
        check(all(nonblank(r[k]) for k in set(r)-{'choices'}) and r['id'] not in ids,'QASC_ID_OR_TEXT');ids.add(r['id'])
        c=r['choices'];check(set(c)=={'text','label'} and c['label']==list('ABCDEFGH') and len(c['text'])==8 and all(nonblank(t) for t in c['text']),'CHOICES')
        check(r['answerKey'] in c['label'],'ANSWER_KEY')
        facts=[dict(field=k,text_sha256=sha(r[k]),grouping_sha256=sha(norm(r[k]))) for k in ('fact1','fact2')]
        keys=[f['grouping_sha256'] for f in facts]
        for key in set(keys):
            if key in owners:parent[find(i)]=find(owners[key])
            else:owners[key]=i
        freq.update(keys);pairs[tuple(sorted(keys))]+=1;questions[sha(norm(r['question']))]+=1
        counts.update(identical_fact_rows=int(keys[0]==keys[1]),combined_equals_fact_rows=int(norm(r['combinedfact']) in [norm(r['fact1']),norm(r['fact2'])]),
                      repeated_option_rows=int(len(set(map(norm,c['text'])))<8))
        cases.append(dict(id=r['id'],question_sha256=sha(r['question']),facts=facts,choices=[dict(label=l,text_sha256=sha(t)) for l,t in zip(c['label'],c['text'])],
                          private_target=dict(answerKey=r['answerKey'],combinedfact_sha256=sha(r['combinedfact'])),formatted_question_sha256=sha(r['formatted_question'])))
    members={}
    for i in range(len(rows)):members.setdefault(find(i),[]).append(i)
    for group in members.values():
        key=min(rows[i]['id'] for i in group)
        for i in group:cases[i]['shared_fact_component']=key
    summary=dict(cases=len(rows),fact_occurrences=2*len(rows),unique_normalized_facts=len(freq),shared_fact_keys=sum(v>1 for v in freq.values()),
        shared_fact_components=len(members),component_size_histogram=histogram(map(len,members.values())),largest_component=max(map(len,members.values())),
        duplicate_unordered_fact_pair_excess=sum(v-1 for v in pairs.values()),duplicate_normalized_question_excess=sum(v-1 for v in questions.values()),**dict(counts),
        relation='provider fact1/fact2 composition; joint necessity unproven',grouping='NFKC/casefold/whitespace identity; not near-duplicate or entity seal',
        original_archive_equivalence=False,source_entity_time_seals=False,negative_options_are_contradictions=False,removed_facts_assign_missing_truth=False)
    return cases,summary


def main():
    p=argparse.ArgumentParser()
    for k in ('multirc','qasc','parent','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();phases={};tick=time.perf_counter()
    m,q=load_sources(a.multirc,a.qasc,a.parent);phases['source_pins_and_training_parse']=time.perf_counter()-tick;tick=time.perf_counter()
    docs,ms=multirc_census(m);cases,qs=qasc_census(q);phases['schema_addresses_and_shared_fact_components']=time.perf_counter()-tick;tick=time.perf_counter()
    publish(a.output/'multirc-documents.json',docs);publish(a.output/'qasc-cases.json',cases)
    summary=dict(multirc=ms,qasc=qs);publish(a.output/'census.json',summary)
    files={n:dict(bytes=(a.output/n).stat().st_size,sha256=file_sha(a.output/n)) for n in ('multirc-documents.json','qasc-cases.json','census.json')}
    phases['persist_and_cold_file_hashes']=time.perf_counter()-tick;wall=time.perf_counter()-start
    report=dict(schema='connected-training-census-v1',source_pins={**PINS,'multirc_parent':MULTIRC_PARENT},files=files,**FLAGS,
        component_assignment_is_split=False,model_inputs_published=False,anchor_fields_are_private=True,measurements='author-checked',
        phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(a.output/'report.json',report);print(json.dumps(dict(report_sha256=file_sha(a.output/'report.json'),summary=summary,seconds=wall)))


if __name__=='__main__':main()
