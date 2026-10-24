"""Cold, independent reconstruction of source addresses and shared-fact graph."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
import unicodedata
import zipfile
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from text.connected_training_census import PINS,FLAGS
from text.connected_training_capture import MULTIRC_PARENT,MEMBER,HF_PIN


def check(ok,code):
    if not ok:raise ValueError('MNLM_CONNECTED_READ_'+code)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def string_hash(value):return hashlib.sha256(value.encode('utf-8')).hexdigest()
def normalize(value):return ' '.join(unicodedata.normalize('NFKC',value).casefold().split())
def hist(values):return {str(k):v for k,v in Counter(values).items()}


def sources(multirc,qasc,parent):
    for root,pin in ((multirc,PINS['multirc']),(qasc,PINS['qasc']),(parent,MULTIRC_PARENT)):
        check(digest(root/'report.json')==pin,'SOURCE_REPORT')
        r=json.loads((root/'report.json').read_bytes());check(r['complete'] and all(r[k] is v for k,v in FLAGS.items()),'SOURCE_STATE')
        for n,v in r['files'].items():
            check(n==Path(n).name and '\\' not in n and ':' not in n,'SOURCE_PATH')
            check((root/n).stat().st_size==v['bytes'] and digest(root/n)==v['sha256'],'SOURCE_BYTES')
    with zipfile.ZipFile(parent/'mutlirc-v2.zip') as z:check(z.read(MEMBER)==(multirc/'train.json').read_bytes(),'ARCHIVE_MEMBER')
    check(digest(qasc/'train.parquet')==HF_PIN,'PROVIDER_HASH')
    return json.loads((multirc/'train.json').read_bytes())['data'],pq.ParquetFile(qasc/'train.parquet').read().to_pylist()


def read_multirc(raw,stored,summary):
    check(len(raw)==len(stored),'DOCUMENT_COUNT');ns=[];nq=na=np=multi=blank=nblank=zero=0;els=[];positives=[];doc_hashes=[]
    for d,s in zip(raw,stored):
        text=d['paragraph']['text'];doc_hashes.append(string_hash(text));check(s['id']==d['id'] and s['text_sha256']==string_hash(text),'DOCUMENT_IDENTITY')
        cursor=0;expected=[]
        while cursor<len(text):
            index=len(expected);prefix=f'<b>Sent {index+1}: </b>';check(text.startswith(prefix,cursor),'SENTENCE_PREFIX')
            lo=cursor+len(prefix);hi=text.find('<br>',lo);check(hi>lo,'SENTENCE_END');end=hi+4
            while end<len(text) and text[end]==' ':end+=1
            expected.append(dict(index=index,display_id=index+1,source_field='paragraph.text',codepoint_span=[lo,hi],envelope_span=[cursor,end],text_sha256=string_hash(text[lo:hi])))
            cursor=end
        check(expected==s['atoms'],'ATOM_ADDRESSES');ns.append(len(expected));qs=d['paragraph']['questions'];check(len(qs)==len(s['questions']),'QUESTION_COUNT')
        for q,v in zip(qs,s['questions']):
            refs=q['sentences_used'];answers=q['answers'];b=sum(not a['text'].strip() for a in answers);pos=sum(a['isAnswer'] for a in answers)
            check(all(type(i) is int and 0<=i<len(expected) for i in refs),'EVIDENCE_RANGE')
            exp=dict(id=q['idx'],text_sha256=string_hash(q['question']),source_evidence_indices=refs,display_evidence_ids=[i+1 for i in refs],
                     multisent=q['multisent'],quarantine_reasons=['EMPTY_ANSWER'] if b else [],
                     answers=[dict(index=i,text_sha256=string_hash(a['text']),isAnswer=a['isAnswer']) for i,a in enumerate(answers)])
            check(v==exp,'QUESTION_BINDING');nq+=1;na+=len(answers);np+=pos;multi+=q['multisent'];blank+=b;nblank+=b>0;zero+=0 in refs
            els.append(len(refs));positives.append(pos)
    expected=dict(documents=len(raw),sentences=sum(ns),questions=nq,answers=na,positive_answers=np,multisent_true=multi,blank_answers=blank,quarantined_questions=nblank,
        sentence_count_histogram=hist(ns),evidence_size_histogram=hist(els),correct_answer_count_histogram=hist(positives),zero_reference_questions=zero,
        duplicate_paragraph_excess=len(raw)-len(set(doc_hashes)),evidence_index_basis='zero-based positions; display IDs are position+1',
        evidence_scope='question-level selected references; not certified answer-specific sufficient sets',licence_scope='research-only',
        original_source_entity_time_seals=False,page_boundaries_available=False)
    check(expected==summary,'MULTIRC_COUNTS_OR_SCOPE');return dict(documents=len(raw),sentences=sum(ns),questions=nq,quarantined_questions=nblank)


def read_qasc(raw,stored,summary):
    check(len(raw)==len(stored),'QASC_COUNT');occ=defaultdict(list);keys=[];pairs=[];questions=[];identical=combined=options=0
    for i,(r,s) in enumerate(zip(raw,stored)):
        fs=[dict(field=k,text_sha256=string_hash(r[k]),grouping_sha256=string_hash(normalize(r[k]))) for k in ('fact1','fact2')]
        ks=[f['grouping_sha256'] for f in fs];keys.append(ks)
        for k in ks:occ[k].append(i)
        c=r['choices'];expected=dict(id=r['id'],question_sha256=string_hash(r['question']),facts=fs,
            choices=[dict(label=l,text_sha256=string_hash(t)) for l,t in zip(c['label'],c['text'])],
            private_target=dict(answerKey=r['answerKey'],combinedfact_sha256=string_hash(r['combinedfact'])),formatted_question_sha256=string_hash(r['formatted_question']))
        check({k:v for k,v in s.items() if k!='shared_fact_component'}==expected,'QASC_BINDING')
        pairs.append(tuple(sorted(ks)));questions.append(normalize(r['question']));identical+=ks[0]==ks[1]
        combined+=normalize(r['combinedfact']) in [normalize(r[k]) for k in ('fact1','fact2')];options+=len(set(map(normalize,c['text'])))<8
    visited=set();usedfacts=set();sizes=[]
    for i in range(len(raw)):
        if i in visited:continue
        stack=[i];visited.add(i);group=[]
        while stack:
            j=stack.pop();group.append(j)
            for k in keys[j]:
                if k in usedfacts:continue
                usedfacts.add(k)
                for other in occ[k]:
                    if other not in visited:visited.add(other);stack.append(other)
        component=min(raw[j]['id'] for j in group);sizes.append(len(group))
        check(all(stored[j]['shared_fact_component']==component for j in group),'SHARED_FACT_COMPONENT')
    expected=dict(cases=len(raw),fact_occurrences=sum(map(len,occ.values())),unique_normalized_facts=len(occ),shared_fact_keys=sum(len(v)>1 for v in occ.values()),
        shared_fact_components=len(sizes),component_size_histogram=hist(sizes),largest_component=max(sizes),
        duplicate_unordered_fact_pair_excess=len(raw)-len(set(pairs)),duplicate_normalized_question_excess=len(raw)-len(set(questions)),
        identical_fact_rows=identical,combined_equals_fact_rows=combined,repeated_option_rows=options,
        relation='provider fact1/fact2 composition; joint necessity unproven',grouping='NFKC/casefold/whitespace identity; not near-duplicate or entity seal',
        original_archive_equivalence=False,source_entity_time_seals=False,negative_options_are_contradictions=False,removed_facts_assign_missing_truth=False)
    check(expected==summary,'QASC_COUNTS_OR_SCOPE');return dict(cases=len(raw),components=len(sizes),largest_component=max(sizes))


def readback(root,multirc,qasc,parent):
    rawm,rawq=sources(multirc,qasc,parent);r=json.loads((root/'report.json').read_bytes())
    check(r['schema']=='connected-training-census-v1' and r['source_pins']=={**PINS,'multirc_parent':MULTIRC_PARENT},'RESULT_SOURCE_BINDING')
    check(all(r[k] is v for k,v in FLAGS.items()) and r['component_assignment_is_split'] is False and r['model_inputs_published'] is False
          and r['anchor_fields_are_private'] is True and r['measurements']=='author-checked','CLAIM_BOUNDARY')
    check(r['wall_seconds']>0 and all(v>=0 for v in r['phases_seconds'].values()) and abs(sum(r['phases_seconds'].values())/r['wall_seconds']-r['phase_coverage'])<1e-12
          and .9<=r['phase_coverage']<=1,'COST_COVERAGE')
    check(set(r['files'])=={'multirc-documents.json','qasc-cases.json','census.json'},'RESULT_FILES')
    for n,v in r['files'].items():check((root/n).stat().st_size==v['bytes'] and digest(root/n)==v['sha256'],'RESULT_BYTES')
    c=json.loads((root/'census.json').read_bytes());check(set(c)=={'multirc','qasc'},'CENSUS_KEYS')
    m=read_multirc(rawm,json.loads((root/'multirc-documents.json').read_bytes()),c['multirc'])
    q=read_qasc(rawq,json.loads((root/'qasc-cases.json').read_bytes()),c['qasc'])
    return dict(report_sha256=digest(root/'report.json'),files=r['files'],multirc=m,qasc=q,passed=True,scientific_admission=False)
