"""Independent raw-field, observation/target and hierarchy reconstruction."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.connected_census_readback import sources
from audit.evaluation_overlap_readback import audit as audit_overlap


def check(ok,code):
    if not ok:raise ValueError('MNLM_CONNECTED_POP_READ_'+code)


def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def identity(value):return sha(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')))
def rows(path):return [json.loads(line) for line in path.read_bytes().splitlines()]


def alias(name):
    s=name.removesuffix('.txt').replace('/','-')
    if s.startswith('Sept11-reports-'):return dict(key='multirc-source:Sept11-reports',method='conservative-collection-group',world_referent_resolved=False)
    for kind,pattern in [('movie',r'wikiMovieSummaries-(\d+)(?:-\d+)?$'),('wiki',r'Wiki_articles-(?:paragraphs-)?wiki(.+)-\d+$'),
         ('gutenberg',r'gutenberg-(\d+)(?:-\d+)?$'),('mctest',r'(mctest-.+)-\d+$'),('science',r'(science-g\d+)-\d+$'),('masc',r'(masc-.+)-\d+$'),('oanc',r'(oanc-.+)-\d+$')]:
        found=re.search(pattern,s)
        if found:return dict(key='multirc-source:'+kind+':'+found[1],method='identifier-pattern-v1',world_referent_resolved=False)
    return dict(key='multirc-document:'+name,method='unresolved-document-only',world_referent_resolved=False)


def verify_views(data,multi,qasc):
    refs=data['references'];units=data['units'];byid={r['id']:r for r in refs};check(len(byid)==len(refs),'DUPLICATE_REF');expected=[];used=set();documents=[];cases=[];targets=[]
    for i,u in enumerate(units):
        check(set(u)=={'row','chunk','entity','text'} and u['row']==i and u['chunk']==0 and u['text'].strip() and u['entity']==sha(u['text']),'PUBLIC_UNIT')
    check(len({u['text'] for u in units})==len(units),'PUBLIC_DUPLICATE_UNIT')
    def ref(family,owner,role,parts,addresses,span=None):
        rid=identity([family,owner,role,addresses,span]);check(rid in byid,'MISSING_NATIVE_REF');r=byid[rid];text='\n'.join(parts)
        check(set(r)=={'id','family','owner','role','unit','text_sha256','measurement_status','source_fields','source_sentence_range','composition'},'PUBLIC_REF_KEYS')
        exp=dict(id=rid,family=family,owner=owner,role=role,unit=r['unit'],text_sha256=sha(text),measurement_status='pending-native' if text.strip() else 'unavailable-empty-surface',
                 source_fields=addresses,source_sentence_range=span,composition='newline-join-unchanged-fields' if len(parts)>1 else 'unchanged-field')
        check(r==exp,'SOURCE_REFERENCE')
        if text.strip():check(type(r['unit']) is int and 0<=r['unit']<len(units) and units[r['unit']]['text']==text,'SOURCE_TEXT');used.add(r['unit'])
        else:check(r['unit'] is None,'EMPTY_MEASURED')
        expected.append(rid);return rid
    for di,d in enumerate(multi):
        owner=d['id'];text=d['paragraph']['text'];start=0;ss=[];spans=[];envelopes=[]
        while start<len(text):
            prefix=f'<b>Sent {len(ss)+1}: </b>';check(text.startswith(prefix,start),'NATIVE_SENTENCE_MARKER')
            lo=start+len(prefix);hi=text.index('<br>',lo);end=hi+4
            while end<len(text) and text[end]==' ':end+=1
            ss.append(text[lo:hi]);spans.append([lo,hi]);envelopes.append([start,end]);start=end
        sentence=[ref('multirc',owner,'sentence',[s],[dict(row=di,field='paragraph.text',codepoint_span=spans[i],envelope_span=envelopes[i])],[i,i+1]) for i,s in enumerate(ss)]
        pairs=[];edges=[]
        for i in range(0,len(ss),2):
            j=min(i+2,len(ss));pid=ref('multirc',owner,'sentence_pair',ss[i:j],[dict(row=di,field='paragraph.text',codepoint_span=spans[k]) for k in range(i,j)],[i,j]);pairs.append(pid)
            edges.extend(dict(origin=pid,target=x,kind='coextensive' if j-i==1 else 'contains') for x in sentence[i:j])
        paragraph=ref('multirc',owner,'paragraph',ss,[dict(row=di,field='paragraph.text',codepoint_span=s) for s in spans],[0,len(ss)])
        edges.extend(dict(origin=paragraph,target=x,kind='contains') for x in sentence+pairs)
        edges.extend(dict(origin=a,target=b,kind='next-source-position') for group in (sentence,pairs) for a,b in zip(group,group[1:]))
        documents.append(dict(id=owner,row=di,sentences=sentence,pairs=pairs,paragraph=paragraph,edges=edges,source_alias=alias(owner),source_text_sha256=sha(text),
                              source_characters=len(text),page_status='absent:no-source-pagination',clock='source-sentence-position; not physical time'))
        for qi,q in enumerate(d['paragraph']['questions']):
            case=identity(['multirc',owner,q['idx']]);question=ref('multirc',owner,'question',[q['question']],[dict(row=di,field=f'paragraph.questions[{qi}].question')])
            options=[ref('multirc',owner,'option',[a['text']],[dict(row=di,field=f'paragraph.questions[{qi}].answers[{i}].text')]) for i,a in enumerate(q['answers'])]
            cases.append(dict(id=case,family='multirc',source_row=di,source_question=qi,document=owner,question=question,options=options,facts=sentence,contexts=[*pairs,paragraph]))
            targets.append(dict(case=case,family='multirc',labels=[a['isAnswer'] for a in q['answers']],source_evidence_indices=q['sentences_used'],evidence=[sentence[i] for i in q['sentences_used']],
                 multisent=q['multisent'],quarantine_reasons=['EMPTY_ANSWER'] if any(not a['text'].strip() for a in q['answers']) else [],
                 semantics='provider per-option correctness; question-level reference selection, sufficiency unproven'))
    for i,r in enumerate(qasc):
        owner=r['id'];case=identity(['qasc',owner]);question=ref('qasc',owner,'question',[r['question']],[dict(row=i,field='question')])
        facts=[ref('qasc',owner,'fact',[r[k]],[dict(row=i,field=k)]) for k in ('fact1','fact2')]
        options=[ref('qasc',owner,'option',[s],[dict(row=i,field=f'choices.text[{j}]')]) for j,s in enumerate(r['choices']['text'])]
        cases.append(dict(id=case,family='qasc',source_row=i,source_question=None,document=owner,question=question,options=options,facts=facts,contexts=[]))
        targets.append(dict(case=case,family='qasc',answer_index=r['choices']['label'].index(r['answerKey']),answer_key=r['answerKey'],combined_statement=r['combinedfact'],combined_sha256=sha(r['combinedfact']),quarantine_reasons=[],
                 semantics='provider answer and two-fact composition; joint necessity unproven; incorrect options are not contradiction labels'))
    check(expected==[r['id'] for r in refs] and used==set(range(len(units))),'OBSERVATION_COVERAGE')
    check(documents==data['documents'],'HIERARCHY');check(cases==data['cases'],'CASES');check(targets==data['targets'],'PRIVATE_TARGETS')
    return dict(public_unique_texts=len(units),references=len(refs),source_views_conserved=True,targets_excluded_from_observations=True,structural_edges=sum(len(d['edges']) for d in documents))


def bound_result(root):
    from text.connected_training_census import PINS
    from text.connected_training_capture import MULTIRC_PARENT
    from text.connected_population import PAYLOAD_PINS
    r=json.loads((root/'report.json').read_bytes());check(r['schema']=='connected-grounding-population-v1' and r['complete'] is True and r['errors']==[],'RESULT_STATE')
    check(r['source_pins']=={**PINS,'multirc_parent':MULTIRC_PARENT} and r['census_payloads']==PAYLOAD_PINS,'SOURCE_PINS')
    names={'connected_population.py','connected_observations.py','connected_leakage.py'}
    check(set(r['code_sha256'])==names and all(digest(Path(__file__).resolve().parents[1]/'text'/n)==r['code_sha256'][n] for n in names),'PRODUCER_CODE')
    for k in ('sealed_evaluation','scientific_admission','served','world_entity_source_time_sealed','cross_source_overlap_sealed','encoder_exposure_certified','full_composition_novelty_certified','empty_observations_measured','class_labels_used_for_split'):
        check(r[k] is False,'FALSE_CLAIM_'+k)
    check(r['native_text_floor']==5 and r['target_identity_used_for_split_only'] is True and r['calibration_only'] is True and r['observations_removed']==0,'SCOPE')
    check(r['private_target_path']=='private/targets.jsonl' and r['public_observation_columns']==['row','chunk','entity','text'],'TARGET_CHANNEL')
    for n,v in r['files'].items():
        p=(root/n).resolve();check(p.is_relative_to(root.resolve()) and p!=root.resolve(),'FILE_PATH')
        check(p.stat().st_size==v['bytes'] and digest(p)==v['sha256'],'RESULT_BYTES')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()};check(actual==set(r['files'])|{'report.json'},'EXTRA_ARTIFACT')
    check(r['wall_seconds']>0 and all(v>=0 for v in r['phases_seconds'].values()) and .9<=r['phase_coverage']<=1 and abs(sum(r['phases_seconds'].values())/r['wall_seconds']-r['phase_coverage'])<1e-12,'COST_COVERAGE')
    return r


def readback(root,multirc,qasc,parent):
    from audit.connected_split_readback import read_splits
    start=time.perf_counter();r=bound_result(root);multi,q=sources(multirc,qasc,parent)
    data=read_data(root)
    public=verify_views(data,multi,q);overlap={}
    for cohort in ('multirc','qasc-query-target','qasc-fact'):overlap[cohort]=audit_overlap(root/'cohorts'/cohort/'overlap',root/'cohorts'/cohort/'source')
    split=read_splits(root,data);summary=json.loads((root/'connected/summary.json').read_bytes())
    check(summary==dict(chunks=len(data['units']),chunks_sha256=digest(root/'connected/chunks/atoms.parquet')),'NATIVE_SUMMARY')
    check(r['families']=={'connected':dict(ready=True,summary_sha256=digest(root/'connected/summary.json'))},'NATIVE_FAMILY')
    return dict(passed=True,report_sha256=digest(root/'report.json'),public=public,split=split,overlap={k:v['counts'] for k,v in overlap.items()},
                wall_seconds=time.perf_counter()-start,measurements='author-checked',scientific_admission=False)


def read_data(root):
    return dict(units=pq.read_table(root/'connected/chunks/atoms.parquet').to_pylist(),references=rows(root/'references.jsonl'),
                cases=rows(root/'cases.jsonl'),documents=rows(root/'documents.jsonl'),targets=rows(root/'private/targets.jsonl'))
