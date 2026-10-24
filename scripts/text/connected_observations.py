"""Source-addressed public views and separate private anchors; no lens or model."""
import hashlib
import json
import re
from text.connected_training_census import check,norm,sha


def identity(value):return sha(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')))


def source_alias(name):
    """Conservative identifier-pattern grouping; never asserted world co-reference."""
    s=name.removesuffix('.txt').replace('/','-')
    patterns=[('movie',r'wikiMovieSummaries-(\d+)(?:-\d+)?$'),('wiki',r'Wiki_articles-(?:paragraphs-)?wiki(.+)-\d+$'),
              ('gutenberg',r'gutenberg-(\d+)(?:-\d+)?$'),('mctest',r'(mctest-.+)-\d+$'),
              ('science',r'(science-g\d+)-\d+$'),('masc',r'(masc-.+)-\d+$'),('oanc',r'(oanc-.+)-\d+$')]
    if s.startswith('Sept11-reports-'):return dict(key='multirc-source:Sept11-reports',method='conservative-collection-group',world_referent_resolved=False)
    for kind,pattern in patterns:
        m=re.search(pattern,s)
        if m:return dict(key='multirc-source:'+kind+':'+m[1],method='identifier-pattern-v1',world_referent_resolved=False)
    return dict(key='multirc-document:'+name,method='unresolved-document-only',world_referent_resolved=False)


def build(multirc,qasc,census_docs):
    units=[];lookup={};refs=[];cases=[];targets=[];documents=[]
    def add(family,owner,role,parts,address,span=None):
        text='\n'.join(parts);key=sha(text)
        if text.strip() and key not in lookup:lookup[key]=len(units);units.append(dict(row=len(units),chunk=0,entity=key,text=text))
        unit=lookup[key] if text.strip() else None
        check(unit is None or units[unit]['text']==text,'OBSERVATION_HASH_COLLISION')
        rid=identity([family,owner,role,address,span])
        refs.append(dict(id=rid,family=family,owner=owner,role=role,unit=unit,text_sha256=key,measurement_status='pending-native' if unit is not None else 'unavailable-empty-surface',
                         source_fields=address,source_sentence_range=span,composition='newline-join-unchanged-fields' if len(parts)>1 else 'unchanged-field'))
        return rid
    for row,(d,c) in enumerate(zip(multirc['data'],census_docs)):
        check(d['id']==c['id'],'POPULATION_DOCUMENT_ORDER');source=d['paragraph']['text'];owner=d['id'];sentences=[];sentence_refs=[]
        for atom in c['atoms']:
            lo,hi=atom['codepoint_span'];text=source[lo:hi];check(sha(text)==atom['text_sha256'],'POPULATION_SENTENCE_SPAN');sentences.append(text)
            sentence_refs.append(add('multirc',owner,'sentence',[text],[dict(row=row,field='paragraph.text',codepoint_span=[lo,hi],envelope_span=atom['envelope_span'])],[atom['index'],atom['index']+1]))
        pair_refs=[];edges=[]
        for i in range(0,len(sentences),2):
            j=min(i+2,len(sentences));addresses=[dict(row=row,field='paragraph.text',codepoint_span=c['atoms'][k]['codepoint_span']) for k in range(i,j)]
            pair=add('multirc',owner,'sentence_pair',sentences[i:j],addresses,[i,j]);pair_refs.append(pair)
            for child in sentence_refs[i:j]:edges.append(dict(origin=pair,target=child,kind='coextensive' if j-i==1 else 'contains'))
        para=add('multirc',owner,'paragraph',sentences,[dict(row=row,field='paragraph.text',codepoint_span=a['codepoint_span']) for a in c['atoms']],[0,len(sentences)])
        for child in sentence_refs+pair_refs:edges.append(dict(origin=para,target=child,kind='contains'))
        for group in (sentence_refs,pair_refs):
            for a,b in zip(group,group[1:]):edges.append(dict(origin=a,target=b,kind='next-source-position'))
        documents.append(dict(id=owner,row=row,sentences=sentence_refs,pairs=pair_refs,paragraph=para,edges=edges,source_alias=source_alias(owner),
                              source_text_sha256=sha(source),source_characters=len(source),page_status='absent:no-source-pagination',clock='source-sentence-position; not physical time'))
        for qi,(q,cq) in enumerate(zip(d['paragraph']['questions'],c['questions'])):
            check(q['idx']==cq['id'],'POPULATION_QUESTION_ORDER');case=identity(['multirc',owner,q['idx']])
            question=add('multirc',owner,'question',[q['question']],[dict(row=row,field=f'paragraph.questions[{qi}].question')])
            options=[add('multirc',owner,'option',[a['text']],[dict(row=row,field=f'paragraph.questions[{qi}].answers[{j}].text')]) for j,a in enumerate(q['answers'])]
            cases.append(dict(id=case,family='multirc',source_row=row,source_question=qi,document=owner,question=question,options=options,
                              facts=sentence_refs,contexts=[*pair_refs,para]))
            targets.append(dict(case=case,family='multirc',labels=[a['isAnswer'] for a in q['answers']],source_evidence_indices=q['sentences_used'],
                evidence=[sentence_refs[j] for j in q['sentences_used']],multisent=q['multisent'],quarantine_reasons=cq['quarantine_reasons'],
                semantics='provider per-option correctness; question-level reference selection, sufficiency unproven'))
    for row,r in enumerate(qasc):
        owner=r['id'];case=identity(['qasc',owner]);addresses=lambda field:[dict(row=row,field=field)]
        question=add('qasc',owner,'question',[r['question']],addresses('question'))
        facts=[add('qasc',owner,'fact',[r[k]],addresses(k)) for k in ('fact1','fact2')]
        options=[add('qasc',owner,'option',[text],addresses(f'choices.text[{j}]')) for j,text in enumerate(r['choices']['text'])]
        cases.append(dict(id=case,family='qasc',source_row=row,source_question=None,document=owner,question=question,options=options,facts=facts,contexts=[]))
        targets.append(dict(case=case,family='qasc',answer_index=r['choices']['label'].index(r['answerKey']),answer_key=r['answerKey'],
            combined_statement=r['combinedfact'],combined_sha256=sha(r['combinedfact']),quarantine_reasons=[],
            semantics='provider answer and two-fact composition; joint necessity unproven; incorrect options are not contradiction labels'))
    check(len({r['id'] for r in refs})==len(refs) and len({c['id'] for c in cases})==len(cases),'POPULATION_DUPLICATE_REFERENCE')
    return dict(units=units,references=refs,cases=cases,targets=targets,documents=documents)


def leakage_catalog(data):
    """Identity/overlap metadata only. Private target text is never an observation."""
    tables={k:{} for k in ('multirc','qasc-query-target','qasc-fact')};units=data['units'];refmap={r['id']:r for r in data['references']}
    def add(cohort,text,owner,role):
        normalized=norm(text);key=sha(normalized);table=tables[cohort]
        if key not in table:table[key]=dict(id=key,text=normalized,owners=set(),roles=set())
        check(table[key]['text']==normalized,'LEAKAGE_HASH_COLLISION');table[key]['owners'].add(owner);table[key]['roles'].add(role)
    for r in data['references']:
        if r['role']=='option':continue
        text=units[r['unit']]['text']
        if r['family']=='multirc':add('multirc',text,r['owner'],r['role'])
    for c,t in zip(data['cases'],data['targets']):
        if c['family']!='qasc':continue
        text=lambda rid:units[refmap[rid]['unit']]['text']
        add('qasc-query-target',text(c['question']),c['id'],'question')
        add('qasc-query-target',t['combined_statement'],c['id'],'private-target')
        for rid in c['facts']:add('qasc-fact',text(rid),c['id'],'fact')
    return {name:[{**v,'owners':sorted(v['owners']),'roles':sorted(v['roles'])} for v in table.values()] for name,table in tables.items()}
