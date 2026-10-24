"""Independent raw-field and native-address readback for private composition targets."""
import json
from pathlib import Path
import pyarrow.parquet as pq
from model.qasc_context import POPULATION,NATIVE,POLICIES,LANES,rows
from text.connected_training_census import sha,norm
from text.evaluation_multiscale import file_sha

def check(ok,code):
    if not ok:raise ValueError('MNLM_TARGET_READBACK_'+code)

def read(root,population,native,raw):
    report=json.loads((root/'report.json').read_bytes())
    check(report['schema']=='qasc-composition-targets-v1' and report['complete'] and report['errors']==[],'STATE')
    check(report['population_sha256']==file_sha(population/'report.json')==POPULATION and report['public_native_sha256']==file_sha(native/'report.json')==NATIVE,'PARENTS')
    check(report['source_parquet_sha256']==file_sha(raw/'qasc-hf-training-v1/train.parquet')=='b9a297b5ab55f1605c7682ffbb7042c26d7ecb9ff1e1aa5a820d4e791c8302d1','RAW_PIN')
    check(all(report[n] is False for n in ('probe_is_target','scientific_admission','served','sealed_evaluation','free_text_composition_proven')) and report['lanes']==list(LANES),'CLAIMS')
    for name,pin in report['files'].items():
        p=(root/name).resolve();check(p.is_relative_to(root.resolve()) and p.stat().st_size==pin['bytes'] and file_sha(p)==pin['sha256'],'FILE')
    check({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==set(report['files'])|{'report.json'},'EXTRA_FILE')
    records=pq.read_table(raw/'qasc-hf-training-v1/train.parquet').to_pylist();public=pq.read_table(population/'connected/chunks/atoms.parquet').to_pylist()
    lookup={r['text']:r['row'] for r in public};targets=rows(root/'targets.jsonl');cases=rows(root/'cases.jsonl');sourcecases=[r for r in rows(population/'cases.jsonl') if r['family']=='qasc']
    fresh=pq.read_table(root/'targets/chunks/atoms.parquet').to_pylist();probe_table=pq.read_table(root/'probe/chunks/atoms.parquet').to_pylist();used=set();fresh_used=set()
    check(len(cases)==len(records)==len(sourcecases)==8134,'CASE_COVERAGE');check(len({t['text'] for t in targets})==len(targets),'TARGET_DEDUP')
    check(set(report['families'])=={'probe','targets'},'FAMILY_SET')
    for i,(m,c,source) in enumerate(zip(cases,sourcecases,records)):
        check(set(m)=={'case','source_row','source_field','target','question_unit','fact_units','equals_current_fact','equals_current_question','surface_codec_available'},'CASE_SCHEMA')
        check(m['case']==c['id'] and m['source_row']==c['source_row']==i and m['source_field']=='combinedfact','SOURCE_ADDRESS')
        check(0<=m['target']<len(targets),'TARGET_INDEX');t=targets[m['target']];used.add(m['target'])
        check(set(t)=={'index','text','sha256','utf8_bytes','native'} and set(t['native'])=={'route','family','row'},'TARGET_SCHEMA')
        check(t['index']==m['target'] and t['text']==source['combinedfact'] and t['sha256']==sha(source['combinedfact']) and t['utf8_bytes']==len(t['text'].encode()),'RAW_TARGET')
        check(public[m['question_unit']]['text']==source['question'] and [public[u]['text'] for u in m['fact_units']]==[source['fact1'],source['fact2']],'INPUT_ADDRESS')
        check(m['equals_current_fact']==(t['text'] in [source['fact1'],source['fact2']]) and m['equals_current_question']==(t['text']==source['question']) and m['surface_codec_available']==(len(t['text'].encode())<=512),'CASE_FLAGS')
        a=t['native']
        if t['text'] in lookup:check(a==dict(route='public',family='connected',row=lookup[t['text']]),'CACHE_ADDRESS')
        else:
            check(a['route']=='fresh' and a['family']=='targets' and 0<=a['row']<len(fresh),'FRESH_ADDRESS');f=fresh[a['row']];fresh_used.add(a['row'])
            check(f==dict(row=a['row'],chunk=0,entity=t['sha256'],text=t['text']),'FRESH_TEXT')
    check(used==set(range(len(targets))) and fresh_used==set(range(len(fresh))),'UNUSED_TARGET')
    probe=json.loads((root/'probe.json').read_bytes());ref={r['id']:r for r in rows(population/'references.jsonl')}[probe['reference']]
    check(ref['role']=='paragraph' and ref['family']=='multirc' and probe['public_unit']==ref['unit'] and probe['role']=='probe-only'
          and probe['model_input'] is False and probe['target'] is False and probe['text_sha256']==ref['text_sha256'],'PROBE_SCOPE')
    text=public[probe['public_unit']]['text'];check(1800<=len(text)<=3000 and probe_table==[dict(row=0,chunk=0,entity=sha(text),text=text)],'PROBE_SOURCE')
    for family,table in [('probe',probe_table),('targets',fresh)]:
        summary=json.loads((root/family/'summary.json').read_bytes());check(summary==dict(chunks=len(table),chunks_sha256=file_sha(root/family/'chunks/atoms.parquet')),'SUMMARY')
        check(report['families'][family]==dict(ready=True,summary_sha256=file_sha(root/family/'summary.json')),'FAMILY')
    policies=json.loads((root/'policies.json').read_bytes());original={(r['policy'],r['case']):r for r in rows(population/'assignments.jsonl') if r['policy'] in POLICIES}
    for policy,rr in policies.items():
        check(policy in POLICIES and len(rr)==8134,'POLICY_COVERAGE');train=[i for i,r in enumerate(rr) if original[policy,r['case']]['partition']=='train']
        inputs={norm(records[i][f]) for i in train for f in ('question','fact1','fact2')};seen={norm(records[i]['combinedfact']) for i in train}
        for i,r in enumerate(rr):
            check(r['case']==cases[i]['case'] and all(r[k]==original[policy,r['case']][k] for k in ('group','partition','eligible')),'PARTITION')
            check(r['target_in_training_generation_inputs']==(norm(records[i]['combinedfact']) in inputs) and r['target_in_training_targets']==(norm(records[i]['combinedfact']) in seen),'EXPOSURE')
            check(r['partition']=='train' or not r['target_in_training_targets'],'TARGET_LEAKAGE')
    counts=dict(cases=len(records),unique_targets=len(targets),fresh_targets=len(fresh),cached_targets=len(targets)-len(fresh),measurement_inputs=len(fresh)+1,
                target_characters=sum(len(r['text']) for r in targets),fresh_characters=sum(len(r['text']) for r in fresh),probe_characters=len(text),
                max_target_utf8_bytes=max(len(r['text'].encode()) for r in targets),over_capacity_cases=sum(len(r['combinedfact'].encode())>512 for r in records),
                equals_current_fact_cases=sum(r['combinedfact'] in [r['fact1'],r['fact2']] for r in records),equals_current_question_cases=sum(r['combinedfact']==r['question'] for r in records),
                exposures={p:{s:dict(cases=sum(r['partition']==s for r in rr),targets_seen_as_training_inputs=sum(r['partition']==s and r['target_in_training_generation_inputs'] for r in rr)) for s in ('train','calibration','evaluation')} for p,rr in policies.items()})
    check(counts==report['counts'],'COUNTS')
    check(.9<=report['phase_coverage']<=1 and abs(sum(report['phases_seconds'].values())/report['wall_seconds']-report['phase_coverage'])<1e-12,'COST')
    return dict(report_sha256=file_sha(root/'report.json'),cases=len(cases),unique_targets=len(targets),fresh_targets=len(fresh),cached_targets=len(targets)-len(fresh),
                probe_only=1,raw_targets_exact=True,cache_addresses_exact=True,all_targets_accounted_for=True,scientific_admission=False,served=False)
