"""Independent source-to-rendered-text and native-cache address readback."""
from collections import Counter
import numpy as np
import pyarrow.parquet as pq
from audit.qasc_region_readback import check,read_json,time_check
from audit.qasc_surface_readback import records
from model.qasc_context import LANES,POLICIES,POPULATION,NATIVE
from model.qasc_generation_data import TARGETS,TARGET_NATIVE
from text.connected_training_census import sha
from text.evaluation_multiscale import file_sha

def read(root,population,native,targets,region,surface):
    r=read_json(root/'report.json');check(r['schema']=='qasc-rendered-native-population-v1' and r['complete'] and r['errors']==[] and r['calibration_only'],'GUARD_POP_STATE');time_check(r)
    check(r['population_sha256']==file_sha(population/'report.json')==POPULATION and r['public_native_sha256']==file_sha(native/'report.json')==NATIVE
          and r['targets_sha256']==file_sha(targets/'report.json')==TARGETS and r['target_native_sha256']==TARGET_NATIVE,'GUARD_POP_PINS')
    check(r['region_report_sha256']==file_sha(region/'report.json') and r['surface_report_sha256']==file_sha(surface/'report.json'),'GUARD_POP_MODELS')
    check(all(r[k] is False for k in ('scientific_admission','served','guard_evaluated','free_text_composition_proven')) and r['lanes']==list(LANES),'GUARD_POP_CLAIMS')
    check({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==set(r['files'])|{'report.json'},'GUARD_POP_FILES')
    for n,pin in r['files'].items():
        p=(root/n).resolve();check(p.is_relative_to(root.resolve()) and p.stat().st_size==pin['bytes'] and file_sha(p)==pin['sha256'],'GUARD_POP_FILE')
    public=pq.read_table(population/'connected/chunks/atoms.parquet').to_pylist();source={v['text']:dict(route='public',row=v['row']) for v in public}
    for t in records(targets/'targets.jsonl'):
        a=t['native'];source.setdefault(t['text'],dict(route='public' if a['route']=='public' else 'target',row=a['row'] if a['route']=='public' else a['row']+1))
    outputs=records(root/'outputs.jsonl');candidates=records(root/'candidates.jsonl');fresh=pq.read_table(root/'rendered/chunks/atoms.parquet').to_pylist();used=set();used_fresh=set()
    expected=[]
    for policy in POLICIES:
        for arm in ('true','gaussian','predicted'):
            for j,x in enumerate(records(surface/policy/arm/'renders.jsonl')):expected.append((policy,arm,j,x))
    check(len(candidates)==len(expected)==3072,'GUARD_POP_CANDIDATES')
    for i,(m,(policy,arm,j,x)) in enumerate(zip(candidates,expected)):
        check(set(m)=={'index','policy','arm','source_file','source_row','case','row','partition','route','status','output','asserted'},'GUARD_POP_CANDIDATE_SCHEMA')
        check(m['index']==i and m['policy']==policy and m['arm']==arm and m['source_file']==policy+'/'+arm+'/renders.jsonl' and m['source_row']==j,'GUARD_POP_SOURCE_ADDRESS')
        check(all(m[k]==x[k] for k in ('case','row','partition','route','status')) and m['asserted'] is False,'GUARD_POP_SOURCE_FIELDS')
        if x['status']!='complete_unasserted':check(m['output'] is None,'GUARD_POP_REFUSAL_HIDDEN');continue
        check(type(m['output'])==int and 0<=m['output']<len(outputs),'GUARD_POP_OUTPUT_INDEX');o=outputs[m['output']];used.add(m['output'])
        check(set(o)=={'index','text','sha256','utf8_bytes','native'} and o['index']==m['output'] and o['text']==x['text']
              and o['sha256']==sha(x['text']) and o['utf8_bytes']==len(x['text'].encode('utf-8'))<=512,'GUARD_POP_OUTPUT_TEXT')
        a=o['native'];check(set(a)=={'route','row'},'GUARD_POP_ADDRESS_SCHEMA')
        if o['text'] in source:check(a==source[o['text']],'GUARD_POP_CACHED_ADDRESS')
        else:
            check(a['route']=='fresh' and type(a['row'])==int and 0<=a['row']<len(fresh),'GUARD_POP_FRESH_ADDRESS');used_fresh.add(a['row'])
            check(fresh[a['row']]==dict(row=a['row'],chunk=0,entity=o['sha256'],text=o['text']),'GUARD_POP_FRESH_TEXT')
    check(used==set(range(len(outputs))) and used_fresh==set(range(len(fresh))) and len({o['text'] for o in outputs})==len(outputs),'GUARD_POP_DEDUP_COVERAGE')
    probe=read_json(root/'probe.json');native_probes=[read_json(native/n/'probe.json') for n in LANES]
    check(set(probe)=={'public_unit','text_sha256','role','model_input','target'} and probe['role']=='probe-only' and probe['model_input'] is False and probe['target'] is False,'GUARD_POP_PROBE_SCOPE')
    text=public[probe['public_unit']]['text'];check(probe['text_sha256']==sha(text) and all(p['row']==probe['public_unit'] and p['family']=='connected' and p['text_sha256']==sha(text) and len(p['trace']['windows'])>1 for p in native_probes),'GUARD_POP_PROBE_SOURCE')
    table=pq.read_table(root/'probe/chunks/atoms.parquet').to_pylist();check(table==[dict(row=0,chunk=0,entity=sha(text),text=text)],'GUARD_POP_PROBE_TEXT')
    check(set(r['families'])=={'probe','rendered'},'GUARD_POP_FAMILIES')
    for family,n in [('probe',1),('rendered',len(fresh))]:
        s=read_json(root/family/'summary.json');check(s==dict(chunks=n,chunks_sha256=file_sha(root/family/'chunks/atoms.parquet'))
            and r['families'][family]==dict(ready=bool(n),summary_sha256=file_sha(root/family/'summary.json')),'GUARD_POP_SUMMARY')
    counts=dict(attempts=len(candidates),statuses=dict(Counter(x['status'] for x in candidates)),unique_complete_outputs=len(outputs),fresh_outputs=len(fresh),cached_outputs=len(outputs)-len(fresh),
        cached_routes=dict(Counter(o['native']['route'] for o in outputs if o['native']['route']!='fresh')),fresh_characters=sum(len(x['text']) for x in fresh),measurement_inputs=len(fresh)+1,
        max_complete_utf8_bytes=max((o['utf8_bytes'] for o in outputs),default=0),probe_characters=len(text))
    check(counts==r['counts'],'GUARD_POP_COUNTS')
    return dict(report_sha256=file_sha(root/'report.json'),counts=counts,all_source_texts_exact=True,all_native_cache_addresses_exact=True,probe_excluded=True,
                scientific_admission=False,served=False,acceptance='author-checked')
