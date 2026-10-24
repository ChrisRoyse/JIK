"""Cold legal-stock byte/index readback and explicit case-ancestry closure."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canon.training_scope import POLICY_SHA256, require_training_source
from claims.process_conditions import write_committed
from text.evaluation_multiscale import file_sha
from text.legal_stock import EXPERIMENT, SOURCES, record


def check(ok, code):
    if not ok:
        raise ValueError('MNLM_LEGAL_READBACK_' + code)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def hash_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def bind(index, raw, source, ordinal, offset):
    row = json.loads(raw)
    metadata = row['metadata']
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    text = row['text']; body = text.encode()
    expected = dict(id=hash_bytes(encode(dict(source=source, ordinal=ordinal))),
        source=source, ordinal=ordinal, offset=offset, line_bytes=len(raw), line_sha256=hash_bytes(raw),
        text_sha256=hash_bytes(body), normalized_sha256=hash_bytes(encode(' '.join(text.casefold().split()))),
        text_bytes=len(body), characters=len(text), metadata_sha256=hash_bytes(encode(metadata)),
        anchors_sha256=hash_bytes(encode(row['anchors'])),
        **{k:metadata.get(k) for k in ('opinion_id','cluster_id','docket_id','court_id','docket_number',
            'date_filed','retrieval_ts','license','source_dataset','source_url','cuyahoga_signal')})
    check(index == expected, 'RECORD_BINDING')
    return metadata


def rejected(action):
    try:
        action()
    except (ValueError, RuntimeError) as error:
        return str(error)
    raise AssertionError('MNLM_LEGAL_READBACK_NEGATIVE_PASSED')


def run(root, pin, output):
    start = tick = time.perf_counter(); output.mkdir(parents=True, exist_ok=False)
    raw = (root/'report.json').read_bytes(); report = json.loads(raw)
    check(hash_bytes(raw) == pin and json.loads((root/'COMMITTED').read_bytes()) == dict(report_sha256=pin), 'COMMIT')
    check(report['complete'] and report['policy_sha256'] == POLICY_SHA256 and not report['training_ready'], 'SCOPE')
    for name, ref in report['files'].items():
        path = root/name
        check(path.stat().st_size == ref['bytes'] and file_sha(path) == ref['sha256'], 'ARTIFACT_'+name)
        if 'source' in ref:
            check(file_sha(Path(ref['source'])) == ref['sha256'], 'ORIGINAL_'+name)
    phases = dict(all_copied_and_original_bytes=time.perf_counter()-tick); tick=time.perf_counter()
    indexed, by_identity, recount = {}, defaultdict(list), {}
    first = None
    with (root/'records.jsonl').open('rb') as saved:
        for source, name, expected_count, expected_pin in SOURCES:
            path=root/'raw'/name; offset=count=total_bytes=chars=maximum=0
            courts, signals, licences, dates = Counter(), Counter(), Counter(), Counter()
            for raw in path.open('rb'):
                line = saved.readline(); check(bool(line), 'INDEX_TOO_SHORT'); index=json.loads(line)
                metadata = bind(index, raw, source+':'+name, count, offset)
                check(index['id'] not in indexed, 'DUPLICATE_ID'); indexed[index['id']]=index
                for key in ('text_sha256','normalized_sha256','opinion_id','cluster_id','docket_id'):
                    by_identity[key+':'+index[key]].append(index['id'])
                courts[index['court_id']]+=1; signals[str(index['cuyahoga_signal'])]+=1
                licences[index['license']]+=1; dates['present' if index['date_filed'] else 'absent']+=1
                count+=1; offset+=len(raw); total_bytes+=index['text_bytes'];chars+=index['characters']
                maximum=max(maximum,index['characters'])
                if first is None: first=(deepcopy(index),raw,source+':'+name)
            check(count==expected_count and (expected_pin is None or file_sha(path)==expected_pin), 'EXPECTED_SOURCE')
            recount[name]=dict(records=count,text_bytes=total_bytes,characters=chars,max_characters=maximum,
                courts=dict(courts),signals=dict(signals),licences=dict(licences),dates=dict(dates))
        check(not saved.read(1), 'INDEX_TOO_LONG')
    check(recount==report['counts'] and len(indexed)==report['records'], 'CENSUS')
    shared={k:v for k,v in by_identity.items() if len(v)>1}
    rows=[json.loads(line) for line in (root/'shared-identities.jsonl').read_bytes().splitlines()]
    check(rows==[dict(identity=k,records=v) for k,v in sorted(shared.items())], 'SHARED_INDEX')
    phases['all_record_source_bindings']=time.perf_counter()-tick; tick=time.perf_counter()
    parent={k:k for k in indexed}
    def find(key):
        while parent[key]!=key:
            parent[key]=parent[parent[key]];key=parent[key]
        return key
    def join(keys):
        if not keys: return
        head=find(keys[0])
        for key in keys[1:]:
            other=find(key)
            if head!=other:parent[other]=head
    for values in shared.values():join(values)
    opinion={k.removeprefix('opinion_id:'):v for k,v in by_identity.items() if k.startswith('opinion_id:')}
    alias_rows=0; alias_links=[]; new_aliases={}
    alias_path=root/'raw/expansion-v5/ohio-sc-whole/opinion_aliases.jsonl'
    for raw in alias_path.open('rb'):
        a=json.loads(raw); canonical=str(a['canonical_opinion_id']); key=str(a['opinion_id'])
        check(canonical in opinion, 'ALIAS_CANONICAL')
        targets=opinion[canonical]
        check(all(indexed[t]['text_sha256']==a['content_sha256'] for t in targets), 'ALIAS_CONTENT')
        if key in new_aliases:check(new_aliases[key]==targets, 'ALIAS_COLLISION')
        new_aliases[key]=targets
        linked=list(targets)+opinion.get(key,[])
        for kind in ('cluster_id','docket_id'):
            linked+=by_identity.get(kind+':'+str(a[kind]),[])
        join(linked);alias_rows+=1
        if len(set(linked))>1:alias_links.append(dict(opinion=key,records=sorted(set(linked))))
    check(alias_rows==96587, 'ALIAS_COUNT')
    drops=[json.loads(line) for line in (root/'raw/ingest/dedup_dropped.jsonl').read_bytes().splitlines()]
    for a in drops:
        targets=opinion[str(a['first_opinion_id'])]
        check(all(indexed[t]['text_sha256']==a['text_sha256'] for t in targets), 'CUY_ALIAS_CONTENT')
        new_aliases[str(a['opinion_id'])]=targets
        join(targets+opinion.get(str(a['opinion_id']),[]))
    components=defaultdict(list)
    for key in sorted(indexed):components[find(key)].append(key)
    groups=sorted((sorted(v) for v in components.values()),key=lambda v:hash_bytes(encode(v)))
    write_committed(output/'case-components.json', dict(components=groups, alias_links=alias_links,
        source_report=pin, joins=['exact_text','casefold_whitespace_text','opinion_id','cluster_id','docket_id','provider_alias'],
        near_duplicate_audit_complete=False, lineage_complete_beyond_retained_metadata=False,
        case_names_used_as_identity=False, assigned_splits=False))
    phases['provider_alias_case_closure']=time.perf_counter()-tick;tick=time.perf_counter()
    selections=[]
    for key in sorted(indexed)[:32]:
        r=indexed[key]; name=r['source'].split(':',1)[1]
        with (root/'raw'/name).open('rb') as inp:
            inp.seek(r['offset']); data=inp.read(r['line_bytes'])
        bind(r,data,r['source'],r['ordinal'],r['offset'])
        selections.append(dict(record=key,line_sha256=hash_bytes(data),text_sha256=r['text_sha256'],
                               model_input='text_only; metadata and target-bearing labels withheld'))
    idx, raw, source=first; wrong=deepcopy(idx);wrong['offset']+=1
    wrong_text=json.loads(raw);wrong_text['text']=''
    wrong_meta=deepcopy(idx);wrong_meta['cluster_id']='invalid'
    controls=dict(changed_offset=rejected(lambda:bind(wrong,raw,source,0,0)),
        changed_ancestry=rejected(lambda:bind(wrong_meta,raw,source,0,0)),
        empty_source=rejected(lambda:record(wrong_text,source,0,0,encode(wrong_text))),
        unauthorized_experiment=rejected(lambda:require_training_source('cuyahoga-eighth-district',experiment='unknown')),
        wrong_experiment_source=rejected(lambda:require_training_source('not-authorized',experiment=EXPERIMENT)),
        default_exclusion_preserved=rejected(lambda:require_training_source('cuyahoga-eighth-district')))
    for source,_,_,_ in SOURCES:require_training_source(source,experiment=EXPERIMENT)
    write_committed(output/'consumer-selections.json',selections)
    phases['consumer_and_negative_controls']=time.perf_counter()-tick;tick=time.perf_counter()
    check(hash_bytes((root/'report.json').read_bytes())==pin, 'FINAL_REPORT_DRIFT')
    files={n:file_sha(output/n) for n in ('case-components.json','consumer-selections.json')}
    phases['final_hashes']=time.perf_counter()-tick;seconds=time.perf_counter()-start
    check(sum(phases.values())/seconds>=.9,'TIMING')
    result=dict(schema='legal-source-stock-readback-v1',source=dict(path=str(root),pin=pin),
        complete=True,records=len(indexed),counts=recount,components=len(groups),
        max_component_records=max(map(len,groups)),component_sizes=dict(Counter(map(len,groups))),
        unique_texts=report['unique_texts'],provider_alias_rows=alias_rows,cuyahoga_alias_rows=len(drops),
        source_bytes_unchanged=True,all_record_bindings_recomputed=True,consumer_cases=len(selections),
        controls=controls,files=files,scientific_admission=False,training_ready=False,
        near_duplicate_audit_complete=False,model_updates=0,acceptance='author-checked',number_status='measured',
        phases_seconds=phases,seconds=seconds,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        code_sha256=file_sha(Path(__file__)))
    receipt=write_committed(output/'report.json',result)
    print(json.dumps(dict(pin=receipt,records=len(indexed),components=len(groups),controls=controls,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.pin,a.output)
