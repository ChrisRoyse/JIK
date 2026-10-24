"""Authenticate original graph generations and retain their original provenance."""
import json
from pathlib import Path

from audit.legal_graph_coverage import code as original_code
from audit.native_graph_coverage import read_shard, replay_samples
from kernel.certified_index_graph import pinned
from kernel.native_graph_recovery import stamp, scope_plan
from text.constellation_corpus import file_sha, need


INCIDENTS={
 'finbert-mean':('phase68-finbert-graph-incident.json',
                 '60f87639dcb93f7fdc97043fc40d16678cb4c91354bf4ea1acada24083eefb99',77),
 'pubmedbert-embeddings':('phase71-pubmed-graph-incident.json',
                 '89cda5def9e3bddb69eaa90c51985238ce83617b650e459147025174fc8b5fcd',166)}


def source_snapshot(root,evidence,lane):
    root=Path(root);evidence=Path(evidence);name=lane.binding['instrument']
    filename,incident_pin,count=INCIDENTS[name]
    incident=pinned(evidence/filename,incident_pin)
    if name=='finbert-mean':
        refs=[dict(path=r['path'],pin=r['report']) for r in incident['outputs'] if r.get('complete')]
    else:refs=[dict(path=r['path'],pin=r['pin']) for r in incident['sealed']]
    need(len(refs)==count,'GRAPH_RECOVERY_INCIDENT_CENSUS')
    events=[json.loads(line) for line in (evidence/'phase52-full-graphs-v1-20261023/events.jsonl').read_text().splitlines()]
    events=[r for r in events if r.get('label')==name+'-full']
    need(len(events)==2 and events[0]['stage']=='running' and events[1]['stage']=='terminal'
         and events[1]['exit_code']==1 and events[0]['command'][-1]==str(root),'GRAPH_RECOVERY_SOURCE_NOT_TERMINAL')
    proc=Path('/proc')/str(events[0]['pid'])/'stat'
    need(not proc.exists() or proc.read_text().rsplit(')',1)[1].split()[19]!=events[0]['starttime'],
         'GRAPH_RECOVERY_SOURCE_LIVE')
    plan_pin=file_sha(root/'plan.json');plan=pinned(root/'plan.json',plan_pin)
    need(plan['scope']=='full' and plan['binding']==lane.binding and plan['code']==original_code()
         and plan['queries']==plan['population']==len(lane.index.uids)
         and plan['policy']==dict(all_pairs_exhaustive=False,candidate_policy='128_per_native_window',
                query_tile=64,score_policy='unchanged_native_window_index',shard_rows=16384,threads=32,top_k=10),
         'GRAPH_RECOVERY_ORIGINAL_PLAN')
    if name=='pubmedbert-embeddings':need(plan_pin==incident['plan_pin'],'GRAPH_RECOVERY_INCIDENT_PLAN')
    files={};records=[]
    def remember(path,ref=None):
        need(path.is_file() and not path.is_symlink(),'GRAPH_RECOVERY_SOURCE_FILE')
        files[str(path.relative_to(root))]=dict(bytes=path.stat().st_size,
            sha256=file_sha(path) if ref is None else ref['sha256'],stamp=stamp(path))
        if ref:need(path.stat().st_size==ref['bytes'],'GRAPH_RECOVERY_SOURCE_SIZE')
    for j,ref in enumerate(refs):
        first=j*16384;last=first+16384;path=root/f'{first:012d}-{last:012d}'
        need(str(path)==ref['path'],'GRAPH_RECOVERY_SOURCE_ADDRESS')
        r=pinned(path/'report.json',ref['pin'])
        need(r['plan_pin']==plan_pin and r['binding']==lane.binding,'GRAPH_RECOVERY_SOURCE_BINDING')
        need(set(p.name for p in path.iterdir())==set(r['files'])|{'report.json','COMMITTED','readback.json'},
             'GRAPH_RECOVERY_SOURCE_SHARD_FILES')
        for item,value in r['files'].items():remember(path/item,value)
        for item in ['report.json','COMMITTED','readback.json']:remember(path/item)
        need(json.loads((path/'COMMITTED').read_bytes())==dict(report_sha256=ref['pin']),
             'GRAPH_RECOVERY_SOURCE_SEAL')
        records.append(dict(path=str(path),pin=ref['pin'],plan_pin=plan_pin,start=first,stop=last,queries=16384))
    partial=root/f'{count*16384:012d}-{(count+1)*16384:012d}'
    need(partial.is_dir() and not (partial/'COMMITTED').exists(),'GRAPH_RECOVERY_PARTIAL_SCOPE')
    partial_files={}
    for p in sorted(partial.iterdir()):
        remember(p);partial_files[p.name]=files[str(p.relative_to(root))]
    expected={Path(r['path']).name for r in records}|{partial.name}
    need({p.name for p in root.iterdir() if p.is_dir()}==expected,'GRAPH_RECOVERY_UNKNOWN_SHARD')
    metadata={p.name for p in root.iterdir() if p.is_file()}
    need({'plan.json','events.jsonl'}<=metadata and not {'report.json','COMMITTED'}&metadata,
         'GRAPH_RECOVERY_SOURCE_STATE')
    failures=[]
    for item in sorted(metadata):
        need(item in ['plan.json','events.jsonl'] or item.startswith('failure-'),'GRAPH_RECOVERY_SOURCE_METADATA')
        remember(root/item)
        if item.startswith('failure-'):
            r=json.loads((root/item).read_bytes())
            need(not r['complete'] and r['code']==plan['code']
                 and r['error']=='ValueError:MNLM_TEXT_CORPUS_GRAPH_BATCH_PHASE_ATTRIBUTION',
                 'GRAPH_RECOVERY_FAILURE_CAUSE')
            failures.append(item)
    need(len(failures)==1,'GRAPH_RECOVERY_FAILURE_COUNT')
    return dict(root=str(root),incident=dict(path=str(evidence/filename),pin=incident_pin),
        original_plan=plan,plan_pin=plan_pin,events=events,imports=records,
        partial=dict(path=str(partial),files=partial_files),files=files,
        numeric_payloads_authenticated=False)


def unchanged(snapshot):
    root=Path(snapshot['root'])
    actual=set()
    for p in root.iterdir():
        if p.is_file():actual.add(p.name)
        elif p.is_dir():actual.update(str(f.relative_to(root)) for f in p.iterdir())
        else:raise ValueError('GRAPH_RECOVERY_UNKNOWN_SOURCE_ENTRY')
    need(actual==set(snapshot['files']),'GRAPH_RECOVERY_SOURCE_FILESET_CHANGED')
    for name,ref in snapshot['files'].items():
        need(stamp(root/name)==ref['stamp'],'GRAPH_RECOVERY_SOURCE_CHANGED')


def import_source(snapshot,lane,selection):
    """Hash every payload, inspect every row, replay the original sample policy."""
    scope_plan(snapshot['imports'],[],len(lane.index.uids));records=[];payload_bytes=0
    expected_samples=set(selection)
    for ref in snapshot['imports']:
        ids=list(range(ref['start'],ref['stop']))
        r,a=read_shard(ref['path'],ref['pin'],ref['plan_pin'],lane,ids)
        chosen=sorted({ids[0],ids[len(ids)//2],ids[-1]} | (set(ids)&expected_samples))
        sample=replay_samples(a,lane,chosen)
        need(json.loads((Path(ref['path'])/'readback.json').read_bytes())==sample,'GRAPH_RECOVERY_ORIGINAL_REPLAY')
        records.append(dict(**ref,native_windows=r['native_windows'],sample=sample,
            original_timing_contract='legacy_publisher_unchanged',recomputed=False))
        payload_bytes+=sum(v['bytes'] for v in r['files'].values())
    unchanged(snapshot)
    return dict(shards=records,queries=sum(r['queries'] for r in records),
        numeric_samples=sum(r['sample']['samples'] for r in records),
        payload_bytes=payload_bytes,numeric_payloads_authenticated=True,
        all_rows_structurally_read=True,all_numeric_rows_replayed=False)
