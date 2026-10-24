"""Independent recovery authority for interrupted graph queues; never repair old receipts."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
import time

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
from audit.legal_graph_coverage import code as original_code
from audit.legal_graph_continuation import ident
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from kernel.certified_index_graph import pinned
from kernel.native_graph_recovery import stamp
from text.constellation_corpus import file_sha,need

QUEUES={'phase52-full-graphs-v1-20261023':(948511,'395931103'),
        'phase73-full-continuation-v1-20261023':(2156766,'398872361')}
PREFIXES={'finbert-mean':77,'pubmedbert-embeddings':166,'specter2-base':281}


def dead(owners):
    for owner in owners:
        need(ident(owner['pid'])!=owner['starttime'],'GRAPH_OBSERVER_OWNER_LIVE')


def lock_graphs(wait=False):
    streams=[]
    try:
        for name in ['legal-native-graph-queue','legal-native-graph']:
            path=Path(__file__).resolve().parents[2]/'.lease'/name
            stream=path.open('a+');streams.append(stream)
            try:fcntl.flock(stream,fcntl.LOCK_EX|(0 if wait else fcntl.LOCK_NB))
            except BlockingIOError:need(False,'GRAPH_OBSERVER_LEASE_BUSY')
        return streams
    except BaseException:
        for stream in streams:stream.close()
        raise


def capture(root,graphs):
    """Bounded metadata census. Numeric payload authentication remains import_source's job."""
    roots={};owners=[];files={}
    def remember(path):
        path=Path(path);need(path.is_file() and not path.is_symlink(),'GRAPH_OBSERVER_FILE')
        files[str(path)]=dict(sha256=file_sha(path),stamp=stamp(path))
    for name,(pid,start) in QUEUES.items():
        directory=root/name;launch=json.loads((directory/'launch.json').read_bytes())
        need((launch['pid'],launch['starttime'])==(pid,start),'GRAPH_OBSERVER_OWNER_ID')
        owners.append(dict(pid=pid,starttime=start))
        events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
        running=[e for e in events if e['stage']=='running']
        for event in running:
            owners.append({k:event[k] for k in ['pid','starttime']})
            ends=[e for e in events if e['stage']=='terminal' and e['label']==event['label']]
            need(len(ends)==1,'GRAPH_OBSERVER_CHILD_EXIT_UNMEASURED')
        for path in sorted(directory.iterdir()):
            need(path.is_file(),'GRAPH_OBSERVER_QUEUE_ENTRY');remember(path)
        if name.startswith('phase52'):
            need(not (directory/'report.json').exists() and events[-1]['stage']=='lane_terminal'
                 and events[-1]['instrument']=='specter2-base'
                 and events[-1]['result']['error']=='BrokenPipeError:[Errno 32] Broken pipe',
                 'GRAPH_OBSERVER_UNEXPECTED_INTERRUPTION')
            state='interrupted_without_queue_terminal_receipt'
        else:
            r=json.loads((directory/'report.json').read_bytes())
            need(r['complete'] is False and events[-1]['stage']=='queue_terminal'
                 and events[-1]['pin']==file_sha(directory/'report.json'),'GRAPH_OBSERVER_FAILED_QUEUE_RECEIPT')
            state='failed_with_terminal_receipt'
        roots[name]=dict(state=state,children=len(running),last_event=events[-1])
    dead(owners)
    sources={}
    old_events=[json.loads(line) for line in (root/'phase52-full-graphs-v1-20261023/events.jsonl').read_text().splitlines()]
    for name,count in PREFIXES.items():
        directory=graphs/f'legal-native-{name}-full-v1-20261023'
        plan=json.loads((directory/'plan.json').read_bytes())
        need(plan['scope']=='full' and plan['population']==6360378 and plan['code']==original_code(),
             'GRAPH_OBSERVER_ORIGINAL_PLAN')
        refs=[]
        for j in range(count):
            lo=j*16384;hi=lo+16384;shard=directory/f'{lo:012d}-{hi:012d}'
            seal=json.loads((shard/'COMMITTED').read_bytes());pin=seal['report_sha256']
            pinned(shard/'report.json',pin)
            refs.append(dict(path=str(shard),pin=pin))
            for item in ['COMMITTED','report.json','readback.json']:remember(shard/item)
        partial=directory/f'{count*16384:012d}-{(count+1)*16384:012d}'
        need(partial.is_dir() and not (partial/'COMMITTED').exists(),'GRAPH_OBSERVER_PARTIAL')
        need({p.name for p in directory.iterdir() if p.is_dir()}==
             {Path(r['path']).name for r in refs}|{partial.name},'GRAPH_OBSERVER_SHARD_CENSUS')
        failures=list(directory.glob('failure-*.json'))
        need(len(failures)==1,'GRAPH_OBSERVER_FAILURE_COUNT')
        failure=json.loads(failures[0].read_bytes())
        need(failure['error']=='ValueError:MNLM_TEXT_CORPUS_GRAPH_BATCH_PHASE_ATTRIBUTION'
             and not failure['complete'],'GRAPH_OBSERVER_FAILURE_CAUSE')
        for p in directory.iterdir():
            if p.is_file():remember(p)
        for p in partial.iterdir():remember(p)
        events=[r for r in old_events if r.get('label')==name+'-full']
        need(len(events)==2 and events[0]['stage']=='running' and events[1]['stage']=='terminal'
             and events[1]['exit_code']==1 and events[0]['command'][-1]==str(directory),
             'GRAPH_OBSERVER_SOURCE_EXIT')
        sources[name]=dict(root=str(directory),plan_pin=file_sha(directory/'plan.json'),
            sealed=refs,partial=str(partial),committed_rows=count*16384,events=events)
    dead(owners)
    return dict(schema='legal-graph-terminal-observation-v1',queues=roots,owners=owners,files=files,
        sources=sources,original_completion=False,recovery_authority_only=True,
        numeric_payloads_authenticated=False,scientific_admission=False)


def validate(path,pin):
    r=pinned(path,pin)
    need(r['schema']=='legal-graph-terminal-observation-v1' and r['recovery_authority_only']
         and not r['original_completion'] and not r['scientific_admission'],'GRAPH_OBSERVER_SCOPE')
    dead(r['owners'])
    for name,ref in r['files'].items():
        need(stamp(name)==ref['stamp'] and file_sha(name)==ref['sha256'],'GRAPH_OBSERVER_CHANGED_EVIDENCE')
    return r


def source_snapshot(root,evidence,lane,observation,observation_pin):
    r=validate(observation,observation_pin);name=lane.binding['instrument'];source=r['sources'][name]
    root=Path(root);need(str(root)==source['root'],'GRAPH_OBSERVER_SOURCE_ROOT')
    plan=pinned(root/'plan.json',source['plan_pin'])
    need(plan['binding']==lane.binding and plan['queries']==plan['population']==len(lane.index.uids)
         and plan['code']==original_code() and plan['policy']==dict(all_pairs_exhaustive=False,
             candidate_policy='128_per_native_window',query_tile=64,score_policy='unchanged_native_window_index',
             shard_rows=16384,threads=32,top_k=10),'GRAPH_OBSERVER_SOURCE_PLAN')
    files={};records=[]
    def remember(p,ref=None):
        need(p.is_file() and not p.is_symlink(),'GRAPH_OBSERVER_SOURCE_FILE')
        if ref:need(p.stat().st_size==ref['bytes'],'GRAPH_OBSERVER_SOURCE_SIZE')
        files[str(p.relative_to(root))]=dict(bytes=p.stat().st_size,
            sha256=ref['sha256'] if ref else file_sha(p),stamp=stamp(p))
    for j,ref in enumerate(source['sealed']):
        lo=j*16384;hi=lo+16384;path=Path(ref['path']);report=pinned(path/'report.json',ref['pin'])
        need(path==root/f'{lo:012d}-{hi:012d}' and report['plan_pin']==source['plan_pin']
             and report['binding']==lane.binding,'GRAPH_OBSERVER_SOURCE_SHARD')
        need({p.name for p in path.iterdir()}==set(report['files'])|{'report.json','COMMITTED','readback.json'},
             'GRAPH_OBSERVER_SOURCE_SHARD_FILES')
        for item,value in report['files'].items():remember(path/item,value)
        for item in ['report.json','COMMITTED','readback.json']:remember(path/item)
        need(json.loads((path/'COMMITTED').read_bytes())==dict(report_sha256=ref['pin']),
             'GRAPH_OBSERVER_SOURCE_SEAL')
        records.append(dict(path=str(path),pin=ref['pin'],plan_pin=source['plan_pin'],start=lo,stop=hi,queries=hi-lo))
    partial=Path(source['partial']);partial_files={}
    need(not (partial/'COMMITTED').exists(),'GRAPH_OBSERVER_PARTIAL_CHANGED')
    for p in sorted(partial.iterdir()):remember(p);partial_files[p.name]=files[str(p.relative_to(root))]
    for p in root.iterdir():
        if p.is_file():remember(p)
    return dict(root=str(root),incident=dict(path=str(observation),pin=observation_pin),original_plan=plan,
        plan_pin=source['plan_pin'],events=source['events'],imports=records,
        partial=dict(path=str(partial),files=partial_files),files=files,numeric_payloads_authenticated=False)


def main(a):
    locks=lock_graphs();r=capture(a.root,a.graphs)
    a.output.mkdir(parents=True,exist_ok=False)
    r.update(observed_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        observer=dict(pid=os.getpid(),starttime=ident(os.getpid())),acceptance='author-checked',number_status='measured')
    pin=write_committed(a.output/'observation.json',r)
    controls={'live_owner':rejected(lambda:dead([r['observer']])),
        'occupied_lease':rejected(lambda:lock_graphs()),
        'changed_pin':rejected(lambda:validate(a.output/'observation.json','0'*64))}
    changed=copy.deepcopy(r);changed['files'][next(iter(changed['files']))]['sha256']='0'*64
    bad=write_committed(a.output/'negative-changed-evidence.json',changed)
    controls['changed_evidence']=rejected(lambda:validate(a.output/'negative-changed-evidence.json',bad))
    need(validate(a.output/'observation.json',pin)==r,'GRAPH_OBSERVER_READBACK')
    write_committed(a.output/'report.json',dict(complete=True,observation_pin=pin,controls=controls,
        code_pin=file_sha(__file__),original_completion=False,recovery_authority_only=True,
        acceptance='author-checked',number_status='measured'))
    for lock in locks:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--graphs',type=Path,default=Path('/zfs/hot/mnlm/associations'))
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
