"""Independently recount physical process-transition evidence and live owner closure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def need(ok, code):
    if not ok:raise ValueError('HANDOFF_READBACK_'+code)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path,pin=None):
    if pin is not None:need(sha(path)==pin,'PIN')
    return json.loads(Path(path).read_bytes())


def kernel(ref):
    root=Path('/proc')/str(ref['pid'])
    try:
        s=(root/'stat').read_text().rsplit(')',1)[1].split()
        children=set()
        for task in (root/'task').iterdir():
            try:children.update(map(int,(task/'children').read_text().split()))
            except FileNotFoundError:pass
        return dict(pid=ref['pid'],starttime=s[19],state=s[0],parent=int(s[1]),children=sorted(children),
                    command=hashlib.sha256((root/'cmdline').read_bytes()).hexdigest())
    except (FileNotFoundError,ProcessLookupError):return None


def exact_scope(rows,refs,stopped):
    for name,ref in refs.items():
        row=rows[name]
        need((row['pid'],row['starttime'])==(ref['pid'],ref['starttime']),'IDENTITY')
        need((row['state']=='T')==(ref['pid'] in stopped),'STOPPED_SET')
        if row['state']!='T':need(row['state'] in ('R','S','D'),'RUNNING_STATE')


def progressed(before,after,refs):
    exact_scope(after,refs,set())
    need(all(after[n]['heartbeat']>before[n]['heartbeat'] for n in refs),'HEARTBEAT')


def run(a):
    began=time.monotonic();a.output.mkdir(exist_ok=False)
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    proof=read(a.input/'report.json',a.pin)
    need(proof['complete'] and proof['boundaries']==27 and proof['all_recovered'] and proof['owners']==6
         and proof['unrelated']==1 and not proof['gpu_used'] and not proof['existing_jobs_signalled']
         and proof['successors_started']==0 and not proof['authorizes_retirement'],'PROOF_SCOPE')
    files={}
    def check(path,pin):need(sha(path)==pin,'FILE_BYTES');files[str(path)]=pin
    check(a.input/'report.json',a.pin)
    for n,h in proof['files'].items():
        path=(a.input/n).resolve();need(path.is_relative_to(a.input.resolve()),'FILE_SCOPE');check(path,h)
    request=read(a.input/'request.json');owners=request['owners'];order=request['order'];ids=set(order)
    need(len(owners)==len(ids)==6 and {r['pid'] for r in owners}==ids,'ROSTER')
    refs={p.name.removesuffix('-launch.json'):read(p) for p in (a.input/'fixtures').glob('*-launch.json')}
    need(len(refs)==7 and len(set(refs)-{'unrelated'})==6 and refs['unrelated']['pid'] not in ids,'FIXTURE_ROSTER')
    expected=['contract']+[v for p in order for v in (f'stop_intent:{p}',f'stopped:{p}')]+['quiescent']
    expected+=[v for p in reversed(order) for v in (f'resume_intent:{p}',f'resumed:{p}')]+['released']
    clean=read(a.input/'clean/report.json');need(clean['markers']==expected,'MARKER_COVERAGE')
    normal=read(a.input/'normal-readback.json');exact_scope(normal['paused'],refs,ids);exact_scope(normal['held'],refs,ids)
    need(all(normal['paused'][n]['heartbeat']==normal['held'][n]['heartbeat'] for n in refs if n!='unrelated')
         and normal['held']['unrelated']['heartbeat']>normal['paused']['unrelated']['heartbeat'],'NORMAL_SCOPE')
    progressed(normal['held'],normal['after'],refs)
    stopped=set();summary=[]
    for index,marker in enumerate(expected,1):
        stage,_,pid=marker.partition(':')
        if stage=='stopped':stopped.add(int(pid))
        elif stage=='resumed':stopped.remove(int(pid))
        row=read(a.input/f'readback-{index:03d}.json');root=a.input/f'crash-{index:03d}'
        contract=read(root/'contract.json',row['contract'])
        # Running/sleeping state changes between launch and capture. Identity and
        # child closure must match; stopped state is recounted from every boundary.
        fields=('pid','starttime','parent','command','children')
        need(contract['order']==order and len(contract['owners'])==len(owners)
             and all(all(actual[k]==expected[k] for k in fields)
                     for actual,expected in zip(contract['owners'],owners,strict=True)),'CONTRACT_ROSTER')
        need(row['boundary']==index and row['marker']==marker and row['exit_code']==9,'CRASH_BOUNDARY')
        events=[read(p) for p in sorted(root.glob('event-*.json'))]
        actual=['contract']+[e['stage']+(':'+str(e['pid']) if 'pid' in e else '') for e in events]
        need(actual==expected[:index],'DURABLE_PREFIX')
        exact_scope(row['stopped'],refs,stopped);progressed(row['stopped'],row['after'],refs)
        recovery=read(a.input/f'recovery-{index:03d}/report.json',row['recovery'])
        need(recovery['complete'] and recovery['contract']==row['contract'] and recovery['terminated']==0
             and recovery['successors']==0 and {r['pid'] for r in recovery['owners']}==ids,'RECOVERY_SCOPE')
        need(all(r['observed']['state'] in ('R','S','D') for r in recovery['owners']),'RECOVERY_RUNNING')
        for name,ref in refs.items():
            need((a.input/'fixtures'/f'{name}-heartbeat').stat().st_size>=row['after'][name]['heartbeat'],'PHYSICAL_HEARTBEAT')
        summary.append(dict(boundary=index,stopped=sorted(stopped),all_resumed=True))
    terminal=read(a.input/'terminal-owner-recovery/report.json');producer=refs['producer']['pid']
    need(next(r for r in terminal['owners'] if r['pid']==producer)==dict(pid=producer,already_terminal=True),'TERMINAL_CONTROL')
    trace=read(a.input/'terminal-owner-readback.json');remaining={n:r for n,r in refs.items() if n!='producer'}
    progressed(trace['before'],trace['after'],remaining)
    for ref in refs.values():
        current=kernel(ref)
        need(current is None or current['starttime']!=ref['starttime'] or current['state']=='Z','FIXTURE_LEAK')
    for name in refs:
        raw=(a.input/'fixtures'/f'{name}-heartbeat').read_bytes()
        need(raw and raw==b'.'*len(raw),'HEARTBEAT_BYTES')
    census=read(a.census/'report.json',a.census_pin);check(a.census/'report.json',a.census_pin)
    need(census['complete'] and len(census['owners'])==6 and len(census['completed_original'])==13
         and census['signals_sent']==census['optimizer_updates']==0 and not census['gpu_used']
         and not census['retirement_authorized'] and not census['gpu_qualified']
         and not census['replacement_stage_chain_wired'] and not census['payloads_reauthenticated'],'CENSUS_SCOPE')
    for n,h in census['files'].items():check(a.census/n,h)
    for path,ref in census['sources'].items():
        need(Path(path).stat().st_size==ref['bytes'],'SOURCE_SIZE');check(Path(path),ref['pin'])
    live={}
    for name,ref in census['owners'].items():
        row=kernel(ref);need(row is not None,'LIVE_OWNER_MISSING')
        need(all(row[k]==ref[k] for k in ('pid','starttime','parent','children','command')),'LIVE_OWNER_CHANGED')
        need(row['state'] in ('R','S','D'),'LIVE_OWNER_STOPPED');live[name]=row
    need(live['driver']['children']==[live['grouped']['pid']] and live['grouped']['children']==[live['producer']['pid']],
         'LIVE_CLOSURE')
    need('UNDECLARED_CHILD' in census['controls']['old_five_owner_snapshot'],'FIVE_OWNER_REFUSAL')
    controls={}
    def reject(name,fn,code):
        try:fn()
        except ValueError as error:
            need(code in str(error),'WRONG_CONTROL_REFUSAL');controls[name]=str(error)
        else:raise ValueError('HANDOFF_READBACK_CONTROL_ACCEPTED:'+name)
    bad=json.loads(json.dumps(normal['paused']));bad['producer']['state']='S'
    reject('wrong_stopped_state',lambda:exact_scope(bad,refs,ids),'STOPPED_SET')
    bad=json.loads(json.dumps(normal['paused']));bad['producer']['starttime']='0'
    reject('wrong_identity',lambda:exact_scope(bad,refs,ids),'IDENTITY')
    reject('no_heartbeat_progress',lambda:progressed(normal['after'],normal['after'],refs),'HEARTBEAT')
    need(sha(a.output/'source.py')==sha(__file__),'SOURCE_CHANGED')
    result=dict(complete=True,input_pin=a.pin,census_pin=a.census_pin,kill_boundaries=summary,
        normal_pause_recounted=True,all_same_identities_resumed=True,terminal_child_recounted=True,
        live_owners=live,physical_files=files,seconds=time.monotonic()-began,gpu_used=False,
        existing_jobs_signalled=False,retirement_authorized=False,controls=controls,acceptance='author-checked')
    path=a.output/'report.json'
    with path.open('xb') as f:
        f.write(json.dumps(result,sort_keys=True,separators=(',',':')).encode());f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(pin=sha(path),boundaries=len(summary),live_owners=len(live))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input','census','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pin',required=True);p.add_argument('--census-pin',required=True);a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():
            with (a.output/'failure.json').open('x') as f:json.dump(dict(error=repr(error)),f);f.flush();os.fsync(f.fileno())
        raise
