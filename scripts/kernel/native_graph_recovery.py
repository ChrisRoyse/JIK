"""Resume accounted graph shards without modifying any incomplete attempt."""
import json
from pathlib import Path

from audit.native_graph_accounted_readback import read_accounted
from audit.native_graph_coverage import replay_samples
from claims.process_conditions import write_committed
from kernel.native_graph_accounted_store import publish
from text.constellation_corpus import digest, file_sha, need


def stamp(path):
    s=Path(path).stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def attempt_files(root):
    """Only one explicitly owned shard directory; never scan a data root."""
    rows={}
    for p in sorted(Path(root).iterdir()):
        need(p.is_file() and not p.is_symlink(),'GRAPH_RECOVERY_ATTEMPT_ENTRY')
        rows[p.name]=dict(bytes=p.stat().st_size,sha256=file_sha(p))
    return rows


def scope_plan(imports, missing, population):
    """Keep original references typed and prove missing IDs do not overlap them."""
    cursor=0
    for r in imports:
        need(r['start']==cursor and r['stop']>cursor and r['queries']==r['stop']-r['start'],
             'GRAPH_RECOVERY_IMPORT_GAP')
        need(r['stop']<=population,'GRAPH_RECOVERY_IMPORT_RANGE')
        cursor=r['stop']
    need(isinstance(missing,list) and missing==sorted(set(missing))
         and all(type(u) is int and cursor<=u<population for u in missing),
         'GRAPH_RECOVERY_MISSING_SCOPE')
    return dict(imported_queries=cursor,missing_queries=len(missing),
                missing_uid_hash=digest(json.dumps(missing,separators=(',',':')).encode()),
                covered_queries=cursor+len(missing),population=population,
                full_graph_complete=cursor+len(missing)==population)


def resume_shard(root,ids,lane,plan_pin,*,threads,chosen,cold=False,interrupt=None):
    """A seal is authoritative only after readback. Torn attempts remain immutable.

    interrupt is an audit fault hook at real durable boundaries, never a provider
    stub. A failed committed attempt refuses; only uncommitted attempts are skipped.
    The caller must hold its publication/qualification lease for this whole call.
    """
    root=Path(root)
    if not cold:root.mkdir(parents=True,exist_ok=True)
    need(root.is_dir(),'GRAPH_RECOVERY_TASK_MISSING')
    attempts=sorted(root.iterdir());sealed=[];incomplete=[]
    for number,p in enumerate(attempts):
        need(p.is_dir() and not p.is_symlink() and p.name==f'attempt-{number:06d}',
             'GRAPH_RECOVERY_ATTEMPT_ORDER')
        if (p/'COMMITTED').exists():sealed.append(p)
        else:incomplete.append(dict(path=str(p),files=attempt_files(p)))
    need(len(sealed)<=1,'GRAPH_RECOVERY_DUPLICATE_COMMIT')
    if sealed:
        path=sealed[0];pin=json.loads((path/'COMMITTED').read_bytes())['report_sha256']
        r,a=read_accounted(path,pin,plan_pin,lane,ids)
        numeric=replay_samples(a,lane,chosen)
        need(json.loads((path/'readback.json').read_bytes())==numeric,'GRAPH_RECOVERY_READBACK_CHANGED')
        reused=True
    else:
        need(not cold,'GRAPH_RECOVERY_COLD_UNSEALED')
        path=root/f'attempt-{len(attempts):06d}'
        pin=publish(path,ids,lane,plan_pin,threads=threads)
        if interrupt:interrupt('report',path)
        r,a=read_accounted(path,pin,plan_pin,lane,ids,committed=False)
        numeric=replay_samples(a,lane,chosen)
        write_committed(path/'readback.json',numeric)
        if interrupt:interrupt('readback',path)
        write_committed(path/'COMMITTED',dict(report_sha256=pin))
        if interrupt:interrupt('commit',path)
        reused=False
    return dict(path=str(path),pin=pin,plan_pin=plan_pin,queries=len(ids),
                native_windows=r['native_windows'],sample=numeric,
                incomplete_attempts=incomplete),a,reused
