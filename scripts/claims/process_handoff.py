"""Identity-bound, reversible process quiescence; never terminates an owner.

The caller must bind the full controller/producer/waiter roster before action.
Recovery resumes only those original identities. This is not a GPU lease or a
training-completion certificate, and has no automatic successor launch.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time


def need(ok, code):
    if not ok: raise ValueError('PROCESS_HANDOFF_'+code)


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def write(path, value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    with Path(path).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    fd=os.open(Path(path).parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return hashlib.sha256(raw).hexdigest()


def process(pid):
    need(type(pid) is int and pid>1,'PID')
    root=Path('/proc')/str(pid)
    try:
        stat=(root/'stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z':return None
        tasks=list((root/'task').iterdir());children=set()
        for task in tasks:
            try:children.update(map(int,(task/'children').read_text().split()))
            except FileNotFoundError:pass  # A finished thread owns no live child here.
        return dict(pid=pid,starttime=stat[19],parent=int(stat[1]),state=stat[0],
                    command=hashlib.sha256((root/'cmdline').read_bytes()).hexdigest(),
                    children=sorted(children),tasks=len(tasks))
    except (FileNotFoundError,ProcessLookupError):return None


def binding(row):
    return {n:row[n] for n in ('pid','starttime','parent','command','children')}


def capture(expected, order):
    need(expected and len({r['pid'] for r in expected})==len(expected),'ROSTER')
    ids={r['pid'] for r in expected}
    need(os.getpid() not in ids and len(order)==len(ids) and set(order)==ids,'ORDER')
    positions={p:i for i,p in enumerate(order)};rows={}
    for ref in expected:
        row=process(ref['pid']);need(row is not None and row['starttime']==ref['starttime'],'IDENTITY')
        need(set(row['children'])<=ids,'UNDECLARED_CHILD')
        need(all(positions[row['pid']]<positions[c] for c in row['children']),'PARENT_ORDER')
        if 'command' in ref:need(row['command']==ref['command'],'COMMAND')
        rows[row['pid']]=row
    for pid,row in rows.items():
        if row['parent'] in ids:need(pid in rows[row['parent']]['children'],'PARENT_CLOSURE')
    return [rows[p] for p in order]


def open_owner(ref):
    try:fd=os.pidfd_open(ref['pid'])
    except ProcessLookupError:
        need(process(ref['pid']) is None,'IDENTITY');return None
    try:
        current=process(ref['pid'])
        if current is None:os.close(fd);return None
        need(current['starttime']==ref['starttime']
             and current['command']==ref['command'],'IDENTITY')
        return fd
    except BaseException:os.close(fd);raise


def wait_state(ref, stopped):
    deadline=time.monotonic()+5
    while True:
        row=process(ref['pid'])
        need(row is not None and row['starttime']==ref['starttime'],'OWNER_EXITED')
        if (row['state']=='T')==stopped:return row
        need(time.monotonic()<deadline,'STATE_NOT_OBSERVED')
        time.sleep(.01)


class Transition:
    def __init__(self, root, expected, order, lock, hook=lambda marker:None):
        self.root=Path(root);self.fds={};self.lock=None;self.rows=[];self.hook=hook;self.index=0
        self.root.mkdir(exist_ok=False)
        try:
            self.lock=Path(lock).open('a+');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.rows=capture(expected,order)
            need(all(r['state'] in ('R','S','D') for r in self.rows),'ALREADY_STOPPED')
            for row in self.rows:
                fd=open_owner(row);need(fd is not None,'OWNER_EXITED');self.fds[row['pid']]=fd
            need([binding(r) for r in capture(self.rows,order)]==[binding(r) for r in self.rows],'CLOSURE_CHANGED')
            me=process(os.getpid())
            self.contract=dict(schema='reversible-process-quiescence-v1',source=sha(__file__),
                boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),owners=self.rows,order=order,
                coordinator=dict(pid=me['pid'],starttime=me['starttime']),lock=str(Path(lock).resolve()),
                may_terminate=False,may_launch_successor=False)
            self.pin=write(self.root/'contract.json',self.contract);self.hook('contract')
        except BaseException:self.close();raise

    def event(self, stage, **values):
        self.index+=1;write(self.root/f'event-{self.index:04d}.json',dict(stage=stage,**values))
        self.hook(stage+(':'+str(values['pid']) if 'pid' in values else ''))

    def pause(self):
        try:
            for row in self.rows:
                pid=row['pid'];current=process(pid)
                need(current is not None and binding(current)==binding(row),'CLOSURE_CHANGED')
                self.event('stop_intent',pid=pid)
                signal.pidfd_send_signal(self.fds[pid],signal.SIGSTOP)
                observed=wait_state(row,True)
                need(binding(observed)==binding(row),'CLOSURE_CHANGED')
                self.event('stopped',pid=pid,observed=observed)
            self.event('quiescent',owners=[process(r['pid']) for r in self.rows])
        except BaseException:
            self.resume();raise

    def resume(self):
        # Producers resume before their controllers. Nothing is killed/replaced.
        for row in reversed(self.rows):
            pid=row['pid'];current=process(pid)
            if current is None:self.event('already_terminal',pid=pid);continue
            need(current['starttime']==row['starttime'] and current['command']==row['command'],'IDENTITY')
            self.event('resume_intent',pid=pid)
            signal.pidfd_send_signal(self.fds[pid],signal.SIGCONT)
            self.event('resumed',pid=pid,observed=wait_state(row,False))
        self.event('released')

    def close(self):
        for fd in self.fds.values():os.close(fd)
        self.fds.clear()
        if self.lock is not None:self.lock.close();self.lock=None


def recover(root, pin, output):
    root,output=Path(root),Path(output)
    need(sha(root/'contract.json')==pin,'CONTRACT_PIN')
    value=json.loads((root/'contract.json').read_bytes())
    need(value['schema']=='reversible-process-quiescence-v1' and value['source']==sha(__file__),'SOURCE')
    need(not value['may_terminate'] and not value['may_launch_successor'],'SCOPE')
    need(value['boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'BOOT')
    owner=process(value['coordinator']['pid'])
    need(owner is None,'COORDINATOR_LIVE_OR_REUSED')
    need(set(value['order'])=={r['pid'] for r in value['owners']} and len(value['order'])==len(value['owners']),
         'ROSTER')
    output.mkdir(exist_ok=False);fds={};rows=value['owners'];events=[]
    with Path(value['lock']).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            # Validate/open the whole roster before sending a signal to any owner.
            for row in rows:fds[row['pid']]=open_owner(row)
            for row in reversed(rows):
                pid=row['pid'];fd=fds[pid]
                if fd is None:events.append(dict(pid=pid,already_terminal=True));continue
                write(output/f'intent-{pid}.json',dict(pid=pid,starttime=row['starttime'],signal='SIGCONT'))
                signal.pidfd_send_signal(fd,signal.SIGCONT)
                events.append(dict(pid=pid,observed=wait_state(row,False)))
            report=dict(complete=True,contract=pin,owners=events,terminated=0,successors=0,
                        source=sha(__file__),acceptance='author-checked')
            write(output/'report.json',report);return report
        finally:
            for fd in fds.values():
                if fd is not None:os.close(fd)
