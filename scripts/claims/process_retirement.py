"""Durable process retirement, separate from reversible quiescence.

The caller owns scientific validation of the pinned readiness artifacts. This
module authenticates their bytes, the complete process closure and the decision.
It neither certifies training completion nor starts a successor or GPU job.
Before decision.json exists, recovery resumes; afterwards recovery retires.
"""
import fcntl
import json
import os
from pathlib import Path
import select
import signal

from claims import process_handoff as base


def need(ok, code):
    if not ok: raise ValueError('PROCESS_RETIREMENT_'+code)


def sources():
    return dict(retirement=base.sha(__file__),quiescence=base.sha(base.__file__))


def receipts(value):
    need(type(value) is dict and set(value)=={'boundary','readback'},'RECEIPT_SCOPE')
    for row in value.values():
        need(set(row)=={'path','pin'} and Path(row['path']).is_absolute(),'RECEIPT_PATH')
        need(base.sha(row['path'])==row['pin'],'RECEIPT_CHANGED')


def decision(root, contract_pin):
    path=Path(root)/'decision.json'
    if not path.exists(): return None
    value=json.loads(path.read_bytes())
    need(value['schema']=='process-retirement-decision-v1' and value['contract']==contract_pin
         and value['action']=='retire' and not value['launch_successor'],'DECISION_SCOPE')
    receipts(value['receipts']);return value


def open_all(rows):
    fds={}
    try:
        for row in rows:fds[row['pid']]=base.open_owner(row)
        return fds
    except BaseException:
        close_all(fds);raise


def close_all(fds):
    for fd in fds.values():
        if fd is not None:os.close(fd)
    fds.clear()


class Events:
    def __init__(self, root, hook=lambda marker:None):
        self.root=Path(root);self.hook=hook;self.index=0

    def event(self, stage, **values):
        self.index+=1
        base.write(self.root/f'event-{self.index:04d}.json',dict(stage=stage,**values))
        self.hook(stage)


def act(rows, fds, retiring, events):
    """Idempotent under process exit; every live handle was opened before action."""
    # Stop controllers first, but retire children first while controllers remain
    # stopped. Killing a parent first can orphan a stopped process group, which
    # permits the kernel to send SIGHUP/SIGCONT to its remaining descendants.
    sequence=list(reversed(rows))
    for row in sequence:
        pid=row['pid'];fd=fds[pid];current=base.process(pid)
        if current is None:
            events.event('already_terminal',pid=pid);continue
        need(current['starttime']==row['starttime'] and current['command']==row['command']
             and fd is not None,'IDENTITY')
        if retiring:
            need(current['state']=='T','RETIRE_OWNER_RUNNING')
            events.event('retire_intent',pid=pid)
            signal.pidfd_send_signal(fd,signal.SIGKILL)
            poll=select.poll();poll.register(fd,select.POLLIN)
            need(bool(poll.poll(5000)),'EXIT_NOT_OBSERVED')
            need(base.process(pid) is None,'EXIT_NOT_OBSERVED')
            events.event('retired',pid=pid)
        else:
            events.event('resume_intent',pid=pid)
            signal.pidfd_send_signal(fd,signal.SIGCONT)
            events.event('resumed',pid=pid,observed=base.wait_state(row,False))
    events.event('all_retired' if retiring else 'released')


class Transition(Events):
    def __init__(self, root, expected, order, lock, hook=lambda marker:None):
        super().__init__(root,hook);self.rows=[];self.fds={};self.lock=None
        self.root.mkdir(exist_ok=False)
        try:
            self.lock=Path(lock).open('a+');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.rows=base.capture(expected,order)
            wanted={r['pid']:base.binding(r) for r in expected}
            need(all(base.binding(r)==wanted[r['pid']] for r in self.rows),'EXPECTED_CLOSURE_CHANGED')
            need(all(r['state'] in ('R','S','D') for r in self.rows),'ALREADY_STOPPED')
            self.fds=open_all(self.rows);need(all(v is not None for v in self.fds.values()),'OWNER_EXITED')
            need([base.binding(r) for r in base.capture(self.rows,order)]==
                 [base.binding(r) for r in self.rows],'CLOSURE_CHANGED')
            me=base.process(os.getpid())
            self.contract=dict(schema='process-retirement-v1',sources=sources(),
                boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                owners=self.rows,order=order,lock=str(Path(lock).resolve()),
                coordinator=dict(pid=me['pid'],starttime=me['starttime']),
                before_decision='resume',after_decision='retire',may_launch_successor=False)
            self.pin=base.write(self.root/'contract.json',self.contract);self.hook('contract')
        except BaseException:self.close();raise

    def pause(self):
        for row in self.rows:
            pid=row['pid'];current=base.process(pid)
            need(current is not None and base.binding(current)==base.binding(row),'CLOSURE_CHANGED')
            self.event('stop_intent',pid=pid)
            signal.pidfd_send_signal(self.fds[pid],signal.SIGSTOP)
            observed=base.wait_state(row,True)
            need(base.binding(observed)==base.binding(row),'CLOSURE_CHANGED')
            self.event('stopped',pid=pid,observed=observed)
        self.quiescent();self.event('quiescent')

    def quiescent(self):
        rows=base.capture(self.rows,self.contract['order'])
        need([base.binding(r) for r in rows]==[base.binding(r) for r in self.rows]
             and all(r['state']=='T' for r in rows),'NOT_QUIESCENT')

    def commit(self, ready):
        self.quiescent();receipts(ready)
        need(self.contract['sources']==sources(),'SOURCE_CHANGED')
        value=dict(schema='process-retirement-decision-v1',contract=self.pin,
                   action='retire',receipts=ready,launch_successor=False)
        # Publish atomically: a torn pending file never becomes a retirement order.
        base.write(self.root/'decision.pending.json',value);self.hook('decision_prepared')
        os.link(self.root/'decision.pending.json',self.root/'decision.json');self.hook('decision_linked')
        fd=os.open(self.root,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        self.hook('decision_committed')
        need(decision(self.root,self.pin)==value,'DECISION_READBACK')
        return base.sha(self.root/'decision.json')

    def resolve(self):
        retiring=decision(self.root,self.pin) is not None
        act(self.rows,self.fds,retiring,self)
        if retiring:need(all(base.process(r['pid']) is None for r in self.rows),'TERMINAL_READBACK')
        return dict(complete=True,contract=self.pin,action='retire' if retiring else 'resume',
                    owners=self.rows,all_terminal=retiring,successors_started=0)

    def close(self):
        close_all(self.fds)
        if self.lock is not None:self.lock.close();self.lock=None


def recover(root, pin, output, hook=lambda marker:None):
    root,output=Path(root),Path(output)
    need(base.sha(root/'contract.json')==pin,'CONTRACT_PIN')
    value=json.loads((root/'contract.json').read_bytes())
    need(value['schema']=='process-retirement-v1' and value['sources']==sources(),'SOURCE')
    need(value['before_decision']=='resume' and value['after_decision']=='retire'
         and not value['may_launch_successor'],'CONTRACT_SCOPE')
    need(value['boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'BOOT')
    need(base.process(value['coordinator']['pid']) is None,'COORDINATOR_LIVE_OR_REUSED')
    rows=value['owners']
    need(value['order']==[r['pid'] for r in rows] and len(set(value['order']))==len(rows)
         and os.getpid() not in value['order'],'ROSTER')
    with Path(value['lock']).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        retiring=decision(root,pin) is not None
        fds=open_all(rows)
        try:
            # Validate every remaining owner before any signal, including after a crash.
            if retiring:
                ids=set(value['order'])
                for row in rows:
                    current=base.process(row['pid'])
                    need(current is None or (current['state']=='T' and set(current['children'])<=ids),
                         'RETIRE_OWNER_RUNNING_OR_UNKNOWN_CHILD')
            output.mkdir(exist_ok=False);events=Events(output,hook)
            events.event('recovery_started',contract=pin,action='retire' if retiring else 'resume')
            act(rows,fds,retiring,events)
            if retiring:need(all(base.process(r['pid']) is None for r in rows),'TERMINAL_READBACK')
            report=dict(complete=True,contract=pin,action='retire' if retiring else 'resume',
                        owners=rows,all_terminal=retiring,successors_started=0,sources=sources())
            base.write(output/'report.json',report);hook('recovery_complete');return report
        finally:close_all(fds)
