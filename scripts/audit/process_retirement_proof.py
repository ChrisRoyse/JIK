"""Real nine-owner retirement and recovery kill sweep; no MNLM job is signalled."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import process_handoff_proof as fixture_code
from claims import process_handoff as base
from claims import process_retirement as retirement

ORDER=('evaluation','full','recovery','source_waiter','epoch_waiter','tile_waiter','driver','grouped','producer')


class Fixtures:
    def __init__(self, root):
        self.root=Path(root);self.root.mkdir();self.children=[];self.refs={}
        try:
            for name in (*[n for n in ORDER if n not in ('grouped','producer')],'unrelated'):
                self.children.append(subprocess.Popen([sys.executable,'-I',fixture_code.__file__,
                    '--mode','fixture','--output',str(self.root),'--name',name],
                    stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL))
            fixture_code.wait_until(lambda:all((self.root/f'{n}-launch.json').exists()
                and (self.root/f'{n}-heartbeat').exists() for n in (*ORDER,'unrelated')))
            self.refs={n:json.loads((self.root/f'{n}-launch.json').read_bytes()) for n in (*ORDER,'unrelated')}
        except BaseException:self.close();raise

    def snapshot(self):
        rows={}
        for name,ref in self.refs.items():
            path=Path('/proc')/str(ref['pid'])/'stat'
            try:
                stat=path.read_text().rsplit(')',1)[1].split()
                retirement.need(stat[19]==ref['starttime'],'PROOF_KERNEL_IDENTITY')
                state=stat[0]
            except FileNotFoundError:state='absent'
            rows[name]=dict(pid=ref['pid'],starttime=ref['starttime'],state=state,
                           heartbeat=(self.root/f'{name}-heartbeat').stat().st_size)
        return rows

    def request(self, root):
        return dict(owners=[self.refs[n] for n in ORDER],order=[self.refs[n]['pid'] for n in ORDER],
                    lock=str(root/'retirement.lock'),fixtures=str(self.root))

    def close(self):
        for ref in self.refs.values():
            fd=base.open_owner(ref)
            if fd is not None:
                try:signal.pidfd_send_signal(fd,signal.SIGCONT)
                finally:os.close(fd)
        for child in self.children:
            if child.poll() is None:child.terminate()
        for child in self.children:child.wait()
        retirement.need(all(base.process(r['pid']) is None for r in self.refs.values()),'PROOF_FIXTURE_LEAK')


def coordinator(a):
    request=json.loads(a.request.read_bytes());markers=[]
    def hook(marker):
        markers.append(marker)
        if a.crash_at==len(markers):os._exit(9)
    if a.mode=='recover':
        retirement.recover(a.transition,base.sha(a.transition/'contract.json'),a.output,hook)
        base.write(a.output/'markers.json',markers);return
    turn=retirement.Transition(a.output,request['owners'],request['order'],request['lock'],hook)
    try:
        turn.pause()
        if a.interactive:
            print('quiescent',flush=True)
            retirement.need(sys.stdin.readline().strip()=='commit','PROOF_COMMIT_COMMAND')
        # These are process-state readiness artifacts, not synthetic training checkpoints.
        rows=[base.process(r['pid']) for r in request['owners']]
        retirement.need(all(r is not None and r['state']=='T' for r in rows),'PROOF_NOT_STOPPED')
        boundary=a.output/'process-boundary.json'
        boundary_pin=base.write(boundary,dict(kind='process-fixture-boundary',owners=rows));hook('boundary_written')
        reread=json.loads(boundary.read_bytes())
        retirement.need(reread['owners']==rows and base.sha(boundary)==boundary_pin,'PROOF_BOUNDARY_READBACK')
        readback=a.output/'process-readback.json'
        back_pin=base.write(readback,dict(kind='process-fixture-readback',boundary=boundary_pin,
            live=[base.process(r['pid']) for r in request['owners']],scientific_checkpoint=False));hook('readback_written')
        turn.commit(dict(boundary=dict(path=str(boundary),pin=boundary_pin),
                         readback=dict(path=str(readback),pin=back_pin)))
        report=turn.resolve()
        base.write(a.output/'report.json',dict(report,markers=[*markers,'complete']));hook('complete')
    except BaseException:
        turn.resolve();raise
    finally:turn.close()


def launch(a, request, out, *, crash=None, transition=None):
    argv=[sys.executable,'-I',__file__,'--mode','recover' if transition else 'coordinator',
          '--request',str(request),'--output',str(out)]
    if crash is not None:argv+=['--crash-at',str(crash)]
    if transition is not None:argv+=['--transition',str(transition)]
    with out.with_suffix('.log').open('xb') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        result=child.wait();log.flush();os.fsync(log.fileno())
    retirement.need(result==(9 if crash is not None else 0),'PROOF_CHILD_EXIT');return result


def outcome(f, prior, retiring):
    def done():
        now=f.snapshot()
        if now['unrelated']['state'] not in ('R','S','D') or now['unrelated']['heartbeat']<=prior['unrelated']['heartbeat']:
            return False
        if retiring:return all(now[n]['state'] in ('absent','Z') for n in ORDER)
        return all(now[n]['state'] in ('R','S','D') and now[n]['heartbeat']>prior[n]['heartbeat'] for n in ORDER)
    fixture_code.wait_until(done);return f.snapshot()


def run(a):
    began=time.perf_counter();a.output.mkdir(exist_ok=False);results=[];controls={}
    source_files={Path(__file__):base.sha(__file__),Path(retirement.__file__):base.sha(retirement.__file__),
                  Path(base.__file__):base.sha(base.__file__),Path(fixture_code.__file__):base.sha(fixture_code.__file__)}
    for path in source_files:
        target=a.output/'source'/path.name;target.parent.mkdir(exist_ok=True);target.write_bytes(path.read_bytes())
    def reject(name, fn, code):
        try:fn()
        except (ValueError,BlockingIOError) as error:
            retirement.need(code in str(error),'PROOF_WRONG_REFUSAL');controls[name]=str(error)
        else:raise ValueError('PROCESS_RETIREMENT_PROOF_ACCEPTED:'+name)
    # Normal before/pause/after boundaries and real lock/identity failures.
    root=a.output/'normal';root.mkdir();f=Fixtures(root/'fixtures');child=None
    try:
        req=f.request(root);request=root/'request.json';base.write(request,req)
        owners=req['owners'];order=req['order'];before=f.snapshot()
        missing=[r for r in owners if r['pid']!=f.refs['grouped']['pid']]
        reject('omitted_controller',lambda:base.capture(missing,[r['pid'] for r in missing]),'UNDECLARED_CHILD')
        bad=deepcopy(owners);bad[-1]['starttime']='0'
        reject('wrong_identity',lambda:base.capture(bad,order),'IDENTITY')
        reject('child_before_parent',lambda:base.capture(owners,list(reversed(order))),'PARENT_ORDER')
        with (root/'coordinator.log').open('xb') as log:
            dest=root/'transition'
            child=subprocess.Popen([sys.executable,'-I',__file__,'--mode','coordinator','--request',str(request),
                '--output',str(dest),'--interactive'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True)
            retirement.need(child.stdout.readline().strip()=='quiescent','PROOF_READY')
            paused=f.snapshot();retirement.need(all(paused[n]['state']=='T' for n in ORDER),'PROOF_PAUSE')
            time.sleep(.12);held=f.snapshot()
            retirement.need(all(held[n]['heartbeat']==paused[n]['heartbeat'] for n in ORDER)
                and held['unrelated']['heartbeat']>paused['unrelated']['heartbeat'],'PROOF_SCOPE')
            reject('duplicate_transition',lambda:retirement.Transition(root/'duplicate',owners,order,req['lock']),
                   'Resource temporarily unavailable')
            pin=base.sha(dest/'contract.json')
            reject('live_coordinator',lambda:retirement.recover(dest,pin,root/'bad-live'),'COORDINATOR_LIVE_OR_REUSED')
            child.stdin.write('commit\n');child.stdin.flush()
            retirement.need(child.wait()==0,'PROOF_CHILD_EXIT')
            child.stdin.close();child.stdout.close();child=None;log.flush();os.fsync(log.fileno())
        after=outcome(f,held,True)
        normal=json.loads((dest/'report.json').read_bytes());markers=normal['markers']
        base.write(root/'readback.json',dict(before=before,paused=paused,held=held,after=after))
        reject('wrong_contract_pin',lambda:retirement.recover(dest,'0'*64,root/'bad-pin'),'CONTRACT_PIN')
        changed=root/'changed-source';changed.mkdir();v=json.loads((dest/'contract.json').read_bytes())
        v['sources']['retirement']='0'*64;h=base.write(changed/'contract.json',v)
        reject('changed_source',lambda:retirement.recover(changed,h,root/'bad-source'),'SOURCE')
        badready=deepcopy(json.loads((dest/'decision.json').read_bytes())['receipts']);badready['readback']['pin']='0'*64
        reject('changed_readback',lambda:retirement.receipts(badready),'RECEIPT_CHANGED')
        retirement.recover(dest,pin,root/'idempotent')
        controls['terminal_recovery']='all terminal; no successor launched'
    finally:
        if child is not None and child.poll() is None:child.kill();child.wait()
        f.close()
    root=a.output/'negative-recovery';root.mkdir();f=Fixtures(root/'fixtures')
    try:
        request=root/'request.json';req=f.request(root);base.write(request,req)
        bad=deepcopy(req['owners']);bad[-1]['parent']=1
        reject('changed_parent_binding',lambda:retirement.Transition(root/'bad-parent',bad,req['order'],req['lock']),
               'EXPECTED_CLOSURE_CHANGED')
        dest=root/'transition';launch(a,request,dest,crash=markers.index('quiescent')+1)
        stopped=f.snapshot();contract=json.loads((dest/'contract.json').read_bytes())
        badroot=root/'wrong-owner';badroot.mkdir();bad=deepcopy(contract);bad['owners'][-1]['starttime']='0'
        pin=base.write(badroot/'contract.json',bad)
        reject('recovery_wrong_identity',lambda:retirement.recover(badroot,pin,root/'bad-identity-readback'),'IDENTITY')
        badroot=root/'wrong-boot';badroot.mkdir();bad=deepcopy(contract);bad['boot']='not-this-boot'
        pin=base.write(badroot/'contract.json',bad)
        reject('recovery_wrong_boot',lambda:retirement.recover(badroot,pin,root/'bad-boot-readback'),'BOOT')
        badroot=root/'wrong-readiness';badroot.mkdir();pin=base.write(badroot/'contract.json',contract)
        ready=deepcopy(json.loads((a.output/'normal/transition/decision.json').read_bytes()))
        ready['contract']=pin;ready['receipts']['readback']['pin']='0'*64
        base.write(badroot/'decision.json',ready)
        reject('recovery_changed_readiness',lambda:retirement.recover(badroot,pin,root/'bad-ready-readback'),'RECEIPT_CHANGED')
        held=f.snapshot()
        retirement.need(all(held[n]['state']=='T' and held[n]['heartbeat']==stopped[n]['heartbeat'] for n in ORDER),
                         'PROOF_REFUSAL_SIGNALLED')
        with (dest/'decision.pending.json').open('xb') as stream:
            stream.write(b'{"torn":');stream.flush();os.fsync(stream.fileno())
        recovery=retirement.recover(dest,base.sha(dest/'contract.json'),root/'torn-pending-recovery')
        retirement.need(recovery['action']=='resume','PROOF_TORN_PENDING_DECISION')
        after=outcome(f,held,False);controls['torn_pending']='incomplete unpublished decision resumes all owners'
        base.write(root/'readback.json',dict(before=stopped,held=held,after=after))
    finally:f.close()
    # Kill the coordinator at every write boundary discovered in the clean real path.
    for index,marker in enumerate(markers,1):
        root=a.output/f'crash-{index:03d}';root.mkdir();f=Fixtures(root/'fixtures')
        try:
            request=root/'request.json';base.write(request,f.request(root));before=f.snapshot();dest=root/'transition'
            launch(a,request,dest,crash=index);stopped=f.snapshot();committed=(dest/'decision.json').exists()
            recovered=retirement.recover(dest,base.sha(dest/'contract.json'),root/'recovery')
            after=outcome(f,stopped,committed)
            retirement.need(recovered['action']==('retire' if committed else 'resume'),'PROOF_DECISION')
            row=dict(boundary=index,marker=marker,before=before,stopped=stopped,after=after,
                     committed=committed,recovery=base.sha(root/'recovery/report.json'))
            base.write(root/'readback.json',row);results.append(dict(path=str(root),boundary=index,marker=marker,committed=committed))
            print(json.dumps(dict(sweep='coordinator',boundary=index,total=len(markers),marker=marker)),flush=True)
        finally:f.close()
    # Discover and then crash every recovery boundary on both sides of the decision.
    recovery_results=[]
    for mode,crash_marker in (('resume','quiescent'),('retire','decision_committed')):
        recovery_markers=None;index=0
        while recovery_markers is None or index<=len(recovery_markers):
            root=a.output/f'recovery-{mode}-{index:03d}';root.mkdir();f=Fixtures(root/'fixtures')
            try:
                request=root/'request.json';base.write(request,f.request(root));dest=root/'transition'
                launch(a,request,dest,crash=markers.index(crash_marker)+1);before=f.snapshot()
                launch(a,request,root/'first',transition=dest,crash=index if index else None)
                middle=f.snapshot()
                if index==0:recovery_markers=json.loads((root/'first/markers.json').read_bytes())
                else:retirement.recover(dest,base.sha(dest/'contract.json'),root/'second')
                after=outcome(f,middle,mode=='retire')
                base.write(root/'readback.json',dict(mode=mode,boundary=index,before=before,middle=middle,after=after))
                recovery_results.append(dict(path=str(root),mode=mode,boundary=index,
                    marker='clean' if index==0 else recovery_markers[index-1]))
                print(json.dumps(dict(sweep=mode,boundary=index,total=len(recovery_markers))),flush=True)
            finally:f.close()
            index+=1
    retirement.need(all(base.sha(p)==pin for p,pin in source_files.items()),'PROOF_SOURCE_CHANGED')
    retirement.need('torch' not in sys.modules,'PROOF_MODEL_IMPORTED')
    files={str(p.relative_to(a.output)):base.sha(p) for p in a.output.rglob('*') if p.is_file()}
    report=dict(complete=True,kind='process-retirement-proof-v1',owners=9,unrelated=1,
        coordinator_boundaries=len(results),recovery_boundaries=len(recovery_results)-2,
        results=results,recovery_results=recovery_results,controls=controls,files=files,
        sources={str(p):pin for p,pin in source_files.items()},seconds=time.perf_counter()-began,
        original_jobs_signalled=False,scientific_checkpoint_qualified=False,gpu_used=False,
        successors_started=0,acceptance='author-checked',number_status='measured')
    print(json.dumps(dict(pin=base.write(a.output/'report.json',report),boundaries=len(results),
                         recovery_boundaries=len(recovery_results)-2)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('proof','coordinator','recover'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--request',type=Path)
    p.add_argument('--transition',type=Path);p.add_argument('--crash-at',type=int);p.add_argument('--interactive',action='store_true')
    a=p.parse_args()
    try:run(a) if a.mode=='proof' else coordinator(a)
    except BaseException as error:
        if a.mode=='proof' and a.output.exists():base.write(a.output/'failure.json',dict(error=repr(error)))
        raise
