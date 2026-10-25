"""Exercise reversible ownership with real processes and a durable-boundary kill sweep."""
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
from claims import process_handoff as handoff

ORDER=('evaluation','full','recovery','driver','grouped','producer')
TREE={'driver':'grouped','grouped':'producer'}


def wait_until(fn):
    deadline=time.monotonic()+10
    while not fn():
        handoff.need(time.monotonic()<deadline,'PROOF_OBSERVATION');time.sleep(.02)


def fixture(a):
    children=[]
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))
    try:
        if a.name in TREE:
            name=TREE[a.name]
            children.append(subprocess.Popen([sys.executable,'-I',__file__,'--mode','fixture',
                '--output',str(a.output),'--name',name],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL))
            wait_until(lambda:(a.output/f'{name}-launch.json').exists())
        me=handoff.process(os.getpid());handoff.write(a.output/f'{a.name}-launch.json',me)
        with (a.output/f'{a.name}-heartbeat').open('ab',buffering=0) as f:
            while True:f.write(b'.');os.fsync(f.fileno());time.sleep(.05)
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
            child.wait()


def coordinator(a):
    request=json.loads(a.request.read_bytes());markers=[]
    def hook(marker):
        markers.append(marker)
        if a.crash_at==len(markers):os._exit(9)
    turn=handoff.Transition(a.output,request['owners'],request['order'],request['lock'],hook)
    try:
        turn.pause();print('quiescent',flush=True)
        if a.crash_at is None:handoff.need(sys.stdin.readline().strip()=='resume','PROOF_RESUME_COMMAND')
        turn.resume();handoff.write(a.output/'report.json',dict(complete=True,contract=turn.pin,markers=markers))
    finally:turn.close()


def independent(root, refs):
    result={}
    for name,ref in refs.items():
        stat=(Path('/proc')/str(ref['pid'])/'stat').read_text().rsplit(')',1)[1].split()
        handoff.need(stat[19]==ref['starttime'],'PROOF_KERNEL_IDENTITY')
        result[name]=dict(pid=ref['pid'],starttime=stat[19],state=stat[0],
                          heartbeat=(root/f'{name}-heartbeat').stat().st_size)
    return result


def progressed(root, refs, before):
    def done():
        now=independent(root,refs)
        return all(v['state'] in ('R','S','D') and v['heartbeat']>before[n]['heartbeat'] for n,v in now.items())
    wait_until(done);return independent(root,refs)


def run(a):
    began=time.monotonic();a.output.mkdir(exist_ok=False);root=a.output/'fixtures';root.mkdir()
    handoff.write(a.output/'source-pins.json',dict(proof=handoff.sha(__file__),helper=handoff.sha(handoff.__file__)))
    (a.output/'source-proof.py').write_bytes(Path(__file__).read_bytes())
    (a.output/'source-helper.py').write_bytes(Path(handoff.__file__).read_bytes())
    children=[];refs={};coordinator_child=None;clean_log=None;results=[];controls={}
    try:
        for name in ('driver','recovery','full','evaluation','unrelated'):
            children.append(subprocess.Popen([sys.executable,'-I',__file__,'--mode','fixture','--output',str(root),
                '--name',name],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL))
        wait_until(lambda:all((root/f'{n}-launch.json').exists() and (root/f'{n}-heartbeat').exists()
                              for n in (*ORDER,'unrelated')))
        refs={n:json.loads((root/f'{n}-launch.json').read_bytes()) for n in (*ORDER,'unrelated')}
        owners=[refs[n] for n in ORDER];order=[r['pid'] for r in owners]
        request=dict(owners=owners,order=order,lock=str(a.output/'transition.lock'))
        request_path=a.output/'request.json';handoff.write(request_path,request)
        def reject(name,fn,code):
            try:fn()
            except (ValueError,BlockingIOError) as error:
                handoff.need(code in str(error),'PROOF_WRONG_REFUSAL');controls[name]=str(error)
            else:raise ValueError('PROCESS_HANDOFF_PROOF_ACCEPTED:'+name)
        missing=[r for r in owners if r['pid']!=refs['grouped']['pid']]
        reject('omitted_controller',lambda:handoff.capture(missing,[r['pid'] for r in missing]),'UNDECLARED_CHILD')
        wrong=deepcopy(owners);wrong[-1]['starttime']='0'
        reject('wrong_identity',lambda:handoff.capture(wrong,order),'IDENTITY')
        reject('child_before_parent',lambda:handoff.capture(owners,list(reversed(order))),'PARENT_ORDER')
        clean=a.output/'clean';before=independent(root,refs)
        clean_log=(a.output/'clean.log').open('xb')
        coordinator_child=subprocess.Popen([sys.executable,'-I',__file__,'--mode','coordinator','--output',str(clean),
            '--request',str(request_path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=clean_log,text=True)
        handoff.need(coordinator_child.stdout.readline().strip()=='quiescent','PROOF_CHILD_READY')
        paused=independent(root,refs)
        handoff.need(all(paused[n]['state']=='T' for n in ORDER),'PROOF_NOT_STOPPED')
        time.sleep(.12);again=independent(root,refs)
        handoff.need(all(again[n]['heartbeat']==paused[n]['heartbeat'] for n in ORDER)
                     and again['unrelated']['heartbeat']>paused['unrelated']['heartbeat'],'PROOF_PAUSE_SCOPE')
        reject('duplicate_transition',lambda:handoff.Transition(a.output/'duplicate',owners,order,request['lock']),
               'Resource temporarily unavailable')
        pin=handoff.sha(clean/'contract.json')
        reject('live_coordinator',lambda:handoff.recover(clean,pin,a.output/'bad-live'),'COORDINATOR_LIVE_OR_REUSED')
        coordinator_child.stdin.write('resume\n');coordinator_child.stdin.flush()
        handoff.need(coordinator_child.wait()==0,'PROOF_CHILD_FAILED');after=progressed(root,refs,again)
        for stream in (coordinator_child.stdin,coordinator_child.stdout):stream.close()
        clean_log.flush();os.fsync(clean_log.fileno());clean_log.close();clean_log=None
        coordinator_child=None;report=json.loads((clean/'report.json').read_bytes());markers=report['markers']
        handoff.write(a.output/'normal-readback.json',dict(before=before,paused=paused,held=again,after=after))
        reject('wrong_contract_pin',lambda:handoff.recover(clean,'0'*64,a.output/'bad-pin'),'CONTRACT_PIN')
        badroot=a.output/'bad-source';badroot.mkdir();bad=json.loads((clean/'contract.json').read_bytes());bad['source']='0'*64
        badpin=handoff.write(badroot/'contract.json',bad)
        reject('changed_source',lambda:handoff.recover(badroot,badpin,a.output/'bad-source-recovery'),'SOURCE')
        badroot=a.output/'bad-identity';badroot.mkdir();bad=json.loads((clean/'contract.json').read_bytes())
        bad['owners'][-1]['starttime']='0';badpin=handoff.write(badroot/'contract.json',bad)
        reject('reused_recovery_owner',lambda:handoff.recover(badroot,badpin,a.output/'bad-identity-recovery'),'IDENTITY')
        for index,marker in enumerate(markers,1):
            dest=a.output/f'crash-{index:03d}';before=independent(root,refs)
            with (a.output/f'crash-{index:03d}.log').open('xb') as log:
                child=subprocess.Popen([sys.executable,'-I',__file__,'--mode','coordinator','--output',str(dest),
                    '--request',str(request_path),'--crash-at',str(index)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                handoff.need(child.wait()==9,'PROOF_CRASH_EXIT');log.flush();os.fsync(log.fileno())
            stopped=independent(root,refs);pin=handoff.sha(dest/'contract.json')
            recovered=handoff.recover(dest,pin,a.output/f'recovery-{index:03d}')
            after=progressed(root,refs,stopped)
            row=dict(boundary=index,marker=marker,exit_code=9,contract=pin,before=before,stopped=stopped,after=after,
                     recovery=handoff.sha(a.output/f'recovery-{index:03d}/report.json'))
            handoff.write(a.output/f'readback-{index:03d}.json',row);results.append(row)
            print(json.dumps(dict(boundary=index,total=len(markers),marker=marker)),flush=True)
        handoff.need(len(results)==len(markers) and len(markers)==27,'PROOF_SWEEP_COVERAGE')
        # A real exited child can remain a zombie while its parent is stopped.
        dest=a.output/'terminal-owner'
        with (a.output/'terminal-owner.log').open('xb') as log:
            child=subprocess.Popen([sys.executable,'-I',__file__,'--mode','coordinator','--output',str(dest),
                '--request',str(request_path),'--crash-at','14'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            handoff.need(child.wait()==9,'PROOF_CRASH_EXIT')
        fd=handoff.open_owner(refs['producer']);handoff.need(fd is not None,'PROOF_TERMINAL_CONTROL')
        try:signal.pidfd_send_signal(fd,signal.SIGKILL)
        finally:os.close(fd)
        wait_until(lambda:handoff.process(refs['producer']['pid']) is None)
        before=independent(root,{n:r for n,r in refs.items() if n!='producer'})
        recovered=handoff.recover(dest,handoff.sha(dest/'contract.json'),a.output/'terminal-owner-recovery')
        handoff.need(next(r for r in recovered['owners'] if r['pid']==refs['producer']['pid'])==
                     dict(pid=refs['producer']['pid'],already_terminal=True),'PROOF_TERMINAL_READBACK')
        after=progressed(root,{n:r for n,r in refs.items() if n!='producer'},before)
        handoff.write(a.output/'terminal-owner-readback.json',dict(before=before,after=after,producer=refs['producer']['pid']))
        controls['exited_child_with_stopped_parent']='observed terminal; remaining owners resumed'
    finally:
        if coordinator_child is not None and coordinator_child.poll() is None:
            coordinator_child.kill();coordinator_child.wait()
        if clean_log is not None:clean_log.close()
        # Cleanup only the fixture identities/handles created by this proof.
        for ref in refs.values():
            try:
                fd=handoff.open_owner(ref)
                if fd is not None:
                    try:signal.pidfd_send_signal(fd,signal.SIGCONT)
                    finally:os.close(fd)
            except ProcessLookupError:pass
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:child.wait()
    handoff.need(all(handoff.process(r['pid']) is None for r in refs.values()),'PROOF_FIXTURE_LEAK')
    handoff.need(json.loads((a.output/'source-pins.json').read_bytes())==
                 dict(proof=handoff.sha(__file__),helper=handoff.sha(handoff.__file__)),'PROOF_SOURCE_CHANGED')
    files={str(p.relative_to(a.output)):handoff.sha(p) for p in a.output.rglob('*') if p.is_file()}
    report=dict(complete=True,owners=6,unrelated=1,boundaries=len(results),all_recovered=True,controls=controls,
        files=files,seconds=time.monotonic()-began,gpu_used=False,existing_jobs_signalled=False,
        successors_started=0,authorizes_retirement=False,acceptance='author-checked',number_status='measured')
    print(json.dumps(dict(pin=handoff.write(a.output/'report.json',report),boundaries=len(results))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('fixture','coordinator','proof'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--name');p.add_argument('--request',type=Path)
    p.add_argument('--crash-at',type=int);a=p.parse_args()
    try:
        if a.mode=='fixture':fixture(a)
        elif a.mode=='coordinator':coordinator(a)
        else:run(a)
    except BaseException as error:
        if a.mode=='proof' and a.output.exists():handoff.write(a.output/'failure.json',dict(error=repr(error)))
        raise
