"""Read the physical idle-owner handoff without importing its implementation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def load(path, pin=None):
    if pin is not None and sha(path) != pin: raise ValueError('PRIORITY_READBACK_PIN')
    return json.loads(Path(path).read_bytes())


def pid_state(pid):
    proc = Path('/proc')/str(pid)
    try: s = (proc/'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError: return None
    return dict(pid=pid, state=s[0], starttime=s[19],
                children=(proc/'task'/str(pid)/'children').read_text().split(),
                wchan=(proc/'wchan').read_text())


def run(a):
    c = load(a.authority, a.pin); before = load(a.handoff/'before.json')
    frozen = load(a.handoff/'frozen.json'); handoff = load(a.handoff/'handoff.json')
    result = load(a.handoff/'report.json'); proof = load(a.probe/'report.json', a.probe_pin)
    assert handoff['complete'] and result['complete'] and handoff['authority'] == result['authority'] == a.pin
    assert proof['complete'] and proof['authority'] == a.pin and proof['signals_sent'] == 0
    assert proof['controls'] == handoff['controls'] and len(proof['controls']) == 21
    assert handoff['terminated'] == ['answer', 'surface', 'grouped']
    assert result['handoff'] == sha(a.handoff/'handoff.json') and not result['numerical_work_complete']
    old_states = {}
    for name, ref in c['old'].items():
        root = Path(ref['path']); launch = load(root/'launch.json', ref['launch_pin'])
        b, f = before['owners'][name], frozen[name]
        assert b['pid'] == f['pid'] == launch['pid'] == ref['pid']
        assert b['starttime'] == f['starttime'] == ref['starttime'] and str(launch['starttime']) == ref['starttime']
        assert b['state'] == 'S' and f['state'] == 'T' and b['children'] == f['children'] == []
        assert b['events'] == f['events'] and b['files'] == f['files']
        assert b['files'] == {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}
        assert not any((root/n).exists() for n in ['report.json', 'failure.json'])
        live = pid_state(ref['pid']); assert live is None or live['state'] == 'Z' or live['starttime'] != ref['starttime']
        old_states[name] = live
    original = load(c['authorities']['answer']['path'], c['authorities']['answer']['pin'])
    revised = load(c['answer_authority']['path'], c['answer_authority']['pin'])
    assert revised['wait'] == c['runtime']; revised['wait'] = original['wait']; assert revised == original
    # All three original worker authorities and their exact script manifests survive.
    repo = Path(__file__).resolve().parents[2]; code_count = 0
    for ref in c['authorities'].values():
        record = load(ref['path'], ref['pin'])
        for name, pin in record['code'].items():
            path = repo/name if name.startswith('scripts/') else repo/'scripts'/name
            assert sha(path) == pin; code_count += 1
    root = Path(c['next_output']); launch = load(root/'launch.json')
    assert launch['authority'] == a.pin and launch['handoff'] == result['handoff']
    assert launch['driver_pin'] == c['driver_pin'] == sha(root/'driver.py')
    assert (launch['pid'], launch['starttime']) == (result['pid'], result['starttime'])
    live = pid_state(launch['pid'])
    assert live and live['starttime'] == launch['starttime'] and live['state'] == 'S' and live['children'] == []
    assert live['wchan'].startswith('poll_schedule_timeout')
    events = [json.loads(x) for x in (root/'events.jsonl').read_text().splitlines()]
    assert len(events) == 1 and events[0]['stage'] == 'waiting_for_runtime'
    assert events[0]['pid'] == c['runtime']['pid'] and events[0]['starttime'] == c['runtime']['starttime']
    protected = [pid_state(r['pid']) for r in c['protected']]
    assert all(row and row['starttime'] == ref['starttime'] for row, ref in zip(protected, c['protected']))
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader'], text=True)
    assert len(gpu.strip().splitlines()) == 1 and int(gpu.split(',')[0]) in {r['pid'] for r in c['protected']}
    report = dict(complete=True, authority=a.pin, handoff_pin=sha(a.handoff/'report.json'),
                  probe_pin=a.probe_pin, controls=21, old_states=old_states, all_original_artifacts_exact=True,
                  numerical_code_references_exact=code_count, only_answer_wait_changed=True,
                  successor=live, successor_launch_pin=sha(root/'launch.json'), protected=protected,
                  nvidia_compute=gpu, new_gpu_children_started=0, numerical_work_complete=False,
                  acceptance='author-checked', scientific_admission=False)
    a.output.mkdir(exist_ok=False)
    with (a.output/'report.json').open('x') as f:
        json.dump(report, f, sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ['authority', 'handoff', 'probe', 'output']: p.add_argument('--'+n, type=Path, required=True)
    for n in ['pin', 'probe-pin']: p.add_argument('--'+n, required=True)
    run(p.parse_args())
