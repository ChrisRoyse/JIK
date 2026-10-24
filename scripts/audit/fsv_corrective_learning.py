"""Fresh per-slot retention, actual loss descent, restart and rollback on captured feedback."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
import torch
from audit.corrective_open import open_session
from audit.fsv_field_world import authenticate
from audit.fsv_process_feedback import write
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from lens.lenscommon import run_cli, take_lease
from model.controlled_learning import ControlledData
from model.corrective_learning import CorrectiveData
from model.corrective_update import increment
from model import field_checkpoint, rehearsal_retention
from senses.contract import SenseError


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cold',action='store_true')
    a = p.parse_args()
    started, root = time.perf_counter(), a.output
    proof = authenticate(a.baseline,a.pin)
    source = json.loads((a.baseline/'source.json').read_bytes())
    echo = Path(source['root'])
    original = json.loads((echo/'snapshot-final.json').read_bytes())
    previous = json.loads((a.baseline/'learning.json').read_bytes())
    spec = dict(base=previous['decision']['spec'],source=dict(root=str(a.baseline),seal=a.pin))
    if not a.cold:
        root.mkdir(exist_ok=False)
        shutil.copyfile(echo/'experience.sqlite3',root/'experience.sqlite3')
        write(root/'source.json',dict(source,baseline=spec['source']))
    stage = 'cold' if a.cold else 'run'
    gpu = subprocess.check_output(['nvidia-smi']).decode()
    take_lease('28','corrective-learning-'+stage)
    with (root/('gpu-'+stage+'.txt')).open('x') as stream:
        stream.write(gpu)
    configure()
    saved = json.loads((root/'snapshot.json').read_bytes()) if a.cold else original
    session, echo_proof = open_session(echo,source['seal'],root,saved)
    phases = dict(authentication_and_replay=time.perf_counter()-started)
    controls = []

    def refuse(name, code, action):
        before = digest(session.snapshot())
        try:
            action()
        except (ValueError,SenseError) as error:
            check(code in str(error),'CORRECTION_CONTROL_CODE_'+name+':'+str(error))
        else:
            raise AssertionError('CORRECTION_CONTROL_ACCEPTED_'+name)
        check(digest(session.snapshot()) == before,'CORRECTION_CONTROL_MUTATION')
        controls.append(dict(name=name,code=code))

    tick = time.perf_counter()
    if not a.cold:
        data = CorrectiveData(session,spec)
        old = ControlledData(session,spec['base'])
        rejected, _, _ = field_checkpoint.load(a.baseline/'candidates/controlled-1',old,expected=previous['candidate_pin'])
        measured = rehearsal_retention.scores(data,rejected)
        violations = [r for r in rehearsal_retention.comparisons(data,measured) if not r['retained']]
        check(violations and all(r['key'] in data.controlled for r in violations),'CORRECTION_KNOWN_REGRESSION')
        write(root/'known-bad-candidate.json',dict(pin=previous['candidate_pin'],violations=violations))
        def altered_ceiling():
            before = copy.deepcopy(data.binding['grounded_retention'])
            try:
                key = next(iter(data.controlled))
                data.binding['grounded_retention']['ceilings'][key]['termination']['common'] += 1.
                data.fresh()
            finally:
                data.binding['grounded_retention'] = before
        refuse('altered_fresh_ceiling','CORRECTION_FIXED_CEILINGS_CHANGED',altered_ceiling)
        context = spec['base']['contexts'][0]
        def missing_feedback():
            entry = session.echo_contexts[context]
            old_status = entry['status']
            try:
                entry['status'] = 'awaiting_correction'
                CorrectiveData(session,spec)
            finally:
                entry['status'] = old_status
        refuse('incomplete_feedback','CONTROLLED_FEEDBACK_INCOMPLETE',missing_feedback)
        phases['negative_controls'] = time.perf_counter()-tick
        tick = time.perf_counter()
        result = increment(session,root/'candidates',name='corrective-1',spec=spec)
        write(root/'learning.json',result)
        write(root/'snapshot.json',session.snapshot())
        phases['corrective_learning'] = time.perf_counter()-tick
    else:
        result = json.loads((root/'learning.json').read_bytes())
        check(session.corrective_candidates['corrective-1'] == result['decision'],'CORRECTION_COLD_DECISION')
        phases['cold_join'] = time.perf_counter()-tick
    tick = time.perf_counter()
    refuse('unadmitted_activation','TOOL_CANDIDATE_NOT_SELECTED',lambda:session.require_selection('corrective-1'))
    if not a.cold:
        broken = root/'negative-checkpoint'
        shutil.copytree(root/'candidates/corrective-1',broken)
        with (broken/'weights.pt').open('ab') as stream:
            stream.write(b'corrupt-control')
        data = CorrectiveData(session,spec)
        manifest = json.loads((broken/'manifest.json').read_bytes())
        refuse('changed_checkpoint_bytes','NATIVE_WORLD_CHECKPOINT_BYTES',lambda:field_checkpoint.load(
            broken,data,expected=result['candidate_pin'],historical=manifest))
    snapshot = session.snapshot()
    excluded = {'head','events','corrective_feedback_candidates'}
    check({k:v for k,v in snapshot.items() if k not in excluded} ==
          {k:v for k,v in original.items() if k not in excluded},'CORRECTION_ORIGINAL_STATE_DRIFT')
    session.close()
    check(sha((echo/'experience.sqlite3').read_bytes()) == source['journal'],'CORRECTION_SOURCE_CHANGED')
    phases['final_readback'] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    report = dict(complete=True,acceptance='author-checked',stage=stage,source=proof,echo=echo_proof,
        head=snapshot['head'],snapshot=digest(snapshot),journal_sha256=sha((root/'experience.sqlite3').read_bytes()),
        controls=controls,phases=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,
        accepted_updates=sum(r['update']['accepted'] for r in result['losses']),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        metric_gate_passed=result['decision']['result']['metric_gate_passed'],serving=False,scientific_admission=False)
    write(root/(stage+'-report.json'),report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    run_cli(main)
