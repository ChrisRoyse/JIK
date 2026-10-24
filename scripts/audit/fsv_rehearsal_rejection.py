"""Exercise actual parameter restoration while a bank-transition repair is pending."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.fsv_grounded_rehearsal import authenticate
from audit.fsv_process_feedback import write
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.process_conditions import ConditionInputs
from claims.tool_temporal_source import ToolSource
from lens.lenscommon import run_cli, take_lease
from model.condition_input import ConditionNative
from model.condition_catalog import NativeConditions
from model.constellation_learning import trainable
from model.loop_training_checkpoint import fingerprint
from model.tool_kernel import ToolKernel
from model.tool_temporal_input import ToolInputs
from model.tool_rehearsal import RehearsalData, RehearsalToolLoop
from model import rehearsal_retention, tool_learner
from model import constellation_training_checkpoint as checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    started = time.perf_counter()
    provenance = json.loads((a.root/'source.json').read_bytes())
    evidence = Path(provenance['root'])
    authenticate(evidence,provenance['seal'])
    summary = json.loads((evidence/'summary.json').read_bytes())
    c = summary['config']
    prior_root = Path(c['output'])
    prior = json.loads((prior_root/'snapshot.json').read_bytes())
    main_before = sha((a.root/'experience.sqlite3').read_bytes())
    root = a.output
    check(root.resolve().parent == a.root.resolve(),'REJECTION_OUTPUT_SCOPE')
    root.mkdir(exist_ok=False)
    shutil.copyfile(prior_root/'experience.sqlite3',root/'experience.sqlite3')
    check(sha((root/'experience.sqlite3').read_bytes()) == summary['journal_sha256'],'REJECTION_PREFIX')
    gpu = subprocess.check_output(['nvidia-smi']).decode()
    take_lease('28','grounded-retention-rejection')
    (root/'gpu.txt').write_text(gpu)
    configure()
    source = ToolSource(Path(c['raw']))
    inputs = ToolInputs(source,Path(c['population']),Path(c['native']),
                        population_pin=c['population_pin'],native_pin=c['native_pin'])
    providers = [ConditionNative(ConditionInputs(Path(v['conditions']),v['condition_pin']),inputs,
        Path(v['condition_population']),Path(v['condition_native']),
        population_pin=v['condition_population_pin'],native_pin=v['condition_native_pin']) for v in c['providers']]
    parent_root = Path(c['parent'])
    check(sha((parent_root/'run.json').read_bytes()) == c['parent_pin'],'REJECTION_PARENT')
    parent = json.loads((parent_root/'run.json').read_bytes())
    routing = ToolKernel(inputs,Path(c['kernel']),(Path(c['kernel'])/'COMMITTED').read_text())
    def open_session(saved):
        return RehearsalToolLoop(root/'experience.sqlite3',source,inputs,parent_root/'initial',model_pin=parent['initial'],
            standard=parent['standard'],world=parent['world'],contexts={'train:1':c['providers'][0]['task']},
            device='cuda',condition_inputs=NativeConditions(providers),routing=routing,growth_root=prior_root/'growth',
            candidate_root=a.root/'candidates',expected_head=saved['head'],expected_contract=saved['contract'])
    session = open_session(prior)
    check(session.snapshot() == prior,'REJECTION_PREFIX_REPLAY')
    decision = json.loads((a.root/'rehearsal-1.json').read_bytes())['decision']
    check(session.consider_rehearsal(decision['candidate'],pin=decision['pin'],train=decision['train'],
        evaluate=decision['evaluate'],parent=decision['parent']) == decision,'REJECTION_CANDIDATE_REPLAY')
    train = [v['forecast'] for v in summary['measured']['scored'][:2]]
    evaluate = [summary['measured']['scored'][2]['forecast']]
    data = RehearsalData(session,train=train,evaluate=evaluate,continuation='rehearsal-1')
    model,opt,_ = tool_learner.start(data)
    saved = session.snapshot()
    state_before = fingerprint(model,opt)
    prepared = rehearsal_retention.prepare(data,model)
    check(any(not r['retained'] for r in rehearsal_retention.comparisons(data,prepared['scores'])),
          'REJECTION_NO_PENDING_REPAIR')
    phases = dict(authenticate_replay_and_prepare=time.perf_counter()-started)
    tick = time.perf_counter()
    # A deliberate no-op proposal cannot be accepted as repairing a real regression.
    before = {n:v.detach().clone() for n,v in trainable(model)}
    result = rehearsal_retention.apply(data,model,before,prepared)
    check(result['accepted'] is False and result['retained'] is False and result['scale'] == 0.
          and len(result['trials']) == 4 and all(not t['advances'] for t in result['trials']),
          'REJECTION_FALSE_REPAIR')
    check(fingerprint(model,opt) == state_before and session.snapshot() == saved,
          'REJECTION_CHANGED_STATE')
    gate = rehearsal_retention.assess(data,data.evaluate(model),dict(metric_gate_passed=True,reasons=[]))
    check(gate['metric_gate_passed'] is False and gate['reasons'] == ['GROUNDED_TRAINING_RETENTION_PENDING'],
          'REJECTION_PENDING_PROMOTION')
    restored_pin = checkpoint.save(root/'restored',model,opt,data,step=0,parent=decision['pin'])
    phases['proposal_restore_and_candidate_gate'] = time.perf_counter()-tick
    tick = time.perf_counter()
    session.close()
    session = open_session(saved)
    check(session.snapshot() == saved,'REJECTION_COLD_REPLAY')
    session.close()
    check(sha((a.root/'experience.sqlite3').read_bytes()) == main_before,'REJECTION_MAIN_CHANGED')
    phases['cold_replay_and_isolation'] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    write(root/'snapshot.json',saved)
    report = dict(complete=True,acceptance='author-checked',binding=data.binding,update=result,gate=gate,
        restored_checkpoint=restored_pin,
        state_before=state_before,state_after=fingerprint(model,opt),head=saved['head'],
        main_journal_unchanged=main_before,phases=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    write(root/'report.json',report)
    print(json.dumps(dict(complete=True,head=saved['head'],restored=True,seconds=wall)),flush=True)


if __name__ == '__main__':
    run_cli(main)
