"""Continue a sealed execution journal through two real rehearsal increments."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.fsv_process_feedback import write
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.process_conditions import ConditionInputs
from claims.tool_temporal_source import ToolSource
from lens.lenscommon import run_cli, take_lease
from model.condition_input import ConditionNative
from model.condition_catalog import NativeConditions
from model.tool_growth import parameters
from model.tool_kernel import ToolKernel
from model.tool_temporal_input import ToolInputs
from model.tool_rehearsal import RehearsalData, RehearsalToolLoop
from model.rehearsal_training import increment
from model import process_anchor, rehearsal_retention, tool_learner
from model import constellation_training_checkpoint as checkpoint
from senses.contract import SenseError


def authenticate(root,pin):
    raw = (root/'seal.json').read_bytes()
    check(sha(raw) == pin,'REHEARSAL_SOURCE_SEAL')
    seal = json.loads(raw)
    for name,entry in seal['files'].items():
        raw = (root/name).read_bytes()
        check(sha(raw) == entry['sha256'] and len(raw) == entry['bytes'],'REHEARSAL_SOURCE_FILE')
    for name,entry in seal['external_native'].items():
        raw = Path(name).read_bytes()
        check(sha(raw) == entry['sha256'] and len(raw) == entry['bytes'],'REHEARSAL_NATIVE_FILE')
    return dict(files=len(seal['files']),external=len(seal['external_native']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--seal',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cold',action='store_true')
    a = p.parse_args()
    started = time.perf_counter()
    source_proof = authenticate(a.evidence,a.seal)
    summary = json.loads((a.evidence/'summary.json').read_bytes())
    c = summary['config']
    prior_root = Path(c['output'])
    prior = json.loads((prior_root/'snapshot.json').read_bytes())
    root = a.output
    if not a.cold:
        root.mkdir(parents=True,exist_ok=False)
        shutil.copyfile(prior_root/'experience.sqlite3',root/'experience.sqlite3')
        check(sha((root/'experience.sqlite3').read_bytes()) == summary['journal_sha256'],
              'REHEARSAL_PREFIX_COPY')
        write(root/'source.json',dict(root=str(a.evidence),seal=a.seal,summary=sha((a.evidence/'summary.json').read_bytes()),
                                    prior_head=prior['head'],author_checked=True))
    gpu = subprocess.check_output(['nvidia-smi']).decode()
    take_lease('28','grounded-rehearsal-cold' if a.cold else 'grounded-rehearsal')
    (root/('gpu-cold.txt' if a.cold else 'gpu.txt')).write_text(gpu)
    configure()
    source = ToolSource(Path(c['raw']))
    inputs = ToolInputs(source,Path(c['population']),Path(c['native']),
                        population_pin=c['population_pin'],native_pin=c['native_pin'])
    providers = [ConditionNative(ConditionInputs(Path(v['conditions']),v['condition_pin']),inputs,
        Path(v['condition_population']),Path(v['condition_native']),
        population_pin=v['condition_population_pin'],native_pin=v['condition_native_pin']) for v in c['providers']]
    conditions = NativeConditions(providers)
    parent_root = Path(c['parent'])
    check(sha((parent_root/'run.json').read_bytes()) == c['parent_pin'],'REHEARSAL_PARENT')
    parent = json.loads((parent_root/'run.json').read_bytes())
    routing = ToolKernel(inputs,Path(c['kernel']),(Path(c['kernel'])/'COMMITTED').read_text())
    def open_session(path, snapshot):
        return RehearsalToolLoop(path,source,inputs,parent_root/'initial',model_pin=parent['initial'],
            standard=parent['standard'],world=parent['world'],contexts={'train:1':c['providers'][0]['task']},
            device='cuda',condition_inputs=conditions,routing=routing,growth_root=prior_root/'growth',
            candidate_root=root/'candidates',expected_head=snapshot['head'],expected_contract=snapshot['contract'])
    saved = json.loads((root/'snapshot.json').read_bytes()) if a.cold else prior
    session = open_session(root/'experience.sqlite3',saved)
    check(session.snapshot() == saved,'REHEARSAL_PREFIX_REPLAY')
    torch.cuda.reset_peak_memory_stats()
    phases = dict(authentication_and_neural_replay=time.perf_counter()-started)
    initial = parameters(session.model)
    controls, results = [], []
    def refuse(name,code,fn):
        before = digest(session.snapshot())
        try:
            fn()
        except (ValueError,SenseError) as e:
            check(code in str(e),'REHEARSAL_CONTROL_'+name+':'+str(e))
        else:
            raise AssertionError('REHEARSAL_CONTROL_ACCEPTED_'+name)
        check(digest(session.snapshot()) == before,'REHEARSAL_CONTROL_STATE')
        controls.append(dict(name=name,code=code))
    train = [v['forecast'] for v in summary['measured']['scored'][:2]]
    evaluate = [summary['measured']['scored'][2]['forecast']]
    if not a.cold:
        tick = time.perf_counter()
        refuse('heldout_used_for_training','TOOL_LEARNING_SPLIT',lambda:RehearsalData(
            session,train=evaluate,evaluate=train))
        data = RehearsalData(session,train=train[:1],evaluate=evaluate)
        baseline = data.evaluate(session.model)
        check(baseline[evaluate[0]]['predictions'] == session.forecasts[evaluate[0]]['record']['predictions']
              and baseline[evaluate[0]]['termination']['prediction'] ==
                  session.forecasts[evaluate[0]]['record']['execution_anchor'],'REHEARSAL_HISTORICAL_EVALUATION')
        actual,observation = data.actual[train[0]]
        refuse('rehearsal_called_prospective','PROCESS_ANCHOR_TARGET_LEAK',lambda:process_anchor.score(
            baseline[train[0]]['termination']['prediction'],actual,observation))
        e = data.examples[evaluate[0]]
        allowed = e['allowed'].clone()
        e['allowed'][0] = False
        refuse('evaluation_mask_changed','TOOL_LEARNING_CUTOFF',lambda:data.forward(session.model,evaluate[0]))
        e['allowed'].copy_(allowed)
        phases['input_and_negative_boundaries'] = time.perf_counter()-tick
        for number in (1,2):
            tick = time.perf_counter()
            if number == 2:
                retained = RehearsalData(session,train=train,evaluate=evaluate,continuation='rehearsal-1')
                candidate, _, _ = tool_learner.start(retained)
                before_scores = rehearsal_retention.scores(retained,candidate)
                transitions = rehearsal_retention.comparisons(retained,before_scores)
                check(any(not r['retained'] for r in transitions),'REHEARSAL_BANK_TRANSITION_NOT_EXERCISED')
                write(root/'bank-transition.json',dict(scores=before_scores,comparisons=transitions,
                    binding=retained.binding['grounded_retention'],parameters=parameters(candidate)))
                refuse('unrepaired_bank_transition','GROUNDED_REPAIR_REQUIRED',lambda:check(
                    rehearsal_retention.advances(retained,before_scores,before_scores),'GROUNDED_REPAIR_REQUIRED'))
                saved_retention = copy.deepcopy(retained.binding['grounded_retention'])
                key = retained.retained_keys[0]
                retained.binding['grounded_retention']['ceilings'][key]['termination']['common'] = 2.
                refuse('forged_retention_ceiling','GROUNDED_RETENTION_BINDING',lambda:rehearsal_retention.validate(retained))
                retained.binding['grounded_retention'] = saved_retention
                rehearsal_retention.validate(retained)
                worse = copy.deepcopy(before_scores)
                worse[key]['termination']['common'] += .01
                refuse('worsened_bank_transition','GROUNDED_REPAIR_REQUIRED',lambda:check(
                    rehearsal_retention.advances(retained,before_scores,worse),'GROUNDED_REPAIR_REQUIRED'))
                del candidate, retained
            name = 'rehearsal-'+str(number)
            result = increment(session,root,name=name,train=train[:number],evaluate=evaluate,
                               continuation=None if number == 1 else 'rehearsal-1')
            write(root/(name+'.json'),result)
            check(parameters(session.model) == initial,'REHEARSAL_ACTIVE_CHANGED')
            check(not result['decision']['result']['selected'],'REHEARSAL_UNADMITTED_SELECTED')
            refuse('selection_'+str(number),'TOOL_CANDIDATE_NOT_SELECTED',lambda:session.require_selection(name))
            results.append(dict(name=name,changed=len(result['changed']),optimizer_steps=result['optimizer_steps'],
                final_loss=result['losses'][-1]['loss'],result=result['decision']['result']))
            phases[name] = time.perf_counter()-tick
            print(json.dumps(dict(stage=name,seconds=phases[name],changed=len(result['changed']),
                                  optimizer_steps=result['optimizer_steps'],selected=False)),flush=True)
        refuse('stale_training_view','TOOL_LEARNING_STALE',data.fresh)
        tick = time.perf_counter()
        saved = session.snapshot()
        write(root/'snapshot.json',saved)
        original = session
        with sqlite3.connect(root/'withdrawal.sqlite3') as db:
            session.journal.db.backup(db)
        session = open_session(root/'withdrawal.sqlite3',saved)
        episode = session.forecasts[evaluate[0]]['execution']['episode']
        withdrawal = session.invalidate(episode,reason='FSV held-out occurrence withdrawn')
        refuse('withdrawn_evaluation','TOOL_LEARNING_INELIGIBLE',lambda:RehearsalData(
            session,train=train,evaluate=evaluate,continuation='rehearsal-2'))
        branch = session.snapshot()
        write(root/'withdrawal-snapshot.json',branch)
        write(root/'withdrawal.json',withdrawal)
        session.close()
        session = open_session(root/'withdrawal.sqlite3',branch)
        check(session.snapshot() == branch,'REHEARSAL_WITHDRAWAL_REPLAY')
        session.close()
        session = original
        check(session.snapshot() == saved,'REHEARSAL_BRANCH_ISOLATION')
        phases['withdrawal_and_cold_replay'] = time.perf_counter()-tick
    tick = time.perf_counter()
    saved = session.snapshot()
    session.close()
    session = open_session(root/'experience.sqlite3',saved)
    check(session.snapshot() == saved,'REHEARSAL_FINAL_NEURAL_REPLAY')
    session.close()
    check(authenticate(a.evidence,a.seal) == source_proof,'REHEARSAL_SOURCE_CHANGED')
    phases['final_replay_and_source_readback'] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    report = dict(complete=True,acceptance='author-checked',source=source_proof,head=saved['head'],
        controls=controls,increments=results,phases=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        serving=False,scientific_admission=False)
    write(root/('cold.json' if a.cold else 'report.json'),report)
    print(json.dumps(dict(complete=True,head=saved['head'],seconds=wall,controls=len(controls))),flush=True)


if __name__ == '__main__':
    run_cli(main)
