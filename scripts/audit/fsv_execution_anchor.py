"""Three prospective process occurrences, observed region growth and termination heads."""
import argparse
import copy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.fsv_process_feedback import write
from audit.tool_execution_readback import readback
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims import forecast_commitment
from claims.process_conditions import ConditionInputs
from claims.tool_temporal_source import ToolSource
from lens.lenscommon import run_cli, take_lease
from model.condition_input import ConditionNative
from model.condition_catalog import NativeConditions
from model.tool_growth import GrowingToolLoop, parameters
from model.tool_kernel import ToolKernel
from model.tool_temporal_input import ToolInputs
from model import process_anchor
from senses.contract import SenseError


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--stage', choices=('first','second','third','finish','audit','cold'), required=True)
    a = p.parse_args()
    started = time.perf_counter()
    c = json.loads(a.config.read_bytes())
    root = Path(c['output'])
    gpu = subprocess.check_output(['nvidia-smi']).decode()
    take_lease('28', 'execution-anchor-'+a.stage)
    configure()
    source = ToolSource(Path(c['raw']))
    inputs = ToolInputs(source, Path(c['population']), Path(c['native']),
        population_pin=c['population_pin'], native_pin=c['native_pin'])
    providers = [ConditionNative(ConditionInputs(Path(v['conditions']),v['condition_pin']), inputs,
        Path(v['condition_population']),Path(v['condition_native']),
        population_pin=v['condition_population_pin'],native_pin=v['condition_native_pin'])
        for v in c['providers']]
    conditions = NativeConditions(providers)
    parent_root = Path(c['parent'])
    check(sha((parent_root/'run.json').read_bytes()) == c['parent_pin'], 'ANCHOR_PARENT')
    parent = json.loads((parent_root/'run.json').read_bytes())
    routing = ToolKernel(inputs,Path(c['kernel']),(Path(c['kernel'])/'COMMITTED').read_text())
    def open_session(path=None, **kw):
        return GrowingToolLoop(path or root/'experience.sqlite3',source,inputs,parent_root/'initial',
            model_pin=parent['initial'],standard=parent['standard'],world=parent['world'],
            contexts={'train:1':c['providers'][0]['task']},device='cuda',condition_inputs=conditions,
            routing=routing,growth_root=root/'growth',candidate_root=root/'candidates',**kw)
    if a.stage == 'first':
        root.mkdir(parents=True,exist_ok=False)
        write(root/'config.json',c)
        session = open_session(create=True)
    else:
        saved = json.loads((root/('snapshot.json' if a.stage in ('audit','cold') else
            {'second':'pending-1.json','third':'pending-2.json','finish':'pending-3.json'}[a.stage])).read_bytes())
        session = open_session(expected_head=saved['head'],expected_contract=saved['contract'])
        check(session.snapshot() == saved,'ANCHOR_REOPEN')
    (root/('gpu-'+a.stage+'.txt')).write_text(gpu)
    phases = dict(load_and_replay=time.perf_counter()-started)
    tick = time.perf_counter()
    controls = []
    initial = parameters(session.model)
    def refuse(name,code,fn):
        before = digest(session.snapshot())
        try:
            fn()
        except (ValueError,SenseError) as e:
            check(code in str(e),'ANCHOR_CONTROL_'+name+':'+str(e))
        else:
            raise AssertionError('ANCHOR_ACCEPTED_'+name)
        check(digest(session.snapshot()) == before,'ANCHOR_CONTROL_STATE')
        controls.append(dict(name=name,code=code))
    def enroll(number,context):
        spec = json.loads(Path(c['executions'][number-1]['enrollment']).read_bytes())
        event = session.enroll_execution(spec)
        key = session.pending[context]
        observation = event['links'][key]
        req = session.forecasts[key]['record']['request']
        session.advance(context,receipt=source.receipt(req['task'],req['turn']+1),execution=observation)
        session.advance(context)
        check('execution_anchor_score' in session.forecasts[key],'ANCHOR_ECHO_NOT_SCORED')
        return key
    result = {}
    if a.stage == 'cold':
        result['snapshot'] = session.snapshot()
    elif a.stage in ('finish','audit'):
        key = (enroll(3,'withheld:0') if a.stage == 'finish' else
               next(iter(json.loads((root/'forecast-3/forecast.json').read_bytes())['forecasts'])))
        prediction = session.forecasts[key]['record']['execution_anchor']
        result.update(forecast=key,score=session.forecasts[key]['execution_anchor_score'])
        bad = copy.deepcopy(prediction)
        bad['common']['probabilities'] = [0.,0.,1.]
        refuse('changed_mass_assignment','PROCESS_ANCHOR_PUSHFORWARD',lambda:process_anchor.validate(bad))
        wrong = copy.deepcopy(prediction)
        row = next(r for r in wrong['bank']['rows'] if r['observation'] is None)
        row['outcome'] = 0
        wrong['bank']['pin'] = digest({k:v for k,v in wrong['bank'].items() if k != 'pin'})
        refuse('reported_text_as_outcome','PROCESS_ANCHOR_UNOBSERVED_LABEL',lambda:process_anchor.validate(wrong))
        binding = copy.copy(session.inputs)
        binding.observed_regions = copy.deepcopy(session.inputs.observed_regions)
        first = next(iter(binding.observed_regions))
        binding.observed_regions[first]['native']['observations'] = '0'*64
        refuse('wrong_native_anchor','PROCESS_ANCHOR_BINDING',
            lambda:process_anchor.bank(binding,[int(first)]))
        bad = copy.deepcopy(prediction)
        bad['common']['probabilities'][0] = float('nan')
        refuse('nonfinite_mass','PROCESS_ANCHOR_MASS',lambda:process_anchor.validate(bad))
        saved = session.snapshot()
        if a.stage == 'finish':
            write(root/'snapshot.json',saved)
        else:
            check(json.loads((root/'snapshot.json').read_bytes()) == saved,'ANCHOR_AUDIT_SOURCE')
        original = session
        branch_path = root/(a.stage+'-withdrawal.sqlite3')
        check(not branch_path.exists(),'ANCHOR_BRANCH_EXISTS')
        with sqlite3.connect(branch_path) as db:
            session.journal.db.backup(db)
        session = open_session(branch_path,expected_head=saved['head'],expected_contract=saved['contract'])
        first_growth = session.growths[0]
        episode = next(iter(first_growth['supports'].values()))['response']
        result['withdrawal'] = session.invalidate(episode,reason='FSV prior anchor withdrawal')
        refuse('withdrawn_anchor_forecast','TOOL_LEARNING_INELIGIBLE',lambda:session.eligible(key))
        branch = session.snapshot()
        write(root/(a.stage+'-withdrawn-before-repair.json'),branch)
        session.close()
        session = original
        refuse('withdrawn_bank_cold','TOOL_EXECUTION_UNAVAILABLE',lambda:open_session(
            branch_path,expected_head=branch['head'],expected_contract=branch['contract']))
        session = open_session(branch_path,expected_head=branch['head'],expected_contract=branch['contract'],recovery=True)
        check(session.snapshot() == branch,'ANCHOR_WITHDRAWAL_RECOVERY')
        result['repair'] = session.repair_bank(a.stage+'-withdrawal-retire')
        check(set(first_growth['supports']).issubset({str(v) for v in result['repair']['removed']}),
              'ANCHOR_WITHDRAWAL_RETIRED')
        refuse('withdrawn_context_after_repair','TOOL_CONTEXT_REVIEW',lambda:session.advance('withheld:0'))
        branch = session.snapshot()
        write(root/(a.stage+'-withdrawal-snapshot.json'),branch)
        session.close()
        session = open_session(branch_path,expected_head=branch['head'],expected_contract=branch['contract'])
        check(session.snapshot() == branch,'ANCHOR_WITHDRAWAL_REPLAY')
        session.close()
        session = original
        check(session.snapshot() == saved,'ANCHOR_BRANCH_ISOLATION')
    else:
        number = ('first','second','third').index(a.stage)+1
        if number > 1:
            key = enroll(number-1,'train:1')
            result['growth'] = session.grow('actual-'+str(number-1),[key])
            result['recall'] = {n:v['recall'] for n,v in session.routing.calibration.items()}
        if number == 3:
            session.start_context('withheld:0',c['providers'][1]['task'])
        context = 'withheld:0' if number == 3 else 'train:1'
        stop = 23 if number == 2 else 13
        provider = providers[int(number == 3)]
        condition = 'without_dependency' if number == 1 else 'with_dependency'
        while session.states[context].sequence < stop:
            seq = session.states[context].sequence
            if seq == stop-2:
                session.bind_execution_conditions(context,provider.conditions.commitment['routing'][condition])
            session.advance(context,receipt=source.receipt(session.contexts[context],3+2*(seq//5))
                            if seq % 5 == 3 else None)
        key = session.pending[context]
        anchor = session.forecasts[key]['record']['execution_anchor']
        check({r['outcome'] for r in anchor['bank']['rows']} == set(range(3)) if number == 3
              else all(r['outcome'] == 2 for r in anchor['bank']['rows']) if number == 1
              else any(r['outcome'] == 1 for r in anchor['bank']['rows']), 'ANCHOR_PRIOR_ROSTER')
        check(not Path(c['executions'][number-1]['observations']).exists(),'ANCHOR_FUTURE_OUTCOME_PRESENT')
        pin = forecast_commitment.publish(session,root/('forecast-'+str(number)),[key])
        saved = session.snapshot()
        write(root/('pending-'+str(number)+'.json'),saved)
        result.update(pin=pin,forecast=key,anchor=anchor)
    check(parameters(session.model) == initial,'ANCHOR_PARAMETERS_CHANGED')
    if a.stage in ('finish','audit','cold'):
        result['readback'] = readback(root/'experience.sqlite3',inputs,session.snapshot(),
            [Path(v['observations']) for v in c['executions']],session.execution_inputs)
    saved = session.snapshot()
    session.close()
    session = open_session(expected_head=saved['head'],expected_contract=saved['contract'])
    check(session.snapshot() == saved,'ANCHOR_NEURAL_REPLAY')
    session.close()
    phases[a.stage] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    write(root/(a.stage+'.json'),dict(complete=True,result=result,controls=controls,
        phases=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,
        model_fitted=False,scientific_admission=False,serving=False))
    print(json.dumps(dict(complete=True,stage=a.stage,wall_seconds=wall,controls=len(controls))))


if __name__ == '__main__':
    run_cli(main)
