"""Continue one response-preserving fit, diagnose exact meanings, and cold replay."""
import argparse
import copy
from functools import partial
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.corrective_open import open_session
from audit.fsv_identity_loop import configure
from audit.proposal_recognition import authenticate, evidence, read
from audit.response_margin import diagnose
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import run_cli, take_lease
from model.corrective_learning import CorrectiveData
from model.response_cycles import CycleData, CycleLoop
from model import bounded_training, field_checkpoint, response_update
from senses.contract import SenseError


def source_recipe(previous, config):
    prior = previous['spec']
    initial, history = prior.get('initial',prior),prior.get('continuations',[])
    spec = config['spec']
    check(spec['initial'] == initial and len(spec['continuations']) == len(history)+1
          and spec['continuations'][:-1] == history, 'RESPONSE_CYCLE_SOURCE_SPEC')
    echo = config.get('echo',dict(root=previous.get('echo_proof'),pin=previous.get('echo_pin')))
    check(isinstance(echo['root'],str) and isinstance(echo['pin'],str), 'RESPONSE_CYCLE_ECHO_SOURCE')
    return Path(echo['root']),echo['pin']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=('train','cold'),required=True)
    a = parser.parse_args()
    started = time.perf_counter()
    config = read(a.config)
    proof = Path(config['proof'])
    seal = authenticate(proof,config['proof_pin'])
    previous = read(proof/'session/config.json')
    echo, echo_pin = source_recipe(previous,config)
    link = config['spec']['continuations'][-1]
    check(evidence(Path(link['evidence']['path'])) == seal['files']['session/learning.json'],
          'RESPONSE_CYCLE_SOURCE_REPORT')
    source = Path(config['source'])
    check(sha((echo/'seal.json').read_bytes()) == echo_pin, 'RESPONSE_CYCLE_ECHO_PIN')
    echo_seal = read(echo/'seal.json')
    for name in ('experience.sqlite3','snapshot-final.json','config.json'):
        check(evidence(source/name) == echo_seal['files']['session/'+name], 'RESPONSE_CYCLE_PREFIX_BYTES')
    root = Path(config['root'])
    prospective = read(Path(config['prospective'])/'config.json')
    ancestor = Path(prospective['source'])
    before = read(source/'snapshot-final.json')
    if a.stage == 'train':
        check(not root.exists(), 'RESPONSE_CYCLE_ROOT_EXISTS')
        root.mkdir()
        (root/'base').mkdir()
        (root/'candidates').mkdir()
        shutil.copyfile(source/'experience.sqlite3',root/'experience.sqlite3')
        shutil.copyfile(ancestor/'experience.sqlite3',root/'base/experience.sqlite3')
        write_committed(root/'config.json',config)
    else:
        check(read(root/'config.json') == config, 'RESPONSE_CYCLE_CONFIG')
    gpu = subprocess.check_output(['nvidia-smi'],text=True)
    take_lease('28','response-cycle-'+a.stage)
    (root/('gpu-'+a.stage+'.txt')).open('x').write(gpu)
    configure()
    original, _ = open_session(ancestor,prospective['source_pin'],root/'base',read(ancestor/'snapshot-final.json'))
    base = CorrectiveData(original,prospective['experiment']['correction'])
    saved = before if a.stage == 'train' else read(root/'snapshot-final.json')
    session, _ = open_session(ancestor,prospective['source_pin'],root,saved,
                             loop_type=partial(CycleLoop,learning_base=base))
    replayed = time.perf_counter()
    print(json.dumps(dict(stage=a.stage,replayed_events=session.journal.count)),flush=True)
    controls, diagnostic_seconds = [], 0.

    def refuse(name, code, fn):
        snapshot = digest(session.snapshot())
        try:
            fn()
        except (ValueError,SenseError) as error:
            check(code in str(error), 'RESPONSE_CYCLE_CONTROL_CODE:'+name+':'+str(error))
        else:
            raise AssertionError('RESPONSE_CYCLE_CONTROL_ACCEPTED:'+name)
        check(digest(session.snapshot()) == snapshot, 'RESPONSE_CYCLE_CONTROL_MUTATION')
        controls.append(dict(name=name,code=code))

    spec = config['spec']
    if a.stage == 'train':
        bad_source = copy.deepcopy(config)
        bad_source['spec']['continuations'] = []
        refuse('missing_lineage','RESPONSE_CYCLE_SOURCE_SPEC',lambda:source_recipe(previous,bad_source))
        for field,code in (('checkpoint','RESPONSE_CYCLE_PARENT_PIN'),('evidence','RESPONSE_CYCLE_REPORT_PIN')):
            bad = copy.deepcopy(spec)
            bad['continuations'][-1][field]['pin'] = '0'*64
            refuse('wrong_'+field,code,lambda bad=bad:CycleData(session,base,bad))
        bad = copy.deepcopy(spec)
        bad['initial']['basis']['pin'] = '0'*64
        refuse('wrong_basis','RESPONSE_BASIS_PIN',lambda:CycleData(session,base,bad))
        data = CycleData(session,base,spec)
        parent, optimizer, _ = field_checkpoint.start(data)
        previous_report = read(Path(link['evidence']['path']))
        check(data.evaluate(parent) == previous_report['after'], 'RESPONSE_CYCLE_PARENT_BEHAVIOUR')
        write_committed(root/'training-binding.json',data.binding)
        arrays, diagnoses = {}, {}
        tick = time.perf_counter()
        diagnoses['before'] = diagnose(data,parent,optimizer,previous_report['after'],'before',arrays)
        diagnostic_seconds += time.perf_counter()-tick

        def review(current, name, pin, recipe):
            nonlocal diagnostic_seconds
            tick = time.perf_counter()
            final_data = CycleData(current,base,recipe)
            model, opt, _ = field_checkpoint.load(root/'candidates'/name,final_data,expected=pin)
            diagnoses['after'] = diagnose(final_data,model,opt,final_data.evaluate(model),'after',arrays)
            diagnostic_seconds += time.perf_counter()-tick
            return current.consider_cycle(name,pin,recipe)

        result = bounded_training.increment(session,root/'candidates',name='cycle-1',spec=spec,
            data_factory=CycleData,update=response_update.step,review=review,
            eligible=lambda data,_model:data.response_basis.names)
        write_committed(root/'learning.json',result)
        np.savez(root/'diagnostic-arrays.npz',**arrays)
        write_committed(root/'diagnostic.json',dict(stages=diagnoses,arrays=evidence(root/'diagnostic-arrays.npz'),
            code={n:sha((Path(__file__).parent/n).read_bytes()) for n in ('response_cycles.py','response_margin.py')},
            seconds=diagnostic_seconds,serving=False,scientific_admission=False))
        refuse('duplicate_decision','CANDIDATE_REVIEW_DUPLICATE',lambda:session.consider_cycle(
            'cycle-1',result['candidate_pin'],spec))
        write_committed(root/'snapshot-final.json',session.snapshot())
    final = session.snapshot()
    check({k:v for k,v in final.items() if k not in ('head','events','response_cycle_candidates')} ==
          {k:v for k,v in before.items() if k not in ('head','events')}, 'RESPONSE_CYCLE_OTHER_STATE_CHANGED')
    session.close()
    original.close()
    finished = time.perf_counter()
    report = dict(stage=a.stage,events=final['events'],head=final['head'],snapshot=digest(final),controls=controls,
        phases=dict(authentication_replay=replayed-started,diagnosis=diagnostic_seconds,
            learning_controls_close=finished-replayed-diagnostic_seconds),wall_seconds=finished-started,
        journal=sha((root/'experience.sqlite3').read_bytes()),serving=False,scientific_admission=False)
    write_committed(root/(a.stage+'-report.json'),report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    run_cli(main)
