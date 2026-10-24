"""Execute and cold-replay a fixed-ceiling proposal correction on real feedback."""
import argparse
import copy
from functools import partial
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.corrective_open import open_session
from audit.fsv_identity_loop import configure
from audit.proposal_recognition import authenticate, evidence, read
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import run_cli, take_lease
from model.corrective_learning import CorrectiveData
from model.slack_learning import SlackData, SlackLearningLoop, increment
from model import field_checkpoint
from senses.contract import SenseError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=('train', 'cold'), required=True)
    a = parser.parse_args()
    started = time.perf_counter()
    config = read(a.config)
    proof = Path(config['proof'])
    seal = authenticate(proof, config['proof_pin'])
    source = Path(config['source'])
    previous_config = read(proof/'session/config.json')
    echo_seal = read(Path(previous_config['proof'])/'seal.json')
    for name in ('experience.sqlite3', 'snapshot-final.json', 'config.json'):
        check(evidence(source/name) == echo_seal['files']['session/'+name], 'SLACK_PREFIX_BYTES')
    check(config['spec']['proposal'] == previous_config['spec'], 'SLACK_PROPOSAL_SOURCE')
    evidence_path = Path(config['spec']['evidence']['path'])
    check(evidence(evidence_path) == seal['files']['session/learning.json'], 'SLACK_LEARNING_SOURCE')
    root = Path(config['root'])
    prospective = read(Path(config['prospective'])/'config.json')
    ancestor = Path(prospective['source'])
    before = read(source/'snapshot-final.json')
    if a.stage == 'train':
        check(not root.exists(), 'SLACK_ROOT_EXISTS')
        root.mkdir()
        (root/'base').mkdir()
        (root/'candidates').mkdir()
        shutil.copyfile(source/'experience.sqlite3', root/'experience.sqlite3')
        shutil.copyfile(ancestor/'experience.sqlite3', root/'base/experience.sqlite3')
        write_committed(root/'config.json', config)
    else:
        check(read(root/'config.json') == config, 'SLACK_CONFIG')
    gpu = subprocess.check_output(['nvidia-smi'], text=True)
    take_lease('28', 'slack-learning-'+a.stage)
    (root/('gpu-'+a.stage+'.txt')).open('x').write(gpu)
    configure()
    original, _ = open_session(ancestor, prospective['source_pin'], root/'base', read(ancestor/'snapshot-final.json'))
    base = CorrectiveData(original, prospective['experiment']['correction'])
    saved = before if a.stage == 'train' else read(root/'snapshot-final.json')
    session, _ = open_session(ancestor, prospective['source_pin'], root, saved,
                             loop_type=partial(SlackLearningLoop, learning_base=base))
    replayed = time.perf_counter()
    print(json.dumps(dict(stage=a.stage, replayed_events=session.journal.count)), flush=True)
    controls = []

    def refuse(name, code, fn):
        snapshot = digest(session.snapshot())
        try:
            fn()
        except (ValueError, SenseError) as error:
            check(code in str(error), 'SLACK_CONTROL_CODE:'+name+':'+str(error))
        else:
            raise AssertionError('SLACK_CONTROL_ACCEPTED:'+name)
        check(digest(session.snapshot()) == snapshot, 'SLACK_CONTROL_MUTATION')
        controls.append(dict(name=name, code=code))

    spec = config['spec']
    if a.stage == 'train':
        bad = copy.deepcopy(spec)
        bad['continuation']['pin'] = '0'*64
        refuse('wrong_parent', 'SLACK_PARENT_PIN', lambda:SlackData(session, base, bad))
        data = SlackData(session, base, spec)
        ceilings = data.binding['grounded_retention']['ceilings']
        key = next(iter(ceilings))
        lane = next(iter(ceilings[key]['native']))
        old = ceilings[key]['native'][lane]
        try:
            ceilings[key]['native'][lane] += .001
            refuse('changed_ceiling', 'PROPOSAL_LEARNING_BINDING_CHANGED', data.fresh)
        finally:
            ceilings[key]['native'][lane] = old
        data.fresh()
        parent, _, _ = field_checkpoint.start(data)
        check(data.evaluate(parent) == read(evidence_path)['after'], 'SLACK_PARENT_BEHAVIOUR')
        refuse('wrong_checkpoint_pin', 'NATIVE_WORLD_CHECKPOINT_PIN', lambda:field_checkpoint.load(
            Path(spec['continuation']['root']), data, expected='0'*64))
        write_committed(root/'training-binding.json', data.binding)
        result = increment(session, root/'candidates', name='slack-1', spec=spec)
        write_committed(root/'learning.json', result)
        refuse('duplicate_decision', 'SLACK_DUPLICATE_DECISION', lambda:session.consider_slack_learning(
            'slack-1', result['candidate_pin'], spec))
        write_committed(root/'snapshot-final.json', session.snapshot())
    final = session.snapshot()
    check({k:v for k,v in final.items() if k not in ('head', 'events', 'slack_learning_candidates')} ==
          {k:v for k,v in before.items() if k not in ('head', 'events')}, 'SLACK_OTHER_STATE_CHANGED')
    session.close()
    original.close()
    finished = time.perf_counter()
    report = dict(stage=a.stage, events=final['events'], head=final['head'], snapshot=digest(final),
        controls=controls, phases=dict(authentication_replay=replayed-started, learning_and_close=finished-replayed),
        wall_seconds=finished-started, journal=sha((root/'experience.sqlite3').read_bytes()),
        serving=False, scientific_admission=False)
    write_committed(root/(a.stage+'-report.json'), report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    run_cli(main)
