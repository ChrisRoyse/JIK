"""Keep the original learning head separate from its candidate and new forecast."""
from functools import partial
from pathlib import Path
import shutil

from audit.corrective_open import open_session
from audit.fresh_learning_open import sources
from audit.proposal_recognition import authenticate, evidence, read
from audit import successor_residual_open as prior_open
from claims.codata_source import check,sha
from model.attempt_experiment import AttemptLoop
from model.successor_residual_data import ResidualData


def recipe(config):
    source = Path(config['source'])
    seal = authenticate(Path(config['proof']['root']),config['proof']['pin'])
    for name in ('config.json','snapshot-final.json','experience.sqlite3',
                 'candidates/brier-1/manifest.json','candidates/successor-1/manifest.json'):
        check(evidence(source/name) == seal['files']['session/'+name], 'ATTEMPT_SOURCE_BYTES')
    prior = read(source/'config.json')
    spec = config['experiment']
    check(spec['learning'] == prior['spec'] and spec['before'] == prior['spec']['checkpoint'],
          'ATTEMPT_SOURCE_RECIPE')
    corrected = spec['corrected']
    check(corrected['root'] == str(source/'candidates/successor-1') and
          sha((Path(corrected['root'])/'manifest.json').read_bytes()) == corrected['pin'],
          'ATTEMPT_CANDIDATE_CHECKPOINT')
    proof = config['input_proof']
    input_seal = authenticate(Path(proof['root']),proof['pin'])
    native = spec['native']
    check(sha((Path(native['native'])/'report.json').read_bytes()) == native['native_pin']
          == input_seal['files']['native/report.json']['sha256'], 'ATTEMPT_NATIVE_PIN')
    reader = config['input_readback']
    check(sha(Path(reader['path']).read_bytes()) == reader['pin']
          == input_seal['files']['attempt-input-readback-20261021.json']['sha256']
          and read(Path(reader['path']))['input'] == native, 'ATTEMPT_INPUT_READBACK')
    return prior,prior_open.authenticate(prior)


def prepare(config,prior,previous,root):
    check(not root.exists(), 'ATTEMPT_OUTPUT_EXISTS')
    root.mkdir()
    prior_open.prepare(prior,previous,root/'parent')
    source = Path(config['source'])
    shutil.copyfile(source/'experience.sqlite3',root/'experience.sqlite3')
    for name in ('brier-1','successor-1'):
        shutil.copytree(source/'candidates'/name,root/'candidates'/name)


def open_attempt(config,prior,previous,root,saved):
    original,sessions = prior_open.open_successor(prior,previous,root/'parent',
        read(Path(prior['source'])/'snapshot-final.json'))
    data = ResidualData(original,original.transfer_data,prior['spec'])
    resolved = sources(previous)
    prospective = resolved[2]
    ancestor = Path(prospective['source'])
    session,_ = open_session(ancestor,prospective['source_pin'],root,saved,
        loop_type=partial(AttemptLoop,learning_base=original.learning_base,
            increment_base=original.increment_base,transfer_data=original.transfer_data,
            attempt_data=data,attempt_spec=prior['spec']))
    return session,(session,*sessions)
