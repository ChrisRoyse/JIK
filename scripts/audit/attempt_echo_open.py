"""Authenticate the observed attempt while retaining its original training prefix."""
from functools import partial
from pathlib import Path
import shutil

from audit.corrective_open import open_session
from audit.fresh_learning_open import sources
from audit.proposal_recognition import authenticate as authenticate_seal, evidence, read
from audit import attempt_open, successor_residual_open as prior_open
from claims.codata_source import check
from model.attempt_echo import AttemptEchoLoop
from model.successor_residual_data import ResidualData


def authenticate(config):
    source = Path(config['source'])
    seal = authenticate_seal(Path(config['proof']),config['proof_pin'])
    for name in ('config.json','snapshot-final.json','experience.sqlite3','feedback.json',
                 'forecast/forecast.json','candidates/brier-1/manifest.json',
                 'candidates/successor-1/manifest.json'):
        check(evidence(source/name) == seal['files']['session/'+name],'ATTEMPT_ECHO_PREFIX_BYTES')
    prior = read(source/'config.json')
    learning,previous = attempt_open.recipe(prior)
    return prior,(learning,previous)


def prepare(config, recipes, root):
    learning,previous = recipes
    check(not root.exists(),'ATTEMPT_ECHO_OUTPUT_EXISTS')
    root.mkdir()
    prior_open.prepare(learning,previous,root/'parent')
    source = Path(config['source'])
    shutil.copyfile(source/'experience.sqlite3',root/'experience.sqlite3')
    for name in ('brier-1','successor-1'):
        shutil.copytree(source/'candidates'/name,root/'candidates'/name)


def open_echo(config, recipes, root, saved):
    learning,previous = recipes
    original,sessions = prior_open.open_successor(learning,previous,root/'parent',
        read(Path(learning['source'])/'snapshot-final.json'))
    data = ResidualData(original,original.transfer_data,learning['spec'])
    prospective = sources(previous)[2]
    ancestor = Path(prospective['source'])
    session,_ = open_session(ancestor,prospective['source_pin'],root,saved,
        loop_type=partial(AttemptEchoLoop,learning_base=original.learning_base,
            increment_base=original.increment_base,transfer_data=original.transfer_data,
            attempt_data=data,attempt_spec=learning['spec']))
    return session,(session,*sessions)
