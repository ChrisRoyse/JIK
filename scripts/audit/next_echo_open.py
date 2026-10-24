"""Keep the immutable learning prefix separate from the next observed successor."""
from functools import partial
from pathlib import Path
import shutil

from audit.corrective_open import open_session
from audit.fresh_learning_open import open_fresh, prepare as prepare_learning, sources
from audit.next_open import recipe
from audit.proposal_recognition import authenticate as authenticate_seal, evidence, read
from claims.codata_source import check
from model.brier_region_session_v3 import BrierData, BrierLoop
from model.next_echo import NextEchoLoop


def authenticate(config):
    source = Path(config['source'])
    seal = authenticate_seal(Path(config['proof']),config['proof_pin'])
    for name in ('config.json','snapshot-final.json','experience.sqlite3','candidates/brier-1/manifest.json'):
        check(evidence(source/name) == seal['files']['session/'+name],'NEXT_ECHO_PREFIX_BYTES')
    prior = read(source/'config.json')
    return prior,recipe(prior)


def prepare(config, previous, root):
    check(not root.exists(),'PROPOSAL_ECHO_OUTPUT_EXISTS')
    root.mkdir()
    prepare_learning(previous,root/'learning')
    source = Path(config['source'])
    shutil.copyfile(source/'experience.sqlite3',root/'experience.sqlite3')
    shutil.copytree(source/'candidates/brier-1',root/'candidates/brier-1')


def open_echo(config, previous, root, saved):
    resolved = sources(previous)
    prefix,base,sessions = open_fresh(previous,root/'learning',loop_type=BrierLoop,sources_value=resolved)
    data = BrierData(prefix,base,previous['spec'])
    prospective = resolved[2]
    ancestor = Path(prospective['source'])
    session,_ = open_session(ancestor,prospective['source_pin'],root,saved,
        loop_type=partial(NextEchoLoop,learning_base=base,increment_base=prefix.increment_base,transfer_data=data))
    return session,(session,*sessions)
