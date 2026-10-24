"""Reconstruct the immutable pre-learning prefix and its exact historical parent."""
from functools import partial
from pathlib import Path
import shutil

from audit.corrective_open import open_session
from audit.proposal_recognition import authenticate, evidence, read
from claims.codata_source import check
from model.corrective_learning import CorrectiveData
from model.fresh_learning_session import FreshLearningLoop
from model.response_cycles import CycleData, CycleLoop


def sources(config):
    source = Path(config['source'])
    seal = authenticate(Path(config['proof']),config['proof_pin'])
    for name in ('experience.sqlite3','snapshot-final.json','config.json'):
        check(evidence(source/name) == seal['files']['session/'+name],'FRESH_OPEN_SOURCE')
    echo = read(source/'config.json')
    upstream = read(Path(echo['source'])/'config.json')
    prospective = read(Path(upstream['prospective'])/'config.json')
    check(config['spec']['checkpoint'] == upstream['experiment']['corrected'],'FRESH_OPEN_PARENT')
    return source,upstream,prospective


def prepare(config, root):
    source,upstream,prospective = sources(config)
    check(not root.exists(),'FRESH_OPEN_EXISTS')
    root.mkdir()
    for name,path in (('base',Path(prospective['source'])),('parent',Path(upstream['source']))):
        (root/name/'candidates').mkdir(parents=True)
        shutil.copyfile(path/'experience.sqlite3',root/name/'experience.sqlite3')
    (root/'candidates').mkdir()
    shutil.copyfile(source/'experience.sqlite3',root/'experience.sqlite3')
    return source,upstream,prospective


def open_fresh(config, root, *, saved=None, loop_type=FreshLearningLoop, sources_value=None):
    source,upstream,prospective = sources(config) if sources_value is None else sources_value
    ancestor = Path(prospective['source'])
    original,_ = open_session(ancestor,prospective['source_pin'],root/'base',read(ancestor/'snapshot-final.json'))
    base = CorrectiveData(original,prospective['experiment']['correction'])
    parent,_ = open_session(ancestor,prospective['source_pin'],root/'parent',
        read(Path(upstream['source'])/'snapshot-final.json'),loop_type=partial(CycleLoop,learning_base=base))
    increment = CycleData(parent,base,upstream['experiment']['learning'])
    session,_ = open_session(ancestor,prospective['source_pin'],root,
        read(source/'snapshot-final.json') if saved is None else saved,
        loop_type=partial(loop_type,learning_base=base,increment_base=increment))
    return session,base,(session,parent,original)
