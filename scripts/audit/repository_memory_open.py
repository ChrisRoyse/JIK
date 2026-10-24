"""Restore the complete prior state-cycle lineage with an additional occurrence reader."""
from functools import partial
from pathlib import Path

from audit.corrective_open import open_session
from audit.fresh_learning_open import sources
from audit.proposal_recognition import read
from audit import state_cycle_open,successor_residual_open as prior_open
from claims.codata_source import check,sha
from model.repository_memory import RepositoryMemoryLoop
from model.successor_residual_data import ResidualData
from senses.trajectory_population import read_pinned


def authenticate(config):
    parent = Path(config['source'])
    proof = read_pinned(Path(config['proof']),config['proof_pin'])
    for name,pin in config['parent_files'].items():
        check(sha((parent/name).read_bytes()) == pin,'REPOSITORY_MEMORY_PARENT_PIN')
    check(config['parent_files']['snapshot-final.json'] == proof['state']['cold_snapshot'],
          'REPOSITORY_MEMORY_COLD_PARENT')
    for name,pin in proof['state']['checkpoint_files'].items():
        p = Path(name)
        check(p.is_relative_to(parent) and dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes())) == pin,
              'REPOSITORY_MEMORY_PARENT_CHECKPOINT')
    return state_cycle_open.authenticate(read(parent/'config.json'))[1]


def prepare(config,recipes,root):
    state_cycle_open.prepare(config,recipes,root)


def open_memory(recipes,root,saved):
    learning,previous = recipes
    original,sessions = prior_open.open_successor(learning,previous,root/'parent',
        read(Path(learning['source'])/'snapshot-final.json'))
    data = ResidualData(original,original.transfer_data,learning['spec'])
    prospective = sources(previous)[2]
    session,_ = open_session(Path(prospective['source']),prospective['source_pin'],root,saved,
        loop_type=partial(RepositoryMemoryLoop,learning_base=original.learning_base,
            increment_base=original.increment_base,transfer_data=original.transfer_data,
            attempt_data=data,attempt_spec=learning['spec']))
    return session,(session,*sessions)
