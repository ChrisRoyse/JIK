"""Restore the existing native-memory lineage with generic cohort intake enabled."""
from functools import partial
from pathlib import Path

from audit.corrective_open import open_session
from audit.fresh_learning_open import sources
from audit.proposal_recognition import read
from audit import repository_native_open,state_cycle_open,successor_residual_open as prior_open
from claims.codata_source import check,sha
from model.repository_cohort_memory import CohortMemoryLoop
from model.successor_residual_data import ResidualData
from senses.trajectory_population import read_pinned


def authenticate(config):
    parent = Path(config['source'])
    proof = read_pinned(Path(config['proof']),config['proof_pin'])
    for name,pin in config['parent_files'].items():
        check(sha((parent/name).read_bytes()) == pin,'COHORT_MEMORY_PARENT_PIN')
    check(config['parent_files']['snapshot-final.json'] == proof['snapshot'] and
          config['parent_files']['experience.sqlite3'] == proof['journal'] and
          proof['cold_replay_equal'],'COHORT_MEMORY_COLD_PARENT')
    for name,pin in proof['immutable_after'].items():
        check(sha(Path(name).read_bytes()) == pin,'COHORT_MEMORY_ANCESTOR_PIN')
    return repository_native_open.authenticate(read(parent/'config.json'))


prepare = state_cycle_open.prepare


def open_memory(recipes,root,saved):
    learning,previous = recipes
    original,sessions = prior_open.open_successor(learning,previous,root/'parent',
        read(Path(learning['source'])/'snapshot-final.json'))
    data = ResidualData(original,original.transfer_data,learning['spec'])
    prospective = sources(previous)[2]
    session,_ = open_session(Path(prospective['source']),prospective['source_pin'],root,saved,
        loop_type=partial(CohortMemoryLoop,learning_base=original.learning_base,
            increment_base=original.increment_base,transfer_data=original.transfer_data,
            attempt_data=data,attempt_spec=learning['spec']))
    return session,(session,*sessions)
