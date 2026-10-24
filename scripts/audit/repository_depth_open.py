"""Restore the existing lineage with depth intake as a versioned event extension."""
from functools import partial
from pathlib import Path

from audit.corrective_open import open_session
from audit.fresh_learning_open import sources
from audit.repository_memory_readback import read
from audit import successor_residual_open as prior_open
from model.repository_depth_memory import DepthMemoryLoop
from model.successor_residual_data import ResidualData


def open_memory(recipes,root,saved):
    learning,previous=recipes
    original,sessions=prior_open.open_successor(learning,previous,root/'parent',
        read(Path(learning['source'])/'snapshot-final.json'))
    data=ResidualData(original,original.transfer_data,learning['spec'])
    prospective=sources(previous)[2]
    session,_=open_session(Path(prospective['source']),prospective['source_pin'],root,saved,
        loop_type=partial(DepthMemoryLoop,learning_base=original.learning_base,
            increment_base=original.increment_base,transfer_data=original.transfer_data,
            attempt_data=data,attempt_spec=learning['spec']))
    return session,(session,*sessions)
