"""Authenticate the historical head, then recover a pending capability commit."""
from functools import partial

from audit.repository_live_bootstrap import open_current
from model.capability_live_memory import LiveCapabilityMemoryLoop


def open_live(recipes, root, population):
    session, sessions = open_current(recipes, root, population,
        loop_type=partial(LiveCapabilityMemoryLoop, recover_on_open=False))
    try:
        if session.task_spec is not None: session.synchronize_tasks()
        return session, sessions
    except BaseException:
        for item in sessions: item.close()
        raise
