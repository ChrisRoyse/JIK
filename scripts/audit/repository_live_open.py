"""Open the retained model lineage with automatic repository update events."""
from audit.repository_live_bootstrap import open_current
from claims.codata_source import check


def recover_updates(session):
    """Startup resumes a durable pending publication before admitting new work."""
    if session.live_spec is not None:return session.synchronize_repository()
    return False


def open_memory(recipes,root,saved,population,*,recover=True):
    session,sessions=open_current(recipes,root,population)
    try:
        if saved is not None:check(session.snapshot()==saved,'LIVE_BOOTSTRAP_SNAPSHOT')
        if recover:recover_updates(session)
        return session,sessions
    except BaseException:
        for prior in sessions:prior.close()
        raise
