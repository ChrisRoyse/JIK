"""Explicit, pinned compatibility transition for the retained native prototype."""
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from claims.codata_source import check,sha
from model.native_world_checkpoint import code,load
from senses.contract import canonical,digest

OLD='1e224a51b6c1a30b9ce4fe5c610714d5a8dc7db0'
CURRENT='d6216e82ab6d3160aae01270e0f200ad735cb295'
CHANGED={'constellation_world.py','identity_loop.py','native_identity_world.py','native_world_checkpoint.py'}


def pins(commit,names):
    raw=subprocess.check_output(['git','archive',commit,'--','scripts/model'])
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        return {n:sha(archive.extractfile('scripts/model/'+n).read()) for n in names}


def transition(source,target,model,*,expected,binding):
    manifest=json.loads((source/'manifest.json').read_bytes())
    check(digest(manifest)==expected==(source/'COMMITTED').read_text(),'OBSERVATION_TRANSITION_PIN')
    check(sha((source/'state.pt').read_bytes())==manifest['sha256'] and
          (source/'state.pt').stat().st_size==manifest['bytes']<=model.config.checkpoint_bytes,
          'OBSERVATION_TRANSITION_BYTES')
    before=manifest['code'];after=code()
    check(before==pins(OLD,before) and after==pins(CURRENT,after),'OBSERVATION_TRANSITION_CODE')
    check({n for n in before if before[n]!=after.get(n)}==CHANGED and
          set(after)-set(before)=={'text_episode_input.py'},'OBSERVATION_TRANSITION_DELTA')
    target.mkdir(exist_ok=False)
    shutil.copy2(source/'state.pt',target/'state.pt')
    rebound=dict(manifest,code=after)
    (target/'manifest.json').write_bytes(canonical(rebound));pin=digest(rebound)
    (target/'COMMITTED').write_text(pin)
    # The unchanged reader checks every architecture/source/identity/tensor contract.
    states=load(target,model,expected=pin,binding=binding)
    check(sha((target/'state.pt').read_bytes())==manifest['sha256'],'OBSERVATION_TRANSITION_PAYLOAD_CHANGED')
    return states,dict(source=str(source),before=expected,after=pin,
        source_commit=OLD,runtime_commit=CURRENT,changed=sorted(CHANGED),
        added=['text_episode_input.py'],payload_sha256=manifest['sha256'],weights_unchanged=True)
