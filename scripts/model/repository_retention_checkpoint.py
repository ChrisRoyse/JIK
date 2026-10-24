"""Bounded repository-retention checkpoints including transactional optimizer clocks."""
import io
import json
from pathlib import Path

import torch

from claims.codata_source import check, sha
from claims.episodic_memory import canonical
from model.loop_training_checkpoint import finite, fingerprint, put
from model import repository_relation_checkpoint as parent
from model.repository_relation_world import BRANCH
from model.repository_retention import POLICY
from production.publication import sync_dir


def code_pins():
    root = Path(__file__).parent
    return dict(parent.code_pins(), **{n:sha((root/n).read_bytes()) for n in (
        'repository_retention.py','repository_retention_checkpoint.py','retention_projection.py','retained_update.py')})


def save(root,model,optimizer,retention,step,attempt,predecessor):
    root = Path(root)
    retention.checked(); retention.data.fresh()
    parent.validate(model,optimizer,retention.data,step)
    check(not root.exists(), 'RETENTION_CHECKPOINT_EXISTS')
    stream = io.BytesIO()
    torch.save(dict(branch=model.query[BRANCH].state_dict(),optimizer=optimizer.state_dict(),
        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(model.core.initial.device)),stream)
    raw = stream.getvalue()
    check(len(raw) <= model.config.checkpoint_bytes, 'RETENTION_CHECKPOINT_BUDGET')
    manifest = dict(schema='repository-retention-checkpoint-v1',contract=retention.contract,
        binding=retention.data.binding,code=code_pins(),policy=POLICY,step=step,attempt=attempt,
        predecessor=predecessor,frozen=parent.frozen(model),legacy=model.legacy,
        fingerprint=fingerprint(model,optimizer),payload=dict(bytes=len(raw),sha256=sha(raw)),
        torch_version=str(torch.__version__),serving=False,scientific_admission=False)
    root.mkdir(parents=True)
    put(root/'delta.pt',raw)
    body=canonical(manifest)
    put(root/'manifest.json',body);put(root/'COMMITTED',sha(body).encode())
    sync_dir(root);sync_dir(root.parent)
    return sha(body)


def load(root,retention,expected):
    root=Path(root)
    retention.checked();retention.data.fresh()
    body=(root/'manifest.json').read_bytes()
    check(sha(body)==expected==(root/'COMMITTED').read_text(), 'RETENTION_CHECKPOINT_PIN')
    m=json.loads(body)
    check(body==canonical(m) and m['schema']=='repository-retention-checkpoint-v1' and
          m['contract']==retention.contract and m['binding']==retention.data.binding and
          m['code']==code_pins() and m['policy']==POLICY and m['torch_version']==str(torch.__version__) and
          not m['serving'] and not m['scientific_admission'], 'RETENTION_CHECKPOINT_CONTRACT')
    model,optimizer=parent.start(retention.data)
    check(m['frozen']==parent.frozen(model) and m['legacy']==model.legacy, 'RETENTION_CHECKPOINT_PARENT')
    check((root/'delta.pt').stat().st_size==m['payload']['bytes'] <= model.config.checkpoint_bytes,
          'RETENTION_CHECKPOINT_BUDGET')
    raw=(root/'delta.pt').read_bytes()
    check(sha(raw)==m['payload']['sha256'], 'RETENTION_CHECKPOINT_BYTES')
    state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    check(finite(state), 'RETENTION_CHECKPOINT_FINITE')
    model.query[BRANCH].load_state_dict(state['branch'],strict=True)
    optimizer.load_state_dict(state['optimizer'])
    parent.validate(model,optimizer,retention.data,m['step'])
    check(fingerprint(model,optimizer)==m['fingerprint'], 'RETENTION_CHECKPOINT_REPLAY')
    torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state(state['cuda_rng'],model.core.initial.device)
    return model,optimizer,m
