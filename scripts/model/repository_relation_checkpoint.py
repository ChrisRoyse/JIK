"""A bounded delta checkpoint, authenticated against the restored active parent."""
import io
import json
from pathlib import Path

import torch

from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from model.loop_training_checkpoint import finite, fingerprint, put, tensor_fingerprint
from model.repository_relation_world import BRANCH, RepositoryWorld, configure
from production.publication import sync_dir

RECIPE = dict(steps_per_increment=8,learning_rate=.001,weight_decay=.01,optimizer='AdamW',
              gradient_clip=1.,selection='final_step_only')


def code_pins():
    root = Path(__file__).parent
    return {n:sha((root/n).read_bytes()) for n in ('repository_relation_input.py',
        'repository_relation_world.py','repository_relation_data.py','repository_relation_checkpoint.py')}


def frozen(model):
    return digest(tensor_fingerprint({n:t for n,t in model.state_dict().items()
                                      if not n.startswith('query.'+BRANCH+'.')}))


def validate(model, optimizer, data, step):
    check(frozen(model) == frozen(data.parent), 'RELATION_PARENT_DRIFT')
    expected = [p for n,p in model.named_parameters() if n.startswith('query.'+BRANCH+'.')]
    check([p for p in model.parameters() if p.requires_grad] == expected, 'RELATION_ELIGIBLE_SCOPE')
    check(len(optimizer.param_groups)==1 and optimizer.param_groups[0]['params']==expected,
          'RELATION_OPTIMIZER_SCOPE')
    for p in expected:
        check(bool(torch.isfinite(p).all()), 'RELATION_PARAMETER_FINITE')
        state = optimizer.state.get(p,{})
        check((not state and step==0) or (int(state.get('step',-1))==step and finite(state)),
              'RELATION_OPTIMIZER_CLOCK')
    model.check_reference()


def start(data):
    model = RepositoryWorld.from_world(data.parent)
    optimizer = configure(model)
    validate(model,optimizer,data,0)
    return model,optimizer


def save(root,model,optimizer,data,step,predecessor):
    root = Path(root)
    data.fresh()
    validate(model,optimizer,data,step)
    check(not root.exists(), 'RELATION_CHECKPOINT_EXISTS')
    torch.cuda.synchronize(model.core.initial.device)
    stream = io.BytesIO()
    torch.save(dict(branch=model.query[BRANCH].state_dict(),optimizer=optimizer.state_dict(),
        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(model.core.initial.device)),stream)
    raw = stream.getvalue()
    check(len(raw) <= model.config.checkpoint_bytes, 'RELATION_CHECKPOINT_BUDGET')
    manifest = dict(schema='repository-relation-checkpoint-v1',binding=data.binding,recipe=RECIPE,
        code=code_pins(),step=step,predecessor=predecessor,frozen=frozen(model),legacy=model.legacy,
        fingerprint=fingerprint(model,optimizer),payload=dict(bytes=len(raw),sha256=sha(raw)),
        torch_version=str(torch.__version__),serving=False,scientific_admission=False)
    root.mkdir(parents=True)
    put(root/'delta.pt',raw)
    body = canonical(manifest)
    put(root/'manifest.json',body)
    put(root/'COMMITTED',sha(body).encode())
    sync_dir(root)
    sync_dir(root.parent)
    return sha(body)


def load(root,data,expected):
    root = Path(root)
    data.fresh()
    body = (root/'manifest.json').read_bytes()
    check(sha(body)==expected==(root/'COMMITTED').read_text(), 'RELATION_CHECKPOINT_PIN')
    m = json.loads(body)
    check(body==canonical(m) and m['schema']=='repository-relation-checkpoint-v1' and
          m['binding']==data.binding and m['code']==code_pins() and m['recipe']==RECIPE and
          m['torch_version']==str(torch.__version__) and not m['serving'] and not m['scientific_admission'],
          'RELATION_CHECKPOINT_CONTRACT')
    model,optimizer = start(data)
    check(m['frozen']==frozen(model) and m['legacy']==model.legacy, 'RELATION_CHECKPOINT_PARENT')
    check((root/'delta.pt').stat().st_size==m['payload']['bytes'] <= model.config.checkpoint_bytes,
          'RELATION_CHECKPOINT_BUDGET')
    raw = (root/'delta.pt').read_bytes()
    check(sha(raw)==m['payload']['sha256'], 'RELATION_CHECKPOINT_BYTES')
    state = torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    check(finite(state['branch']) and finite(state['optimizer']), 'RELATION_CHECKPOINT_FINITE')
    model.query[BRANCH].load_state_dict(state['branch'],strict=True)
    optimizer.load_state_dict(state['optimizer'])
    validate(model,optimizer,data,m['step'])
    check(fingerprint(model,optimizer)==m['fingerprint'], 'RELATION_CHECKPOINT_READBACK')
    torch.set_rng_state(state['cpu_rng'])
    torch.cuda.set_rng_state(state['cuda_rng'],model.core.initial.device)
    return model,optimizer,m
