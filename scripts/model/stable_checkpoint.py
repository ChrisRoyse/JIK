"""Checkpoint the separated computation, frozen parent matrices and projector buffers."""
import io
import json
from pathlib import Path

import torch

from claims.codata_source import check, sha
from claims.episodic_memory import canonical
from model.loop_training_checkpoint import finite, fingerprint, put
from model import field_checkpoint, response_basis, stable_response
from production.publication import sync_dir


def code_pins():
    return {n:sha((Path(__file__).parent/n).read_bytes()) for n in
            ('stable_checkpoint.py','stable_response.py','field_checkpoint.py')}


def validate(model, optimizer, data, step):
    field_checkpoint.validate(model,optimizer,data,step)
    check(stable_response.buffers(model) == data.binding['stable_response']['static_buffers'],
          'STABLE_CHECKPOINT_BUFFERS')


def start(data):
    model,optimizer,migration = field_checkpoint.start(data)
    before = fingerprint(model,optimizer)
    stable_response.install(model,data.response_basis)
    validate(model,optimizer,data,0)
    return model,optimizer,dict(kind=stable_response.RULE,parent=migration,plain_parent=before,
        static_buffers=data.binding['stable_response']['static_buffers'])


def save(root, model, optimizer, data, *, step, parent):
    root = Path(root)
    data.fresh()
    validate(model,optimizer,data,step)
    check(not root.exists(),'STABLE_CHECKPOINT_EXISTS')
    weights = model.state_dict()
    training = dict(optimizer=optimizer.state_dict(),cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(model.core.initial.device))
    payloads = {}
    for name,value in (('weights.pt',weights),('training.pt',training)):
        stream = io.BytesIO()
        torch.save(value,stream)
        payloads[name] = stream.getvalue()
    check(sum(map(len,payloads.values())) <= model.config.checkpoint_bytes,'STABLE_CHECKPOINT_BUDGET')
    manifest = dict(schema='stable-response-learning-v1',binding=data.binding,step=step,parent=parent,
        frozen=data.frozen,names=[n for n,_ in field_checkpoint.trainable(model)],
        fingerprint=fingerprint(model,optimizer),offsets=data.binding['native_world']['offsets'],
        files={n:dict(bytes=len(v),sha256=sha(v)) for n,v in payloads.items()},code=code_pins(),
        torch_version=str(torch.__version__),serving=False,scientific_admission=False)
    root.mkdir(parents=True)
    for name,raw in payloads.items():
        put(root/name,raw)
    raw = canonical(manifest)
    put(root/'manifest.json',raw)
    put(root/'COMMITTED',sha(raw).encode())
    sync_dir(root)
    sync_dir(root.parent)
    return sha(raw)


def load(root, data, *, expected):
    root = Path(root)
    data.fresh()
    check((root/'manifest.json').stat().st_size <= 1 << 20,'STABLE_CHECKPOINT_BUDGET')
    raw = (root/'manifest.json').read_bytes()
    check(sha(raw) == expected == (root/'COMMITTED').read_text(),'STABLE_CHECKPOINT_PIN')
    manifest = json.loads(raw)
    check(raw == canonical(manifest) and manifest['schema'] == 'stable-response-learning-v1'
          and manifest['code'] == code_pins() and manifest['torch_version'] == str(torch.__version__)
          and manifest['serving'] is False and manifest['scientific_admission'] is False,
          'STABLE_CHECKPOINT_CONTRACT')
    check(manifest['binding'] == data.binding and manifest['frozen'] == data.frozen,
          'STABLE_CHECKPOINT_BINDING')
    model,optimizer,_ = start(data)
    check(set(manifest['files']) == {'weights.pt','training.pt'} and
          sum(v['bytes'] for v in manifest['files'].values()) <= model.config.checkpoint_bytes,
          'STABLE_CHECKPOINT_BUDGET')
    payloads = {}
    for name,item in manifest['files'].items():
        check((root/name).stat().st_size == item['bytes'],'STABLE_CHECKPOINT_BYTES')
        content = (root/name).read_bytes()
        check(sha(content) == item['sha256'],'STABLE_CHECKPOINT_BYTES')
        payloads[name] = torch.load(io.BytesIO(content),weights_only=True,map_location='cpu')
    weights,training = payloads['weights.pt'],payloads['training.pt']
    check(finite(weights) and finite(training['optimizer']),'STABLE_CHECKPOINT_FINITE')
    template = model.state_dict()
    check(set(weights) == set(template) and all(weights[n].shape == v.shape and weights[n].dtype == v.dtype
          for n,v in template.items()),'STABLE_CHECKPOINT_STATE')
    model.load_state_dict(weights,strict=True)
    for module in response_basis.modules(model).values():
        module.seal()
    check(manifest['names'] == [n for n,_ in field_checkpoint.trainable(model)],'STABLE_CHECKPOINT_ORDER')
    optimizer.load_state_dict(training['optimizer'])
    validate(model,optimizer,data,manifest['step'])
    check(fingerprint(model,optimizer) == manifest['fingerprint'],'STABLE_CHECKPOINT_READBACK')
    torch.set_rng_state(training['cpu_rng'])
    torch.cuda.set_rng_state(training['cuda_rng'],model.core.initial.device)
    return model,optimizer,manifest
