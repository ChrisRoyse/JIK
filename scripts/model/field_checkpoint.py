"""Versioned native-world weights and per-parameter Adam clocks; legacy is immutable."""
import copy
import io
import json
from pathlib import Path

import torch

from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from model.constellation_learning import configure, frozen_pin, trainable
from model.loop_training_checkpoint import put, finite, fingerprint, tensor_fingerprint
from model.native_world import NativeWorld as PreviousWorld
from model.field_world import FieldWorld as NativeWorld
from model import world_checkpoint, native_residual_checkpoint
from production.publication import sync_dir


def code_pins():
    return dict(native_residual_checkpoint.code_pins(), **{n:sha((Path(__file__).parent/n).read_bytes()) for n in
            ('field_native.py','field_world.py','field_checkpoint.py','field_learning.py','field_retention.py')})


def optimizer_for(model, plan):
    configure(model)
    return torch.optim.AdamW([p for _,p in trainable(model)],lr=plan['learning_rate'],
                             weight_decay=plan['weight_decay'],foreach=False,fused=False)


def validate(model, optimizer, data, step):
    check(type(step) is int and 0 <= step <= data.plan['steps'], 'NATIVE_WORLD_CURSOR')
    named = trainable(model)
    offsets = data.binding['native_world']['offsets']
    check(set(offsets) == {n for n,_ in named}, 'NATIVE_WORLD_PARAMETER_ORDER')
    check(type(optimizer) is torch.optim.AdamW and len(optimizer.param_groups) == 1, 'NATIVE_WORLD_OPTIMIZER')
    group = optimizer.param_groups[0]
    expected = dict(lr=data.plan['learning_rate'],weight_decay=data.plan['weight_decay'],
        betas=(.9,.999),eps=1e-8,amsgrad=False,maximize=False,foreach=False,fused=False,
        capturable=False,differentiable=False)
    check(all(group[k] == v for k,v in expected.items()) and len(group['params']) == len(named)
          and all(p is q for (_,p),q in zip(named,group['params'])), 'NATIVE_WORLD_OPTIMIZER')
    check(len(optimizer.state) == sum(offsets[n]+step > 0 for n,_ in named), 'NATIVE_WORLD_OPTIMIZER_COVERAGE')
    for name,parameter in named:
        clock = offsets[name]+step
        if clock == 0:
            check(parameter not in optimizer.state, 'NATIVE_WORLD_NEW_MOMENTS')
            continue
        state = optimizer.state[parameter]
        check(set(state) == {'step','exp_avg','exp_avg_sq'} and float(state['step']) == clock,
              'NATIVE_WORLD_CURSOR')
        check(all(state[k].shape == parameter.shape and state[k].dtype == parameter.dtype
                  and state[k].device == parameter.device for k in ('exp_avg','exp_avg_sq'))
              and finite(state), 'NATIVE_WORLD_MOMENTS')
    model.check_reference()
    check(frozen_pin(model) == data.frozen, 'NATIVE_WORLD_FROZEN_DRIFT')


def start(data):
    if data.field_predecessor is not None:
        root,pin,m = data.field_predecessor
        model,opt,_ = load(root,data,expected=pin,historical=m)
        validate(model,opt,data,0)
        return model,opt,dict(kind='field_continuation',pin=pin)
    from model.native_world_learning import NativeData
    previous = NativeData(data.session,train=data.plan['train'],evaluate=data.plan['evaluate'],
                          continuation=data.continuation_name)
    old,old_opt,continuity = native_residual_checkpoint.start(previous)
    model = NativeWorld.from_world(old)
    optimizer = optimizer_for(model,data.plan)
    old_params = dict(trainable(old))
    new_params = dict(trainable(model))
    for name,parameter in old_params.items():
        optimizer.state[new_params[name]] = copy.deepcopy(old_opt.state[parameter])
    check(all(torch.equal(p,new_params[n]) for n,p in old_params.items()), 'NATIVE_WORLD_MIGRATION_WEIGHTS')
    old_states = {n:old_opt.state[p] for n,p in old_params.items()}
    new_states = {n:optimizer.state[new_params[n]] for n in old_params}
    check(digest(tensor_fingerprint(old_states)) == digest(tensor_fingerprint(new_states)),
          'NATIVE_WORLD_MIGRATION_MOMENTS')
    validate(model,optimizer,data,0)
    return model,optimizer,dict(kind='native_to_fields_migration',continuity=continuity,
        retained_parameters=len(old_params), new_parameters=len(new_params)-len(old_params),
        retained_moments=digest(tensor_fingerprint(old_states)))


def save(root, model, optimizer, data, *, step, parent):
    root = Path(root)
    data.fresh()
    validate(model,optimizer,data,step)
    check(not root.exists(), 'NATIVE_WORLD_CHECKPOINT_EXISTS')
    weights = model.state_dict()
    training = dict(optimizer=optimizer.state_dict(),cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(model.core.initial.device))
    check(sum(v.numel()*v.element_size() for v in weights.values()) <= model.config.checkpoint_bytes,
          'NATIVE_WORLD_CHECKPOINT_BUDGET')
    payloads = {}
    for name,value in (('weights.pt',weights),('training.pt',training)):
        buf = io.BytesIO()
        torch.save(value,buf)
        payloads[name] = buf.getvalue()
    check(sum(map(len,payloads.values())) <= model.config.checkpoint_bytes, 'NATIVE_WORLD_CHECKPOINT_BUDGET')
    m = dict(schema='field-world-learning-v1',binding=data.binding,step=step,parent=parent,
        frozen=data.frozen,names=[n for n,_ in trainable(model)],fingerprint=fingerprint(model,optimizer),
        offsets=data.binding['native_world']['offsets'],
        files={n:dict(sha256=sha(v),bytes=len(v)) for n,v in payloads.items()},code=code_pins(),
        torch_version=str(torch.__version__),serving=False,scientific_admission=False)
    root.mkdir(parents=True)
    for name,raw in payloads.items():
        put(root/name,raw)
    raw = canonical(m)
    put(root/'manifest.json',raw)
    put(root/'COMMITTED',sha(raw).encode())
    sync_dir(root)
    sync_dir(root.parent)
    return sha(raw)


def load(root, data, *, expected, historical=None):
    root = Path(root)
    data.fresh()
    check((root/'manifest.json').stat().st_size <= 1 << 20, 'NATIVE_WORLD_CHECKPOINT_BUDGET')
    raw = (root/'manifest.json').read_bytes()
    check(sha(raw) == expected == (root/'COMMITTED').read_text(), 'NATIVE_WORLD_CHECKPOINT_PIN')
    m = json.loads(raw)
    check(raw == canonical(m) and m['schema'] == 'field-world-learning-v1'
          and m['code'] == code_pins() and m['torch_version'] == str(torch.__version__)
          and m['serving'] is False and m['scientific_admission'] is False, 'NATIVE_WORLD_CHECKPOINT_CONTRACT')
    binding = data.binding if historical is None else historical['binding']
    check(m['binding'] == binding and m['frozen'] == data.frozen and (historical is None or m == historical),
          'NATIVE_WORLD_CHECKPOINT_BINDING')
    foundation = binding['native_world']['foundation']
    old,_,_ = world_checkpoint.load(Path(foundation['root']),expected=foundation['pin'],
        standard=binding['standard'],world=binding['world'],device=data.session.device)
    model = NativeWorld.from_world(PreviousWorld.from_world(old))
    check(set(m['files']) == {'weights.pt','training.pt'} and
          sum(v['bytes'] for v in m['files'].values()) <= model.config.checkpoint_bytes,
          'NATIVE_WORLD_CHECKPOINT_BUDGET')
    payload = {}
    for name,entry in m['files'].items():
        check((root/name).stat().st_size == entry['bytes'], 'NATIVE_WORLD_CHECKPOINT_BYTES')
        raw = (root/name).read_bytes()
        check(sha(raw) == entry['sha256'], 'NATIVE_WORLD_CHECKPOINT_BYTES')
        payload[name] = torch.load(io.BytesIO(raw),weights_only=True,map_location='cpu')
    check(finite(payload['weights.pt']) and finite(payload['training.pt']['optimizer']), 'NATIVE_WORLD_NONFINITE')
    model.load_state_dict(payload['weights.pt'],strict=True)
    optimizer = optimizer_for(model,data.plan)
    check(m['names'] == [n for n,_ in trainable(model)], 'NATIVE_WORLD_PARAMETER_ORDER')
    optimizer.load_state_dict(payload['training.pt']['optimizer'])
    if historical is None:
        validate(model,optimizer,data,m['step'])
    else:
        check(all(data.binding['native_world']['offsets'][n] == m['offsets'][n]+m['step'] for n in m['names']),
              'NATIVE_WORLD_CONTINUATION_CLOCK')
        validate(model,optimizer,data,0)
    check(fingerprint(model,optimizer) == m['fingerprint'], 'NATIVE_WORLD_CHECKPOINT_READBACK')
    torch.set_rng_state(payload['training.pt']['cpu_rng'])
    torch.cuda.set_rng_state(payload['training.pt']['cuda_rng'],model.core.initial.device)
    return model,optimizer,m
