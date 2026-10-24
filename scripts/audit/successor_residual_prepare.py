"""Measure migration from the real parent before a successor can be trained."""
import copy
import time

import torch

from audit.proposal_recognition import evidence
from claims.codata_source import check
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from model.loop_training_checkpoint import fingerprint,tensor_fingerprint
from model.successor_box_data import BoxData
from model.successor_residual_data import code_pins
from model import successor_checkpoint,successor_residual,response_basis,field_checkpoint


def prepare_residual(session,config,root):
    started = time.perf_counter()
    data = BoxData(session,session.transfer_data,config['spec'])
    model,optimizer,_ = successor_checkpoint.start(data)
    before = data.evaluate(model)
    original = fingerprint(model,optimizer)
    names = [(n,id(p),p.data_ptr()) for n,p in model.named_parameters()]
    weights = tensor_fingerprint(dict(model.named_parameters()))
    buffer_bytes = sum(v.numel()*v.element_size() for v in model.buffers())
    static = successor_residual.install(model,data.response_basis)
    check(names == [(n,id(p),p.data_ptr()) for n,p in model.named_parameters()] and
          weights == tensor_fingerprint(dict(model.named_parameters())) and
          fingerprint(model,optimizer)['optimizer'] == original['optimizer'],
          'SUCCESSOR_RESIDUAL_PARAMETER_IDENTITY')
    field_checkpoint.validate(model,optimizer,data,0)
    after = data.evaluate(model)
    check(before == after,'SUCCESSOR_RESIDUAL_ZERO_UPDATE_BEHAVIOUR')
    scores,drift = data.response_basis.measure(data,model)
    check(all(v['equal'] for v in drift.values()),'SUCCESSOR_RESIDUAL_ZERO_UPDATE_RESPONSE')
    controls = []
    module = next(iter(response_basis.modules(model).values()))
    for name in ('parent_weight','successor_projector'):
        value = getattr(module,name)
        saved = value.detach().clone()
        with torch.no_grad():
            value.view(-1)[0].add_(1)
        try:
            module.checked()
        except ValueError as error:
            check('SUCCESSOR_RESIDUAL_FROZEN_BUFFER' in str(error),'SUCCESSOR_RESIDUAL_CONTROL_CODE')
        else:
            raise AssertionError('SUCCESSOR_RESIDUAL_MUTATION_ACCEPTED')
        with torch.no_grad():
            value.copy_(saved)
        module.seal()
        controls.append(dict(name='mutated_'+name,code='SUCCESSOR_RESIDUAL_FROZEN_BUFFER'))
    check(successor_residual.buffers(model) == static and data.evaluate(model) == before,
          'SUCCESSOR_RESIDUAL_CONTROL_RESTORE')
    result = dict(rule=successor_residual.RULE,code=code_pins(),parent=data.parent_pin,
        parent_binding=digest(data.binding),basis=config['spec']['basis'],before=before,after=after,
        parent_fingerprint=original,parameter_identity_retained=True,static_buffers=static,
        response_drift=drift,scores=scores,controls=controls,
        added_buffer_bytes=sum(v.numel()*v.element_size() for v in model.buffers())-buffer_bytes,
        checkpoint_limit=model.config.checkpoint_bytes,wall_seconds=time.perf_counter()-started,
        serving=False,scientific_admission=False)
    write_committed(root/'migration.json',result)
    prepared = copy.deepcopy(config)
    prepared['spec']['residual'] = dict(path=str(root/'migration.json'),pin=evidence(root/'migration.json')['sha256'])
    write_committed(root/'prepared-config.json',prepared)
    write_committed(root/'snapshot-prepared.json',session.snapshot())
    return controls
