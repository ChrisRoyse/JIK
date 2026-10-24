"""One source-document update with complete forward history and detached prior KV."""
import torch

from model.loop_training_checkpoint import validate_optimizer
from model.native_population_training import cpu, fingerprint, save
from text.constellation_corpus import file_sha, need


def check_item(item):
    need(item['side'] == 'train', 'DOCUMENT_TRAIN_NONTRAIN')
    need(len(item['frames']) >= 2 and len(item['codes']) == len(item['frames'])-1,
         'DOCUMENT_TRAIN_TRANSITIONS')
    need(fingerprint(item['frames']) == item['native_pin'], 'DOCUMENT_TRAIN_NATIVE_PIN')
    need(fingerprint((item['codes'], item['counts'])) == item['label_pin'], 'DOCUMENT_TRAIN_LABEL_PIN')


def branches(values, roster):
    prefixes = ['encoder.', 'region.']+['projectors.'+n+'.' for n in roster]+['counts.'+n+'.' for n in roster]
    return {p:fingerprint({n:x for n,x in values.items() if n.startswith(p)}) for p in prefixes}


def update(model, optimizer, item, step, policy):
    check_item(item)
    validate_optimizer(model, optimizer, policy, step)
    before = cpu(model.state_dict()); device = model.encoder.query.device
    frames = tuple({n:x.to(device) for n,x in f.items()} for f in item['frames'])
    codes = item['codes'].to(device); counts = {n:x.to(device) for n,x in item['counts'].items()}
    optimizer.zero_grad(set_to_none=True); stream = model.stream(); state = None; total = len(codes)
    loss_sum = 0.; context = 'document:'+str(item['document']); consumed = 0
    for lo in range(0, len(frames), policy['tile']):
        hi = min(lo+policy['tile'], len(frames))
        hidden, state = stream.step(frames[lo:hi], context=context, offset=lo, state=state, detach_history=True)
        stop = min(hi, total); targets = stop-lo
        if targets > 0:
            out = model.loss(hidden[:targets], codes[lo:stop], {n:x[lo:stop] for n,x in counts.items()})
            need(bool(torch.isfinite(out['loss'])), 'DOCUMENT_TRAIN_NONFINITE_LOSS')
            weighted = out['loss']*(targets/total); weighted.backward()
            loss_sum += float(weighted.detach()); consumed += targets
    need(consumed == total and state.next_atom == len(frames), 'DOCUMENT_TRAIN_INCOMPLETE_HISTORY')
    gradients = cpu({n:p.grad for n,p in model.named_parameters()})
    need(all(x is not None and bool(torch.isfinite(x).all()) for x in gradients.values()), 'DOCUMENT_TRAIN_GRADIENT')
    need(all(any(bool(x.abs().max()>0) for n,x in gradients.items() if n.startswith('projectors.'+lane+'.'))
             for lane in model.roster), 'DOCUMENT_TRAIN_DISCONNECTED_INSTRUMENT')
    optimizer.step(); validate_optimizer(model, optimizer, policy, step+1)
    after = cpu(model.state_dict()); moments = cpu(optimizer.state_dict())
    need(all(bool(torch.isfinite(x).all()) for x in after.values()), 'DOCUMENT_TRAIN_PARAMETER')
    need(len(moments['state']) == len(list(model.parameters())), 'DOCUMENT_TRAIN_MOMENT_COVERAGE')
    need(fingerprint(cpu(frames)) == item['native_pin'] == fingerprint(item['frames']), 'DOCUMENT_TRAIN_NATIVE_CHANGED')
    try:stream.step(frames[-1:],context=context,offset=len(frames),state=state,detach_history=True)
    except (ValueError,RuntimeError) as error: stale=str(error)
    else:raise AssertionError('DOCUMENT_TRAIN_STALE_CACHE_ACCEPTED')
    need('WEIGHTS_CHANGED' in stale, 'DOCUMENT_TRAIN_STALE_CACHE_REASON')
    return dict(step=step+1,document=item['document'],atoms=len(frames),targets=total,loss=loss_sum,
                before=fingerprint(before),after=fingerprint(after),gradient=fingerprint(gradients),
                optimizer=fingerprint(moments),branch_before=branches(before,model.roster),
                branch_after=branches(after,model.roster),branch_gradient=branches(gradients,model.roster),
                native=item['native_pin'],labels=item['label_pin'],stale_cache_refusal=stale,
                complete_forward_history=True,backward_history='detached_prior_tiles')


def checkpoint(path, model, optimizer, binding, step, rows, ledger):
    pin = save(path, model, optimizer, binding, step, rows, ledger)
    value = torch.load(path, map_location='cpu', weights_only=True)
    return dict(pin=pin,state=fingerprint(value),rows=rows)


def load(path, pin, model, optimizer, binding, step, rows, policy):
    need(file_sha(path) == pin, 'DOCUMENT_TRAIN_CHECKPOINT_PIN')
    value = torch.load(path, map_location='cpu', weights_only=True)
    need(value['binding'] == binding and value['step'] == step and value['rows'] == rows,
         'DOCUMENT_TRAIN_CHECKPOINT_CURSOR')
    model.load_state_dict(value['model'], strict=True); optimizer.load_state_dict(value['optimizer'])
    validate_optimizer(model, optimizer, policy, step)
    return value
