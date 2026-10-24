"""Source-bound reciprocal updates; each update visits every declared mask."""
import torch

from model.loop_training_checkpoint import validate_optimizer
from model.native_document_training import branches
from model.native_population_training import cpu, fingerprint
from text.constellation_corpus import need


def masks(roster):
    return [(hidden, visible) for hidden in roster
            for visible in [tuple(n for n in roster if n != hidden)]
            + [(name,) for name in roster if name != hidden]]


def check_item(item, roster):
    rows, donors = item['rows'], item['donor_rows']
    frames, wrong = item['frames'], item['wrong']
    need(0 < len(rows) <= 64 and len(rows) == len(donors) == len(frames) == len(wrong),
         'RECIPROCAL_TRAIN_ROWS')
    need(all(r['side'] == d['side'] == 'train' for r, d in zip(rows, donors)),
         'RECIPROCAL_TRAIN_NONTRAIN')
    need(all(r['uid'] != d['uid'] and r['component'] != d['component']
             for r, d in zip(rows, donors)), 'RECIPROCAL_TRAIN_DONOR_IDENTITY')
    need(all(set(a) == set(b) == set(roster)
             and all(a[n].shape == b[n].shape for n in roster)
             for a, b in zip(frames, wrong)), 'RECIPROCAL_TRAIN_DONOR_LAYOUT')
    need(fingerprint((frames, wrong)) == item['native_pin'], 'RECIPROCAL_TRAIN_NATIVE_PIN')
    need(fingerprint((item['codes'], item['counts'])) == item['label_pin'],
         'RECIPROCAL_TRAIN_LABEL_PIN')
    need(fingerprint((rows, donors)) == item['source_pin'], 'RECIPROCAL_TRAIN_SOURCE_PIN')
    need(item['codes'].shape == (len(rows), 3)
         and item['counts'].shape == (len(rows), len(roster)), 'RECIPROCAL_TRAIN_TARGET_SHAPE')


def update(model, optimizer, item, step, policy, arm):
    need(arm in ('paired', 'wrong_pair'), 'RECIPROCAL_TRAIN_ARM')
    check_item(item, model.roster)
    validate_optimizer(model, optimizer, policy, step)
    before = cpu(model.state_dict()); device = model.encoder.query.device
    source = item['frames'] if arm == 'paired' else item['wrong']
    frames = tuple({n:x.to(device) for n,x in f.items()} for f in source)
    codes, counts = item['codes'].to(device), item['counts'].to(device)
    optimizer.zero_grad(set_to_none=True); objectives = masks(model.roster); losses = []
    for hidden, visible in objectives:
        context = model.encode_visible(tuple({n:f[n] for n in visible} for f in frames),
                                       hidden=hidden, visible=visible)
        loss = model.loss_withheld(context, codes, counts[:, model.roster.index(hidden)], hidden=hidden)
        need(bool(torch.isfinite(loss['loss'])), 'RECIPROCAL_TRAIN_NONFINITE_LOSS')
        (loss['loss']/len(objectives)).backward()
        losses.append(float(loss['loss'].detach()))
    gradients = cpu({n:p.grad for n,p in model.named_parameters()})
    need(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients.values()),
         'RECIPROCAL_TRAIN_DISCONNECTED_GRADIENT')
    need(all(any(bool(g.abs().max()>0) for n,g in gradients.items()
                 if n.startswith('projectors.'+lane+'.')) for lane in model.roster),
         'RECIPROCAL_TRAIN_DISCONNECTED_INSTRUMENT')
    optimizer.step(); validate_optimizer(model, optimizer, policy, step+1)
    after, moments = cpu(model.state_dict()), cpu(optimizer.state_dict())
    need(all(bool(torch.isfinite(v).all()) for v in after.values()), 'RECIPROCAL_TRAIN_NONFINITE_PARAMETER')
    need(len(moments['state']) == len(list(model.parameters())), 'RECIPROCAL_TRAIN_MOMENT_COVERAGE')
    check_item(item, model.roster)
    return dict(step=step+1, arm=arm, targets=len(frames)*len(objectives), masks=len(objectives),
                mask_losses=losses, before=fingerprint(before), after=fingerprint(after),
                gradient=fingerprint(gradients), optimizer=fingerprint(moments),
                branch_before=branches(before,model.roster), branch_after=branches(after,model.roster),
                branch_gradient=branches(gradients,model.roster), native=item['native_pin'],
                labels=item['label_pin'], source=item['source_pin'])
