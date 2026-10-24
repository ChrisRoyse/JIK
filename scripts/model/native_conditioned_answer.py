"""Learn only the answer suffix and its ending; preload all observed context.

Prior positions retain the established detached-history recipe. This module
does not invent event order from prompt layout or add a role/clock encoder.
"""
import torch

from model.answer_training_schedule import decisions
from model.native_answer_sequence import versions
from model.native_device_transport import transfer
from model.native_placement_checked import validate_optimizer
from model.native_population_training import cpu, fingerprint
from text.constellation_corpus import need


def document_item(original, observed):
    """Bind a real legal continuation for runtime qualification, not task QA."""
    need(type(observed) is int and 0 < observed < len(original['frames']), 'CONDITIONED_PREFIX')
    need(fingerprint(original['frames']) == original['native_pin']
         and fingerprint((original['codes'], original['counts'])) == original['label_pin'],
         'CONDITIONED_PARENT_PIN')
    frames = original['frames']; codes = original['codes'][observed-1:]
    counts = {n:x[observed-1:] for n,x in original['counts'].items()}
    item = dict(document=original['document'], side=original['side'],
        context=tuple(frames[:observed]), answers=tuple(frames[observed:]), codes=codes, counts=counts,
        provenance=dict(parent_native=original['native_pin'], parent_labels=original['label_pin'],
            component=original['component'], observed=observed,
            position_kind='attested_document_order', scope='legal_continuation_not_capability_task'))
    item['native_pin'] = fingerprint((item['context'], item['answers']))
    item['label_pin'] = fingerprint((codes, counts))
    item['binding'] = fingerprint((item['document'], item['side'], item['provenance']))
    return item


def check(item):
    need(item['side'] == 'train', 'CONDITIONED_NONTRAIN')
    c, a = len(item['context']), len(item['answers'])
    rows = decisions(c, a)
    need(len(item['codes']) == a and all(len(v) == a for v in item['counts'].values()), 'CONDITIONED_LABEL_COUNT')
    need(fingerprint((item['context'], item['answers'])) == item['native_pin'], 'CONDITIONED_NATIVE_PIN')
    need(fingerprint((item['codes'], item['counts'])) == item['label_pin'], 'CONDITIONED_LABEL_PIN')
    need(fingerprint((item['document'], item['side'], item['provenance'])) == item['binding']
         and item['provenance']['observed'] == c, 'CONDITIONED_SOURCE_BINDING')
    return rows


def update(model, optimizer, item, step, policy, *, target):
    schedule = check(item); validate_optimizer(model, optimizer, policy, step)
    before = fingerprint(model.state_dict()); frozen = versions(target)
    optimizer.zero_grad(set_to_none=True); device = next(model.parameters()).device
    frames = transfer((*item['context'], *item['answers']), device)
    c, a = len(item['context']), len(item['answers'])
    codes = item['codes'].to(device); counts = {n:x.to(device) for n,x in item['counts'].items()}
    noise = torch.randn(a, policy['samples'], policy['noise_width'], device=device,
        generator=torch.Generator(device=device).manual_seed(policy['noise_seed']+step))
    stream = model.forecaster.predictor.stream(); state = None; records = []
    # Unlabelled prompt positions need forward memory, not unused backward graphs.
    for position, frame in enumerate(frames):
        with torch.set_grad_enabled(position >= c-1):
            hidden, state = stream.step((frame,), context='conditioned-answer:'+str(item['document']),
                offset=position, state=state, detach_history=True)
        if position < c-1: continue
        index = position-(c-1); row = schedule[index]; stop = model.stop_loss(hidden, row['end'])
        loss = stop['loss']/(a+1); continuation = None
        if not row['end']:
            continuation = model.forecaster.losses(hidden, codes[index:index+1],
                {n:x[index:index+1] for n,x in counts.items()}, frames[position+1:position+2],
                noise[index:index+1], target=target)
            loss = loss+continuation['loss']/a
        need(bool(torch.isfinite(loss)), 'CONDITIONED_NONFINITE')
        records.append(dict(**row, stop=cpu(stop),
            continuation=None if continuation is None else float(continuation['loss'].detach()),
            weighted_loss=float(loss.detach())))
        loss.backward()
    need(len(records) == a+1 and sum(r['end'] for r in records) == 1, 'CONDITIONED_COVERAGE')
    gradients = cpu({n:p.grad for n,p in model.named_parameters()})
    need(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients.values()), 'CONDITIONED_GRADIENTS')
    optimizer.step(); validate_optimizer(model, optimizer, policy, step+1)
    need(all(bool(torch.isfinite(x).all()) for x in model.state_dict().values()), 'CONDITIONED_PARAMETERS')
    need(versions(target) == frozen, 'CONDITIONED_FROZEN_CHANGED'); check(item)
    return dict(document=item['document'], context_positions=c, answer_targets=a, decisions=a+1,
        end_labels=1, prompt_transition_labels=0, before=before, after=fingerprint(model.state_dict()),
        gradients=fingerprint(gradients), optimizer=fingerprint(optimizer.state_dict()),
        native=item['native_pin'], labels=item['label_pin'], binding=item['binding'], records=records,
        backward_history='detached_prior_positions'), gradients
