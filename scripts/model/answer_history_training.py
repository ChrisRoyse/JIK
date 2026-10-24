"""Matched complete-answer training with explicit, detached model feedback.

Training continues to every attested target and END even if an intermediate
stop prediction is wrong. This is a history-robustness experiment, not an
autonomous answer or a statistically consistent distribution-learning claim.
"""
import torch
from torch.nn import functional as F

from model.native_answer_sequence import versions
from model.native_conditioned_answer import check
from model.native_device_transport import transfer
from model.native_placement_checked import validate_optimizer
from model.native_population_training import cpu, fingerprint
from model.native_request_input import NativeRequest
from model.native_request_stream import bind
from model.native_stochastic_placement import energy_loss
from text.constellation_corpus import need


def model_feedback(model, head, hidden, proposal, *, target, generator):
    """No answer, teacher region/count, future length or stop override is input."""
    need(proposal in model.proposals, 'ANSWER_HISTORY_PROPOSAL')
    placement = model.forecaster.placement[proposal]
    with torch.no_grad():
        forecast = head.forecast(hidden)
        noise = torch.randn(1, 1, placement.noise_width, device=hidden.device, generator=generator)
        sample = placement(hidden, forecast['codes'], forecast['counts'], noise, target=target)[0]
        frame = {n: sample[n][0].detach().clone() for n in model.roster}
    return frame, dict(codes=cpu(forecast['codes']), counts=cpu(forecast['counts']),
                      noise=cpu(noise), slots=cpu(frame), source='predicted', validity='unassessed')


def predicted_position(index, step, enabled):
    need(type(enabled) is bool and type(index) is int and index >= 0
         and type(step) is int and step >= 0, 'ANSWER_HISTORY_POLICY')
    return enabled and (index+step) % 2 == 0


def update(model, optimizer, item, step, policy, *, request, target, predicted):
    schedule = check(item)
    predicted_position(0, step, predicted)
    need(isinstance(request, NativeRequest) and request.observed == len(item['context'])
         and item['provenance']['request'] == request.pin, 'ANSWER_HISTORY_REQUEST')
    validate_optimizer(model, optimizer, policy, step)
    before = fingerprint(model.state_dict()); frozen = versions(target)
    optimizer.zero_grad(set_to_none=True); device = next(model.parameters()).device
    frames = transfer((*item['context'], *item['answers']), device)
    c, a = len(item['context']), len(item['answers'])
    codes = item['codes'].to(device); counts = {n: x.to(device) for n, x in item['counts'].items()}
    noise = torch.randn(a, policy['samples'], policy['noise_width'], device=device,
        generator=torch.Generator(device=device).manual_seed(policy['noise_seed']+step))
    records = []; proposals = len(model.proposals)
    for proposal in model.proposals:
        original, select, readout = model._stream(proposal)
        stream = bind(original, model.request_projection, request)
        state = hidden = None
        # Same stream of placement noise for each candidate, as at inference.
        generator = torch.Generator(device=device).manual_seed(policy['noise_seed']+1000000+step)
        context = 'history-train:'+str(item['document'])
        try:
            for position in range(c):
                with torch.set_grad_enabled(position == c-1):
                    hidden, state = stream.step((select(frames[position]),), context=context,
                        offset=position, state=state, detach_history=True)
            input_source = 'observed_context'; input_pin = fingerprint(frames[c-1])
            for index, label in enumerate(schedule):
                head, h = readout(hidden)
                stop_logits = model.stop_logits(h, proposal)
                stop = F.cross_entropy(stop_logits.double(),
                    torch.tensor([int(label['end'])], dtype=torch.long, device=device))
                loss = stop/(a+1)/proposals
                continuation = feedback = next_frame = None; selected = False
                if not label['end']:
                    # Forecast before labels are passed to the supervised objective.
                    forecast_frame, feedback = model_feedback(model, head, h, proposal,
                        target=target, generator=generator)
                    selected = predicted_position(index, step, predicted)
                    next_frame = forecast_frame if selected else frames[c+index]
                    discrete = head.loss(h, codes[index:index+1], {n: x[index:index+1] for n, x in counts.items()})
                    samples = model.forecaster.placement[proposal](h, codes[index:index+1],
                        {n: x[index:index+1] for n, x in counts.items()}, noise[index:index+1], target=target)
                    native = energy_loss(samples, frames[c+index:c+index+1],
                        model.forecaster.placement[proposal].variances())
                    continuation = discrete['loss']+native['loss']
                    loss = loss+continuation/a/proposals
                need(bool(torch.isfinite(loss)), 'ANSWER_HISTORY_NONFINITE')
                records.append(dict(**label, proposal=proposal, input_source=input_source, input=input_pin,
                    stop_logits=cpu(stop_logits), predicted_end=bool(stop_logits.argmax(-1).item()),
                    forced_training_continuation=not label['end'],
                    feedback=feedback, selected_prediction=selected,
                    next_input=None if next_frame is None else fingerprint(next_frame),
                    continuation=None if continuation is None else float(continuation.detach()),
                    weighted_loss=float(loss.detach())))
                loss.backward()
                if next_frame is not None:
                    hidden, state = stream.step((select(next_frame),), context=context,
                        offset=c+index, state=state, detach_history=True)
                    input_source = 'predicted' if selected else 'observed_answer'
                    input_pin = fingerprint(next_frame)
            need(len(records) % (a+1) == 0, 'ANSWER_HISTORY_DECISIONS')
        finally:
            state = hidden = stream = original = None
    need(len(records) == (a+1)*proposals and sum(r['end'] for r in records) == proposals,
         'ANSWER_HISTORY_COVERAGE')
    gradients = cpu({n: p.grad for n, p in model.named_parameters()})
    need(all(g is not None and bool(g.isfinite().all()) for g in gradients.values()), 'ANSWER_HISTORY_GRADIENTS')
    prefixes = ['forecaster.predictor.encoder.']
    prefixes += [kind+n+'.' for n in model.roster for kind in
                 ('forecaster.predictor.projectors.', 'request_projection.lanes.')]
    prefixes += ['ending.'+p+'.'+n+'.' for p in model.proposals for n in model.roster]
    prefixes += ['forecaster.placement.'+p+'.'+kind+'.'+n+'.'
                 for p in model.proposals for kind in ('context', 'region', 'noise', 'position', 'residual')
                 for n in model.roster]
    branches = {}
    for prefix in prefixes:
        group = {n:g for n,g in gradients.items() if n.startswith(prefix)}
        need(group and any(bool(g.abs().any()) for g in group.values()), 'ANSWER_HISTORY_DISCONNECTED:'+prefix)
        branches[prefix] = fingerprint(group)
    optimizer.step(); validate_optimizer(model, optimizer, policy, step+1)
    need(all(bool(x.isfinite().all()) for x in model.state_dict().values()), 'ANSWER_HISTORY_PARAMETERS')
    need(versions(target) == frozen, 'ANSWER_HISTORY_FROZEN_CHANGED'); check(item)
    return dict(document=item['document'], request=request.pin, context_positions=c, answer_targets=a,
        decisions=len(records), end_labels=proposals, prompt_transition_labels=0,
        rule='alternating_model_history' if predicted else 'teacher_history',
        predicted_inputs=sum(r['selected_prediction'] for r in records),
        before=before, after=fingerprint(model.state_dict()), gradients=fingerprint(gradients),
        branch_gradients=branches,
        optimizer=fingerprint(optimizer.state_dict()), native=item['native_pin'], labels=item['label_pin'],
        binding=item['binding'], records=records, backward_history='detached_prior_positions'), gradients
