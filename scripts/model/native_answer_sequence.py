"""Ordered native answer trajectories with learned ending and autonomous feedback.

An end prediction is a sequence boundary, not a validity/guard verdict. The
generator accepts only an observed prefix; no future length or target enters it.
"""
import torch
from torch import nn
from torch.nn import functional as F

from model.attention import Linear
from model.native_placement_checked import validate_optimizer
from model.native_device_transport import transfer
from model.native_placement_training import check_item
from model.native_population_training import cpu, fingerprint
from model.native_prefix_stream import NativePrefixStream
from text.constellation_corpus import need


def versions(model):
    return tuple((id(t), t._version) for t in model.state_dict(keep_vars=True).values())


class NativeAnswerSequence(nn.Module):
    def __init__(self, forecaster):
        super().__init__()
        self.forecaster = forecaster
        self.roster, self.proposals = forecaster.roster, forecaster.proposals
        width = forecaster.predictor.config['width']
        self.ending = nn.ModuleDict({p:nn.ModuleDict({n:Linear(width, 2) for n in self.roster})
                                    for p in self.proposals})
        with torch.no_grad():
            for group in self.ending.values():
                for head in group.values(): head.weight.zero_(); head.bias.zero_()

    def stop_logits(self, context, proposal):
        need(proposal in self.proposals and context.ndim == 3
             and context.shape[1] == len(self.roster), 'ANSWER_STOP_CONTEXT')
        # Combine categorical decision logits only; native readings stay typed.
        return torch.stack([self.ending[proposal][n](context[:, i])
                            for i, n in enumerate(self.roster)]).mean(0)

    def stop_loss(self, hidden, end):
        need(type(end) is bool, 'ANSWER_END_LABEL')
        rows = {}
        for p, (_, context) in self.forecaster.contexts(hidden).items():
            score = self.stop_logits(context, p)
            label = torch.full((len(score),), int(end), dtype=torch.long, device=score.device)
            rows[p] = dict(logits=score, loss=F.cross_entropy(score.double(), label))
        return dict(loss=torch.stack([r['loss'] for r in rows.values()]).mean(), proposals=rows)

    def _stream(self, proposal):
        predictor = self.forecaster.predictor
        if self.forecaster.architecture != 'whole_proposals':
            return predictor.stream(), lambda f:f, lambda h:(predictor, h)
        # This candidate depends on one source instrument. Do not encode five
        # unrelated histories merely to discard their hidden values afterwards.
        branch = predictor.encoder.lanes[proposal]
        stream = NativePrefixStream(branch, nn.ModuleDict({proposal:predictor.projectors[proposal]}),
                                    cache_bytes=predictor.cache_bytes//len(self.roster))
        def readout(hidden):
            value = hidden[:, 0]
            context = torch.stack([predictor.destinations[proposal][n](value) for n in self.roster], dim=1)
            return predictor.readouts[proposal], context
        return stream, lambda f:{proposal:f[proposal]}, readout

    def rollout(self, prefix, *, target, proposal, context, seed):
        """Yield native atoms then an end event; consumers may close an observation.

        There is no automatic length cutoff. A physical-cache refusal is not an
        end event, and closing the generator does not claim a completed answer.
        """
        need(proposal in self.proposals and isinstance(prefix, (tuple, list)) and prefix
             and type(seed) is int and seed >= 0 and type(context) is str and context,
             'ANSWER_REQUEST')
        need(all(set(f) == set(self.roster) for f in prefix), 'ANSWER_PREFIX_ROSTER')
        placement = self.forecaster.placement[proposal]; placement.check_target(target)
        pin = versions(self); stream, select, readout = self._stream(proposal)
        device = next(self.parameters()).device
        rng = torch.Generator(device=device).manual_seed(seed)
        state = None; hidden = None; offset = 0; generated = 0
        try:
            with torch.no_grad():
                for frame in prefix:
                    hidden, state = stream.step((select(frame),), context=context, offset=offset,
                                                state=state, detach_history=True)
                    offset += 1
            while True:
                with torch.no_grad():
                    need(versions(self) == pin, 'ANSWER_WEIGHTS_CHANGED')
                    placement.check_target(target)
                    head, value = readout(hidden)
                    logits = self.stop_logits(value, proposal).detach().cpu()
                    ended = bool(logits.argmax(-1).item())
                    if ended:
                        event = dict(kind='end', proposal=proposal, generated=generated,
                                     stop_logits=logits.clone(), validity='unassessed')
                    else:
                        forecast = head.forecast(value)
                        noise = torch.randn(1, 1, placement.noise_width, generator=rng, device=device)
                        sample = placement(value, forecast['codes'], forecast['counts'], noise, target=target)[0]
                        frame = {n:sample[n][0].detach() for n in self.roster}
                        # Consume the predicted readings before exposing owned copies.
                        hidden, state = stream.step((select(frame),), context=context, offset=offset,
                                                    state=state, detach_history=True)
                        offset += 1
                        event = dict(kind='constellation', proposal=proposal, index=generated,
                                     slots={n:x.clone() for n,x in frame.items()}, codes=forecast['codes'].clone(),
                                     counts={n:x.clone() for n,x in forecast['counts'].items()},
                                     stop_logits=logits.clone(), feedback_index=offset-1, source='predicted',
                                     validity='unassessed')
                        generated += 1
                yield event
                if ended: return
        finally:
            state = hidden = stream = None


def update_answer(model, optimizer, item, step, policy, *, target):
    """One complete attested continuation and end; no unobserved ordering labels."""
    check_item(item); validate_optimizer(model, optimizer, policy, step)
    before = fingerprint(model.state_dict()); frozen = versions(target)
    optimizer.zero_grad(set_to_none=True); device = next(model.parameters()).device
    frames = transfer(item['frames'], device); total = len(frames)-1
    codes = item['codes'].to(device); counts = {n:x.to(device) for n,x in item['counts'].items()}
    noise = torch.randn(total, policy['samples'], policy['noise_width'],
                        generator=torch.Generator(device=device).manual_seed(policy['noise_seed']+step), device=device)
    stream = model.forecaster.predictor.stream(); state = None; records = []
    for index, frame in enumerate(frames):
        hidden, state = stream.step((frame,), context='answer-train:'+str(item['document']), offset=index,
                                    state=state, detach_history=True)
        stop = model.stop_loss(hidden, index == total)
        loss = stop['loss']/len(frames); continuation = None
        if index < total:
            continuation = model.forecaster.losses(hidden, codes[index:index+1],
                {n:x[index:index+1] for n,x in counts.items()}, frames[index+1:index+2], noise[index:index+1], target=target)
            loss = loss+continuation['loss']/total
        need(bool(torch.isfinite(loss)), 'ANSWER_TRAIN_NONFINITE')
        records.append(dict(index=index, end=index == total, stop=cpu(stop),
                            continuation=None if continuation is None else float(continuation['loss'].detach()),
                            weighted_loss=float(loss.detach())))
        loss.backward()
    gradients = cpu({n:p.grad for n,p in model.named_parameters()})
    need(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients.values()), 'ANSWER_TRAIN_GRADIENTS')
    optimizer.step(); validate_optimizer(model, optimizer, policy, step+1)
    need(all(bool(torch.isfinite(x).all()) for x in model.state_dict().values()), 'ANSWER_TRAIN_PARAMETERS')
    need(versions(target) == frozen, 'ANSWER_TRAIN_FROZEN_CHANGED'); check_item(item)
    return dict(document=item['document'], transitions=total, end_labels=1, decisions=len(frames),
                before=before, after=fingerprint(model.state_dict()), gradients=fingerprint(gradients),
                optimizer=fingerprint(optimizer.state_dict()), records=records), gradients
