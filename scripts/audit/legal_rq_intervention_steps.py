"""Actual optimizer-stage transitions and independent loss arithmetic on real windows."""
import numpy as np
import torch

from audit import legal_rq_growth as growth
from audit.text_sigreg_readback import integral
from model.loop_training_checkpoint import tensor_fingerprint
from model.native_rq_data import matched_donors
from model.native_rq_interventions import SCHEDULE
from model.text_sigreg import statistic
from senses.contract import digest
from text.constellation_corpus import need


def fingerprint(value):
    return digest(tensor_fingerprint(value))


def frozen_check(model, optimizer, frozen, before):
    need(fingerprint({n:p for n, p in model.named_parameters() if n in frozen}) == before, 'RQ_INTERVENTION_FROZEN_CHANGED')
    need(all(p not in optimizer.state for n, p in model.named_parameters() if n in frozen), 'RQ_INTERVENTION_FROZEN_MOMENTS')


def transition(model, optimizer, reader, uids):
    if model.arm != 'continuous_then_rq' or model.step_number != SCHEDULE['continuous_warmup']:
        return None
    need(all(p not in optimizer.state for p in model.books), 'RQ_INTERVENTION_BOOK_MOMENTS')
    before = fingerprint(model.state_dict()); moments = fingerprint(optimizer.state_dict())
    other = fingerprint({n:p for n, p in model.named_parameters() if not n.startswith('books.')})
    model.initialized.fill_(False); latent_pin = growth.initialize(model, reader, uids)
    need(other == fingerprint({n:p for n, p in model.named_parameters() if not n.startswith('books.')})
         and moments == fingerprint(optimizer.state_dict()), 'RQ_INTERVENTION_TRANSITION_CHANGED_OTHER_STATE')
    return dict(before=before, after=fingerprint(model.state_dict()), optimizer=moments, train_latent_pin=latent_pin)


def step(model, optimizer, reader, uids, cases, number):
    model.set_step(number); model.check_statistics(); before = fingerprint(model.state_dict())
    changed = transition(model, optimizer, reader, uids)
    positions = growth.batch_indices(uids, number); ids = [uids[j] for j in positions]
    frames = reader.frames(ids); components = [cases[j] for j in positions]
    donors, eligible = matched_donors(frames, ids, components, reader.roster)
    native = fingerprint(frames); frozen = model.frozen_names()
    protected = fingerprint({n:p for n, p in model.named_parameters() if n in frozen})
    optimizer.zero_grad(set_to_none=True); out = model.training_loss(frames, donors, eligible); out['loss'].backward()
    gradients = {n:p.grad for n, p in model.named_parameters()}
    need(all((g is None) if n in frozen else (g is not None and bool(torch.isfinite(g).all()))
             for n, g in gradients.items()), 'RQ_INTERVENTION_GRADIENT_POLICY')
    branches = ['encoder.', 'books.', 'to_latent.']+['projectors.'+n+'.' for n in reader.roster]+['decoders.'+n+'.' for n in reader.roster]
    norms = {b:sum(float(g.double().square().sum()) for n, g in gradients.items() if n.startswith(b) and g is not None)**.5 for b in branches}
    gradient_pin = fingerprint(gradients); optimizer.step(); frozen_check(model, optimizer, frozen, protected)
    need(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'RQ_INTERVENTION_PARAMETER')
    need(native == fingerprint(frames), 'RQ_INTERVENTION_NATIVE_CHANGED'); model.check_statistics()
    cursors = {n:int(optimizer.state[p]['step']) if p in optimizer.state else 0 for n, p in model.named_parameters()}
    for n, cursor in cursors.items():
        expected = number+1
        if not n.startswith('decoders.'):
            if model.arm == 'fixed_encoder': expected = 0
            elif model.arm == 'decoder_warmup': expected = max(0, number+1-SCHEDULE['decoder_warmup'])
            elif model.arm == 'continuous_then_rq' and n.startswith('books.'):
                expected = max(0, number+1-SCHEDULE['continuous_warmup'])
        need(cursor == expected, 'RQ_INTERVENTION_OPTIMIZER_CURSOR')
    return dict(step=number+1, ids=ids, donors=donors.tolist(), eligible=eligible.tolist(),
        stage=out['stage'], before=before, after=fingerprint(model.state_dict()), gradient=gradient_pin,
        optimizer=fingerprint(optimizer.state_dict()), gradient_norms=norms, frozen_count=len(frozen),
        frozen_pin=protected, transition=changed, optimizer_cursors=cursors,
        **{n:float(out[k].detach()) for n,k in [('loss','loss'),('reconstruction','reconstruction'),
              ('contrast','contrast'),('quantizer','quantizer_loss'),('regularizer','regularizer')]})


@torch.no_grad()
def loss_reference(model, reader, uids, cases, root):
    ids = uids[:8]; frames = reader.frames(ids)
    donors, eligible = matched_donors(frames, ids, cases[:8], reader.roster)
    out = model.training_loss(frames, donors, eligible)
    wrong = model.reconstruct(out['values'][donors], [{n:len(f[n]) for n in reader.roster} for f in frames])
    recon = []; contrasts = []
    for i, n in enumerate(reader.roster):
        correct = []; shuffled = []
        for d, bad, f in zip(out['decoded'], wrong, frames):
            target = f[n].numpy().astype(np.float64)
            correct.append(np.mean((d[n].numpy().astype(np.float64)-target)**2))
            shuffled.append(np.mean((bad[n].numpy().astype(np.float64)-target)**2))
        variance = float(getattr(model, 'variance_'+str(i)))
        recon.append(float(np.mean(correct))/variance)
        if bool(eligible.any()):
            delta = (np.asarray(correct)-np.asarray(shuffled))/variance
            contrasts.append(float(np.logaddexp(0., delta[eligible.numpy()]).mean()))
    expected = dict(reconstruction=float(np.mean(recon)), contrast=float(np.mean(contrasts)) if contrasts else 0.)
    quantizer = 0.
    if out['stage'] == 'quantized_joint':
        residual = out['latent'].numpy().astype(np.float64)
        for j, book in enumerate(model.books):
            array = book.detach().numpy().astype(np.float64); chosen = array[out['codes'][:,j].numpy()]
            quantizer += float(np.mean((chosen-residual)**2))*(1. if model.arm == 'no_encoder_quantizer' else 1.25)
            residual = residual-chosen
    expected['quantizer_loss'] = quantizer; expected['regularizer'] = 0.; sigreg = None
    if model.arm == 'latent_sigreg':
        values = out['latent'][None]; matrix = model.projection(out['latent'])
        actual = statistic(values, matrix).numpy(); separate = integral(values.numpy(), matrix.numpy())
        need(np.allclose(actual, separate, atol=3e-5, rtol=1e-6), 'RQ_INTERVENTION_SIGREG_REFERENCE')
        reversed_value = statistic(values[:,torch.arange(values.shape[1]-1,-1,-1)], matrix).numpy()
        need(np.allclose(actual, reversed_value, atol=3e-5, rtol=1e-6), 'RQ_INTERVENTION_SIGREG_ORDER')
        collapsed = values.mean(1, keepdim=True).expand_as(values)
        collapsed_torch = statistic(collapsed, matrix).numpy(); collapsed_reference = integral(collapsed.numpy(), matrix.numpy())
        need(np.allclose(collapsed_torch, collapsed_reference, atol=3e-5, rtol=1e-6), 'RQ_INTERVENTION_SIGREG_COLLAPSE_REFERENCE')
        np.savez(root/'sigreg.npz', values=values.numpy(), projection=matrix.numpy(), actual=actual,
                 collapsed=collapsed.numpy(), collapsed_actual=collapsed_torch)
        expected['regularizer'] = float(separate[0])
        sigreg = dict(actual=actual.tolist(), independent=separate.tolist(), collapsed=collapsed_reference.tolist())
    expected['loss'] = expected['reconstruction']+expected['contrast']+quantizer+SCHEDULE['sigreg_strength']*expected['regularizer']
    errors = {n:abs(float(out[n])-v)/max(1.,abs(v)) for n,v in expected.items()}
    need(max(errors.values()) <= growth.POLICY['normalized_loss_relative_error'], 'RQ_INTERVENTION_LOSS_REFERENCE')
    return dict(reference=expected, relative_errors=errors, sigreg=sigreg)
