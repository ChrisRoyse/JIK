"""Read-only derivative census on committed native region fits; no updates."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit.legal_rq_growth import make, pins, pinned, POLICY
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model.legal_learning_source import LegalLearningSource
from model.loop_training_checkpoint import tensor_fingerprint
from model.native_rq_data import NativeRQData, matched_donors, training_components
from senses.contract import digest
from text.constellation_corpus import file_sha, need
from text.legal_continuation_input import Continuations
from text.legal_reuse_input import ReuseInputs

TERMS = ('reconstruction', 'quantizer_loss', 'contrast', 'loss')


def unchanged(value, pin):
    need(digest(tensor_fingerprint(value)) == pin, 'RQ_DERIVATIVE_STATE_CHANGED')


def recount(arrays, norms):
    errors = {}
    for name in TERMS:
        a = arrays[name].astype(np.float64)
        need(a.shape == (8, 32) and np.isfinite(a).all(), 'RQ_DERIVATIVE_SHAPE')
        reference = math.sqrt(math.fsum(float(x)*float(x) for x in a.flat))
        error = abs(reference-norms[name])/max(1., abs(reference))
        need(error <= POLICY['normalized_loss_relative_error'], 'RQ_DERIVATIVE_NORM')
        errors[name] = error
    additive = float(np.max(np.abs(arrays['loss'].astype(np.float64)-sum(
        arrays[k].astype(np.float64) for k in TERMS[:-1]))))
    need(additive <= POLICY['arithmetic_error'], 'RQ_DERIVATIVE_ADDITIVITY')
    return dict(norm_relative_errors=errors, additive_error=additive)


def run(a):
    start = tick = time.perf_counter(); a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    fit = pinned(a.fit/'report.json', a.fit_pin)
    need(fit['complete'] and fit['mode'] == 'fit' and fit['code'] == pins(), 'RQ_DERIVATIVE_FIT')
    q = Path(fit['qualification']['path']); qualified = pinned(q/'report.json', fit['qualification']['pin'])
    selection = pinned(a.fit/'selection.json', fit['selection_pin'])
    baseline = pinned(a.baseline/'report.json', a.baseline_pin)
    source = ReuseInputs(baseline['source']['path'], baseline['source']['pin'])
    c = Continuations(Path(baseline['continuations']['path']), baseline['continuations']['pin'])
    split, cold = c.report['split'], c.report['cold']; bank = baseline['bank']
    data = LegalLearningSource(source, split['path'], split['pin'], cold['path'], cold['pin'], bank['path'], bank['pin'])
    need(data.binding == fit['source_binding'], 'RQ_DERIVATIVE_SOURCE')
    reader = NativeRQData(data, Path(baseline['proof']['path']), baseline['proof']['pin'])
    statistics = {}; cases = {}
    for population, uids in selection['populations'].items():
        path = q/(population+'-statistics.pt')
        need(file_sha(path) == qualified['statistic_pins'][population], 'RQ_DERIVATIVE_STATISTICS')
        statistics[population] = torch.load(path, map_location='cpu', weights_only=True)
        _, cases[population] = training_components(data, uids[:8], c.presence)
    phases = dict(source_and_statistics=time.perf_counter()-tick); tick = time.perf_counter()
    results = {}; arrays = {}
    for ordinal, (arm, record) in enumerate(fit['results'].items()):
        population, recipe = arm.split('-', 1); path = a.fit/arm/'final.pt'
        need(file_sha(path) == record['files']['final.pt'], 'RQ_DERIVATIVE_CHECKPOINT_PIN')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        model, _ = make(fit['dimensions'], recipe, statistics[population]); model.load_state_dict(saved['model'])
        before = digest(tensor_fingerprint(model.state_dict())); model.check_statistics()
        uids = selection['populations'][population][:8]; frames = reader.frames(uids)
        native_before = digest(tensor_fingerprint(frames))
        donors, eligible = matched_donors(frames, uids, cases[population], reader.roster)
        output = model.training_loss(frames, donors, eligible)
        gradients = {name: torch.autograd.grad(output[name], output['latent'], retain_graph=True)[0]
                     for name in TERMS}
        norms = {name: float(g.double().norm()) for name, g in gradients.items()}
        values = {name: g.detach().numpy().copy() for name, g in gradients.items()}
        checked = recount(values, norms)
        residual = {}
        for lane in reader.roster:
            errors = [d[lane].detach().numpy().astype(np.float64)-f[lane].numpy().astype(np.float64)
                      for d, f in zip(output['decoded'], frames)]
            mean_error = np.mean([x.mean(0) for x in errors], axis=0)
            mse = float(np.mean([np.mean(x*x) for x in errors])); bias = float(np.mean(mean_error*mean_error))
            need(abs(mse-float(output['per_lane'][lane])) <= POLICY['arithmetic_error'], 'RQ_DERIVATIVE_MSE')
            residual[lane] = dict(mse=mse, common_offset_mse=bias, common_offset_fraction=bias/mse)
            arrays[str(ordinal)+'_bias_'+lane] = mean_error
        unchanged(model.state_dict(), before); unchanged(frames, native_before); model.check_statistics()
        need(file_sha(path) == record['files']['final.pt'], 'RQ_DERIVATIVE_CHECKPOINT_MUTATED')
        for name, value in values.items(): arrays[str(ordinal)+'_'+name] = value
        arrays[str(ordinal)+'_latent'] = output['latent'].detach().numpy().copy()
        results[arm] = dict(ordinal=ordinal, checkpoint_pin=record['files']['final.pt'], weight_pin=before,
            uids=uids, native_pin=native_before, window_counts=[[len(f[n]) for n in reader.roster] for f in frames],
            eligible_contrast_rows=int(eligible.sum()), scalars={n:float(output[n].detach()) for n in TERMS},
            latent_gradient_norms=norms, independent_recount=checked, residual=residual)
    phases['fixed_batch_derivatives']=time.perf_counter()-tick; tick=time.perf_counter()
    path=a.output/'derivatives.npz';np.savez(path, **arrays)
    loaded=np.load(path, allow_pickle=False)
    for arm, row in results.items():
        recount({n:loaded[str(row['ordinal'])+'_'+n] for n in TERMS}, row['latent_gradient_norms'])
    altered={n:values[n].copy() for n in TERMS}; altered['loss'][0,0]+=1
    changed=dict(frames[0]);changed[reader.roster[0]]=changed[reader.roster[0]].clone()+1
    controls=dict(wrong_fit_pin=rejected(lambda:pinned(a.fit/'report.json','0'*64)),
                  altered_derivative=rejected(lambda:recount(altered,norms)),
                  changed_native=rejected(lambda:unchanged((changed,*frames[1:]),native_before)))
    source.fresh();data.fresh();c.fresh()
    phases['persisted_recount_controls_freshness']=time.perf_counter()-tick
    seconds=time.perf_counter()-start;need(sum(phases.values())/seconds>=.9,'RQ_DERIVATIVE_PHASE_ATTRIBUTION')
    report=dict(schema='legal-rq-gradient-census-v1',fit_pin=a.fit_pin,source_binding=data.binding,
        code=pins(),reader_code_pin=file_sha(Path(__file__)),results=results,controls=controls,
        arrays_pin=file_sha(path),phases_seconds=phases,seconds=seconds,optimizer_updates=0,
        number_status='measured',acceptance='author-checked',scientific_admission=False,model_promotion=False)
    pin=write_committed(a.output/'report.json',report)
    print(json.dumps(dict(pin=pin,seconds=seconds,gradient_norms={k:v['latent_gradient_norms'] for k,v in results.items()})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['baseline','fit']:
        p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-pin',required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
