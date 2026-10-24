"""Qualify a separate batched student runtime against persisted native windows."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit import legal_rq_growth as growth, legal_rq_residual as residual
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model.loop_training_checkpoint import tensor_fingerprint
from model.native_rq_batch import BatchedInterventionNativeRQ
from model.native_rq_interventions import InterventionNativeRQ
from senses.contract import digest
from text.constellation_corpus import file_sha, need

ROOT = Path('/zfs/archive/mnlm/evidence/fsv/29')
COST_PIN = 'cac15f126f7f405dd81d1dfe12c1245010fcd293eb808fdd9564d88782b99069'
FIT_PIN = 'efd936f65ce47cbf220998f93aa6bfa61b3f2d737e450dcf151dfceb5a867fb1'
ERROR = 5e-6  # Existing RQ arithmetic-check tolerance; admission gates do not change.


def flat(outputs):
    return dict(latent=outputs['latent'], codes=outputs['codes'],
                **{n:torch.cat([row[n] for row in outputs['decoded']]) for n in outputs['decoded'][0]})


def difference(a, b):
    need(set(a) == set(b), 'RQ_BATCH_READBACK_KEYS')
    errors = {}
    for n in a:
        need(a[n].shape == b[n].shape and torch.isfinite(a[n]).all() and torch.isfinite(b[n]).all(),
             'RQ_BATCH_READBACK_LAYOUT')
        if a[n].dtype == torch.int64:
            need(torch.equal(a[n], b[n]), 'RQ_BATCH_READBACK_CODES'); errors[n] = 0.
        else:
            error = float((a[n].double()-b[n].double()).abs().max())
            need(error <= ERROR*max(1., float(a[n].abs().max())), 'RQ_BATCH_READBACK_ARITHMETIC')
            errors[n] = error
    return errors


def make(kind, payload, state):
    cls = InterventionNativeRQ if kind == 'scalar' else BatchedInterventionNativeRQ
    torch.manual_seed(growth.POLICY['seed'])
    model = cls(payload['dimensions'], payload['config'], payload['sizes'], 'decoder_warmup', payload['statistics'])
    model.load_state_dict(state, strict=True); model.set_step(3306); return model


def measure(model, frames, tile):
    before = time.perf_counter(); cpu = time.process_time(); parts = dict(latent=0., quantize=0., reconstruct=0., collect=0.)
    zs = []; cs = []; rows = []
    with torch.no_grad():
        for lo in range(0, len(frames), tile):
            f = frames[lo:lo+tile]; tick = time.perf_counter(); z = model.latent(f)
            parts['latent'] += time.perf_counter()-tick; tick = time.perf_counter()
            codes, values, _ = model.quantize(z); parts['quantize'] += time.perf_counter()-tick; tick = time.perf_counter()
            decoded = model.reconstruct(values, [{n:len(x[n]) for n in model.roster} for x in f])
            parts['reconstruct'] += time.perf_counter()-tick; tick = time.perf_counter()
            zs.append(z); cs.append(codes); rows.extend(decoded); parts['collect'] += time.perf_counter()-tick
    seconds = time.perf_counter()-before; used = time.process_time()-cpu
    need(sum(parts.values())/seconds >= .9, 'RQ_BATCH_READBACK_PHASES')
    return flat(dict(latent=torch.cat(zs), codes=torch.cat(cs), decoded=rows)), dict(seconds=seconds, cpu_seconds=used, phases_seconds=parts)


def one_update(model, frames):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01, foreach=False, fused=False)
    out = model(frames); out['loss'].backward()
    gradients = {n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
    need(len(gradients) == len(list(model.parameters())), 'RQ_BATCH_READBACK_MISSING_GRADIENT')
    optimizer.step()
    return dict(loss=float(out['loss'].detach()), gradients=gradients,
                model={n:v.detach().clone() for n,v in model.state_dict().items()}, optimizer=optimizer.state_dict())


def run(a):
    start = tick = time.perf_counter(); a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    cost_root = ROOT/'legal-native-student-cost-v2-20261023'; cost = growth.pinned(cost_root/'report.json', COST_PIN)
    need(cost['complete'] and cost['code'] == residual.pins(), 'RQ_BATCH_READBACK_SOURCE_CODE')
    path = cost_root/'reference.pt'; need(file_sha(path) == cost['reference_pin'], 'RQ_BATCH_READBACK_NATIVE_PIN')
    payload = torch.load(path, map_location='cpu', weights_only=True); frames = payload['frames']
    native_pin = digest(tensor_fingerprint(frames)); need(native_pin == cost['native_pin'], 'RQ_BATCH_READBACK_NATIVE')
    fit_root = ROOT/'legal-rq-residual-fit-v1-20261023'; fit = growth.pinned(fit_root/'report.json', FIT_PIN)
    final_path = fit_root/'position_residual/final.pt'; need(file_sha(final_path) == fit['results']['position_residual']['files']['final.pt'], 'RQ_BATCH_READBACK_FIT_PIN')
    trained = torch.load(final_path, map_location='cpu', weights_only=True)['model']
    code = dict(residual.pins(), **{n:file_sha(Path(__file__).parents[1]/n) for n in
                ['model/native_rq_batch.py', 'model/native_rq_interventions.py', 'audit/legal_rq_batch_readback.py']})
    phases = dict(authenticate=time.perf_counter()-tick); tick = time.perf_counter()
    arrays = {}; records = {}; steps = {}; snapshot_controls = {}
    for label, state in [('initial', payload['model']), ('trained', trained)]:
        models = {n:make(n,payload,state) for n in ['scalar','batched']}
        weight_pin = digest(tensor_fingerprint(state)); reference, _ = measure(models['scalar'],frames,1)
        if label == 'initial': need(tensor_fingerprint(reference) == tensor_fingerprint(flat(payload['reference'])), 'RQ_BATCH_READBACK_PRIOR_OUTPUT')
        timings = []; errors = {}
        for kind, model in models.items():
            for tile in [1,8,32,128]:
                first = None
                for repetition in range(5):
                    value, timing = measure(model,frames,tile)
                    if first is None: first = value
                    need(tensor_fingerprint(first) == tensor_fingerprint(value), 'RQ_BATCH_READBACK_REPEAT')
                    errors[kind+':'+str(tile)] = difference(reference,value)
                    timings.append(dict(kind=kind,tile=tile,repetition=repetition,**timing))
                for name, value in first.items(): arrays[label+':'+kind+':'+str(tile)+':'+name] = value.detach().numpy()
            need(digest(tensor_fingerprint(model.state_dict())) == weight_pin, 'RQ_BATCH_READBACK_MUTATION')
        # Reverse, duplicate, and vary a separate batch member: no cross-atom computation.
        batched = models['batched']; order = list(range(len(frames)-1,-1,-1))
        actual = batched.latent([frames[i] for i in order]).detach(); original = batched.latent(frames).detach()
        difference({'latent':original}, {'latent':actual[order]})
        pair = batched.latent([frames[0],frames[0],frames[-1]]).detach()
        difference({'latent':original[[0,0,-1]]}, {'latent':pair})
        gradient_ids = sorted(set(range(8)) | {i for i,f in enumerate(frames) if any(len(x)>1 for x in f.values())})
        gradient_frames = [frames[i] for i in gradient_ids]
        updates = {kind:one_update(make(kind,payload,state),gradient_frames) for kind in models}
        gradients = difference(updates['scalar']['gradients'],updates['batched']['gradients'])
        batched_repeat = one_update(make('batched',payload,state),gradient_frames)
        need(tensor_fingerprint(updates['batched']) == tensor_fingerprint(batched_repeat), 'RQ_BATCH_READBACK_UPDATE_REPEAT')
        steps[label] = updates['batched']
        records[label] = dict(timings=timings,errors=errors,gradient_errors=gradients,gradient_uids=[payload['uids'][i] for i in gradient_ids],
                              batched_update_pin=digest(tensor_fingerprint(updates['batched'])))
        snapshot_controls[label] = dict(oversized_counts=rejected(lambda:batched.groups([{n:1<<20 for n in batched.roster}])),
            missing_slot=rejected(lambda:batched.latent([{n:v for n,v in frames[0].items() if n!=batched.roster[0]}])),
            wrong_latent=rejected(lambda:batched.reconstruct(torch.full((1,32),float('nan')),payload['counts'][:1])),
            changed_codes=rejected(lambda:difference(reference,dict(reference,codes=reference['codes']+1))))
    phases['runtime_outputs_gradients_and_controls']=time.perf_counter()-tick; tick=time.perf_counter()
    need(digest(tensor_fingerprint(frames)) == native_pin, 'RQ_BATCH_READBACK_NATIVE_MUTATION')
    np.savez(a.output/'outputs.npz',**arrays); torch.save(steps,a.output/'updates.pt')
    with np.load(a.output/'outputs.npz',allow_pickle=False) as saved:
        need(set(saved.files)==set(arrays) and all(np.array_equal(saved[n],v) for n,v in arrays.items()), 'RQ_BATCH_READBACK_PERSISTED_ARRAYS')
    need(tensor_fingerprint(torch.load(a.output/'updates.pt',map_location='cpu',weights_only=True))==tensor_fingerprint(steps), 'RQ_BATCH_READBACK_PERSISTED_UPDATE')
    if a.previous is not None:
        previous=growth.pinned(a.previous/'report.json',a.previous_pin)
        need(previous['complete'] and previous['code']==code and previous['source_pin']==COST_PIN, 'RQ_BATCH_READBACK_COLD_BINDING')
        for n,pin in previous['files'].items(): need(file_sha(a.previous/n)==pin, 'RQ_BATCH_READBACK_COLD_FILE')
        with np.load(a.previous/'outputs.npz',allow_pickle=False) as saved:
            need(set(saved.files)==set(arrays) and all(np.array_equal(saved[n],v) for n,v in arrays.items()), 'RQ_BATCH_READBACK_COLD_ARRAYS')
        prior_steps=torch.load(a.previous/'updates.pt',map_location='cpu',weights_only=True)
        need(tensor_fingerprint(prior_steps)==tensor_fingerprint(steps), 'RQ_BATCH_READBACK_COLD_STEPS')
    phases['persisted_and_cold_readback']=time.perf_counter()-tick; seconds=time.perf_counter()-start
    need(sum(phases.values())/seconds>=.9, 'RQ_BATCH_READBACK_PHASE_COVERAGE')
    report=dict(complete=True,source_pin=COST_PIN,fit_pin=FIT_PIN,code=code,native_pin=native_pin,inputs=len(frames),
        previous_pin=a.previous_pin,records=records,controls=snapshot_controls,phases_seconds=phases,seconds=seconds,
        files={n:file_sha(a.output/n) for n in ['outputs.npz','updates.pt']},
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,optimizer_updates=6,
        model_promotion=False,scientific_admission=False,number_status='measured',acceptance='author-checked')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),seconds=seconds)),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--previous',type=Path);p.add_argument('--previous-pin');a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.is_dir(): write_committed(a.output/'failure.json',dict(error=type(exc).__name__+':'+str(exc)))
        raise
