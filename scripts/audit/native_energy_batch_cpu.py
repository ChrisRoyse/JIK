"""CPU-only equality qualification on the retained real legal placement cohort."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch

from audit import legal_grouped_placement_training as source
from model import native_energy_batched as candidate
from model import native_placement_training as base
from model import native_placement_efficient as efficient
from model.native_population_training import cpu, fingerprint
from model.native_rq_packed import PackedNativeRQ
from text.constellation_corpus import file_sha, need


def save(path, value):
    with path.open('xb') as f:
        torch.save(cpu(value),f); f.flush(); os.fsync(f.fileno())
    return dict(sha256=file_sha(path),state=fingerprint(torch.load(path,map_location='cpu',weights_only=True)))


def sources(root):
    report,bundle = source.source_bundle(root)
    target = PackedNativeRQ(bundle['dimensions'],bundle['config'],bundle['sizes'],
                            'no_encoder_quantizer',bundle['statistics'])
    target.load_state_dict(bundle['target'],strict=True); target.eval().requires_grad_(False)
    need(fingerprint(target.state_dict()) == bundle['target_fingerprint'], 'ENERGY_TARGET_PIN')
    fitting = [i for i in bundle['items'] if i['side'] == 'train']
    need(len(fitting) == 5 and sum(len(i['codes']) for i in fitting) == 66, 'ENERGY_REAL_COHORT')
    return report,bundle,target,fitting


def code():
    root = Path(__file__).resolve().parents[1]; paths = {}
    for module in tuple(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name and Path(name).resolve().is_relative_to(root):
            path = Path(name).resolve(); paths[str(path.relative_to(root))] = file_sha(path)
    return paths


def controls(probe):
    s,t,v = deepcopy(probe); result = {}
    reject = source.runtime.reject
    result['empty'] = reject(lambda:candidate.energy_loss([],[],v))
    result['missing_slot'] = reject(lambda:candidate.energy_loss([{k:x for k,x in s[0].items()
                                       if k != next(iter(v))}],t[:1],v))
    result['memory'] = reject(lambda:candidate.energy_loss(s,t,v,working_bytes=1))
    result['variance'] = reject(lambda:candidate.energy_loss(s,t,{n:float('nan') for n in v}))
    broken = deepcopy(s); broken[0][next(iter(v))].view(-1)[0] = float('nan')
    result['nonfinite'] = reject(lambda:candidate.energy_loss(broken,t,v))
    result['shape'] = reject(lambda:candidate.energy_loss(s,t[:-1],v))
    return result


def isolated(probe, original):
    saved = []
    for empirical in (False,True):
        for name,fn in (('reference',original),('candidate',candidate.energy_loss)):
            rows = []; last = None
            # Alternate complete invocations; every tensor is a real retained sample.
            for _ in range(7):
                s,t,v = deepcopy(probe)
                for row in s:
                    for x in row.values(): x.requires_grad_(True)
                began = time.perf_counter(); out = fn(s,t,v,empirical=empirical); out['loss'].backward()
                rows.append(time.perf_counter()-began)
                last = dict(output=cpu(out),gradient=cpu([{n:x.grad for n,x in row.items()} for row in s]))
            saved.append(dict(name=name,empirical=empirical,seconds=rows,payload=last))
    return saved


def run(a):
    began = time.perf_counter(); cpu0 = time.process_time(); a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    report,bundle,target,items = sources(a.root); policy = dict(bundle['policy'],device='cpu')
    frozen = fingerprint(target.state_dict()); phases = {'authentication':time.perf_counter()-began}
    original = base.energy_loss; files = {}; records = {}; first_probe = []
    for architecture in base.ARCHITECTURES:
        records[architecture] = {}
        for arm,fn in (('reference',original),('candidate',candidate.energy_loss)):
            model,opt = source.runtime.make(bundle,target,architecture,'cpu'); rows = []
            name = architecture+'-'+arm+'-initial.pt'
            files[name] = save(a.output/name,dict(model=model.state_dict(),optimizer=opt.state_dict()))
            def wrapped(samples,targets,variances,**kw):
                if not first_probe: first_probe.append(cpu((samples,targets,variances)))
                return fn(samples,targets,variances,**kw)
            base.energy_loss = wrapped
            try:
                for step,item in enumerate(items):
                    t0 = time.perf_counter(); c0 = time.process_time()
                    update,gradients,noise = efficient.update(model,opt,item,step,policy,target=target)
                    elapsed = time.perf_counter()-t0; cputime = time.process_time()-c0
                    payload = dict(update=update,gradients=gradients,noise=noise,
                                   model=model.state_dict(),optimizer=opt.state_dict())
                    name = f'{architecture}-{arm}-{step+1}.pt'
                    files[name] = save(a.output/name,payload)
                    rows.append(dict(step=step+1,wall=elapsed,cpu=cputime,file=name,**files[name]))
                held = next(i for i in bundle['items'] if i['side'] == 'query')
                negative = source.runtime.controls(model,opt,items[0],held,target,len(items),policy)
            finally:
                base.energy_loss = original
            records[architecture][arm] = dict(rows=rows,controls=negative)
            print(json.dumps(dict(architecture=architecture,arm=arm,wall=sum(r['wall'] for r in rows))),flush=True)
    phases['updates_controls_and_persistence'] = time.perf_counter()-began-sum(phases.values())
    tick = time.perf_counter(); probe = first_probe[0]
    files['probe.pt'] = save(a.output/'probe.pt',probe)
    bench = isolated(probe,original); files['isolated.pt'] = save(a.output/'isolated.pt',bench)
    negative = controls(probe)
    equality = {n:[records[n]['reference']['rows'][i]['state'] == records[n]['candidate']['rows'][i]['state']
                   for i in range(len(items))] for n in records}
    isolated_equal = [fingerprint(bench[i]['payload']) == fingerprint(bench[i+1]['payload']) for i in (0,2)]
    need(fingerprint(target.state_dict()) == frozen, 'ENERGY_TARGET_CHANGED')
    phases['isolated_and_controls'] = time.perf_counter()-tick
    result = dict(complete=True,device='cpu',cuda_qualified=False,source=source.SOURCE,source_state=report['bundle_state'],
        code=code(),records=records,files=files,equality=equality,isolated_equal=isolated_equal,controls=negative,
        scope=dict(train_documents=5,train_targets=66,architectures=3,arms=2,updates=30,
                   native_dimensions=bundle['dimensions']),
        isolated_times=[{k:v for k,v in r.items() if k != 'payload'} for r in bench],
        candidate_eligible=all(all(x) for x in equality.values()) and all(isolated_equal),
        phases=phases,wall=time.perf_counter()-began,cpu=time.process_time()-cpu0,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,acceptance='author-checked only')
    with (a.output/'report.json').open('x') as f:
        json.dump(result,f,sort_keys=True,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    print(json.dumps(dict(report=file_sha(a.output/'report.json'),equality=equality,isolated_equal=isolated_equal)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    try: run(a)
    except BaseException as e:
        if a.output.is_dir():
            with (a.output/'failure.json').open('x') as f: json.dump(dict(error=type(e).__name__,message=str(e)),f)
        raise
