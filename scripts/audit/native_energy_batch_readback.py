"""Cold equality/restart readback and alternating complete-update CPU timings."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch

from audit import native_energy_batch_cpu as producer
from model import native_placement_training as base
from model import native_placement_efficient as efficient
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha,need


def checked(path,ref):
    need(file_sha(path) == ref['sha256'],'ENERGY_COLD_FILE')
    value = torch.load(path,map_location='cpu',weights_only=True)
    need(fingerprint(value) == ref['state'],'ENERGY_COLD_STATE')
    return value


def equal(a,b):
    need(fingerprint(a) == fingerprint(b),'ENERGY_COLD_DIFFERENT')


def run(a):
    began = time.perf_counter(); c0 = time.process_time(); a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    need(file_sha(a.source/'report.json') == a.pin,'ENERGY_COLD_REPORT')
    r = json.loads((a.source/'report.json').read_text()); repo = Path(__file__).resolve().parents[1]
    for n,h in r['code'].items(): need(file_sha(repo/n) == h,'ENERGY_COLD_CODE')
    need(r['complete'] and r['device'] == 'cpu' and not r['cuda_qualified'],'ENERGY_COLD_SCOPE')
    for n,ref in r['files'].items(): checked(a.source/n,ref)
    source,bundle,target,items = producer.sources(a.root); policy = dict(bundle['policy'],device='cpu')
    need(source['bundle_state'] == r['source_state'],'ENERGY_COLD_SOURCE')
    target_pin = fingerprint(target.state_dict()); original = base.energy_loss
    timings = []; cold = []; restarts = []; files = {}
    tick = time.perf_counter(); phases = dict(authentication=tick-began)
    for architecture in base.ARCHITECTURES:
        # Reference/candidate labels have no impact on the order of training items.
        for repetition in range(6):
            order = ('reference','candidate') if repetition % 2 == 0 else ('candidate','reference')
            for arm in order:
                model,opt = producer.source.runtime.make(bundle,target,architecture,'cpu')
                name = architecture+'-'+arm+'-initial.pt'
                initial = checked(a.source/name,r['files'][name])
                equal(dict(model=model.state_dict(),optimizer=opt.state_dict()),initial)
                base.energy_loss = original if arm == 'reference' else producer.candidate.energy_loss
                seconds = []
                try:
                    for step,item in enumerate(items):
                        before = time.perf_counter()
                        update,gradients,noise = efficient.update(model,opt,item,step,policy,target=target)
                        seconds.append(time.perf_counter()-before)
                        actual = dict(update=update,gradients=gradients,noise=noise,
                                      model=model.state_dict(),optimizer=opt.state_dict())
                        saved = checked(a.source/f'{architecture}-{arm}-{step+1}.pt',
                                        r['files'][f'{architecture}-{arm}-{step+1}.pt'])
                        equal(actual,saved)
                        peer = checked(a.source/f'{architecture}-reference-{step+1}.pt',
                                       r['files'][f'{architecture}-reference-{step+1}.pt'])
                        equal(actual,peer)
                        if repetition == 0: cold.append(dict(architecture=architecture,arm=arm,step=step+1,state=fingerprint(actual)))
                        if repetition == 5 and step == 2:
                            name = architecture+'-'+arm+'-restart.pt'; files[name] = producer.save(a.output/name,actual)
                            reopened = checked(a.output/name,files[name])
                            model,opt = producer.source.runtime.make(bundle,target,architecture,'cpu')
                            model.load_state_dict(reopened['model'],strict=True); opt.load_state_dict(reopened['optimizer'])
                            efficient.validate_optimizer(model,opt,policy,step+1)
                            restarts.append(dict(architecture=architecture,arm=arm,step=step+1,state=files[name]['state']))
                finally:
                    base.energy_loss = original
                timings.append(dict(architecture=architecture,arm=arm,repetition=repetition,
                                    order=order,seconds=seconds,total=sum(seconds)))
        print(json.dumps(dict(architecture=architecture,repetitions=6,equal=True,restarts=2)),flush=True)
    phases['replay_equality_and_timing'] = time.perf_counter()-tick; tick = time.perf_counter()
    probe = checked(a.source/'probe.pt',r['files']['probe.pt'])
    isolated = checked(a.source/'isolated.pt',r['files']['isolated.pt'])
    for i in (0,2): equal(isolated[i]['payload'],isolated[i+1]['payload'])
    for n,h in r['code'].items(): need(file_sha(repo/n) == h,'ENERGY_COLD_CODE')
    equal(target.state_dict(),bundle['target'])
    need(fingerprint(target.state_dict()) == target_pin,'ENERGY_COLD_TARGET_CHANGED')
    reject = producer.source.runtime.reject
    negative = dict(bad_file=reject(lambda:checked(a.source/'probe.pt',dict(r['files']['probe.pt'],sha256='0'*64))))
    artifact = checked(a.source/'shared-candidate-1.pt',r['files']['shared-candidate-1.pt'])
    broken = deepcopy(artifact); next(iter(broken['gradients'].values())).view(-1)[0] += 1
    negative['gradient'] = reject(lambda:equal(broken,artifact))
    broken = deepcopy(artifact); next(iter(broken['optimizer']['state'].values()))['exp_avg'].view(-1)[0] += 1
    negative['moment'] = reject(lambda:equal(broken,artifact))
    speed = {}
    for architecture in base.ARCHITECTURES:
        # First repetition warms the path; remaining repetitions are paired by order.
        selected = [x for x in timings if x['architecture'] == architecture and x['repetition'] > 0]
        ref = [x['total'] for x in selected if x['arm'] == 'reference']
        new = [x['total'] for x in selected if x['arm'] == 'candidate']
        speed[architecture] = dict(reference_seconds=ref,candidate_seconds=new,
            reference_median=statistics.median(ref),candidate_median=statistics.median(new),
            ratios=[b/a for a,b in zip(ref,new)],median_ratio=statistics.median(new)/statistics.median(ref))
    phases['controls_and_summary'] = time.perf_counter()-tick
    result = dict(complete=True,source=a.pin,own_code=producer.code(),cold_steps=cold,restarts=restarts,
        files=files,controls=negative,isolated_exact=True,paired_timings=timings,speed=speed,
        real_documents=5,real_targets=66,independent_updates=15,replayed_updates=180,
        device='cpu',cuda_qualified=False,phases=phases,wall=time.perf_counter()-began,cpu=time.process_time()-c0,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,acceptance='author-checked only')
    with (a.output/'report.json').open('x') as f:
        json.dump(result,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
    print(json.dumps(dict(report=file_sha(a.output/'report.json'),speed=speed)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('root','source','output'): p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--pin',required=True); a = p.parse_args()
    try: run(a)
    except BaseException as e:
        if a.output.is_dir():
            with (a.output/'failure.json').open('x') as f: json.dump(dict(error=type(e).__name__,message=str(e)),f)
        raise
