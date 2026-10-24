"""Export an authenticated native proof once for device-specific placement workers."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch

from audit import legal_optimizer_census_readback as previous
from claims.process_conditions import write_committed
from model.native_population_training import cpu, fingerprint
from text.constellation_corpus import file_sha, need


def run(a):
    started = time.perf_counter(); a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    prior = previous.prior
    q = prior.prior.branch.proof.pinned(a.root/previous.QUAL[0], previous.QUAL[1])
    c = prior.prior.branch.proof.pinned(a.root/previous.COLD[0], previous.COLD[1])
    need(q['complete'] and c['complete'] and c['previous_pin'] == previous.QUAL[1]
         and q['binding'] == c['binding'] and q['results_pin'] == c['results_pin']
         and q['binding']['code'] == prior.code(), 'PLACEMENT_RUNTIME_SOURCE_AUTHORITY')
    target, ref, _, items, source = prior.prior.source(a.root)
    need(ref == q['binding']['target'] and source == q['binding']['source'], 'PLACEMENT_RUNTIME_SOURCE_CHANGED')
    for item in items:
        item.pop('hidden', None)
        item['label_pin'] = fingerprint((item['codes'], item['counts']))
    statistics = {kind: {n: getattr(target, prefix+str(i)).detach().clone() for i, n in enumerate(target.roster)}
                  for kind, prefix in [('centers', 'center_'), ('variance', 'variance_'), ('positions', 'position_')]}
    result_path = a.root/previous.QUAL[0]/'results.json'
    need(file_sha(result_path) == q['files']['results.json'] == q['results_pin'], 'PLACEMENT_RUNTIME_RESULTS_PIN')
    results = json.loads(result_path.read_bytes())
    bundle = cpu(dict(target=target.state_dict(), config=target.config, dimensions=target.dimensions,
                      sizes=target.sizes, statistics=statistics, target_ref=ref, items=items, policy=prior.POLICY,
                      source=source, cpu_reference=results, target_fingerprint=fingerprint(target.state_dict())))
    # Prior summary embeds the complete per-update records; hash the authority before using them.
    need(set(bundle['cpu_reference']) == {'shared', 'own_slot', 'whole_proposals'}, 'PLACEMENT_RUNTIME_ARMS')
    f = a.output/'bundle.pt'; torch.save(bundle, f)
    restored = torch.load(f, map_location='cpu', weights_only=True)
    need(fingerprint(restored) == fingerprint(bundle), 'PLACEMENT_RUNTIME_BUNDLE_READBACK')
    pins = previous.code()
    for n in ['audit/legal_placement_runtime_source.py', 'model/native_placement_efficient.py',
              'model/native_device_transport.py', 'audit/legal_placement_runtime_readback.py']:
        pins[n] = file_sha(Path(__file__).resolve().parents[1]/n)
    for n, pin in pins.items():
        f = a.output/'source'/n; f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes((Path(__file__).resolve().parents[1]/n).read_bytes())
        need(file_sha(f) == pin, 'PLACEMENT_RUNTIME_SOURCE_COPY')
    need(not torch.cuda.is_initialized(), 'PLACEMENT_RUNTIME_SOURCE_GPU')
    report = dict(complete=True, bundle_sha256=file_sha(a.output/'bundle.pt'), bundle_state=fingerprint(bundle),
                  code=pins, qualification=previous.QUAL[1], cold=previous.COLD[1], source=source,
                  documents=len(items), atoms=sum(len(i['frames']) for i in items),
                  train_documents=sum(i['side']=='train' for i in items),
                  train_targets=sum(len(i['codes']) for i in items if i['side']=='train'),
                  seconds=time.perf_counter()-started, number_status='measured', acceptance='author-checked')
    pin = write_committed(a.output/'report.json', report)
    print(json.dumps(dict(complete=True, pin=pin, seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try: run(a)
    except BaseException as error:
        if a.output.exists(): write_committed(a.output/'failure.json', dict(error=repr(error)))
        raise
