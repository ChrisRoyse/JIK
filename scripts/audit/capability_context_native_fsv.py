"""Read native source windows independently, then prove complete typed coding context and masks."""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lens'))
from audit.capability_revision_fsv import rejected
from audit.capability_delta_native_fsv import registry_check
from audit.window_readback import readback
from claims.capability_population import need
from claims.process_conditions import write_committed
from kernel.capability_context_registry import sources
from kernel.capability_registry import freeze_together
from model.capability_context_input import ContextNative, ContextStore
from model.contracts import ModelConfig
from senses.trajectory_population import read_pinned
from text.evaluation_multiscale import file_sha


def build(a):
    start = tick = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    write_committed(a.output / 'window-readback.json', readback(a.native,a.plan))
    phases = dict(native_token_offset_readback=time.perf_counter()-tick); tick = time.perf_counter()
    store = ContextStore(a.plan,file_sha(a.plan/'report.json'),a.native,file_sha(a.native/'report.json'))
    pop = next(iter(store.populations.values())); provider = ContextNative(pop.root,pop.pin,store=store)
    arrays, traces = {}, {}
    for bank, root in store.roots.items():
        for spec in store.specs:
            with np.load(root/spec.name/'slots.npz',allow_pickle=False) as z:
                arrays[bank,spec.name] = z['vectors'].copy(),z['offsets'].copy()
            with gzip.open(root/spec.name/'traces.jsonl.gz','rt',encoding='utf-8') as stream:
                traces[bank,spec.name] = [json.loads(line) for line in stream]
    phases['shared_store_and_separate_disk_reads'] = time.perf_counter()-tick; tick = time.perf_counter()
    fields = {}
    for key, field in pop.fields.items():
        fields[key] = {}
        for spec in store.specs:
            actual, desc = provider.field(key,spec.name)
            pieces = []
            for part in desc['parts']:
                bank,row = part['origin'],part['row']; values, offsets = arrays[bank,spec.name]
                lo,hi = map(int,offsets[row:row+2]); trace = traces[bank,spec.name][row]
                raw = field['text'].encode()[slice(*part['source_byte_span'])]
                need(trace['vector_span'] == part['source_window_span'] == [lo,hi]
                     and trace['trace']['surface_sha256'] == part['source_sha256']
                     == hashlib.sha256(raw).hexdigest(), 'CONTEXT_NATIVE_SOURCE_JOIN')
                pieces.append(values[lo:hi])
            need(np.array_equal(actual,np.concatenate(pieces)), 'CONTEXT_NATIVE_WINDOW_CONSERVATION')
            fields[key][spec.name] = dict(windows=len(actual),dimensions=actual.shape[1],
                sha256=hashlib.sha256(actual.astype('<f4').tobytes()).hexdigest(),description=desc)
    def width(key,side):
        return max(sum(fields[f][s.name]['windows'] for f in pop.tasks[key][side]) for s in store.specs)
    selected = {next(iter(pop.tasks)),max(pop.tasks,key=lambda k:width(k,'context')),
                max(pop.tasks,key=lambda k:width(k,'targets'))}
    selected.add(next(k for k,t in pop.tasks.items() if t['provenance']['parent_kind']=='contrast_failure_with_supplied_repair'))
    tensor_cases = {}
    for i,key in enumerate(sorted(selected)):
        saved, descriptions = {}, {}
        for side in ('context','targets'):
            slots, desc = provider.batch(key,side=side,config=ModelConfig(),device='cpu')
            for spec in store.specs:
                actual = slots[spec.name].values.numpy()[0]
                expected = np.concatenate([provider.field(f,spec.name)[0] for f in pop.tasks[key][side]])
                need(np.array_equal(actual,expected),'CONTEXT_NATIVE_TENSOR_CONSERVATION')
                saved[side+'__'+spec.name] = actual
            descriptions[side] = desc
        name = 'tensor-'+str(i)+'.npz'; np.savez(a.output/name,**saved)
        with np.load(a.output/name,allow_pickle=False) as z:
            need(set(z.files)==set(saved) and all(np.array_equal(z[n],v) for n,v in saved.items()),
                 'CONTEXT_NATIVE_TENSOR_PERSISTENCE')
        tensor_cases[key] = dict(file=name,descriptions=descriptions)
    provider.fresh()
    write_committed(a.output/'field-receipts.json',{pop.pin:fields})
    write_committed(a.output/'tensor-cases.json',{pop.pin:tensor_cases})
    phases['all_field_and_extreme_tensor_joins'] = time.perf_counter()-tick; tick = time.perf_counter()
    registry = json.loads(a.registry.read_bytes()); prior = registry['previous']
    populations, history = sources(Path(prior['path']),prior['pin'],pop.root,pop.pin)
    replay = freeze_together(populations,history)
    need(all(registry[k]==v for k,v in replay.items() if k!='wall_seconds'),'CONTEXT_REGISTRY_REPLAY')
    registry_check(registry,populations,history)
    bad = deepcopy(registry); split = bad['splits'][pop.pin]; split['train'].append(split['queries'].pop())
    controls = dict(split_leak=rejected(lambda:registry_check(bad,populations,history)))
    entry = store.mapping['populations'][pop.pin]['units'][0]; original = entry['row']
    try:
        entry['row'] = original+1; controls['changed_native_row'] = rejected(provider.fresh)
    finally: entry['row'] = original
    provider.fresh()
    phases['registry_replay_and_controls'] = time.perf_counter()-tick; tick = time.perf_counter()
    files = {p.name:dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in a.output.iterdir() if p.is_file()}
    phases['final_hash_census'] = time.perf_counter()-tick; wall = time.perf_counter()-start
    result = dict(schema='capability-context-native-fsv-v1',complete=True,author_checked=True,
        plan=store.plan_pin,native=store.native_pin,registry=file_sha(a.registry),
        populations={pop.pin:dict(fields=len(fields),tasks=len(pop.tasks),binding=provider.binding,
            tensor_cases=len(tensor_cases),max_context_windows=max(width(k,'context') for k in pop.tasks),
            field_windows={s.name:sum(f[s.name]['windows'] for f in fields.values()) for s in store.specs})},
        instruments={s.name:s.identity for s in store.specs},all_native_windows_preserved=True,
        historical_split_membership_unchanged=True,controls=controls,files=files,
        phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,
        scientific_admission=False,learned_capability=False,live_kernel_connected=False,
        producer_sha256=file_sha(Path(__file__)))
    need(result['phase_coverage']>=.9,'CONTEXT_NATIVE_TIMING')
    pin=write_committed(a.output/'report.json',result)
    print(json.dumps(dict(report_sha256=pin,populations=result['populations'],controls=controls,wall_seconds=wall)))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','native','registry','output'): p.add_argument('--'+name,type=Path,required=True)
    build(p.parse_args())
