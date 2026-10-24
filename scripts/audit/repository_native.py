"""Bounded native measurement through the retained window runtime and cold reader."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.window_readback import readback
from claims.codata_source import check,sha
from claims.process_conditions import write_committed
from claims.repository_population import RepositoryPopulation,build
from lens.catalogue import by_name
from lens.learned import snapshot_dir
from lens.lenscommon import run_cli,take_lease,LensError
from text.window_readings import load_chunks,measure,RUNTIME
from transformers import AutoTokenizer
import torch


def census(pop,lenses):
    records,_ = load_chunks(pop.root,pop.pin)
    texts = [r['text'] for r in records]
    lanes = {}
    for name in lenses:
        spec = by_name(name)
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot_dir(spec['repo'])),trust_remote_code=False)
        encoded = tokenizer([spec.get('prefix','')+t for t in texts],truncation=True,
            max_length=spec['max_len'],return_overflowing_tokens=True,padding=False)
        counts = Counter(encoded['overflow_to_sample_mapping'])
        check(set(counts) == set(range(len(texts))) and max(counts.values()) <= 8,
              'REPOSITORY_NATIVE_WINDOW_BOUND')
        lanes[name] = dict(windows=[counts[i] for i in range(len(texts))],
            tokens=sum(map(len,encoded['input_ids'])),revision=snapshot_dir(spec['repo']).name)
    check(any(1800 <= len(t) <= 3000 for t in texts),'REPOSITORY_NATIVE_REAL_PROBE')
    return dict(population=pop.pin,inputs=len(texts),utf8_bytes=sum(len(t.encode()) for t in texts),lanes=lanes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=('build','census','measure','read'),required=True)
    a = parser.parse_args()
    started = time.perf_counter()
    c = json.loads(a.config.read_bytes())
    if a.stage == 'build':
        pin = build(c['observations'],c['observation_pin'],c['population'])
        pop = RepositoryPopulation(c['population'],pin)
        print(json.dumps(dict(pin=pin,inputs=pop.report['inputs'],unique_bytes=pop.report['unique_utf8_bytes'],
            fields=pop.report['field_occurrences'],occurrence_bytes=pop.report['occurrence_bytes'],
            wall_seconds=time.perf_counter()-started)),flush=True)
        return
    pin = (Path(c['population'])/'COMMITTED').read_text()
    pop = RepositoryPopulation(c['population'],pin)
    value = census(pop,c['lenses'])
    if a.stage == 'census':
        value.update(code=sha(Path(__file__).read_bytes()),wall_seconds=time.perf_counter()-started)
        print(write_committed(Path(c['census']),value),flush=True)
        print(json.dumps(value),flush=True)
        return
    plan = json.loads(Path(c['census']).read_bytes())
    check(plan['code'] == sha(Path(__file__).read_bytes()) and
          value == {k:v for k,v in plan.items() if k not in ('code','wall_seconds')},
          'REPOSITORY_NATIVE_CENSUS_CHANGED')
    root = Path(c['native'])
    if a.stage == 'read':
        result = readback(root,pop.root)
        import numpy as np
        for name in c['lenses']:
            with np.load(root/name/'slots.npz',allow_pickle=False) as arrays:
                check(np.diff(arrays['offsets']).tolist() == plan['lanes'][name]['windows'],
                      'REPOSITORY_NATIVE_WINDOW_COUNT')
        result.update(plan=sha(Path(c['census']).read_bytes()),native=sha((root/'report.json').read_bytes()),
                      wall_seconds=time.perf_counter()-started,acceptance='author-checked')
        print(write_committed(Path(c['readback']),result),flush=True)
        return
    check(not root.exists() and torch.cuda.is_available(),'REPOSITORY_NATIVE_RUNTIME')
    gpu = subprocess.check_output(['nvidia-smi'],text=True)
    take_lease('28','repository-native')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records,sources = load_chunks(pop.root,pin)
    root.mkdir()
    (root/'gpu-before.txt').write_text(gpu)
    ready = time.perf_counter()
    published,errors = {},[]
    for name in c['lenses']:
        try:
            published[name] = measure(name,records,root)
            check(published[name]['windows'] == sum(plan['lanes'][name]['windows']),
                  'REPOSITORY_NATIVE_WINDOW_COUNT')
        except (ValueError,RuntimeError,OSError,LensError) as exc:
            errors.append(dict(lens=name,error=str(exc)))
            print(json.dumps(errors[-1]),flush=True)
            torch.cuda.empty_cache()
    report = dict(schema=RUNTIME,calibration_report_sha256=pin,sources=sources,inputs=len(records),
        requested=c['lenses'],lenses=published,errors=errors,complete=not errors,
        scientific_admission=False,served=False,phases=dict(census_and_setup=ready-started,
        measurement_and_publication=time.perf_counter()-ready))
    print(write_committed(root/'report.json',report),flush=True)
    check(not errors,'REPOSITORY_NATIVE_INCOMPLETE')


if __name__ == '__main__':
    run_cli(main)
