"""Census, measure and cold-read a proposal with the unchanged frozen instruments."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.window_readback import readback
from claims.codata_source import check, sha
from claims.next_echo_population import NextEchoPopulation
from claims.process_conditions import write_committed
from lens.catalogue import by_name
from lens.learned import snapshot_dir
from lens.lenscommon import run_cli, take_lease
from text.window_readings import load_chunks, measure, RUNTIME
from transformers import AutoTokenizer
import torch


def census(config):
    p = NextEchoPopulation(config['population'],config['population_pin'])
    records,_ = load_chunks(p.root,p.pin)
    texts = [r['text'] for r in records]
    lanes = {}
    for name in config['lenses']:
        spec = by_name(name)
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot_dir(spec['repo'])),trust_remote_code=False)
        encoded = tokenizer([spec.get('prefix','')+t for t in texts],truncation=True,
            max_length=spec['max_len'],return_overflowing_tokens=True,padding=False)
        counts = Counter(encoded['overflow_to_sample_mapping'])
        check(set(counts) == set(range(len(texts))) and max(counts.values()) <= 8,'NEXT_NATIVE_CENSUS')
        lanes[name] = dict(windows=[counts[i] for i in range(len(texts))],
            tokens=sum(map(len,encoded['input_ids'])),revision=snapshot_dir(spec['repo']).name)
    return dict(population=p.pin,lanes=lanes,inputs=len(texts),utf8_bytes=sum(len(t.encode()) for t in texts),
        stream_references=len(p.refs),probe=p.probe['unit'],scientific_admission=False,serving=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=('census','measure','read'),required=True)
    a = parser.parse_args()
    started = time.perf_counter()
    config = json.loads(a.config.read_bytes())
    root = Path(config['native'])
    census_path = Path(config['census'])
    if a.stage == 'census':
        value = census(config)
        check(not census_path.exists(),'NEXT_NATIVE_CENSUS_EXISTS')
        value.update(wall_seconds=time.perf_counter()-started,code=sha(Path(__file__).read_bytes()))
        print(write_committed(census_path,value),flush=True)
        print(json.dumps(value),flush=True)
        return
    plan = json.loads(census_path.read_bytes())
    check(plan['code'] == sha(Path(__file__).read_bytes()) and
          {k:v for k,v in plan.items() if k not in ('wall_seconds','code')} == census(config),
          'NEXT_NATIVE_PLAN_CHANGED')
    if a.stage == 'read':
        value = readback(root,Path(config['population']))
        report = json.loads((root/'report.json').read_bytes())
        check(report['calibration_report_sha256'] == plan['population'],'NEXT_NATIVE_SOURCE')
        for name in config['lenses']:
            import numpy as np
            with np.load(root/name/'slots.npz',allow_pickle=False) as arrays:
                check(np.diff(arrays['offsets']).tolist() == plan['lanes'][name]['windows'],
                      'NEXT_NATIVE_WINDOWS')
        value.update(plan=sha(census_path.read_bytes()),native=sha((root/'report.json').read_bytes()),
                     wall_seconds=time.perf_counter()-started,acceptance='author-checked')
        print(write_committed(Path(config['readback']),value),flush=True)
        return
    check(not root.exists() and torch.cuda.is_available(),'NEXT_NATIVE_RUNTIME')
    gpu = subprocess.check_output(['nvidia-smi'],text=True)
    take_lease('28','next-echo-native')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records,sources = load_chunks(Path(config['population']),plan['population'])
    root.mkdir()
    (root/'gpu-before.txt').write_text(gpu)
    published,errors = {},[]
    for name in config['lenses']:
        try:
            published[name] = measure(name,records,root)
            check(published[name]['windows'] == sum(plan['lanes'][name]['windows']),'NEXT_NATIVE_COUNT')
        except (ValueError,RuntimeError,OSError) as error:
            errors.append(dict(lens=name,error=str(error)))
            print(json.dumps(errors[-1]),flush=True)
            torch.cuda.empty_cache()
    value = dict(schema=RUNTIME,calibration_report_sha256=plan['population'],sources=sources,
        inputs=len(records),requested=config['lenses'],lenses=published,errors=errors,
        complete=not errors,scientific_admission=False,served=False)
    print(write_committed(root/'report.json',value),flush=True)
    check(not errors,'NEXT_NATIVE_INCOMPLETE')


if __name__ == '__main__':
    run_cli(main)
