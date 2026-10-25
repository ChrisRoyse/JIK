"""Canonical CUDA remeasurement of free-generated atoms, with separate cold replay."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'lens')]
import torch
from audit.free_guard_native_contract import owner_fd
from claims.process_conditions import write_committed as write
from lenscommon import take_lease, release_lease
from text.answer_native_measure import lane
from text.answer_native_plan import read
from text.constellation_corpus import need
from text.free_guard_native_plan import FreeGuardNativePlan, code


def run(a):
    began = tick = time.perf_counter(); a.output.mkdir(exist_ok=False); leased = False
    results = {}; errors = []
    try:
        owner_fd(a.owner_fd)
        plan = FreeGuardNativePlan(a.plan, a.pin); reference = None
        if a.reference is not None:
            old = read(a.reference / 'report.json', a.reference_pin)
            need(old['complete'] and old['schema'] == 'free-guard-native-measurements-v1'
                 and old['plan'] == dict(path=str(a.plan), pin=a.pin) and old['code'] == code(), 'FREE_NATIVE_COLD_SOURCE')
            reference = a.reference
        phases = dict(authenticate=time.perf_counter() - tick); tick = time.perf_counter()
        residents = subprocess.check_output(['nvidia-smi'], text=True)
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
        need(not active.strip(), 'FREE_NATIVE_GPU_BUSY')
        need(torch.cuda.is_available(), 'FREE_NATIVE_GPU_REQUIRED')
        take_lease('29', 'free-generated-surface-native-measurement'); leased = True
        (a.output / 'gpu-before.txt').write_text(residents)
        torch.set_num_threads(32); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        phases['lease'] = time.perf_counter() - tick; tick = time.perf_counter()
        for name in plan.report['roster']:
            try:
                owner_fd(a.owner_fd); results[name] = lane(plan, name, a.output / name, reference)
            except Exception as exc:
                failure = dict(name=name, error=repr(exc)); errors.append(failure)
                write(a.output / (name + '-failure.json'), failure); torch.cuda.empty_cache()
        phases['canonical_lanes'] = time.perf_counter() - tick; tick = time.perf_counter()
        plan.fresh(); need(plan.report['code'] == code(), 'FREE_NATIVE_CODE_CHANGED'); owner_fd(a.owner_fd)
        phases['seal'] = time.perf_counter() - tick; seconds = time.perf_counter() - began
        need(.9 <= sum(phases.values()) / seconds <= 1.001, 'FREE_NATIVE_PHASES')
        result = dict(schema='free-guard-native-measurements-v1', complete=not errors, errors=errors,
            plan=dict(path=str(a.plan), pin=a.pin), code=code(), instruments=results, roster=plan.report['roster'],
            inputs=len(plan.records), source_counts=plan.report['source_counts'],
            cold_reference=None if reference is None else dict(path=str(reference), pin=a.reference_pin),
            cold_all_inputs_exact=reference is not None and not errors, gpu_used=True, optimizer_updates=0,
            scientific_admission=False, serving=False, semantic_validity_established=False,
            phases=phases, seconds=seconds, acceptance='author-checked')
        pin = write(a.output / 'report.json', result)
        print(json.dumps(dict(complete=not errors, pin=pin, seconds=seconds)), flush=True)
        need(not errors, 'FREE_NATIVE_MEASUREMENT_REFUSED')
    finally:
        if leased:
            release_lease()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('plan', 'output'):
        p.add_argument('--' + n, type=Path, required=True)
    p.add_argument('--pin', required=True); p.add_argument('--reference', type=Path)
    p.add_argument('--reference-pin'); p.add_argument('--owner-fd', type=int, required=True)
    a = p.parse_args()
    try:
        run(a)
    except BaseException as exc:
        if a.output.exists():
            write(a.output / 'failure.json', dict(error=repr(exc)))
        raise
