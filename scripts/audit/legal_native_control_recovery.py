"""Finish failed filesystem control setup using already-published cold lane receipts.

No encoder invocation: authenticate each old lane receipt against its original
stdout pin and the unchanged production shard before reusing its evidence.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
from audit.legal_native_bank_readback import controls
from claims.process_conditions import write_committed
from text.constellation_corpus import file_sha, need
from text.legal_native_store import NativeBank
from text.legal_reuse_input import ReuseInputs


def run(a):
    start = time.perf_counter(); a.output.mkdir(parents=True, exist_ok=False)
    need(file_sha(a.publication/'report.json') == a.pin, 'NATIVE_RECOVERY_PUBLICATION_PIN')
    report = json.loads((a.publication/'report.json').read_bytes())
    need(report['complete_scope'] and not report['errors'] and report['stage'] == 'prefix', 'NATIVE_RECOVERY_SCOPE')
    bank = NativeBank(Path(report['bank']['path']), report['bank']['configuration_pin'])
    source = ReuseInputs(Path(bank.config['source']['path']), bank.config['source']['pin'])
    logged = {}
    for line in a.log.read_text().splitlines():
        if line.startswith('{'):
            item = json.loads(line)
            if 'pin' in item and 'name' in item:
                need(item['name'] not in logged, 'NATIVE_RECOVERY_DUPLICATE_RECEIPT')
                logged[item['name']] = item
    need(set(logged) == set(bank.config['roster']) and not (a.old/'report.json').exists(), 'NATIVE_RECOVERY_LANES')
    results = {}; stamps = {}; first = None
    for name in bank.config['roster']:
        path = a.old/(name+'.json'); need(file_sha(path) == logged[name]['pin'], 'NATIVE_RECOVERY_LANE_PIN')
        lane = json.loads(path.read_bytes())
        need(lane['all_vectors_finite_unit'] and lane['all_source_bindings_checked'] and lane['cold_arrays_exact']
             and lane['cold_inputs'] >= 16 and lane['inputs'] == logged[name]['inputs'], 'NATIVE_RECOVERY_PRIOR_CHECKS')
        expected = report['instruments'][name]['shards']
        need([(s['path'], s['pin']) for s in lane['shards']] == [(s['path'], s['pin']) for s in expected],
             'NATIVE_RECOVERY_PRIOR_POPULATION')
        for ref in lane['shards']:
            lo, hi = map(int, ref['path'].split('-')); saved = bank.shard(name, lo)
            need(saved.pin == ref['pin'] and saved.report['end'] == hi, 'NATIVE_RECOVERY_UNCHANGED')
            stamps.update({str(p): list(value) for p, value in saved.stamps.items()})
            if first is None:
                row = saved.table.read_row_group(0).slice(1, 1).to_pylist()[0]
                first = (row, source.input(row['index']), int(saved.offsets[2]-saved.offsets[1]), saved)
        results[name] = dict(path=str(path), pin=logged[name]['pin'], inputs=lane['inputs'], windows=lane['windows'])
    negative = controls(bank, first, a.output); source.fresh()
    write_committed(a.output/'bank-file-stamps.json', stamps)
    result = dict(schema='legal-native-prefix-control-recovery-v1', complete=True, errors=[],
        publication=dict(path=str(a.publication), pin=a.pin), bank=report['bank'], instruments=results,
        controls=negative, original_log=dict(path=str(a.log), sha256=file_sha(a.log)),
        prior_reader_code=dict(path=str(a.old_code), sha256=file_sha(a.old_code)),
        original_failure='cross-device hard link during corruption-control setup; all lane receipts already published',
        new_encoder_calls=0, unchanged_shards_reauthenticated=True, code_sha256=file_sha(Path(__file__)),
        control_code_sha256=file_sha(Path(__file__).with_name('legal_native_bank_readback.py')),
        seconds=time.perf_counter()-start, model_updates=0, full_corpus_measured=False,
        scientific_admission=False, acceptance='author-checked', number_status='measured')
    pin = write_committed(a.output/'report.json', result)
    print(json.dumps(dict(pin=pin, complete=True, controls=negative, new_encoder_calls=0)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('publication', 'old', 'log', 'old-code', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--pin', required=True); run(p.parse_args())
