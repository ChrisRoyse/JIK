"""Cold replay of legal canonical window evidence, with independent token census."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit.legal_native_probe import setup_gpu
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from lens.catalogue import by_name
from lens.learned import LearnedLens, weights_sha256, preprocessing_sha256
from lenscommon import release_lease
import text_windows
from text.canonical_windows import contract, exact_reference, forward
from text.constellation_corpus import canonical, digest, file_sha, need
from text.legal_reuse_input import ReuseInputs


def token_census(lens, text, trace):
    # Separate unpadded tokenizer call; every active saved token and its source
    # offset must agree, including overflow windows beyond the historical cap.
    encoded = lens.tok(lens.spec.get('prefix', '')+text, truncation=True, max_length=lens.spec['max_len'],
        padding=False, return_overflowing_tokens=True, return_offsets_mapping=True, return_special_tokens_mask=True)
    need(len(encoded['input_ids']) == len(trace['windows']), 'NATIVE_TOKEN_WINDOW_COVERAGE')
    for i, window in enumerate(trace['windows']):
        active = int(sum(window['attention_mask']))
        for key in ('input_ids', 'attention_mask', 'special_tokens_mask', 'offset_mapping'):
            expected = encoded[key][i]
            if key == 'offset_mapping': expected = [list(v) for v in expected]
            need(window[key][:active] == expected, 'NATIVE_ACTIVE_TOKEN_IDENTITY')
        need(not any(window['attention_mask'][active:]), 'NATIVE_PADDING')
    text_windows.validate(trace, 'dense', len(trace['windows']), text=text)
    return len(trace['windows'])


def lane(name, root, reference, texts, output):
    start = tick = time.perf_counter(); destination = root/name
    need(file_sha(destination/'report.json') == reference['pin'], 'NATIVE_LANE_REPORT_PIN')
    report = json.loads((destination/'report.json').read_bytes())
    for filename, item in report['files'].items():
        need((destination/filename).stat().st_size == item['bytes']
             and file_sha(destination/filename) == item['sha256'], 'NATIVE_LANE_FILE')
    traces = json.loads((destination/'traces.json').read_bytes())
    with np.load(destination/'reference.npz', allow_pickle=False) as saved:
        values, bounds = saved['vectors'].copy(), saved['bounds'].copy()
    lens = LearnedLens(by_name(name), 'cuda'); identity = contract(lens)
    need(identity == report['contract'] and digest(canonical(identity)) == reference['lens_id'], 'NATIVE_CONTRACT')
    phases = dict(authenticate_and_load=time.perf_counter()-tick); tick = time.perf_counter()
    windows = sum(token_census(lens, text, trace) for text, trace in zip(texts, traces))
    need(windows == len(values) == report['windows'] and len(traces) == len(texts), 'NATIVE_ALL_WINDOWS')
    phases['independent_tokens_and_spans'] = time.perf_counter()-tick; tick = time.perf_counter()
    replay, offsets, actual, timing = forward(lens, texts)
    exact_reference(values, bounds, traces, replay, offsets, actual, list(range(len(texts))))
    phases['cold_native_replay'] = time.perf_counter()-tick; tick = time.perf_counter()
    need(weights_sha256(lens.snap)[0] == lens.sha256 and preprocessing_sha256(lens.snap) == lens.preprocessing,
         'NATIVE_FROZEN_DRIFT')
    corrupt_source = deepcopy(traces[0]); corrupt_source['surface_sha256'] = '0'*64
    corrupt_window = deepcopy(traces[0]); corrupt_window['windows'] = []
    shifted = replay.copy(); shifted[0, 0] = np.nextafter(shifted[0, 0], np.float32(np.inf))
    reversed_order = np.arange(len(texts))[::-1].tolist()
    controls = dict(source_binding=rejected(lambda: text_windows.validate(corrupt_source, 'dense', len(traces[0]['windows']), text=texts[0])),
        missing_window=rejected(lambda: text_windows.validate(corrupt_window, 'dense', len(traces[0]['windows']), text=texts[0])),
        one_ulp_coordinate_drift=rejected(lambda: exact_reference(values, bounds, traces, shifted, offsets, actual, list(range(len(texts))))),
        wrong_context_order=rejected(lambda: exact_reference(values, bounds, traces, replay, offsets, actual, reversed_order)))
    phases['frozen_controls_and_pins'] = time.perf_counter()-tick
    result = dict(name=name, lens_id=reference['lens_id'], producer_pin=reference['pin'], inputs=len(texts), windows=windows,
        maximum_windows=int(np.diff(bounds).max()), all_active_tokens_checked=True, all_spans_checked=True,
        cold_arrays_byte_identical=True, frozen_unchanged=True, controls=controls, timing=timing,
        phases_seconds=phases, seconds=time.perf_counter()-start, scientific_admission=False)
    need(sum(phases.values())/result['seconds'] >= .9, 'NATIVE_READBACK_TIMING')
    pin = write_committed(output/(name+'.json'), result)
    del lens; torch.cuda.empty_cache()
    print(json.dumps(dict(name=name, pin=pin, windows=windows, seconds=result['seconds'])), flush=True)
    return dict(pin=pin, lens_id=reference['lens_id'])


def run(root, pin, output):
    start = time.perf_counter(); output.mkdir(parents=True, exist_ok=False)
    need(file_sha(root/'report.json') == pin, 'NATIVE_REPORT_PIN')
    report = json.loads((root/'report.json').read_bytes())
    need(report['complete'] and not report['errors'], 'NATIVE_PRODUCER_INCOMPLETE')
    for filename, item in report['files'].items():
        need((root/filename).stat().st_size == item['bytes'] and file_sha(root/filename) == item['sha256'], 'NATIVE_INPUT_PIN')
    bank = ReuseInputs(Path(report['source']['path']), report['source']['pin'])
    inputs = json.loads((root/'inputs.json').read_bytes())
    need(len(inputs) == report['inputs'] and len({r['index'] for r in inputs}) == len(inputs), 'NATIVE_INPUT_COUNT')
    for row in inputs: need(row == bank.input(row['index']), 'NATIVE_SOURCE_BYTES')
    resident = setup_gpu(); results = {}; errors = []
    try:
        (output/'gpu-before.txt').write_text(resident)
        for name in report['roster']:
            try: results[name] = lane(name, root, report['instruments'][name], [r['text'] for r in inputs], output)
            except Exception as exc:
                error = dict(name=name, error=type(exc).__name__+':'+str(exc)); errors.append(error)
                write_committed(output/(name+'-failure.json'), error); print(json.dumps(error), flush=True)
                torch.cuda.empty_cache()
        bank.fresh()
        result = dict(schema='legal-canonical-native-cold-readback-v1', complete=not errors, errors=errors,
            source=dict(path=str(root), pin=pin), instruments=results, inputs=len(inputs),
            code_sha256=file_sha(Path(__file__)), seconds=time.perf_counter()-start,
            all_selected_source_bytes_checked=True, full_corpus_measured=False, model_updates=0,
            scientific_admission=False, number_status='measured', acceptance='author-checked')
        result_pin = write_committed(output/'report.json', result)
        print(json.dumps(dict(pin=result_pin, complete=not errors)), flush=True)
        need(not errors, 'NATIVE_COLD_REPLAY_REFUSED')
    finally: release_lease()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--pin', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.root, a.pin, a.output)
