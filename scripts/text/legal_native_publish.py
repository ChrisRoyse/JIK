"""Publish one complete native range; unchanged proof inputs reuse frozen readings."""
import json
import os
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from claims.process_conditions import write_committed
from text.canonical_windows import forward
from text.constellation_corpus import canonical, digest, file_sha, need
from text.legal_native_store import IO_INPUTS, SCHEMA, durable_directory, shard_name

TRACE_SCHEMA = pa.schema([('index', pa.int64()), ('input_sha256', pa.string()), ('provenance', pa.binary())])


def proof_cache(root, name, proof, inputs):
    lane = root/name; raw = (lane/'report.json').read_bytes()
    need(digest(raw) == proof['instruments'][name]['pin'], 'NATIVE_PROOF_PIN')
    report = json.loads(raw)
    for filename, item in report['files'].items():
        need((lane/filename).stat().st_size == item['bytes'] and file_sha(lane/filename) == item['sha256'],
             'NATIVE_PROOF_FILE')
    traces = json.loads((lane/'traces.json').read_bytes())
    with np.load(lane/'reference.npz', allow_pickle=False) as saved:
        values, bounds = saved['vectors'].copy(), saved['bounds'].copy()
    need(len(traces) == len(inputs) and len(bounds) == len(inputs)+1, 'NATIVE_PROOF_COUNT')
    cache = {}
    for i, row in enumerate(inputs):
        cache[row['index']] = (values[bounds[i]:bounds[i+1]], traces[i], row['text_sha256'])
    return report, cache


def publish(lens, source, root, configuration_pin, lens_id, start, end, cache):
    began = time.perf_counter(); cpu = time.process_time()
    destination = root/shard_name(start, end); pending = root/(destination.name+'.pending')
    need(not destination.exists() and not pending.exists(), 'NATIVE_SHARD_RECOVERY_REQUIRED')
    pending.mkdir(); durable_directory(root)
    write_committed(pending/'intent.json', dict(configuration_pin=configuration_pin, lens_id=lens_id, start=start, end=end))
    phases = dict(source=0., measurement=0., packing=0., persistence=0.)
    measured = reused = windows = active_tokens = padding_tokens = 0; max_windows = 0
    forward_phases = dict(tokenize=0., forward=0.)
    physical_shapes = {}; vector_blocks = []; offsets = [0]
    try:
        tick = time.perf_counter()
        g0, g1 = start//source.report['group_rows'], (end-1)//source.report['group_rows']
        source_rows = source.table.read_row_groups(list(range(g0, g1+1))).to_pylist()
        need(len(source_rows) == end-start and [r['index'] for r in source_rows] == list(range(start, end)),
             'NATIVE_SOURCE_RANGE')
        phases['source'] += time.perf_counter()-tick
        with pq.ParquetWriter(pending/'traces.parquet', TRACE_SCHEMA, compression='zstd') as writer:
            for offset in range(start, end, IO_INPUTS):
                tick = time.perf_counter(); stop = min(offset+IO_INPUTS, end)
                rows = source_rows[offset-start:stop-start]
                fresh = [r for r in rows if r['index'] not in cache]
                phases['source'] += time.perf_counter()-tick; tick = time.perf_counter()
                if fresh:
                    values, bounds, traces, timing = forward(lens, [r['text'] for r in fresh])
                    measured += len(fresh)
                    active_tokens += timing['active_source_tokens']; padding_tokens += timing['padding_copy_tokens']
                    for key, seconds in timing['phases_seconds'].items(): forward_phases[key] += seconds
                    for shape in timing['shapes']:
                        key = f"{shape['rows']}x{shape['width']}"; physical_shapes[key] = physical_shapes.get(key, 0)+1
                phases['measurement'] += time.perf_counter()-tick; tick = time.perf_counter()
                records = []; j = 0
                for row in rows:
                    uid = row['index']; expected = row['text_sha256']
                    if uid in cache:
                        native, trace, old_hash = cache[uid]; need(old_hash == expected, 'NATIVE_CACHE_INPUT')
                        reused += 1
                    else:
                        native, trace = values[bounds[j]:bounds[j+1]], traces[j]; j += 1
                    need(trace['surface_sha256'] == expected and len(native) == len(trace['windows']), 'NATIVE_INPUT_TRACE')
                    vector_blocks.append(native); windows += len(native); offsets.append(windows); max_windows = max(max_windows, len(native))
                    records.append(dict(index=uid, input_sha256=expected, provenance=canonical(trace)))
                phases['packing'] += time.perf_counter()-tick; tick = time.perf_counter()
                writer.write_table(pa.Table.from_pylist(records, schema=TRACE_SCHEMA), row_group_size=IO_INPUTS)
                phases['persistence'] += time.perf_counter()-tick
        tick = time.perf_counter()
        for name, value in [('vectors', np.concatenate(vector_blocks)), ('offsets', np.asarray(offsets, dtype=np.int64))]:
            with (pending/(name+'.npy')).open('xb') as f:
                np.save(f, value, allow_pickle=False); f.flush(); os.fsync(f.fileno())
        with (pending/'traces.parquet').open('rb') as f: os.fsync(f.fileno())
        phases['persistence'] += time.perf_counter()-tick; tick = time.perf_counter()
        need(measured+reused == end-start and len(offsets) == end-start+1, 'NATIVE_SHARD_COVERAGE')
        source.fresh()
        files = {name: dict(bytes=(pending/name).stat().st_size, sha256=file_sha(pending/name))
                 for name in ('vectors.npy', 'offsets.npy', 'traces.parquet')}
        phases['pins_and_source_check'] = time.perf_counter()-tick
        report = dict(schema=SCHEMA, configuration_pin=configuration_pin, lens_id=lens_id, complete=True,
            start=start, end=end, inputs=end-start, measured_inputs=measured, reused_inputs=reused,
            dimensions=lens.dim, windows=windows, maximum_windows=max_windows, files=files,
            active_measured_tokens=active_tokens, padding_copy_tokens=padding_tokens, physical_shapes=physical_shapes,
            forward_phases_seconds=forward_phases, phases_seconds=phases, seconds=time.perf_counter()-began,
            cpu_seconds=time.process_time()-cpu, code_sha256=file_sha(Path(__file__)),
            scientific_admission=False, model_updates=0, acceptance='author-checked', number_status='measured')
        need(sum(phases.values())/report['seconds'] >= .9, 'NATIVE_PUBLICATION_TIMING')
        pin = write_committed(pending/'report.json', report)
        write_committed(pending/'COMMITTED', dict(report_sha256=pin)); durable_directory(pending)
        pending.rename(destination); durable_directory(root)
        return dict(path=destination.name, pin=pin, inputs=end-start, windows=windows,
                    measured_inputs=measured, reused_inputs=reused, seconds=report['seconds'])
    except BaseException as exc:
        write_committed(pending/'failure.json', dict(error=type(exc).__name__+':'+str(exc), phases_seconds=phases))
        raise
