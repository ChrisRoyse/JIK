"""Explicit cache execution adapter for the unchanged predictor experiment.

This separate entry point leaves running/queued source pins untouched. It replaces
only the reader factory for the duration of one run and records that dependency
in execution artifacts. Existing CUDA, resume, ledger and cold-update gates stay
in the original driver. Full execution is restricted to its repeated-data arm.
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]

from audit import legal_efficient_predictor_training as driver
from claims.process_conditions import write_committed
from lenscommon import release_lease
from model.legal_native_reading_cache import ReadingNativeFrames
from text.constellation_corpus import file_sha, need


def authority(proof, proof_pin, readback, readback_pin):
    need(file_sha(proof/'report.json') == proof_pin and file_sha(readback) == readback_pin,
         'READING_EXECUTION_PROOF_PIN')
    q = json.loads((proof/'report.json').read_bytes()); c = json.loads(readback.read_bytes())
    need(q['complete'] and c['complete'] and q['all_native_bytes_exact']
         and c['all_native_bytes_exact'] and c['input_pin'] == proof_pin
         and c['reuse_source_reads'] == 0 and q['all_handles_and_cache_closed'],
         'READING_EXECUTION_PROOF')
    for name,pin in q['files'].items():
        need(file_sha(proof/name) == pin, 'READING_EXECUTION_PHYSICAL')
    root = Path(__file__).resolve().parents[1]
    for name,pin in q['binding']['code'].items():
        need(file_sha(root/name) == pin, 'READING_EXECUTION_SOURCE_CHANGED')
    need(c['code'] == file_sha(root/'audit/legal_reading_cache_readback.py'),
         'READING_EXECUTION_COLD_SOURCE')
    return q,c


def run(a):
    q,c = authority(a.reading_proof, a.reading_proof_pin, a.reading_readback, a.reading_readback_pin)
    need(not a.reference and (a.scope == 'qualify' or a.arm == 'repeated'), 'READING_EXECUTION_SCOPE')
    need(type(a.reading_cache_bytes) is int and 1 << 20 <= a.reading_cache_bytes <= 32 << 30,
         'READING_EXECUTION_CAPACITY')
    old_factory, old_code = driver.ResidentNativeFrames, driver.code
    root = Path(__file__).resolve().parents[1]
    extra = {name:file_sha(root/name) for name in [
        'model/legal_native_reading_cache.py', 'audit/legal_reading_cached_training.py',
        'audit/legal_reading_cache.py', 'audit/legal_reading_cache_readback.py']}
    created = []

    class QualifiedReader(ReadingNativeFrames):
        def __init__(self, data, audit_root, audit_pin):
            need(not created and data.binding == q['binding']['source'], 'READING_EXECUTION_SOURCE')
            mem = {k:int(v.split()[0])*1024 for k,v in
                   (line.split(':') for line in Path('/proc/meminfo').read_text().splitlines())}
            # Existing job state plus a 4 GiB workspace reserve must fit before
            # committing a cache capacity. Capacity is never raised on failure.
            need(mem['MemAvailable'] >= a.reading_cache_bytes+(4 << 30), 'READING_EXECUTION_RAM')
            super().__init__(data,audit_root,audit_pin,cache_bytes=a.reading_cache_bytes)
            created.append(self)
            write_committed(a.output/'reading-cache-execution.json', dict(
                proof=a.reading_proof_pin, readback=a.reading_readback_pin,
                cache_bytes=a.reading_cache_bytes, available_bytes=mem['MemAvailable'],
                workspace_reserve=4 << 30, source=data.binding, code=extra,
                arithmetic='unchanged', source_checks='certified_frames_before_and_after'))

        def close(self):
            if self.closed: return
            super().close()
            write_committed(a.output/'reading-cache-usage.json', dict(
                cache=self.cache_stats, io=self.io_stats, limit=self.cache_limit,
                closed=self.closed, cached_entries=len(self.readings), owned_bytes=self.owned_bytes,
                open_handles=len(self.maps), number_status='measured'))

    driver.ResidentNativeFrames = QualifiedReader
    driver.code = lambda:dict(old_code(), **extra)
    try:
        driver.run(a)
    finally:
        driver.ResidentNativeFrames, driver.code = old_factory, old_code
        for reader in created:
            if not reader.closed: reader.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scope',choices=['qualify','full'],required=True)
    p.add_argument('--arm',choices=['breadth','repeated'],default='repeated')
    for name in ['qualification','cold','cache','cache-cold','schedule','schedule-cold',
                 'reading-proof','reading-readback']:
        p.add_argument('--'+name,type=Path,required=True); p.add_argument('--'+name+'-pin',required=True)
    for name in ['previous','full-qualification','full-cold','resume','speed-qualification',
                 'speed-cold','io-qualification','io-cold']:
        p.add_argument('--'+name,type=Path); p.add_argument('--'+name+'-pin')
    p.add_argument('--reference',action='store_true')
    p.add_argument('--reading-cache-bytes',type=int,default=24 << 30)
    a = p.parse_args()
    try: run(a)
    finally: release_lease()
