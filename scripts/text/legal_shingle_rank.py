"""One exact global tuple dictionary and rare-first rank bank for the legal join."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.process_conditions import write_committed
from text.constellation_corpus import file_sha, need
from text.legal_shingles_input import Shingles

SCHEMA = 'legal-exact-shingle-rank-v1'


def available_memory():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'): return int(line.split()[1])*1024
    raise ValueError('MNLM_SHINGLE_MEMAVAILABLE_ABSENT')


def run(source, pin, cold, cold_pin, output):
    start = tick = time.perf_counter(); cpu = time.process_time()
    need(file_sha(cold/'report.json') == cold_pin, 'RANK_COLD_PIN')
    proof = json.loads((cold/'report.json').read_bytes())
    need(proof['complete'] and proof['source']['pin'] == pin
         and proof['all_original_documents_reconstructed'], 'RANK_COLD_SCOPE')
    bank = Shingles(source, pin); n = len(bank.documents); h = len(bank.values)
    memory = available_memory(); projected = 160*h+32*n
    need(memory >= projected, 'RANK_PHYSICAL_MEMORY_REQUIRES_EXTERNAL_SORT')
    output.mkdir(parents=True, exist_ok=False)
    write_committed(output/'before.json', dict(available_bytes=memory,
                    projected_working_bytes=projected, inputs={str(p):s for p,s in bank.stamps.items()}))
    phases = dict(authenticate=time.perf_counter()-tick); tick = time.perf_counter()
    unique, inverse, counts = np.unique(bank.values.view('V20').reshape(-1),
                                       return_inverse=True, return_counts=True)
    s = len(unique); need(s < 2**32, 'RANK_ID_WIDTH_REQUIRES_MIGRATION')
    phases['global_exact_tuple_sort'] = time.perf_counter()-tick; tick = time.perf_counter()
    order = np.argsort(counts, kind='stable')
    ranks = np.empty(s, dtype='<u4'); ranks[order] = np.arange(s, dtype='<u4')
    need(int(counts.sum()) == h and int(counts.max()) <= n, 'RANK_DOCUMENT_FREQUENCY')
    phases['frequency_rank'] = time.perf_counter()-tick; tick = time.perf_counter()
    for name, array in [('dictionary.npy',unique), ('document-frequency.npy',counts), ('ranks.npy',ranks)]:
        with (output/name).open('xb') as stream:
            np.save(stream, array, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
    with (output/'rank.u32').open('xb') as stream:
        for i in range(n):
            lo, hi = map(int, bank.offsets[i:i+2])
            np.sort(ranks[inverse[lo:hi]]).astype('<u4', copy=False).tofile(stream)
        stream.flush(); os.fsync(stream.fileno())
    for name, array in [('offsets.i64',bank.offsets), ('eligible.u8',(bank.counts>=20).astype('u1'))]:
        with (output/name).open('xb') as stream:
            array.tofile(stream); stream.flush(); os.fsync(stream.fileno())
    phases['rank_bank_persist'] = time.perf_counter()-tick; tick = time.perf_counter()
    bank.fresh()
    names = ('dictionary.npy','document-frequency.npy','ranks.npy','rank.u32','offsets.i64','eligible.u8','before.json')
    files = {name:dict(bytes=(output/name).stat().st_size,sha256=file_sha(output/name)) for name in names}
    phases['final_hashes'] = time.perf_counter()-tick
    seconds = time.perf_counter()-start; need(sum(phases.values())/seconds>=.9,'RANK_TIMING')
    result = dict(schema=SCHEMA,complete=True,source=dict(path=str(source),pin=pin),
                  cold=dict(path=str(cold),pin=cold_pin),files=files,
                  counts=dict(documents=n,memberships=h,global_unique_shingles=s,
                              maximum_frequency=int(counts.max()),single_document_shingles=int((counts==1).sum())),
                  policy=bank.report['policy'], phases_seconds=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  near_join_complete=False,splits_frozen=False,scientific_admission=False,model_updates=0,
                  acceptance='author-checked',number_status='measured',code_sha256=file_sha(Path(__file__)))
    receipt=write_committed(output/'report.json',result)
    write_committed(output/'COMMITTED',dict(report_sha256=receipt))
    print(json.dumps(dict(pin=receipt,counts=result['counts'],seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','cold','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('pin','cold-pin'):p.add_argument('--'+name,required=True)
    a=p.parse_args();run(a.source,a.pin,a.cold,a.cold_pin,a.output)
