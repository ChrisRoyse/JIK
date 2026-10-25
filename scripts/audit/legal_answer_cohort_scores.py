"""Post-generation private scoring worker. No predictors or encoders are loaded."""
import argparse
import json
import os
from pathlib import Path
import re
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit.legal_answer_public_pack import authenticate
from audit.complete_answer_evaluation import read,BASE,ROOT
from claims.process_conditions import write_committed
from model import complete_answer_cohort_scores as scoring
from model.legal_answer_public_pack import PublicPack
from model.native_population_training import fingerprint
from text.constellation_corpus import canonical,file_sha,need
from text.legal_native_store import SHARD_INPUTS

CODE = ('model/complete_answer_summary.py','model/complete_answer_cohort_scores.py',
        'audit/legal_answer_cohort_scores.py','audit/legal_answer_cohort_scores_proof.py')


def native_exact(data, reader, uids, frames):
    windows = {n:0 for n in data.roster}; total = 0
    groups = {}
    for i,u in enumerate(uids):groups.setdefault(u//SHARD_INPUTS*SHARD_INPUTS,[]).append((i,u))
    for name in data.roster:
        for start,items in sorted(groups.items()):
            d = reader.descriptor(name,start); data.native.descriptor_fresh(d)
            values = np.load(d['root']/'vectors.npy',mmap_mode='r',allow_pickle=False)
            try:
                for i,u in items:
                    lo,hi = map(int,d['offsets'][u-start:u-start+2]); x = frames[i][name].numpy(); y = values[lo:hi]
                    need(x.shape == y.shape and x.tobytes() == y.tobytes(), 'COHORT_SCORE_DIRECT_NATIVE')
                    windows[name] += len(y); total += y.nbytes
            finally:values._mmap.close()
            data.native.descriptor_fresh(d)
    return dict(windows=windows,bytes=total)


class References:
    def __init__(self, source, data, reader, pack, output):
        self.source,self.data,self.reader,self.pack,self.output = source,data,reader,pack,Path(output)
        self.calls = []; self.maximum_bytes = 0; self.maximum_allowance = None

    def __call__(self, ordinal, public):
        row = self.pack.rows[ordinal]; doc = int(row['document']); lo,hi = map(int,self.source.documents[doc])
        n = int(row['observed']); start = lo+n; self.source.fresh()
        need(int(self.source.sides[doc]) in (1,2) and n == min(2,hi-lo-1)
             and np.array_equal(self.source.mapping[lo:start],row['uids'][:n]), 'COHORT_SCORE_REFERENCE_BOUNDARY')
        uids = self.source.mapping[start:hi].tolist()
        payload = int(self.source.counts[uids].sum())*768*4
        available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        allowance = available//4 if self.maximum_allowance is None else min(available//4,self.maximum_allowance)
        need(payload+len(uids)*4096 <= allowance, 'COHORT_SCORE_REFERENCE_MEMORY')
        frames = tuple(f for k in range(0,len(uids),128) for f in self.reader.frames(uids[k:k+128]))
        need(len(frames) == hi-start and all([len(frame[name]) for name in self.data.roster] == self.source.counts[uid].tolist()
             for uid,frame in zip(uids,frames,strict=True)), 'COHORT_SCORE_REFERENCE_WINDOWS')
        direct = native_exact(self.data,self.reader,uids,frames)
        identity = dict(source=fingerprint(self.source.binding),side=('train','development','query')[int(row['side'])],
            document=doc,component=int(row['component']),collection=self.source.collections[int(row['collection'])],
            exposure=self.source.binding['exposure'],reference=fingerprint((uids,frames)),reference_atoms=len(frames))
        receipt = dict(ordinal=ordinal,document=doc,identity=identity,**direct)
        with (self.output/'reference-reads.jsonl').open('ab') as stream:
            stream.write(canonical(receipt)+b'\n');stream.flush();os.fsync(stream.fileno())
        self.calls.append(receipt); self.maximum_bytes=max(self.maximum_bytes,payload); self.source.fresh()
        return dict(identity=identity,future=frames)


def worker(a):
    began = tick = time.perf_counter(); cpu = time.process_time(); a.output.mkdir(exist_ok=True)
    need(re.fullmatch('[a-z0-9-]+',a.receipt) and not (a.output/(a.receipt+'.json')).exists(), 'COHORT_SCORE_RECEIPT')
    report = read(a.cohort/'report.json',a.cohort_pin); pack = PublicPack(report['pack']['path'],report['pack']['pin'])
    # Validate coverage before setting up any private reference reader.
    scoring.authority(a.cohort,a.cohort_pin,pack,a.scope)
    source,rows,data,continuation,reader = authenticate(a.root)
    need(pack.report['binding']['source'] == source.binding, 'COHORT_SCORE_SOURCE_AUTHORITY')
    refs = References(source,data,reader,pack,a.output); code={n:file_sha(BASE/n) for n in CODE}
    phases = dict(authentication=time.perf_counter()-tick); tick=time.perf_counter()
    def stop(_):os._exit(73)
    result = scoring.run(a.cohort,a.cohort_pin,pack,refs,output=a.output/'scores',code=code,
        source=source.binding,scope=a.scope,after_case=stop if a.fault else None)
    phases['score_run'] = time.perf_counter()-tick; tick=time.perf_counter()
    reader.close();source.close();rows._mmap.close();continuation.fresh();pack.close()
    need(not torch.cuda.is_initialized() and all(file_sha(BASE/n)==h for n,h in code.items()),'COHORT_SCORE_FINAL')
    phases['final_authentication']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9 <= sum(phases.values())/seconds <= 1.001,'COHORT_SCORE_WORKER_PHASES')
    result.update(code=code,reference_calls=refs.calls,maximum_reference_bytes=refs.maximum_bytes,
        timings=phases,worker_seconds=seconds,cpu_seconds=time.process_time()-cpu,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        model_loads=0,optimizer_updates=0,encoder_calls=0,gpu_used=False,admitted_senses=0)
    pin=write_committed(a.output/(a.receipt+'.json'),result)
    print(json.dumps(dict(complete=True,pin=pin,actions=result['actions'],cases=result['declared_cases'],seconds=seconds)),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--cohort-pin',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--receipt',required=True);p.add_argument('--scope',choices=('proof','all'),required=True)
    p.add_argument('--fault',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    try:worker(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/(a.receipt+'-failure.json'),dict(error=type(exc).__name__,detail=str(exc)))
        raise
