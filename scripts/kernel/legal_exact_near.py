"""Exact prefix join, checked against real all-pairs proof and exact tuple replay."""
import argparse
from collections import Counter
from itertools import combinations
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.process_conditions import write_committed
from text.constellation_corpus import file_sha, need
from text.constellation_input import stamp
from text.legal_shingles_input import Shingles
from text.legal_shingle_rank import SCHEMA, available_memory


def load_rank(root, pin):
    need(sys.byteorder == 'little', 'JOIN_ENDIAN')
    need(file_sha(root/'report.json') == pin and json.loads((root/'COMMITTED').read_bytes()) ==
         dict(report_sha256=pin), 'JOIN_RANK_COMMIT')
    r = json.loads((root/'report.json').read_bytes())
    need(r['schema'] == SCHEMA and r['complete'], 'JOIN_RANK_SCOPE')
    seen = {root/n:stamp(root/n) for n in ('report.json','COMMITTED')}
    for name, ref in r['files'].items():
        path = root/name
        need(path.stat().st_size == ref['bytes'] and file_sha(path) == ref['sha256'], 'JOIN_RANK_FILE')
        seen[path] = stamp(path)
    n = r['counts']['documents']; h = r['counts']['memberships']; s = r['counts']['global_unique_shingles']
    arrays = {name: np.load(root/name, mmap_mode='r',allow_pickle=False)
              for name in ('dictionary.npy','document-frequency.npy','ranks.npy')}
    arrays.update(values=np.memmap(root/'rank.u32',mode='r',dtype='<u4',shape=(h,)),
                  offsets=np.memmap(root/'offsets.i64',mode='r',dtype='<i8',shape=(n+1,)),
                  eligible=np.memmap(root/'eligible.u8',mode='r',dtype='u1',shape=(n,)))
    need(arrays['dictionary.npy'].dtype == np.dtype('V20') and arrays['dictionary.npy'].shape == (s,)
         and arrays['ranks.npy'].dtype == np.dtype('<u4') and arrays['ranks.npy'].shape == (s,)
         and arrays['document-frequency.npy'].shape == (s,), 'JOIN_RANK_SHAPE')
    return r, arrays, seen


def replay_rank(bank, arrays):
    dictionary, ranks, frequency = (arrays[k] for k in ('dictionary.npy','ranks.npy','document-frequency.npy'))
    s = len(dictionary)
    need(np.array_equal(np.sort(ranks), np.arange(s,dtype='<u4')), 'JOIN_RANK_PERMUTATION')
    inverse = np.empty(s,dtype='<u4'); inverse[ranks] = np.arange(s,dtype='<u4')
    need(np.all(np.diff(frequency[inverse]) >= 0), 'JOIN_RANK_FREQUENCY_ORDER')
    need(np.array_equal(arrays['offsets'],bank.offsets) and
         np.array_equal(arrays['eligible'],(bank.counts>=20).astype('u1')), 'JOIN_RANK_SOURCE_ADDRESS')
    observed = np.zeros(s,dtype='<i8')
    for i in range(len(bank.documents)):
        lo,hi=map(int,bank.offsets[i:i+2]); values=arrays['values'][lo:hi]
        need(not len(values) or (int(values[-1])<s and np.all(values[1:]>values[:-1])), 'JOIN_RANK_VALUES')
        keys=inverse[values]
        need(np.array_equal(np.sort(dictionary[keys]),bank.values[lo:hi].view('V20').reshape(-1)),
             'JOIN_EXACT_TUPLE_REPLAY')
        observed[keys]+=1
    need(np.array_equal(observed,frequency),'JOIN_FREQUENCY_RECOUNT')
    return dict(documents=len(bank.documents),memberships=len(bank.values),all_exact_tuples_replayed=True,
                all_global_frequencies_recounted=True)


def invoke(binary, root, n, s, output):
    command = [str(binary),str(root/'rank.u32'),str(root/'offsets.i64'),str(root/'eligible.u8'),str(n),str(s),str(output)]
    p = subprocess.run(command,capture_output=True,text=True)
    need(p.returncode==0,'JOIN_NATIVE_FAILURE:'+p.stderr[-2000:])
    with output.open('rb') as stream:os.fsync(stream.fileno())
    stats=json.loads(p.stdout)
    need(output.stat().st_size==stats['edges']*12,'JOIN_PAIR_SIZE')
    return stats


def pair_map(path, n):
    raw=np.fromfile(path,dtype='<u4')
    need(len(raw)%3==0,'JOIN_PAIR_FORMAT'); result={}
    for a,b,c in raw.reshape(-1,3):
        a,b,c=map(int,(a,b,c));need(0<=a<b<n and (a,b) not in result,'JOIN_PAIR_ID')
        result[a,b]=c
    return result


def equal_pairs(actual, expected):
    need(actual==expected,'JOIN_EXHAUSTIVE_PROOF_MISMATCH')


def exact_proof(binary, arrays, bank, root):
    root.mkdir(); selected=set(sorted(range(len(bank.documents)),key=lambda i:bank.documents[i]['record'])[:256])
    # Include real identical normalized shingle sets, selected before any join score.
    twins={}
    for i,d in enumerate(bank.documents):
        if arrays['eligible'][i]:
            lo,hi=map(int,arrays['offsets'][i:i+2])
            key=hashlib.sha256(arrays['values'][lo:hi].tobytes()).hexdigest()
            twins.setdefault(key,[]).append(i)
    for _,group in sorted((k,v) for k,v in twins.items() if len(v)>1)[:32]:selected.update(group[:2])
    chosen=sorted(selected); values=[];offsets=[0];eligible=[]
    for i in chosen:
        lo,hi=map(int,arrays['offsets'][i:i+2]);v=arrays['values'][lo:hi]
        values.append(np.array(v));offsets.append(offsets[-1]+len(v));eligible.append(arrays['eligible'][i])
    np.concatenate(values).astype('<u4').tofile(root/'rank.u32')
    np.asarray(offsets,dtype='<i8').tofile(root/'offsets.i64');np.asarray(eligible,dtype='u1').tofile(root/'eligible.u8')
    stats=invoke(binary,root,len(chosen),len(arrays['ranks.npy']),root/'pairs.u32')
    sets=[set(map(int,v)) for v in values];expected={}; comparisons=0
    for a,b in combinations(range(len(chosen)),2):
        if not eligible[a] or not eligible[b]:continue
        common=len(sets[a]&sets[b]);comparisons+=1
        if 9*common>=4*(len(sets[a])+len(sets[b])):expected[a,b]=common
    actual=pair_map(root/'pairs.u32',len(chosen));equal_pairs(actual,expected)
    need(expected,'JOIN_PROOF_POSITIVE_ABSENT')
    negative={}
    def reject(name,value):
        try:equal_pairs(value,expected)
        except ValueError as exc:negative[name]=str(exc);return
        raise AssertionError('JOIN_NEGATIVE_PASSED')
    missed=dict(actual);missed.pop(next(iter(missed)));reject('dropped_true_edge',missed)
    altered=dict(actual);altered[next(iter(altered))]+=1;reject('changed_intersection',altered)
    absent=next(pair for pair in combinations(range(len(chosen)),2) if pair not in expected)
    extra=dict(actual);extra[absent]=1;reject('extra_wrong_edge',extra)
    return dict(indices=chosen,source_records=[bank.documents[i]['record'] for i in chosen],
                candidate_policy='256 source-hash ordered plus first 32 identical normalized-shingle-set pairs',
                pairs=len(expected),brute_force_pairs_compared=comparisons,native=stats,controls=negative)


def run(root,pin,output):
    start=tick=time.perf_counter();cpu=time.process_time()
    r,arrays,stamps=load_rank(root,pin)
    bank=Shingles(Path(r['source']['path']),r['source']['pin']);s=r['counts']['global_unique_shingles']
    need(available_memory()>40*s+20*len(bank.values),'JOIN_PHYSICAL_MEMORY')
    output.mkdir(parents=True,exist_ok=False)
    phases=dict(authenticate=time.perf_counter()-tick);tick=time.perf_counter()
    replay=replay_rank(bank,arrays);write_committed(output/'rank-readback.json',replay)
    phases['all_rank_tuple_replay']=time.perf_counter()-tick;tick=time.perf_counter()
    cpp=Path(__file__).with_name('legal_prefix_join.cpp');binary=output/'legal-prefix-join'
    command=['g++','-O3','-std=c++20','-Wall','-Wextra','-Werror',str(cpp),'-o',str(binary)]
    compilation=subprocess.run(command,capture_output=True,text=True)
    write_committed(output/'compiler.json',dict(command=command,stdout=compilation.stdout,stderr=compilation.stderr,
                    returncode=compilation.returncode,version=subprocess.check_output(['g++','--version'],text=True)))
    need(compilation.returncode==0,'JOIN_COMPILE')
    phases['compile']=time.perf_counter()-tick;tick=time.perf_counter()
    proof=exact_proof(binary,arrays,bank,output/'proof')
    write_committed(output/'proof.json',proof)
    phases['exhaustive_real_proof']=time.perf_counter()-tick;tick=time.perf_counter()
    stats=invoke(binary,root,len(bank.documents),s,output/'pairs.u32')
    phases['full_native_join']=time.perf_counter()-tick;tick=time.perf_counter()
    pairs=pair_map(output/'pairs.u32',len(bank.documents))
    for (a,b),common in pairs.items():
        x=bank.values[bank.offsets[a]:bank.offsets[a+1]].view('V20').reshape(-1)
        y=bank.values[bank.offsets[b]:bank.offsets[b+1]].view('V20').reshape(-1)
        actual=len(np.intersect1d(x,y,assume_unique=True))
        need(actual==common and 9*common>=4*(len(x)+len(y)),'JOIN_FULL_EXACT_PAIR_RECOUNT')
    phases['all_accepted_exact_pair_recount']=time.perf_counter()-tick;tick=time.perf_counter()
    bank.fresh();need(all(stamp(p)==v for p,v in stamps.items()),'JOIN_SOURCE_DRIFT')
    files={str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=file_sha(p))
           for p in sorted(output.rglob('*')) if p.is_file()}
    phases['final_hashes']=time.perf_counter()-tick;seconds=time.perf_counter()-start
    need(sum(phases.values())/seconds>=.9,'JOIN_TIMING')
    report=dict(schema='legal-exact-near-join-v1',complete=True,source=dict(path=str(root),pin=pin),
                counts=stats,rank_replay=replay,proof=dict(documents=len(proof['indices']),pairs=proof['pairs'],controls=proof['controls']),
                all_accepted_pairs_recounted=True,policy=r['policy'],files=files,
                phases_seconds=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                code_sha256=file_sha(Path(__file__)),native_source_sha256=file_sha(cpp),
                near_relation='exact_for_declared_normalized_word_shingle_relation',semantic_closure_complete=False,
                lineage_complete_beyond_retained_metadata=False,splits_frozen=False,model_updates=0,
                scientific_admission=False,acceptance='author-checked',number_status='measured')
    receipt=write_committed(output/'report.json',report);write_committed(output/'COMMITTED',dict(report_sha256=receipt))
    print(json.dumps(dict(pin=receipt,counts=stats,proof=report['proof'],seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pin',required=True);a=p.parse_args();run(a.source,a.pin,a.output)
