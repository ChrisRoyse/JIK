"""Full public-cohort native packing, independent source readback and cold consumption."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit import legal_training_statistics as training
from audit.complete_answer_evaluation import read
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model import legal_answer_evaluation_plan as plan
from model import legal_answer_public_pack as packing
from model.complete_answer_evaluation import public_inputs
from model.legal_native_resident_stream import ResidentNativeFrames
from model.native_population_training import fingerprint
from model.native_request_input import legal
from text.constellation_corpus import canonical, digest, file_sha, need
from text.legal_native_store import SHARD_INPUTS

PLAN = ('legal-answer-evaluation-plan-v2-20261024', 'f0ffd23c01f9d9db5d388783fd414c9840bf93dff9f8ce872564146565f9a1a7')
COLD = ('legal-answer-evaluation-plan-cold-v2-20261024', '75362d95e072a272d6626f5a87ba2e705cf49e20cf7fba86ed7d4f5a77869f9e')
PRIOR = ('legal-answer-cohort-proof-v1-20261024', 'b3f844f5bcd2dc5dbdcaa96ddbbd80d42728ea3e6bc0de4168a23ccfd89b0312')
BASE = Path(__file__).resolve().parents[1]
CODE = ('model/legal_answer_public_pack.py', 'audit/legal_answer_public_pack.py')


def authenticate(root):
    source = plan.authenticate(root)
    r,c = [read(root/n/'report.json', h) for n,h in (PLAN,COLD)]
    need(c['previous_pin'] == PLAN[1] and c['summary'] == r['summary']
         and c['controls'] == r['controls'] and c['binding'] == r['binding'] == source.binding,
         'PUBLIC_PACK_SCHEDULE_AUTHORITY')
    path = root/PLAN[0]/'evaluation-plan.npy'; need(file_sha(path) == r['plan_sha256'], 'PUBLIC_PACK_PLAN_PIN')
    rows = np.load(path, mmap_mode='r', allow_pickle=False)
    public = plan.prefixes(source); index = plan.buckets(source, public)
    schedule = plan.Schedule(source,rows,public,index)
    for i in range(len(rows)):schedule.request(i); schedule.donor_request(i)
    # This existing authority path creates no model or optimizer and reads no target vectors.
    _,_,data,continuation,_,_,_,_ = training.authenticate(root)
    need(data.native.pin == source.binding['native_bank'] and data.roster == source.roster,
         'PUBLIC_PACK_NATIVE_BINDING')
    name,pin = training.DEFAULTS['audit']; reader = ResidentNativeFrames(data,root/name,pin)
    return source,rows,data,continuation,reader


def publish(root, output, source, rows, reader):
    output.mkdir(); uids = packing.union(rows); counts = source.counts[uids]
    np.save(output/'uids.npy', uids, allow_pickle=False); np.save(output/'rows.npy', rows, allow_pickle=False)
    offsets = {}; vectors = {}; windows = counts.sum(axis=0, dtype=np.int64)
    binding = dict(source=source.binding, plan=PLAN[1], cold_plan=COLD[1],
        native_audit=training.DEFAULTS['audit'][1], code={n:file_sha(BASE/n) for n in CODE})
    phases = dict(native_read=0., write=0., case_seal=0.); began = tick = time.perf_counter()
    for lane,name in enumerate(source.roster):
        o = np.r_[np.int64(0),np.cumsum(counts[:,lane],dtype='<i8')]
        np.save(output/(name+'.offsets.npy'),o,allow_pickle=False); offsets[name] = o
        vectors[name] = np.lib.format.open_memmap(output/(name+'.windows.npy'), mode='w+',
            dtype='<f4',shape=(int(windows[lane]),768))
    phases['write'] += time.perf_counter()-tick
    try:
        for lo in range(0,len(uids),128):
            tick = time.perf_counter(); ids = uids[lo:lo+128].tolist(); frames = reader.frames(ids)
            phases['native_read'] += time.perf_counter()-tick; tick = time.perf_counter()
            for j,frame in enumerate(frames):
                for name in source.roster:
                    a,b = map(int,offsets[name][lo+j:lo+j+2]); x = frame[name].numpy()
                    need(x.shape == (b-a,768), 'PUBLIC_PACK_SOURCE_WINDOWS'); vectors[name][a:b] = x
            del frames
            phases['write'] += time.perf_counter()-tick
            if lo % 8192 == 0:print(json.dumps(dict(stage='pack',atoms=lo+len(ids),total=len(uids))),flush=True)
        tick = time.perf_counter()
        for v in vectors.values():v.flush()
        phases['write'] += time.perf_counter()-tick; tick = time.perf_counter()
        with (output/'cases.jsonl').open('xb') as stream:
            for ordinal in range(len(rows)):
                public = packing.value(rows,uids,offsets,vectors,source.roster,ordinal)
                stream.write(canonical(packing.case(output,binding,rows,ordinal,public))+b'\n')
            stream.flush(); os.fsync(stream.fileno())
        phases['case_seal'] = time.perf_counter()-tick
    finally:
        for v in vectors.values():v._mmap.close()
    for path in output.iterdir():
        with path.open('rb') as stream:os.fsync(stream.fileno())
    files = {p.name:dict(sha256=file_sha(p),bytes=p.stat().st_size) for p in output.iterdir()}
    summary = dict(cases=len(rows),unique_atoms=len(uids),observed_occurrences=2*int(rows['observed'].sum()),
        windows=windows.tolist(),payload_bytes=int(windows.sum())*768*4,
        uid_sha256=digest(uids.tobytes()),source_shard_ranges=len(np.unique(uids//SHARD_INPUTS)))
    pin = write_committed(output/'report.json',dict(schema=packing.SCHEMA, complete=True,binding=binding,
        roster=list(source.roster),files=files,summary=summary,policy=packing.POLICY))
    return pin,phases,time.perf_counter()-began


def direct(pack, source, reader):
    """Separate NumPy shard path: recount every UID, offset, window and byte."""
    need(np.array_equal(pack.uids,packing.union(pack.rows)), 'PUBLIC_PACK_DIRECT_UNION')
    total = 0; bytes_read = 0
    for lane,name in enumerate(pack.roster):
        counts = np.diff(pack.offsets[name])
        need(np.array_equal(counts,source.counts[pack.uids,lane]), 'PUBLIC_PACK_DIRECT_COUNTS')
        for start in sorted(set((pack.uids//SHARD_INPUTS*SHARD_INPUTS).tolist())):
            d = reader.descriptor(name,start); reader.data.native.descriptor_fresh(d)
            values = np.load(d['root']/'vectors.npy',mmap_mode='r',allow_pickle=False)
            try:
                lo,hi = np.searchsorted(pack.uids,[start,start+SHARD_INPUTS])
                for j in range(int(lo),int(hi)):
                    uid = int(pack.uids[j]); a,b = map(int,d['offsets'][uid-start:uid-start+2])
                    c,e = map(int,pack.offsets[name][j:j+2]); x=pack.vectors[name][c:e]; y=values[a:b]
                    need(x.shape == y.shape and x.tobytes() == y.tobytes(), 'PUBLIC_PACK_DIRECT_BYTES')
                    total += len(y); bytes_read += y.nbytes
            finally:values._mmap.close()
            reader.data.native.descriptor_fresh(d)
        print(json.dumps(dict(stage='direct-native',lane=name,windows=total)),flush=True)
    return dict(all_windows_exact=True,windows=total,bytes=bytes_read)


def controls(pack, output):
    output.mkdir(); out = {}
    out['wrong_report_pin'] = rejected(lambda:packing.PublicPack(pack.root,'0'*64))
    ref = dict(pack.cases[0]['public'],future_length=1)
    out['private_reference_field'] = rejected(lambda:pack.load(ref))
    out['changed_case_pin'] = rejected(lambda:pack.load(dict(pack.cases[0]['public'],pin='0'*64)))
    for fault in ('uid','offset','missing_lane','private_field'):
        dest = output/fault; dest.mkdir(); r = deepcopy(pack.report)
        # Hard links change the source inode ctime and correctly trip freshness.
        # Fault artifacts must own their bytes without touching authoritative state.
        for name in r['files']:shutil.copyfile(pack.root/name,dest/name)
        if fault == 'missing_lane':
            name = pack.roster[0]+'.windows.npy'; (dest/name).unlink(); del r['files'][name]
        elif fault == 'private_field':
            name = 'cases.jsonl'; rows = deepcopy(pack.cases); rows[0]['future_length'] = 1
            (dest/name).unlink(); (dest/name).write_bytes(b''.join(canonical(v)+b'\n' for v in rows))
            r['files'][name] = dict(sha256=file_sha(dest/name),bytes=(dest/name).stat().st_size)
        else:
            name = 'uids.npy' if fault == 'uid' else pack.roster[0]+'.offsets.npy'
            values = np.load(dest/name,allow_pickle=False).copy(); values[0] = -1 if fault == 'uid' else 1
            (dest/name).unlink(); np.save(dest/name,values,allow_pickle=False)
            r['files'][name] = dict(sha256=file_sha(dest/name),bytes=(dest/name).stat().st_size)
        pin = write_committed(dest/'report.json',r)
        out[fault] = rejected(lambda:packing.PublicPack(dest,pin))
    expected = dict(wrong_report_pin='PUBLIC_PACK_REPORT_PIN',private_reference_field='PUBLIC_PACK_REFERENCE_FIELDS',
        changed_case_pin='PUBLIC_PACK_REFERENCE',uid='PUBLIC_PACK_PUBLIC_UNION',offset='PUBLIC_PACK_LAYOUT',
        missing_lane='PUBLIC_PACK_FILES',private_field='ANSWER_COHORT_PUBLIC_FIELDS')
    need(all(out[k] == 'MNLM_TEXT_CORPUS_'+v for k,v in expected.items()), 'PUBLIC_PACK_CONTROL_REASON')
    return out


def consume(pack):
    began = time.perf_counter(); gaps = 0; chain = '0'*64
    for row in pack.cases:
        public = pack.load(row['public']); _,pins = public_inputs(public,pack.roster)
        gaps += pins['correct']['prefix'] == pins['train_donor']['prefix']
        chain = digest(bytes.fromhex(chain)+canonical(dict(ordinal=row['ordinal'],state=fingerprint(public))))
    return dict(cases=len(pack.cases),native_identical_controls=gaps,chain=chain,seconds=time.perf_counter()-began)


def baseline(pack, reader):
    began = time.perf_counter(); calls = reader.stats['calls']; chain = '0'*64
    for ordinal,row in enumerate(pack.rows):
        n = int(row['observed']); a = reader.frames(row['uids'][:n].tolist()); b = reader.frames(row['donor_uids'][:n].tolist())
        public = {m:dict(frames=f,descriptor=legal(n)) for m,f in [('correct',a),('reversed',a[::-1]),('train_donor',b)]}
        public_inputs(public,pack.roster)
        state = fingerprint(public); need(state == pack.cases[ordinal]['public']['state'], 'PUBLIC_PACK_BASELINE_BYTES')
        chain = digest(bytes.fromhex(chain)+canonical(dict(ordinal=ordinal,state=state)))
        if ordinal % 4096 == 0:print(json.dumps(dict(stage='baseline',cases=ordinal)),flush=True)
    return dict(cases=len(pack.rows),chain=chain,calls=reader.stats['calls']-calls,seconds=time.perf_counter()-began)


def run(a):
    began = tick = time.perf_counter(); cpu = time.process_time(); a.output.mkdir(exist_ok=False)
    source,rows,data,continuation,reader = authenticate(a.root)
    phases = dict(authenticate=time.perf_counter()-tick); tick = time.perf_counter()
    previous = read(a.previous/'report.json',a.previous_pin) if a.previous else None
    if previous:
        pack_root = Path(previous['pack']['path']); pin = previous['pack']['pin']; publishing = None
    else:
        pack_root = a.output/'pack'; pin,detail,seconds = publish(a.root,pack_root,source,rows,reader)
        publishing = dict(phases=detail,seconds=seconds,reader=dict(reader.stats),handles=dict(reader.io_stats))
    phases['publish'] = time.perf_counter()-tick; tick = time.perf_counter()
    pack = packing.PublicPack(pack_root,pin)
    need(np.array_equal(pack.rows,rows) and pack.report['binding']['source'] == source.binding
         and pack.report['binding']['code'] == {n:file_sha(BASE/n) for n in CODE}, 'PUBLIC_PACK_SOURCE_AUTHORITY')
    phases['pack_authenticate'] = time.perf_counter()-tick; tick = time.perf_counter()
    exact = direct(pack,source,reader)
    phases['independent_native_readback'] = time.perf_counter()-tick; tick = time.perf_counter()
    old = baseline(pack,reader) if previous is None else None
    phases['matched_original_reader'] = time.perf_counter()-tick; tick = time.perf_counter()
    consumed = consume(pack)
    need(consumed['chain'] == (previous['consumed']['chain'] if previous else old['chain']), 'PUBLIC_PACK_CASE_CHAIN')
    phases['pack_consumer'] = time.perf_counter()-tick; tick = time.perf_counter()
    prior = read(a.root/PRIOR[0]/'report.json',PRIOR[1]); m = read(a.root/PRIOR[0]/'request.json',prior['files']['request.json'])
    matched = []
    for c in m['cases']:
        saved = torch.load(c['public']['path'],map_location='cpu',weights_only=True)
        actual = pack.load(pack.cases[c['ordinal']]['public'])
        need(fingerprint(saved) == fingerprint(actual) == c['public']['state'], 'PUBLIC_PACK_PRIOR_CASE')
        matched.append(c['document'])
    negatives = controls(pack,a.output/'controls')
    phases['compatibility_and_controls'] = time.perf_counter()-tick; tick = time.perf_counter()
    pack.close(); reader.close(); source.close(); continuation.fresh(); rows._mmap.close()
    phases['final_authentication'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
    need(.9 <= sum(phases.values())/seconds <= 1.001 and not torch.cuda.is_initialized(), 'PUBLIC_PACK_PHASES')
    report = dict(complete=True,mode='cold' if previous else 'publish',previous_pin=a.previous_pin,
        pack=dict(path=str(pack_root),pin=pin),publishing=publishing,summary=pack.report['summary'],
        exact=exact,baseline=old,consumed=consumed,controls=negatives,prior_exact_documents=matched,
        native=reader.stats,source_authentication=data.native.stats,policy=packing.POLICY,
        phases_seconds=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        code={n:file_sha(BASE/n) for n in CODE},admitted_senses=0,architecture_winner=False)
    result = write_committed(a.output/'report.json',report)
    print(json.dumps(dict(complete=True,pin=result,seconds=seconds,consumed=consumed)),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output',type=Path,required=True); p.add_argument('--previous',type=Path); p.add_argument('--previous-pin')
    a = p.parse_args(); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise
