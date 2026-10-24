"""CPU-only batch-reader proof against saved scalar bytes and independent native replay."""
import argparse
from copy import copy
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from text.constellation_corpus import canonical,digest,file_sha,need
from text.constellation_input import stamp
import text.legal_native_batch as implementation
import text.legal_native_store as storage
from text.legal_reuse_input import ReuseInputs


def fingerprint(uid,values,trace):
    return dict(uid=uid,windows=len(values),shape=list(values.shape),
                values_sha256=digest(values.tobytes()),trace_sha256=digest(canonical(trace)))


def fingerprints(batch):
    return [fingerprint(uid,*batch.input(i)) for i,uid in enumerate(batch.uids)]


def timed_read(session,name,uids):
    stats=dict(file_hash_calls=0,file_hash_bytes=0,file_hash_seconds=0.)
    original=storage.file_sha
    def measured(path):
        began=time.perf_counter();value=original(path)
        stats['file_hash_seconds']+=time.perf_counter()-began
        stats['file_hash_calls']+=1;stats['file_hash_bytes']+=Path(path).stat().st_size
        return value
    storage.file_sha=measured;began=time.perf_counter()
    try:batch=session.read_many(name,uids)
    finally:storage.file_sha=original
    return batch,dict(seconds=time.perf_counter()-began,**stats)


def fd_count():return len(list(Path('/proc/self/fd').iterdir()))


def controls(session,source,output):
    name=session.config['roster'][0]
    wrong=copy(source);wrong.pin='0'*64
    result=dict(wrong_source=rejected(lambda:implementation.NativeBatchSession(session.root,session.pin,wrong)),
        wrong_configuration_pin=rejected(lambda:implementation.NativeBatchSession(session.root,'0'*64,source)),
        invalid_uid=rejected(lambda:session.read_many(name,[session.config['inputs']])),
        wrong_instrument=rejected(lambda:session.read_many('unknown',[0])),
        insufficient_memory_tile=rejected(lambda:session.read_many(name,[0],memory_bytes=1)),
        implicit_training=rejected(session.training_authority))
    missing=output/'uncommitted-bank';missing.mkdir()
    shutil.copy2(session.root/'configuration.json',missing/'configuration.json')
    absent=implementation.NativeBatchSession(missing,session.pin,source)
    result['uncommitted_shard']=rejected(lambda:absent.read_many(name,[0]))
    altered=output/'changed-real-bank';altered.mkdir()
    shutil.copy2(session.root/'configuration.json',altered/'configuration.json')
    descriptor=session.descriptor(name,0);target=altered/name/descriptor['root'].name
    target.mkdir(parents=True)
    for filename in ('COMMITTED','report.json','vectors.npy','offsets.npy','traces.parquet'):
        shutil.copy2(descriptor['root']/filename,target/filename)
    reader=implementation.NativeBatchSession(altered,session.pin,source);before=reader.read_many(name,[0])
    lost=target/'COMMITTED';lost.rename(target/'COMMITTED.removed')
    result['removed_commit_in_warm_session']=rejected(lambda:reader.read_many(name,[0]))
    (target/'COMMITTED.removed').rename(lost)
    reader=implementation.NativeBatchSession(altered,session.pin,source);reader.read_many(name,[0])
    path=target/'vectors.npy';original=file_sha(path)
    with path.open('r+b') as handle:
        handle.seek(-1,2);old=handle.read(1);handle.seek(-1,2);handle.write(bytes([old[0]^1]))
        handle.flush();os.fsync(handle.fileno())
    result['changed_vector_in_warm_session']=rejected(lambda:reader.read_many(name,[0]))
    cold=implementation.NativeBatchSession(altered,session.pin,source)
    result['corrupted_vector_in_cold_session']=rejected(lambda:cold.read_many(name,[0]))
    result['mutation']=dict(path=str(path),before_sha256=original,after_sha256=file_sha(path),
                            original_read_sha256=digest(before.values.tobytes()))
    return result


def proof_inputs(session,source):
    root=Path(session.config['runtime_proof']['path']);pin=session.config['runtime_proof']['pin']
    need(file_sha(root/'report.json')==pin,'BATCH_AUDIT_PROOF_PIN')
    proof=json.loads((root/'report.json').read_bytes())
    for filename,ref in proof['files'].items():need(file_sha(root/filename)==ref['sha256'],'BATCH_AUDIT_PROOF_FILE')
    inputs=json.loads((root/'inputs.json').read_bytes());results={};scalar=storage.NativeBank(session.root,session.pin)
    for name in session.config['roster']:
        lane=root/name;need(file_sha(lane/'report.json')==proof['instruments'][name]['pin'],'BATCH_AUDIT_LANE_PIN')
        lane_report=json.loads((lane/'report.json').read_bytes())
        for filename,ref in lane_report['files'].items():need(file_sha(lane/filename)==ref['sha256'],'BATCH_AUDIT_LANE_FILE')
        longest=max(inputs,key=lambda row:len(row['text']))['index']
        selected=[(j,row) for j,row in enumerate(inputs) if row['index']<storage.SHARD_INPUTS
                  or (name==session.config['roster'][0] and row['index']==longest)]
        batch=session.read_many(name,[int(row['index']) for _,row in selected])
        traces=json.loads((lane/'traces.json').read_bytes())
        with np.load(lane/'reference.npz',allow_pickle=False) as reference:
            for k,(j,row) in enumerate(selected):
                values,trace=batch.input(k);a,b=map(int,reference['bounds'][j:j+2])
                need(np.array_equal(values,reference['vectors'][a:b]) and trace==traces[j],'BATCH_AUDIT_NATIVE_PROOF')
        uids=[0,1023,1024,16383,0];gather=session.read_many(name,uids)
        for j,uid in enumerate(uids):
            values,trace=scalar.input(name,uid,source.arrays['hashes'][uid].tobytes().hex())
            need(fingerprint(uid,values,trace)==fingerprints(gather)[j],'BATCH_AUDIT_SCALAR_BYTES')
        results[name]=dict(lens_id=batch.lens_id,prior_native_inputs=len(selected),scalar_inputs=len(uids),
                           windows=len(batch.values),output=digest(batch.values.tobytes()))
    for saved in scalar.cached.values():
        saved.table.close();saved.vectors._mmap.close();saved.offsets._mmap.close()
    return results


def run(a):
    began=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    need(file_sha(a.baseline/'report.json')==a.pin,'BATCH_AUDIT_BASELINE_PIN')
    baseline=json.loads((a.baseline/'report.json').read_bytes());root=Path(baseline['root'])
    config=json.loads((root/'configuration.json').read_bytes())
    source=ReuseInputs(Path(config['source']['path']),config['source']['pin'])
    need(source.pin==baseline['source_pin'],'BATCH_AUDIT_SOURCE_PIN')
    session=implementation.NativeBatchSession(root,baseline['config_pin'],source)
    paths={Path(p):tuple(s) for p,s in baseline['unchanged_source_stamps'].items()}
    need(all(stamp(p)==s for p,s in paths.items()),'BATCH_AUDIT_BASELINE_CHANGED')
    descriptors_before=fd_count();name=baseline['instrument'];requests=baseline['requests']
    batch,cold=timed_read(session,name,requests)
    need(fingerprints(batch)==baseline['outputs'],'BATCH_AUDIT_COLD_BYTES')
    np.savez(a.output/'cold-batch.npz',vectors=batch.values,offsets=batch.offsets,uids=np.array(batch.uids,dtype='<i8'))
    write_committed(a.output/'cold-traces.json',[batch.input(i)[1] for i in range(len(requests))])
    traces=json.loads((a.output/'cold-traces.json').read_bytes())
    with np.load(a.output/'cold-batch.npz',allow_pickle=False) as saved:
        for j,uid in enumerate(saved['uids'].tolist()):
            lo,hi=map(int,saved['offsets'][j:j+2])
            need(fingerprint(uid,saved['vectors'][lo:hi],traces[j])==baseline['outputs'][j],'BATCH_AUDIT_PERSISTED_BYTES')
    warm_batch,warm=timed_read(session,name,requests)
    need(fingerprints(warm_batch)==baseline['outputs'] and warm['file_hash_calls']==0,'BATCH_AUDIT_WARM_BYTES')
    contexts={};expected={r['uid']:r for r in baseline['outputs']}
    for label,uids in [('reversed',requests[::-1]),('duplicate',[requests[1]]*3),('singleton',[requests[1]])]:
        sample,timing=timed_read(session,name,uids)
        need(fingerprints(sample)==[expected[i] for i in uids] and timing['file_hash_calls']==0,'BATCH_AUDIT_CONTEXT')
        contexts[label]=dict(uids=uids,windows=len(sample.values),**timing)
    descriptors_after=fd_count()
    need(descriptors_before==descriptors_after and session.stats['authenticated_shards']==2,'BATCH_AUDIT_HANDLES')
    need(cold['file_hash_bytes']*16==baseline['stats']['file_hash_bytes'],'BATCH_AUDIT_AUTHENTICATION_ONCE')
    lanes=proof_inputs(session,source);descriptors_after_lanes=fd_count()
    need(descriptors_before==descriptors_after_lanes,'BATCH_AUDIT_LANE_HANDLES')
    controls_result=controls(session,source,a.output)
    source.fresh();session.fresh()
    need(all(stamp(p)==s for p,s in paths.items()),'BATCH_AUDIT_ORIGINAL_CHANGED')
    for descriptor in session.descriptors.values():session.descriptor_fresh(descriptor)
    files={p.name:dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in (a.output/'cold-batch.npz',a.output/'cold-traces.json')}
    report=dict(schema='legal-native-batch-cold-readback-v1',complete=True,baseline=dict(path=str(a.baseline),pin=a.pin),
        source=dict(path=str(source.root),pin=source.pin),bank=dict(path=str(root),pin=session.pin),files=files,
        requests=len(requests),all_window_and_provenance_bytes_exact=True,original_seconds=baseline['seconds'],
        original_stats=baseline['stats'],cold=cold,warm=warm,contexts=contexts,lanes=lanes,controls=controls_result,
        file_descriptors=dict(before=descriptors_before,after=descriptors_after,after_all_lanes=descriptors_after_lanes,
                              limits=list(resource.getrlimit(resource.RLIMIT_NOFILE))),
        reader_stats=session.stats,unchanged_source_stamps={str(p):s for p,s in paths.items()},
        resident_offset_bytes=sum(d['offsets'].nbytes for d in session.descriptors.values()),
        seconds=time.perf_counter()-began,code_sha256=file_sha(Path(__file__)),reader_sha256=file_sha(Path(implementation.__file__)),
        new_encoder_calls=0,model_updates=0,scientific_admission=False,training_ready=False,
        acceptance='author-checked',number_status='measured')
    pin=write_committed(a.output/'report.json',report)
    print(json.dumps(dict(pin=pin,cold=cold,warm=warm,lanes=lanes,controls=controls_result)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--pin',required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
