"""Full TRAIN normalization with native-bank authentication and cold readback."""
import argparse
import fcntl
import json
from pathlib import Path
import resource
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch

from audit.legal_rq_growth import pinned
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model import native_training_statistics as stats
from model.legal_learning_source import LegalLearningSource
from model.native_rq_data import NativeRQData, statistics
from model.native_rq_batch import BatchedInterventionNativeRQ
from audit.legal_residual_codebook_readback import CONFIG, POLICY as RQ_POLICY
from text.constellation_corpus import file_sha, need
from text.constellation_input import stamp
from text.legal_continuation_input import Continuations
from text.legal_native_store import NativeBank, NativeShard, ranges
from text.legal_reuse_input import ReuseInputs

POLICY = dict(tile=stats.TILE, memory_bytes=stats.MEMORY, arithmetic_error=stats.ERROR,
              weighting='one_per_TRAIN_present_unique_input_then_1_over_native_windows',
              order='ascending_source_uid', device='cpu', threads=1,
              qualification='prior_small_TRAIN_union_all_multiwindow_TRAIN',
              native_encoder_calls=0, model_updates=0)
DEFAULTS = {
    'publication': ('legal-native-full-v1-20261022', 'b3cd5977b4cf35cea37091091d14990a8a2bbaa62b31001239f493aafb68954c'),
    'audit': ('legal-native-full-audit-v1-20261023', '76753f8063eaa4141779e85f892c91dd78cdd41ff8759c65e5262cad6c978b90'),
    'census': ('legal-native-window-census-v1-20261023', 'aa31d96102544f3f48edecb194d5832981ca9c63a0adee526c234b24d1c19052'),
    'baseline': ('legal-native-associations-v1-20261022', 'a5d76f4e0a7f62ddbc2ec18ada60d2bcb77a588fa0d7e608d46ee6ac68ac5f0f'),
    'growth': ('legal-rq-growth-v1-20261022', '18abb4b9631dfb5140d30831dd1446875507b6b6d8cc45f3790395931da22e75')}


def code_pins():
    root = Path(__file__).resolve().parents[1]
    return {n:file_sha(root/n) for n in ['audit/legal_training_statistics.py',
        'model/native_training_statistics.py', 'model/native_rq_data.py',
        'model/native_rq_batch.py', 'model/native_rq_interventions.py', 'model/native_rq_residual.py',
        'text/legal_native_store.py', 'model/legal_learning_source.py']}


def training_only(ids, presence):
    need(ids.dtype == np.int64 and ids.ndim == 1 and len(ids) > 0
         and bool(((ids >= 0) & (ids < len(presence))).all())
         and bool((np.diff(ids) > 0).all()), 'TRAIN_STATS_UID_LAYOUT')
    need(bool((presence[ids] & 1).all()), 'TRAIN_STATS_NON_TRAIN')


def save_uids(path, presence, selected):
    n = len(selected) if selected is not None else int(np.count_nonzero(presence & 1))
    result = np.lib.format.open_memmap(path, mode='w+', dtype='<i8', shape=(n,))
    cursor = 0
    for lo in range(0, len(presence), 16384):
        hi = min(lo+16384, len(presence))
        if selected is None: values = lo+np.flatnonzero(presence[lo:hi] & 1)
        else: values = selected[np.searchsorted(selected, lo):np.searchsorted(selected, hi)]
        if len(values):
            training_only(values, presence); result[cursor:cursor+len(values)] = values
        cursor += len(values)
    need(cursor == n, 'TRAIN_STATS_UID_COVERAGE'); result.flush()
    return result


def authenticate(root):
    records = {n:pinned(root/path/'report.json', pin) for n,(path,pin) in DEFAULTS.items()}
    pub, audit, census = (records[n] for n in ['publication','audit','census'])
    need(pub['full_corpus_complete'] and pub['complete_scope'] and not pub['errors']
         and audit['complete'] and audit['full_corpus_readback']
         and audit['binding']['publication_pin'] == DEFAULTS['publication'][1]
         and census['publication_pin'] == DEFAULTS['publication'][1], 'TRAIN_STATS_FULL_AUTHORITY')
    bank = NativeBank(pub['bank']['path'], pub['bank']['configuration_pin'])
    source = ReuseInputs(bank.config['source']['path'], bank.config['source']['pin'])
    need(audit['binding']['source_pin'] == census['source_pin'] == source.pin
         and audit['binding']['bank_pin'] == census['configuration_pin'] == bank.pin,
         'TRAIN_STATS_INPUT_BINDING')
    base = records['baseline']; c = Continuations(Path(base['continuations']['path']), base['continuations']['pin'])
    split, cold = c.report['split'], c.report['cold']
    data = LegalLearningSource(source,split['path'],split['pin'],cold['path'],cold['pin'],bank.root,bank.pin)
    need(c.report['source']['pin'] == source.pin and split['pin'] == census['split_pin'],
         'TRAIN_STATS_SPLIT_BINDING')
    multi = pinned(root/DEFAULTS['census'][0]/'multiwindow.json', census['files']['multiwindow.json'])
    growth = records['growth']
    old = pinned(root/DEFAULTS['growth'][0]/'selection.json', growth['selection_pin'])
    selected = np.asarray(sorted(set(old['populations']['small']) | {r['uid'] for r in multi if r['presence'] & 1}), dtype='<i8')
    for row in multi:
        need(row['presence'] == int(c.presence[row['uid']])
             and row['source_sha256'] == source.arrays['hashes'][row['uid']].tobytes().hex(),
             'TRAIN_STATS_MULTI_SOURCE')
    held = {side:sorted(set(old['held'][side]) | {r['uid'] for r in multi if r['presence'] == bit})
            for side,bit in [('development',2),('query',4)]}
    need(all(bool((c.presence[ids] == bit).all()) for ids,bit in [(held['development'],2),(held['query'],4)]),
         'TRAIN_STATS_HELD_CONTAMINATION')
    scans = {}
    for name in bank.config['roster']:
        item = audit['scans'][name]
        scans[name] = pinned(root/DEFAULTS['audit'][0]/name/'stored/scan.json',item['pin'])
        need(item['all_source_bindings'] and item['all_windows_finite_unit']
             and scans[name]['inputs'] == bank.config['inputs'], 'TRAIN_STATS_AUDIT_SCOPE')
        expected = [(r['path'],r['pin']) for r in pub['instruments'][name]['shards']]
        need([(r['path'],r['pin']) for r in scans[name]['shards']] == expected
             and [r[0] for r in expected] == [f'{lo:012d}-{hi:012d}' for lo,hi in ranges(bank.config['inputs'])],
             'TRAIN_STATS_AUDIT_SHARDS')
    binding = dict(code=code_pins(), policy=POLICY, sources={n:pin for n,(_,pin) in DEFAULTS.items()},
                   source=data.binding, continuation=c.pin if hasattr(c,'pin') else base['continuations']['pin'])
    return records, bank, data, c, selected, held, scans, binding


def close(saved):
    saved.table.close(); saved.vectors._mmap.close(); saved.offsets._mmap.close()


def numpy_stats(path):
    with np.load(path, allow_pickle=False) as values: return {k:values[k].copy() for k in values.files}


def lane_scan(name, lane, bank, references, uids, counts, maximum, output, reference):
    total = None; parts = []; used_refs = []; peak = 0; phases = dict(authentication=0., reduction=0., persistence=0.)
    first_layout = None; stamps = {}
    for ref in references:
        lo,hi = map(int,ref['path'].split('-')); a,b = np.searchsorted(uids, [lo,hi]).tolist()
        if a == b: continue
        tick = time.perf_counter()
        saved = NativeShard(bank.root/name/ref['path'],bank.pin,bank.config['instruments'][name]['lens_id'],lo,hi)
        try:
            need(saved.pin == ref['pin'], 'TRAIN_STATS_SHARD_PIN')
            stamps.update({str(path):list(value) for path,value in saved.stamps.items()})
            phases['authentication'] += time.perf_counter()-tick; tick = time.perf_counter()
            selected = np.asarray(uids[a:b]-lo, dtype='<i8')
            partial, windows, memory = stats.reduce(saved.vectors,saved.offsets,selected,maximum,reference=reference)
            if total is None: total = stats.empty(saved.vectors.shape[1],maximum)
            stats.add(total,partial); counts[a:b,lane] = windows
            need(int(windows.max()) <= np.iinfo(counts.dtype).max, 'TRAIN_STATS_COUNT_STORAGE')
            peak = max(peak,memory); parts.append(partial); used_refs.append(dict(path=ref['path'],pin=saved.pin,rows=b-a))
            if first_layout is None:
                # Retain a real tiny window block for refusal probes, never fake readings.
                index = int(selected[0]); x,y = map(int,saved.offsets[index:index+2])
                first_layout = (saved.vectors[x:y].copy(),np.array([0,y-x],dtype='<i8'))
            saved.fresh(); phases['reduction'] += time.perf_counter()-tick
        finally: close(saved)
        if len(parts)%64 == 0: print(json.dumps(dict(stage='statistics',instrument=name,shards=len(parts),inputs=int(total['inputs']))),flush=True)
    tick=time.perf_counter(); need(total is not None and int(total['inputs']) == len(uids), 'TRAIN_STATS_LANE_COVERAGE')
    result = stats.finish(total)
    np.savez(output/(name+'.npz'), **result)
    np.savez(output/(name+'-partials.npz'), **{k:np.stack([p[k] for p in parts]) for k in total})
    write_committed(output/(name+'-stamps.json'), stamps)
    stats.compare(numpy_stats(output/(name+'.npz')), result)
    phases['persistence'] += time.perf_counter()-tick
    return result,dict(shards=used_refs,peak_tile_bytes=peak,phases_seconds=phases,
                      inputs=int(total['inputs']),windows=int(total['windows'])),first_layout


def scalar_readback(records, data, presence, uids, results):
    base=records['baseline'];reader=NativeRQData(data,Path(base['proof']['path']),base['proof']['pin'])
    old=statistics(reader,uids.tolist(),presence); errors={}
    for name in data.roster:
        converted=dict(center=old['centers'][name].numpy(),variance=old['variance'][name].numpy(),
                       positions=old['positions'][name].numpy(),weights=np.asarray(old['position_weights'][name]))
        errors[name]=stats.compare({k:results[name][k] for k in converted},converted)
    return dict(errors=errors,native_sha256=old['native_sha256'],windows=old['window_counts'].numpy(),
                maximum_read_bytes=reader.maximum_read_bytes,peak_cache_bytes=reader.peak_cache_bytes)


def run(a):
    start=tick=time.perf_counter();cpu=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    lock=(Path('/home/croyse/mnlm/.lease')/'legal-training-statistics').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    records,bank,data,c,selected,held,scans,binding=authenticate(a.root)
    previous=None;scope=a.mode
    if a.mode in ['full','cold']:
        need(a.previous is not None and a.previous_pin is not None,'TRAIN_STATS_PREVIOUS_REQUIRED')
        previous=pinned(a.previous/'report.json',a.previous_pin)
        need(previous['complete'] and previous['binding']==binding,'TRAIN_STATS_PREVIOUS_BINDING')
        if a.mode=='full':
            proof=pinned(a.cold_proof/'report.json',a.cold_pin)
            need(previous['mode']=='qualify' and proof['mode']=='cold' and proof['complete']
                 and proof['previous_pin']==a.previous_pin and proof['binding']==binding,'TRAIN_STATS_QUALIFICATION')
        else:
            scope=previous['scope'];need(scope in ['qualify','full'],'TRAIN_STATS_COLD_SCOPE')
    uids=save_uids(a.output/'uids.npy',c.presence,selected if scope=='qualify' else None)
    counts=np.lib.format.open_memmap(a.output/'window-counts.npy',mode='w+',dtype='<u4',shape=(len(uids),len(data.roster)))
    selection_pin=write_committed(a.output/'selection.json',dict(qualification_uids=selected.tolist(),held=held,
        full_train_unique=int(np.count_nonzero(c.presence&1)),exposure='development_exposed_case_split_not_untouched'))
    phases=dict(bindings_selection=time.perf_counter()-tick);results={};lanes={};first=None
    for lane,name in enumerate(data.roster):
        tick=time.perf_counter(); maximum=records['census']['lanes'][name]['maximum_windows']
        results[name],lanes[name],layout=lane_scan(name,lane,bank,scans[name]['shards'],uids,counts,maximum,a.output,a.mode=='cold')
        if first is None:first=layout
        phases[name]=time.perf_counter()-tick
        print(json.dumps(dict(stage='lane_complete',instrument=name,inputs=lanes[name]['inputs'],seconds=phases[name])),flush=True)
    tick=time.perf_counter();counts.flush();uids.flush(); comparisons={};scalar=None
    if scope=='qualify':
        scalar=scalar_readback(records,data,c.presence,uids,results)
        need(np.array_equal(scalar.pop('windows'),counts),'TRAIN_STATS_SCALAR_WINDOWS')
    if a.mode=='cold':
        for name in data.roster:
            need(file_sha(a.previous/(name+'.npz'))==previous['files'][name+'.npz'],'TRAIN_STATS_PREVIOUS_FILE')
            comparisons[name]=stats.compare(results[name],numpy_stats(a.previous/(name+'.npz')))
            partial=numpy_stats(a.output/(name+'-partials.npz'))
            need(file_sha(a.previous/(name+'-partials.npz'))==previous['files'][name+'-partials.npz'],'TRAIN_STATS_PREVIOUS_PARTIALS')
            comparisons[name+'_partials']=stats.compare(partial,numpy_stats(a.previous/(name+'-partials.npz')))
        for filename in ['uids.npy','window-counts.npy','selection.json']:
            need(file_sha(a.output/filename)==previous['files'][filename],'TRAIN_STATS_COLD_SELECTION')
    tensors=dict(centers={n:torch.from_numpy(v['center'].copy()) for n,v in results.items()},
        variance={n:torch.from_numpy(v['variance'].copy()) for n,v in results.items()},
        positions={n:torch.from_numpy(v['positions'].copy()) for n,v in results.items()},
        position_weights={n:v['weights'].tolist() for n,v in results.items()})
    torch.save(tensors,a.output/'statistics.pt')
    loaded=torch.load(a.output/'statistics.pt',map_location='cpu',weights_only=True)
    for name in data.roster:
        need(torch.equal(loaded['positions'][name],tensors['positions'][name]),'TRAIN_STATS_MODEL_READBACK')
    # The actual typed decoder must accept the newly observed position support.
    torch.manual_seed(RQ_POLICY['seed'])
    model=BatchedInterventionNativeRQ({n:len(loaded['centers'][n]) for n in data.roster},
                                      CONFIG,RQ_POLICY['sizes'],'no_encoder_quantizer',loaded)
    layouts=[{n:len(loaded['positions'][n]) for n in data.roster}]
    with torch.no_grad():
        decoded=model.reconstruct(torch.zeros((1,CONFIG['width'])),layouts)[0]
    need(all(torch.equal(decoded[n],loaded['positions'][n].float()) for n in data.roster),
         'TRAIN_STATS_MODEL_BASELINE')
    model.check_statistics()
    wrong=first[1].copy();wrong[-1]=0
    shifted={k:v.copy() for k,v in results[data.roster[0]].items()};shifted['center'][0]+=1
    controls=dict(nontrain=rejected(lambda:training_only(np.array(held['query'][:1],dtype='<i8'),c.presence)),
        lost_window=rejected(lambda:stats.reduce(first[0],wrong,np.array([0],dtype='<i8'),len(first[0]))),
        changed_statistics=rejected(lambda:stats.compare(shifted,results[data.roster[0]])),
        wrong_audit=rejected(lambda:pinned(a.root/DEFAULTS['audit'][0]/'report.json','0'*64)))
    data.fresh();data.source.fresh();c.fresh()
    for name in data.roster:
        source_stamps=json.loads((a.output/(name+'-stamps.json')).read_bytes())
        need(all(list(stamp(Path(path)))==value for path,value in source_stamps.items()),
             'TRAIN_STATS_SOURCE_CHANGED')
    phases['readback_controls']=time.perf_counter()-tick;tick=time.perf_counter()
    files={p.name:file_sha(p) for p in a.output.iterdir() if p.is_file()}
    phases['final_hashes']=time.perf_counter()-tick;seconds=time.perf_counter()-start
    need(sum(phases.values())/seconds>=.9,'TRAIN_STATS_TIMING_COVERAGE')
    report=dict(schema='legal-training-statistics-v1',mode=a.mode,scope=scope,complete=True,binding=binding,
        previous_pin=a.previous_pin,inputs=len(uids),roster=list(data.roster),lanes=lanes,files=files,
        scalar_readback=scalar,cold_comparisons=comparisons,controls=controls,selection_pin=selection_pin,
        model_baseline_windows=layouts[0],model_baseline_exact=True,
        phases_seconds=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        number_status='measured',acceptance='author-checked',scientific_admission=False,model_updates=0)
    pin=write_committed(a.output/'report.json',report)
    print(json.dumps(dict(complete=True,pin=pin,inputs=len(uids),seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--mode',choices=['qualify','full','cold'],required=True);p.add_argument('--output',type=Path,required=True)
    for name in ['previous','cold-proof']:p.add_argument('--'+name,type=Path)
    for name in ['previous-pin','cold-pin']:p.add_argument('--'+name)
    a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.is_dir():write_committed(a.output/'failure.json',dict(error=type(exc).__name__+':'+str(exc)))
        raise
