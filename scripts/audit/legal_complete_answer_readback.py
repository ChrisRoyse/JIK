"""Publish and independently consume every legal complete-answer supervision tile."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from claims.process_conditions import write_committed
from model import legal_complete_answer_plan as plan
from text.constellation_corpus import file_sha, need


def code():
    root = Path(__file__).resolve().parents[1]
    return {n:file_sha(root/n) for n in ('model/legal_complete_answer_plan.py',
        'audit/legal_complete_answer_readback.py','text/constellation_corpus.py',
        'text/constellation_input.py','claims/process_conditions.py')}


def check_tile(source, document, value):
    lo,hi = map(int,source.documents[document]); observed = min(2,hi-lo-1)
    current = value['current']; target = value['target']; ending = value['end']
    need(current.ndim == 1 and 0 < len(current) <= 128 and int(current[0]) >= lo+observed-1
         and int(current[-1]) < hi and np.array_equal(current,np.arange(current[0],current[-1]+1))
         and value['source_start'] == lo and value['decision_start'] == int(current[0])
         and value['requires_carried_state'] == (int(current[0]) != lo+observed-1), 'ANSWER_READBACK_TILE_SCOPE')
    need(np.array_equal(ending,current == hi-1) and np.array_equal(target,np.where(current < hi-1,current+1,-1)),
         'ANSWER_READBACK_TRUE_TERMINAL')
    expected = np.full(len(current),-1,dtype='<i8')
    selected = np.flatnonzero(current < hi-1)
    expected[selected] = source.mapping[current[selected]+1]
    need(np.array_equal(expected,value['target_uid']), 'ANSWER_READBACK_TARGET_ADDRESS')


def recount(source, rows):
    consumer = plan.Plan(source,rows); groups = {}; ends = decisions = tiles = 0; choices = {}
    for i,value in enumerate(rows):
        lo,hi = map(int,source.documents[i]); n = hi-lo; context = 1 if n == 2 else 2
        side = int(source.sides[i]); collection = source.metadata[i]['source_path']
        # Derive expectations from source metadata and scalar ranges, not producer row().
        expected = (i,lo,lo+context,hi,side,int(source.components[i]),source.collections.index(collection))
        need(tuple(int(value[k]) for k in plan.DTYPE.names[:7]) == expected, 'ANSWER_READBACK_ROW_BOUNDARY')
        ids = source.mapping[lo:hi]
        need(value['addresses'].tobytes() == hashlib.sha256(ids.tobytes()).digest(), 'ANSWER_READBACK_ORDER')
        whole = [int(source.counts[ids,j].sum(dtype=np.int64)) for j in range(6)]
        answer = [int(source.counts[ids[context:],j].sum(dtype=np.int64)) for j in range(6)]
        need(whole == value['windows'].tolist() and answer == value['answer_windows'].tolist(),
             'ANSWER_READBACK_NATIVE_WINDOWS')
        request = consumer.request(i)
        need(request == dict(operation='continue_observed_source',position_kind='attested_document_order',
            observed_uids=tuple(int(source.mapping[p]) for p in range(lo,lo+context))), 'ANSWER_READBACK_PRIVATE_TARGETS')
        cursor = lo+context-1; count = terminal = 0
        for tile in consumer.supervision(i,fitting=side==0):
            check_tile(source,i,tile)
            need(int(tile['current'][0]) == cursor, 'ANSWER_READBACK_TILE_GAP')
            cursor = int(tile['current'][-1])+1; count += len(tile['current']); terminal += int(tile['end'].sum()); tiles += 1
        need(cursor == hi and count == n-context+1 and terminal == 1, 'ANSWER_READBACK_DECISION_COVERAGE')
        ends += terminal; decisions += count
        for key in (plan.SIDES[side], plan.SIDES[side]+':'+collection):
            g = groups.setdefault(key,dict(documents=0,atoms=0,contexts=0,answers=0,decisions=0,ends=0,
                windows=[0]*6,answer_windows=[0]*6,lengths=Counter(),components=set(),max_native_document_bytes=0,
                sum_attention_positions_squared=0))
            for k,x in [('documents',1),('atoms',n),('contexts',context),('answers',n-context),('decisions',count),('ends',1)]:g[k]+=x
            g['lengths'][n] += 1; g['components'].add(int(source.components[i]))
            for j in range(6):g['windows'][j]+=whole[j];g['answer_windows'][j]+=answer[j]
            g['max_native_document_bytes'] = max(g['max_native_document_bytes'],sum(whole)*768*4)
            g['sum_attention_positions_squared'] += (sum(whole)+n*6)**2
        for key,rank in [('shortest',(n,i)),('longest',(-n,i)),('most_windows',(-sum(whole),i))]:
            if key not in choices or rank < choices[key][0]:choices[key]=(rank,i)
    for side in plan.SIDES:
        support = source.reports['split']['support'][side]; g = groups[side]
        need(all(g[k] == support[k] for k in ('documents','atoms')) and len(g['components']) == support['components'],
             'ANSWER_READBACK_SPLIT_DENOMINATOR')
    for g in groups.values():
        g['components'] = len(g['components']);g['lengths']={str(k):v for k,v in sorted(g['lengths'].items())}
    return dict(documents=len(rows),ends=ends,decisions=decisions,tiles=tiles,groups=groups,
                boundary_examples={k:v[1] for k,v in choices.items()},all_source_targets_accounted=True,
                fit_uses_only_train=True,terminal_count_is_document_count=True)


def reject(call):
    try: call()
    except ValueError as e: return str(e)
    raise AssertionError('ANSWER_READBACK_NEGATIVE_ACCEPTED')


def controls(source,rows,summary,output):
    consumer=plan.Plan(source,rows); result={}
    document=summary['boundary_examples']['longest']; original=rows[document]
    for name in ('stop','answer_start','side','component','addresses'):
        bad=original.copy()
        if name=='addresses':bad[name]=bytes(32)
        else:bad[name]=int(bad[name])+1
        result['changed_'+name]=reject(lambda:plan.validate(source,bad))
    result['missing_document']=reject(lambda:plan.Plan(source,rows[:-1]))
    repeated=rows.copy();repeated[1]=repeated[0]
    result['repeated_document']=reject(lambda:plan.Plan(source,repeated));del repeated
    held=int(np.flatnonzero(source.sides!=0)[0])
    result['held_out_fit']=reject(lambda:next(consumer.supervision(held,fitting=True)))
    interior=next(consumer.supervision(document,fitting=int(source.sides[document])==0,tile=1))
    interior['end'][:]=True;interior['target'][:]=-1;interior['target_uid'][:]=-1
    result['interior_tile_end']=reject(lambda:check_tile(source,document,interior))
    retiled={}
    for number in sorted(set(summary['boundary_examples'].values())):
        digests={}
        for width in (1,128):
            hs={n:hashlib.sha256() for n in ('current','target','target_uid','end')}
            for tile in consumer.supervision(number,fitting=int(source.sides[number])==0,tile=width):
                check_tile(source,number,tile)
                for name,h in hs.items():h.update(tile[name].tobytes())
            digests[str(width)]={n:h.hexdigest() for n,h in hs.items()}
        need(digests['1']==digests['128'],'ANSWER_READBACK_RETILING');retiled[str(number)]=digests
    # Real changed persisted bytes; do not touch any authoritative source file.
    path=output/'changed-source-control.json';path.write_bytes(b'{"observed":1}')
    correct=file_sha(path);source.bind(path,correct);path.write_bytes(b'{"observed":2}')
    # Same-size immediate writes can retain a filesystem timestamp. The required
    # final byte readback, not the cheap identity guard, must reject that case.
    result['same_size_rewrite_final_hash']=reject(lambda:source.fresh(physical=True))
    replacement=output/'source-control-replacement.json';replacement.write_bytes(b'{"observed":3}')
    os.replace(replacement,path)
    result['replaced_source_identity']=reject(source.fresh)
    del source.stamps[path];del source.files[str(path)]
    return dict(refusals=result,retiled=retiled)


def run(a):
    start=tick=time.perf_counter();cpu=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    pins=code()
    for name,pin in pins.items():
        dest=a.output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes())
        need(file_sha(dest)==pin,'ANSWER_READBACK_CODE_COPY')
    source=plan.Source(a.root);phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    if a.previous:
        need(a.previous_pin and file_sha(a.previous/'report.json')==a.previous_pin,'ANSWER_READBACK_PREVIOUS_PIN')
        previous=json.loads((a.previous/'report.json').read_bytes())
        need(previous['complete'] and previous['mode']=='publish' and previous['binding']==source.binding
             and previous['code']==pins,'ANSWER_READBACK_PREVIOUS_BINDING')
        for name,pin in previous['files'].items():need(file_sha(a.previous/name)==pin,'ANSWER_READBACK_PUBLICATION_PIN')
        path=a.previous/'plan.npy'
    else:
        plan.build(source,a.output/'plan.npy');path=a.output/'plan.npy'
    rows=np.load(path,mmap_mode='r',allow_pickle=False)
    phases['publication_open']=time.perf_counter()-tick;tick=time.perf_counter()
    summary=recount(source,rows);phases['complete_independent_consumption']=time.perf_counter()-tick;tick=time.perf_counter()
    negatives=controls(source,rows,summary,a.output)
    if a.previous:
        need(summary==previous['summary'] and negatives==previous['controls'],'ANSWER_READBACK_COLD_RECOUNT')
    refs=dict(source.files);source.close();rows._mmap.close()
    need(code()==pins,'ANSWER_READBACK_CODE_CHANGED')
    phases['controls_and_final_physical_readback']=time.perf_counter()-tick;tick=time.perf_counter()
    files={str(f.relative_to(a.output)):file_sha(f) for f in a.output.rglob('*') if f.is_file()}
    phases['output_hashes']=time.perf_counter()-tick;seconds=time.perf_counter()-start
    need(sum(phases.values())/seconds>=.9,'ANSWER_READBACK_PHASE_ATTRIBUTION')
    result=dict(complete=True,mode='cold' if a.previous else 'publish',previous_pin=a.previous_pin,
        binding=source.binding,code=pins,source_files=refs,files=files,summary=summary,controls=negatives,
        plan_sha256=file_sha(path),plan_bytes=path.stat().st_size,seconds=seconds,cpu_seconds=time.process_time()-cpu,
        phases_seconds=phases,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        native_encoder_calls=0,optimizer_updates=0,model_promotion=False,acceptance='author-checked',number_status='measured')
    pin=write_committed(a.output/'report.json',result);(a.output/'COMMITTED').write_text(pin+'\n')
    print(json.dumps(dict(pin=pin,seconds=seconds,documents=summary['documents'],decisions=summary['decisions'],
        ends=summary['ends'],plan_bytes=result['plan_bytes'],peak_rss_bytes=result['peak_rss_bytes'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path);p.add_argument('--previous-pin')
    a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(e),code=code()))
        raise
