"""Physical source-to-learner proof for full legal continuations, without training."""
import argparse
from collections import OrderedDict
from copy import deepcopy
import json
from pathlib import Path
import resource
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit import legal_target_cache as existing
from audit.legal_grouped_placement_training import source_bundle
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model import legal_complete_answer_plan as plan
from model.legal_complete_answer_source import CompleteAnswerCorpus
from model.legal_cached_targets import LegalCachedTargets
from model.legal_native_resident_stream import ResidentNativeFrames
from model.native_population_training import fingerprint
from model.native_request_input import NativeRequest
from text.constellation_corpus import file_sha,need
from text.legal_native_store import SHARD_INPUTS


def prepare(root):
    data,continuation,old,labels,_,authority = existing.prepare(root); old.close()
    ref = existing.qualification.training.DEFAULTS['audit']
    reader = ResidentNativeFrames(data,root/ref[0],ref[1])
    n,p = plan.REFS['cache']; cn,cp = plan.REFS['cache_cold']
    cache = LegalCachedTargets(root/n,p,root/cn,cp,data,authority)
    corpus = CompleteAnswerCorpus(root,data,reader,cache)
    original,bundle = source_bundle(root)
    base = Path(__file__).resolve().parents[1]
    names = set(existing.code()) | set(original['code']) | {
        'model/legal_complete_answer_plan.py','model/legal_complete_answer_source.py',
        'model/legal_cached_documents.py','model/legal_cached_targets.py','model/legal_native_resident_stream.py',
        'model/legal_grouped_documents.py','audit/legal_grouped_io_authority.py',
        'model/native_conditioned_answer.py','model/native_request_input.py','audit/legal_complete_answer_consumer.py'}
    code = {n:file_sha(base/n) for n in sorted(names)}
    return corpus,labels,continuation,bundle,code


def selection(corpus,bundle):
    chosen = {v['document']:{'prior_exposed_bundle'} for v in bundle['items']}
    rows = corpus.rows
    for side in range(3):
        indices = np.flatnonzero(rows['side'] == side)
        lengths = rows['stop'][indices]-rows['start'][indices]
        windows = rows['windows'][indices].sum(1)
        for reason,local in [('shortest',int(lengths.argmin())),('longest',int(lengths.argmax())),
                             ('most_windows',int(windows.argmax()))]:
            chosen.setdefault(int(indices[local]),set()).add(plan.SIDES[side]+':'+reason)
    return [dict(document=n,reasons=sorted(reasons)) for n,reasons in sorted(chosen.items())]


def independent(corpus,item,request):
    """Gather directly from immutable vector/offset arrays, not the resident reader."""
    data=corpus.data;number=item['document'];lo,hi=map(int,data.documents[number]);observed=min(2,hi-lo-1)
    uids=list(map(int,data.mapping[lo:hi]));frames=(*item['context'],*item['answers']);maps=OrderedDict()
    windows={n:0 for n in data.roster};total=0
    try:
        for uid,frame in zip(uids,frames,strict=True):
            start=uid//SHARD_INPUTS*SHARD_INPUTS
            for name in data.roster:
                d=data.native.descriptor(name,start);data.native.descriptor_fresh(d);key=(name,start)
                if key not in maps:
                    if len(maps)>=6:maps.popitem(last=False)[1]._mmap.close()
                    maps[key]=np.load(d['root']/'vectors.npy',mmap_mode='r',allow_pickle=False)
                maps.move_to_end(key);a,b=map(int,d['offsets'][uid-start:uid-start+2])
                value=maps[key][a:b];actual=frame[name].numpy()
                need(value.shape==actual.shape and value.dtype==actual.dtype
                     and value.tobytes()==actual.tobytes(),'LEGAL_ANSWER_DIRECT_NATIVE')
                windows[name]+=b-a;total+=value.nbytes;data.native.descriptor_fresh(d)
        target=uids[observed:];codes=np.asarray(corpus.cache.maps[0][target],dtype=np.int64)
        counts=np.asarray(corpus.cache.maps[1][target],dtype=np.int64)
        need(np.array_equal(codes,item['codes'].numpy()),'LEGAL_ANSWER_DIRECT_CODES')
        for j,name in enumerate(data.roster):
            need(np.array_equal(counts[:,j],item['counts'][name].numpy())
                 and counts[:,j].tolist()==[len(f[name]) for f in item['answers']],'LEGAL_ANSWER_DIRECT_COUNTS')
        need(request.observed==observed and len(frames)==hi-lo,'LEGAL_ANSWER_DIRECT_BOUNDARY')
        decisions=0;ends=0
        for tile in corpus.plan.supervision(number,fitting=False):
            for current,target_uid,end in zip(tile['current'],tile['target_uid'],tile['end'],strict=True):
                need(int(current)==lo+observed-1+decisions and bool(end)==(current==hi-1)
                     and int(target_uid)==(-1 if end else int(data.mapping[current+1])),'LEGAL_ANSWER_DIRECT_END')
                decisions+=1;ends+=int(end)
        need(decisions==len(item['answers'])+1 and ends==1,'LEGAL_ANSWER_DIRECT_DECISIONS')
        return dict(document=number,side=item['side'],context=observed,answers=len(item['answers']),
            decisions=decisions,ends=ends,windows=windows,native_bytes=total,native=item['native_pin'],
            labels=item['label_pin'],request=request.pin,binding=item['binding'])
    finally:
        for value in maps.values():value._mmap.close()


def controls(corpus,item,request):
    out={};number=item['document'];old=corpus.allowance;calls=corpus.reader.stats['calls']
    try:
        corpus.allowance=1
        out['physical_memory_before_read']=rejected(lambda:corpus.example(number,fitting=False))
        need(corpus.reader.stats['calls']==calls,'LEGAL_ANSWER_MEMORY_READ_OCCURRED')
    finally:corpus.allowance=old
    held=int(np.flatnonzero(corpus.rows['side']!=0)[0])
    out['held_out_fit']=rejected(lambda:corpus.example(held,fitting=True))
    for field in ('source_stop','addresses'):
        changed=deepcopy(item)
        changed['provenance'][field]=changed['provenance'][field]-1 if field=='source_stop' else '0'*64
        out[field]=rejected(lambda:corpus.validate(changed,request,fitting=False))
    changed=deepcopy(item);changed['answers']=changed['answers'][:-1]
    out['truncated_answer']=rejected(lambda:corpus.validate(changed,request,fitting=False))
    changed=deepcopy(item);changed['answers'][0][corpus.roster[0]][0,0]+=1
    out['changed_native']=rejected(lambda:corpus.validate(changed,request,fitting=False))
    changed=deepcopy(item);changed['codes'][0,0]+=1
    out['changed_labels']=rejected(lambda:corpus.validate(changed,request,fitting=False))
    changed=NativeRequest(json.loads(request.descriptor));changed.features[0,0]+=1
    out['changed_request']=rejected(lambda:corpus.validate(item,changed,fitting=False))
    bad=corpus.rows[number].copy();bad['addresses']=bytes(32)
    out['changed_source_addresses']=rejected(lambda:plan.validate(corpus.source,bad))
    out['corrupted_plan_authority']=rejected(lambda:corpus.source.report(corpus.source.root/plan.REFS['cache'][0],'0'*64))
    return out


def warm_pair(corpus,chosen):
    from model.legal_cached_documents import read_document as temporal
    from model.legal_grouped_documents import read_document as grouped
    ref=existing.qualification.training.DEFAULTS['audit'];readers={
        n:ResidentNativeFrames(corpus.data,corpus.source.root/ref[0],ref[1]) for n in ('temporal','grouped')}
    functions=dict(temporal=temporal,grouped=grouped);rows=[]
    try:
        for index,choice in enumerate(chosen):
            number=choice['document'];side=plan.SIDES[int(corpus.rows[number]['side'])];values={};times={}
            for name in (('temporal','grouped') if index%2==0 else ('grouped','temporal')):
                tick=time.perf_counter();value,_=functions[name](corpus.data,readers[name],corpus.cache,number,side=side,fit=False)
                times[name]=time.perf_counter()-tick;values[name]=value
            need(fingerprint(values['temporal'])==fingerprint(values['grouped']),'LEGAL_ANSWER_WARM_PAIR_EXACT')
            rows.append(dict(document=number,seconds=times,native=values['temporal']['native_pin'],
                             labels=values['temporal']['label_pin']))
            del values,value
        return dict(rows=rows,seconds={n:sum(v['seconds'][n] for v in rows) for n in readers},
                    io={n:dict(r.io_stats) for n,r in readers.items()},all_items_exact=True,
                    descriptor_authority_shared=True,alternating_first_arm=True)
    finally:
        for reader in readers.values():reader.close()


def run(a):
    began=tick=time.perf_counter();cpu=time.process_time();a.output.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    need(not torch.cuda.is_initialized(),'LEGAL_ANSWER_UNEXPECTED_GPU')
    corpus,labels,continuation,bundle,code=prepare(a.root);phases=dict(authentication=time.perf_counter()-tick)
    prior=None;baseline=None
    if a.baseline:
        need(a.baseline_pin and a.baseline_cold and a.baseline_cold_pin,'LEGAL_ANSWER_BASELINE_ARGUMENTS')
        need(file_sha(a.baseline/'report.json')==a.baseline_pin
             and file_sha(a.baseline_cold/'report.json')==a.baseline_cold_pin,'LEGAL_ANSWER_BASELINE_PIN')
        baseline=json.loads((a.baseline/'report.json').read_bytes());bc=json.loads((a.baseline_cold/'report.json').read_bytes())
        need(baseline['complete'] and bc['complete'] and bc['previous_pin']==a.baseline_pin
             and baseline['binding']==bc['binding'] and baseline['rows']==bc['rows']
             and baseline['code']==bc['code'] and baseline['controls']==bc['controls'],'LEGAL_ANSWER_BASELINE_COLD')
        need(all(baseline['binding'][n]==corpus.binding[n] for n in ('plan','cold','source','cache')),
             'LEGAL_ANSWER_BASELINE_SOURCE')
    if a.previous:
        need(file_sha(a.previous/'report.json')==a.previous_pin,'LEGAL_ANSWER_PREVIOUS_PIN')
        prior=json.loads((a.previous/'report.json').read_bytes())
        need(prior['complete'] and prior['binding']==corpus.binding and prior['code']==code,'LEGAL_ANSWER_PREVIOUS_BINDING')
    chosen=selection(corpus,bundle);old={v['document']:v for v in bundle['items']};rows=[];negatives=None
    if prior:need(prior['selection']==chosen,'LEGAL_ANSWER_SELECTION_CHANGED')
    if baseline:need(baseline['selection']==chosen,'LEGAL_ANSWER_BASELINE_SELECTION')
    tick=time.perf_counter()
    for choice in chosen:
        number=choice['document'];item,request,timing=corpus.example(number,fitting=bool(corpus.rows[number]['side']==0))
        prefix,public=corpus.public(number)
        need(fingerprint(prefix)==fingerprint(item['context']) and request.pin==public.pin,'LEGAL_ANSWER_PUBLIC_PREFIX')
        if number in old:
            reference=old[number];c=len(prefix)
            need(fingerprint((*item['context'],*item['answers']))==reference['native_pin']
                 and torch.equal(item['codes'],reference['codes'][c-1:])
                 and all(torch.equal(item['counts'][n],reference['counts'][n][c-1:]) for n in corpus.roster),
                 'LEGAL_ANSWER_OLD_BUNDLE_EXACT')
        payload=dict(item=item,descriptor=json.loads(request.descriptor),features=request.features)
        path=a.output/f'document-{number}.pt';torch.save(payload,path)
        saved=torch.load(path,map_location='cpu',weights_only=True);need(fingerprint(saved)==fingerprint(payload),'LEGAL_ANSWER_SAVED_BYTES')
        restored=NativeRequest(saved['descriptor']);corpus.validate(saved['item'],restored,fitting=False)
        result=independent(corpus,saved['item'],restored);result['old_bundle_exact']=number in old;rows.append(result)
        if prior:
            name=path.name;need(file_sha(a.previous/name)==prior['files'][name],'LEGAL_ANSWER_COLD_ITEM_PIN')
            previous=torch.load(a.previous/name,map_location='cpu',weights_only=True)
            need(fingerprint(previous)==fingerprint(saved),'LEGAL_ANSWER_COLD_ITEM_EXACT')
        if baseline:
            name=path.name;need(file_sha(a.baseline/name)==baseline['files'][name],'LEGAL_ANSWER_BASELINE_ITEM_PIN')
            reference=torch.load(a.baseline/name,map_location='cpu',weights_only=True)
            need(fingerprint(reference)==fingerprint(saved),'LEGAL_ANSWER_GROUPED_ITEM_EXACT');del reference
        if negatives is None:negatives=controls(corpus,saved['item'],restored)
        del item,payload,saved,prefix
        if prior:del previous
    phases['materialize_publish_and_direct_readback']=time.perf_counter()-tick;tick=time.perf_counter()
    if prior:need(rows==prior['rows'] and negatives==prior['controls'],'LEGAL_ANSWER_COLD_RECOUNT')
    if baseline:need(rows==baseline['rows'] and negatives==baseline['controls'],'LEGAL_ANSWER_GROUPED_RECOUNT')
    paired=warm_pair(corpus,chosen)
    phases['alternating_warm_io_comparison']=time.perf_counter()-tick;tick=time.perf_counter()
    io=dict(corpus.reader.io_stats);reader=dict(corpus.reader.stats);source_files=dict(corpus.source.files)
    corpus.close();corpus.cache.close();corpus.reader.close();labels.fresh();continuation.fresh()
    need(all(file_sha(Path(__file__).resolve().parents[1]/n)==pin for n,pin in code.items()),'LEGAL_ANSWER_CODE_CHANGED')
    need(not torch.cuda.is_initialized(),'LEGAL_ANSWER_UNEXPECTED_GPU')
    phases['close_and_final_authentication']=time.perf_counter()-tick;tick=time.perf_counter()
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['hashes']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(sum(phases.values())/seconds>=.9,'LEGAL_ANSWER_PHASE_ATTRIBUTION')
    report=dict(complete=True,mode='cold' if prior else 'publish',previous_pin=a.previous_pin,binding=corpus.binding,
        baseline=None if not baseline else dict(path=str(a.baseline),pin=a.baseline_pin,cold=str(a.baseline_cold),
            cold_pin=a.baseline_cold_pin,all_learner_items_exact=True),
        selection=chosen,rows=rows,controls=negatives,code=code,source_files=source_files,files=files,
        warm_comparison=paired,
        reader=reader,io=io,seconds=seconds,cpu_seconds=time.process_time()-cpu,phases_seconds=phases,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,optimizer_updates=0,native_encoder_calls=0,
        target_inference_calls=0,acceptance='author-checked',scientific_admission=False,capability_demonstrated=False)
    pin=write_committed(a.output/'report.json',report)
    print(json.dumps(dict(pin=pin,seconds=seconds,documents=len(rows),atoms=sum(r['context']+r['answers'] for r in rows),
        native_bytes=sum(r['native_bytes'] for r in rows),peak_rss_bytes=report['peak_rss_bytes'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path);p.add_argument('--previous-pin')
    p.add_argument('--baseline',type=Path);p.add_argument('--baseline-pin');p.add_argument('--baseline-cold',type=Path)
    p.add_argument('--baseline-cold-pin')
    a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(e)))
        raise
