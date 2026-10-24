"""All-occurrence legal schedule census and real typed/withheld-view readback, CPU only."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model.legal_learning_source import LegalLearningSource
from text.constellation_corpus import digest,file_sha,need
from text.legal_native_store import NativeBank,SHARD_INPUTS
from text.legal_reuse_input import ReuseInputs

PLAN=np.dtype([('document','<u4'),('side','u1'),('start','<i8'),('stop','<i8'),('prefix_start','<i8'),('prefix_stop','<i8')])
SIDES=('train','development','query')


def all_tiles(data,output):
    n=len(data.mapping);atoms=np.zeros(n,dtype='u1');edges=np.zeros(n,dtype='u1');plan=[]
    for side in SIDES:
        for tile in data.tiles(side,atoms_per_tile=256):
            need(tile.requires_carried_state==(tile.start!=tile.document_start),'LEARNING_AUDIT_STATE_RESET')
            atoms[tile.start:tile.stop]+=1
            edges[tile.start:min(tile.stop,tile.document_stop-1)]+=1
            plan.append((tile.document,SIDES.index(side),tile.start,tile.stop,*tile.earlier_prefix))
    expected=np.zeros(n,dtype='u1');expected[data.source.arrays['successors']]=1
    need(bool((atoms==1).all()) and np.array_equal(edges,expected),'LEARNING_AUDIT_FULL_DENOMINATOR')
    values=np.array(plan,dtype=PLAN);values.sort(order='start');np.save(output/'plan.npy',values,allow_pickle=False)
    # Cold file and separate source document metadata; no generated tile object is trusted here.
    saved=np.load(output/'plan.npy',mmap_mode='r',allow_pickle=False)
    need(saved['start'][0]==0 and saved['stop'][-1]==n
         and np.array_equal(saved['start'][1:],saved['stop'][:-1]),'LEARNING_AUDIT_COLD_CONTIGUOUS')
    corpus=Path(data.source.report['corpus']['path'])
    need(file_sha(corpus/'report.json')==data.source.report['corpus']['pin'],'LEARNING_AUDIT_CORPUS_PIN')
    report=json.loads((corpus/'report.json').read_bytes());ref=report['files']['documents.jsonl']
    expected_pin=ref if isinstance(ref,str) else ref['sha256']
    need(file_sha(corpus/'documents.jsonl')==expected_pin,'LEARNING_AUDIT_DOCUMENTS_PIN')
    support={side:dict(documents=0,atoms=0,successors=0,tiles=0,carried_tiles=0) for side in SIDES};cursor=0
    with (corpus/'documents.jsonl').open('rb') as handle:
        for document,line in enumerate(handle):
            source=json.loads(line);lo,hi=source['atom_start'],source['atom_end'];side=int(data.doc_sides[document])
            begin=cursor
            while cursor<len(saved) and saved['document'][cursor]==document:
                row=saved[cursor]
                need(row['side']==side and lo<=row['start']<row['stop']<=hi
                     and row['prefix_start']==lo and row['prefix_stop']==row['start'],'LEARNING_AUDIT_COLD_SOURCE')
                cursor+=1
            need(cursor>begin and saved['start'][begin]==lo and saved['stop'][cursor-1]==hi,'LEARNING_AUDIT_COLD_DOCUMENT')
            totals=support[SIDES[side]];totals['documents']+=1;totals['atoms']+=hi-lo
            totals['successors']+=hi-lo-1;totals['tiles']+=cursor-begin;totals['carried_tiles']+=cursor-begin-1
    need(cursor==len(saved) and document+1==len(data.documents),'LEARNING_AUDIT_COLD_COVERAGE')
    for side,counts in support.items():
        need(all(counts[k]==data.report['support'][side][k] for k in ('documents','atoms','successors')),
             'LEARNING_AUDIT_SPLIT_DENOMINATOR')
    return dict(atoms=n,successors=int(edges.sum()),reciprocal_instrument_targets=n*len(data.roster),
                tiles=len(saved),tile_atoms=256,support=support,all_atoms_exactly_once=True,all_successors_exactly_once=True)


def compare_slots(data,slots,occurrences,scalar):
    windows={}
    for name,batch in slots.items():
        expected=data.mapping[np.asarray(occurrences,dtype='<i8')].tolist()
        need(list(batch.uids)==expected,'LEARNING_AUDIT_ADDRESS')
        for i,uid in enumerate(expected):
            values,trace=scalar.input(name,uid,data.source.arrays['hashes'][uid].tobytes().hex())
            actual,provenance=batch.input(i)
            need(np.array_equal(actual,values) and provenance==trace,'LEARNING_AUDIT_TYPED_BYTES')
        windows[name]=len(batch.values)
    return windows


def native_views(data,output):
    # Fixed previously accepted prefix, not a score-dependent selection or a training population.
    train=np.flatnonzero((data.sides==0)&(data.mapping<SHARD_INPUTS));a=int(train[0])
    doc=np.searchsorted(data.documents[:,1],train,side='right')
    other=(data.components[doc]!=data.components[doc[0]])&(data.mapping[train]!=data.mapping[a])
    b=int(train[np.flatnonzero(other)[0]]);positions=[a,b,a]
    scalar=NativeBank(data.native.root,data.native.pin);observed=data.observations(positions,side='train',fit=True)
    results=dict(observation_windows=compare_slots(data,observed,positions,scalar),positions=positions,reciprocal={})
    for name in data.roster:
        positive=data.reciprocal([a,b],name,side='train',fit=True)
        control=data.reciprocal([a,b],name,side='train',fit=True,donors=[b,a])
        need(positive.control is None and control.control and name not in positive.inputs and name not in control.inputs,
             'LEARNING_AUDIT_WITHHELD')
        for sample in (positive,control):
            compare_slots(data,sample.inputs,sample.input_occurrences,scalar)
            compare_slots(data,sample.targets,sample.target_occurrences,scalar)
        results['reciprocal'][name]=dict(input_instruments=list(positive.inputs),target_instruments=list(positive.targets),
            positive_target_sha256=digest(positive.targets[name].values.tobytes()),
            shuffled_target_sha256=digest(control.targets[name].values.tobytes()),donors=list(control.target_occurrences))
    left=data.source.arrays['successors']
    eligible=left[(data.sides[left]==0)&(data.mapping[left]<SHARD_INPUTS)&(data.mapping[left+1]<SHARD_INPUTS)]
    need(len(eligible)>0,'LEARNING_AUDIT_NO_REAL_PAIR')
    chosen=eligible[:8].tolist();pair=data.next_atoms(chosen,side='train',fit=True)
    need(pair.context_scope=='single_source_atom','LEARNING_AUDIT_HISTORY_SCOPE')
    results['next_atoms']=dict(available_fixed_prefix_pairs=len(eligible),occurrences=chosen,
        context_windows=compare_slots(data,pair.inputs,pair.input_occurrences,scalar),
        target_windows=compare_slots(data,pair.targets,pair.target_occurrences,scalar))
    for saved in scalar.cached.values():
        saved.table.close();saved.vectors._mmap.close();saved.offsets._mmap.close()
    write_committed(output/'native-views.json',results)
    return results


def controls(data,a,positions,output):
    name=data.roster[0];left=np.asarray([positions[0]],dtype='<i8');last=np.asarray([int(data.documents[0,1])-1],dtype='<i8')
    dev=int(np.flatnonzero(data.sides==1)[0]);doc=int(np.searchsorted(data.documents[:,1],left[0],side='right'))
    lo,hi=map(int,data.documents[doc]);same=np.flatnonzero(data.mapping[lo:hi]!=data.mapping[left[0]])
    need(len(same)>0,'LEARNING_AUDIT_COMPONENT_CONTROL')
    same=np.asarray([lo+int(same[0])],dtype='<i8')
    result=dict(future_target=rejected(lambda:data.validate_next(left,left+2)),
        cross_document_target=rejected(lambda:data.validate_next(last,last+1)),
        non_train_fitting=rejected(lambda:data.observations([dev],side='development',fit=True)),
        wrong_split=rejected(lambda:data.observations([dev],side='train',fit=True)),
        visible_target=rejected(lambda:data.assert_withheld(dict.fromkeys(data.roster),{name:None},name)),
        identical_donor=rejected(lambda:data.validate_donors(left,left)),
        same_component_donor=rejected(lambda:data.validate_donors(left,same)),
        non_train_shuffle=rejected(lambda:data.reciprocal([dev],name,side='development',donors=[int(left[0])])),
        inadequate_memory=rejected(lambda:data.observations(positions,side='train',memory_bytes=1)),
        implicit_training=rejected(data.training_authority))
    missing=output/'uncommitted-bank';missing.mkdir();shutil.copy2(data.native.root/'configuration.json',missing/'configuration.json')
    unavailable=LegalLearningSource(data.source,a.split,a.split_pin,a.cold,a.cold_pin,missing,a.bank_pin)
    result['unavailable_native_shard']=rejected(lambda:unavailable.observations(positions,side='train'))
    changed=output/'changed-real-split';changed.mkdir()
    for filename in ['report.json','COMMITTED',*data.report['files']]:shutil.copy2(a.split/filename,changed/filename)
    stale=LegalLearningSource(data.source,changed,a.split_pin,a.cold,a.cold_pin,a.bank,a.bank_pin);stale.fresh()
    path=changed/'document-side.npy';before=file_sha(path)
    with path.open('r+b') as handle:
        handle.seek(-1,2);old=handle.read(1);handle.seek(-1,2);handle.write(bytes([old[0]^1]));handle.flush();os.fsync(handle.fileno())
    result['changed_split']=rejected(lambda:stale.observations(positions,side='train'))
    result['split_mutation']=dict(path=str(path),before=before,after=file_sha(path))
    return result


def run(a):
    started=tick=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    source=ReuseInputs(a.source,a.source_pin)
    data=LegalLearningSource(source,a.split,a.split_pin,a.cold,a.cold_pin,a.bank,a.bank_pin)
    phases=dict(authenticate=time.perf_counter()-tick);tick=time.perf_counter()
    counts=all_tiles(data,a.output);phases['all_occurrence_plan_and_cold_readback']=time.perf_counter()-tick;tick=time.perf_counter()
    views=native_views(data,a.output);phases['real_typed_views']=time.perf_counter()-tick;tick=time.perf_counter()
    negative=controls(data,a,views['positions'],a.output);data.fresh()
    phases['negative_controls']=time.perf_counter()-tick;tick=time.perf_counter()
    files={name:dict(bytes=(a.output/name).stat().st_size,sha256=file_sha(a.output/name)) for name in ('plan.npy','native-views.json')}
    phases['artifact_readback']=time.perf_counter()-tick;seconds=time.perf_counter()-started
    need(sum(phases.values())/seconds>=.9,'LEARNING_AUDIT_PHASE_ATTRIBUTION')
    result=dict(schema='legal-learning-source-readback-v1',complete=True,binding=data.binding,counts=counts,
        native_views=views,controls=negative,files=files,phases_seconds=phases,seconds=seconds,
        code_sha256=file_sha(Path(__file__)),adapter_sha256=file_sha(Path(__file__).parents[1]/'model/legal_learning_source.py'),
        no_pooling=True,implicit_clock=False,all_native_bank_complete=False,scientific_admission=False,training_ready=False,
        new_encoder_calls=0,model_updates=0,acceptance='author-checked',number_status='measured')
    pin=write_committed(a.output/'report.json',result)
    print(json.dumps(dict(pin=pin,counts=counts,controls=negative,seconds=seconds)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','split','cold','bank'):
        parser.add_argument('--'+name,type=Path,required=True);parser.add_argument('--'+name+'-pin',required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
