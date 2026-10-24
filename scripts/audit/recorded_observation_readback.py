"""Read source bytes and observation journals without importing their producer.

PC-07/37: authenticate native files once; reuse the resulting per-lane receipt
index for the real journal and deliberately rehashed physical fault copies.
"""
import argparse
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from audit.native_session_readback import array,digest,encode,file_sha,pinned


def need(ok,code):
    if not ok:raise ValueError('MNLM_OBSERVATION_READBACK_'+code)


FLAGS=dict(source_measurements=True,parameter_update=False,
           scientific_admission=False,serving=False,learned_capability=False)


def journal(path,warm,contract,moments,traces):
    db=sqlite3.connect(f'file:{path.as_posix()}?mode=ro',uri=True)
    try:
        need(db.execute('SELECT contract FROM metadata').fetchall()==[(encode(contract),)],'CONTRACT')
        states=copy.deepcopy(warm['before']['contexts']);head='0'*64;used=set();operations=[]
        for expected,row in enumerate(db.execute('SELECT seq,previous,body,hash FROM events ORDER BY seq')):
            seq,previous,raw,pin=row;body=json.loads(raw)
            need(seq==expected and previous==head and raw==encode(body) and
                 pin==digest(dict(seq=seq,previous=previous,body=body)),'CHAIN')
            need(set(body)=={'kind','request','payload','output','flags'} and
                 body['kind']=='recorded_observation' and body['flags']==FLAGS,'EVENT')
            r=body['request'];context=r['context'];index=r['index']
            need(context in states and type(index) is int and 0<=index<len(moments),'CONTEXT')
            m=moments[index];state=states[context]
            need(r['episode']==m['id'] and r['source']==m['source'] and
                 r['event_time']==m['event_time'] and r['situation']==m['content'] and
                 r['input_binding']==warm['input_binding'],'SOURCE_JOIN')
            need(r['origin']=='source_observation' and r['purpose']=='recorded_observation','PURPOSE')
            need(r['prior_head']==head and r['previous_event']==state['last_event'] and
                 r['identity_sequence']==state['identity_sequence']==0 and
                 r['observation']==state['observations'] and m['id'] not in state['episodes'],'HISTORY')
            if state['episodes']:
                prior=next(x for x in moments if x['id']==state['episodes'][-1])
                need(all(prior['source'][k]==m['source'][k] for k in ('bundle','clock_id','split_group'))
                     and Fraction(*prior['event_time']['at'])<=Fraction(*m['event_time']['at']),'CLOCK')
            item=db.execute('SELECT data FROM payloads WHERE hash=?',(body['payload'],)).fetchone()
            need(item is not None and hashlib.sha256(item[0]).hexdigest()==body['payload'],'PAYLOAD_PIN')
            p=json.loads(item[0]);need(encode(p)==item[0],'PAYLOAD_CANONICAL')
            array(p['world'],(16,32));array(p['identity'],(6,32))
            need(p['identity_sequence']==0 and p['last_event']==digest(r),'STATE')
            actual=sorted(p['traversals'],key=lambda x:encode(x))
            wanted=sorted(traces[m['id']]*2,key=lambda x:encode(x))
            need(actual==wanted,'TRAVERSAL')
            need(p['world']['sha256']!=state['world'],'NO_OBSERVATION_UPDATE')
            states[context]=dict(observations=state['observations']+1,
                episodes=state['episodes']+[m['id']],identity_sequence=0,last_event=digest(r),
                world=p['world']['sha256'],identity=p['identity']['sha256'])
            if context=='scene':need(states[context]==warm['steps'][r['observation']],'STEP')
            operations.append(dict(seq=seq,context=context,index=index,native_traversals=len(actual)))
            used.add(body['payload']);head=pin
        need(head==warm['after']['retained']['head'] and states==warm['after']['contexts'],'FINAL')
        need(used=={r[0] for r in db.execute('SELECT hash FROM payloads')},'ORPHAN')
        need(len(operations)==5 and states['scene']['observations']==4 and
             states['isolated']['observations']==1 and
             states['scene']['world']!=states['isolated']['world'],'HISTORY_EFFECT')
        return dict(operations=operations,head=head,contexts=states)
    finally:db.close()


def corrupt(source,target,mutation):
    """Rehash semantic defects, so rejection must inspect meaning-bearing fields."""
    need(not target.exists(),'CONTROL_EXISTS');shutil.copy2(source,target)
    db=sqlite3.connect(target)
    try:
        if mutation=='truncate':
            db.execute('DELETE FROM events WHERE seq=(SELECT MAX(seq) FROM events)')
        elif mutation=='payload_bytes':
            pin,raw=db.execute('SELECT hash,data FROM payloads LIMIT 1').fetchone()
            db.execute('UPDATE payloads SET data=? WHERE hash=?',(raw+b' ',pin))
        else:
            seq,_,raw,_=db.execute('SELECT seq,previous,body,hash FROM events ORDER BY seq LIMIT 1').fetchone()
            body=json.loads(raw)
            if mutation=='situation':body['request']['situation']={'invented':'unattested person'}
            elif mutation=='purpose':body['request']['purpose']='assert_world_truth'
            elif mutation=='history':body['request']['observation']=7
            elif mutation=='traversal':
                old=body['payload'];p=json.loads(db.execute('SELECT data FROM payloads WHERE hash=?',(old,)).fetchone()[0])
                p['traversals'].pop();body['payload']=digest(p)
                db.execute('DELETE FROM payloads WHERE hash=?',(old,))
                db.execute('INSERT INTO payloads VALUES (?,?)',(body['payload'],encode(p)))
            else:raise ValueError('unknown fault')
            db.execute('UPDATE events SET body=? WHERE seq=?',(encode(body),seq))
            rows=db.execute('SELECT seq,body FROM events ORDER BY seq').fetchall();head='0'*64
            for seq,raw in rows:
                body=json.loads(raw);pin=digest(dict(seq=seq,previous=head,body=body))
                db.execute('UPDATE events SET previous=?,hash=? WHERE seq=?',(head,pin,seq));head=pin
        db.commit()
    finally:db.close()


def run(a):
    start=time.perf_counter();r=a.run;c=json.loads((r/'config.json').read_bytes());s=c['source']
    warm=pinned(r/'warm.json',a.warm_pin);cold=pinned(r/'cold.json',a.cold_pin)
    need(warm['flags']==cold['flags']==FLAGS and cold['warm']==a.warm_pin and
         cold['after']==warm['after'] and cold['exact'] is True,'COLD')
    need(warm['source_config']==cold['source_config']==c,'CONFIG')
    for report in (warm,cold):
        need(sum(report['phases'].values())/report['wall_seconds']>=.9,'PHASE_COVERAGE')
    need(file_sha(r/'observations.sqlite')==warm['journal']==cold['journal'],'JOURNAL_BYTES')
    original=pinned(Path(s['inputs'])/'report.json',s['input_pin']);moments=original['moments']
    previous=pinned(Path(s['previous'])/'report.json',s['previous_pin'])
    bank=pinned(Path(s['bank'])/'bank.json',s['bank_pin'])
    records=pinned(r/'source-records.json',warm['records'])
    need(records==pinned(Path(s['previous'])/'episode-records.json',previous['records']),'RECORDS')
    need(file_sha(r/'memory.sqlite')==warm['source_memory']==s['memory_pin']==
         file_sha(Path(s['inputs'])/'rollback.sqlite'),'SOURCE_MEMORY')
    cp=json.loads((r/'initial/manifest.json').read_bytes())
    old=json.loads((Path(s['previous'])/'initial/manifest.json').read_bytes())
    transition=pinned(r/'transition.json',warm['transition'])
    need(digest(cp)==warm['checkpoint']==transition['after']==(r/'initial/COMMITTED').read_text() and
         digest(old)==previous['initial']==transition['before'],'CHECKPOINT')
    need({k:v for k,v in cp.items() if k!='code'}=={k:v for k,v in old.items() if k!='code'} and
         file_sha(r/'initial/state.pt')==file_sha(Path(s['previous'])/'initial/state.pt')==
         old['sha256']==transition['payload_sha256'],'UNCHANGED_WEIGHTS')
    payload=torch.load(r/'initial/state.pt',map_location='cpu',weights_only=True)
    weights=digest({n:dict(shape=list(t.shape),dtype=str(t.dtype),
        sha256=hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()) for n,t in payload['model'].items()})
    need(weights==warm['weights'],'MODEL_WEIGHTS')
    for name,pin in cp['code'].items():need(file_sha(a.repo/'scripts/model'/name)==pin,'MODEL_CODE')
    for field,path in (('driver','audit/recorded_observation_witness.py'),
                       ('session_code','model/recorded_observation_session.py'),
                       ('transition_code','audit/observation_checkpoint_transition.py')):
        need(file_sha(a.repo/'scripts'/path)==warm[field]==cold[field],'RUNTIME_CODE')
    db=sqlite3.connect(f'file:{(r/"observations.sqlite").as_posix()}?mode=ro',uri=True)
    contract=json.loads(db.execute('SELECT contract FROM metadata').fetchone()[0]);db.close()
    need(digest(contract)==warm['after']['retained']['contract']==warm['before']['retained']['contract']
         and contract['flags']==FLAGS and contract['checkpoint']==warm['checkpoint']
         and contract['input_binding']==warm['input_binding']==previous['binding']
         and contract['standard']==previous['standard'] and contract['world']==previous['world']
         and contract['instruments']==[x['identity'] for x in cp['instruments']],'BINDING')
    need(contract['memory']==pinned(Path(s['inputs'])/'rollback-retained.json',s['retained_pin']),'RETENTION')
    for name,pin in contract['code'].items():need(file_sha(a.repo/'scripts'/name)==pin,'JOURNAL_CODE')
    for spec in cp['instruments']:
        definition=json.loads(spec['definition_json'])
        need(digest(definition)==spec['identity'] and
             definition['measurement_contract']==bank['instruments'][spec['name']],'INSTRUMENT')
    grouped={};total=0;readings=0;bank_root=Path(s['bank'])
    roots=[bank_root.resolve(),*(x.resolve() for x in a.shared_bank)]
    for lane,entries in bank['readings'].items():
        for entry in entries:
            path=(bank_root/lane/entry['file']).resolve()
            need(any(path.is_relative_to(root) for root in roots),'STORAGE_ROOT')
            need(file_sha(path)==entry['sha256'] and
                 path.stat().st_size==entry['shape'][0]*entry['shape'][1]*4,'NATIVE_BYTES')
            total+=path.stat().st_size;readings+=1
            grouped.setdefault((lane,entry['input']['segment']),[]).append(entry)
    census={};traces={}
    for m in moments:
        census[m['id']]={};traces[m['id']]=[]
        for name,record in records[m['id']].items():
            entries=grouped[name,m['source']['segment']];count=0;h=hashlib.sha256();size=0
            need(len(entries)==len(record['pieces']) and record['source']==m['source'],'NATIVE_ROSTER')
            for entry,timing in zip(entries,record['pieces'],strict=True):
                need(timing['reading_sha256']==entry['sha256'] and timing['source']==m['source']
                     and timing['row']==m['id'] and timing['bank']==s['bank_pin'],'NATIVE_JOIN')
                n,dims=entry['shape'];size+=n*dims*4
                h.update(encode(dict(file=entry['file'],sha256=entry['sha256'],
                    rows=[count,count+n],timing=digest(timing))));count+=n
            census[m['id']][name]=dict(rows=count,readings=len(entries),dims=dims)
            traces[m['id']].append(dict(row=m['id'],lane=name,count=count,
                readings=len(entries),bytes=size,traversal=h.hexdigest()))
    need(census==pinned(r/'census.json',warm['census']) and total==warm['source_bytes']==2960381952
         and readings==1934 and all(len(v)==16 for v in census.values()),'CENSUS')
    authenticated=time.perf_counter()
    result=journal(r/'observations.sqlite',warm,contract,moments,traces)
    control_codes={'truncate':'FINAL','payload_bytes':'PAYLOAD_PIN','situation':'SOURCE_JOIN',
                   'purpose':'PURPOSE','history':'HISTORY','traversal':'TRAVERSAL'}
    controls=[]
    for name,code in control_codes.items():
        path=r/('readback-fault-'+name+'.sqlite');corrupt(r/'observations.sqlite',path,name)
        try:journal(path,warm,contract,moments,traces)
        except ValueError as exc:
            need(str(exc)=='MNLM_OBSERVATION_READBACK_'+code,'FAULT_REASON:'+str(exc))
        else:raise ValueError('fault accepted:'+name)
        controls.append(dict(name=name,code=code,file=path.name,sha256=file_sha(path)))
    need(file_sha(r/'observations.sqlite')==warm['journal'],'READBACK_MUTATED')
    finished=time.perf_counter()
    result.update(schema='recorded-observation-readback-v1',acceptance='author-checked',
        number_status='measured',flags=FLAGS,warm=a.warm_pin,cold=a.cold_pin,controls=controls,
        native_bytes=total,native_readings=readings,checkpoint=warm['checkpoint'],
        code=file_sha(Path(__file__)),phases=dict(authentication=authenticated-start,
        journal_and_controls=finished-authenticated),wall_seconds=finished-start)
    with a.output.open('x') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
    print(json.dumps(dict(report=file_sha(a.output),operations=len(result['operations']),
        controls=len(controls),native_bytes=total,wall_seconds=result['wall_seconds'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','repo','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('warm-pin','cold-pin'):p.add_argument('--'+name,required=True)
    p.add_argument('--shared-bank',type=Path,action='append',default=[])
    run(p.parse_args())
