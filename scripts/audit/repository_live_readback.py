"""Independently join live journal transitions, native sources, checkpoints and model scores."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.repository_kernel_connection_readback import tensor
from audit.repository_memory_readback import chain
from audit.repository_update_readback import inspect
from claims.codata_source import check,sha
from claims.episodic_memory import canonical,digest
from claims.process_conditions import write_committed
from kernel.repository_population import KINDS,RepositoryPopulation


def read(path):return json.loads(Path(path).read_bytes())


def audit(base_rows,rows,reference):
    check(rows[:len(base_rows)]==base_rows,'LIVE_READER_PREFIX')
    native={};groups=0
    def intake(body,payload):
        for condition,binding in payload['bindings'].items():
            for side in ('input','echo'):
                receipt=binding[side];key=digest(dict(episode=payload['episodes'][condition],receipt=receipt))
                check(key not in native and key in reference.records and
                      reference.records[key]['receipt']==receipt,'LIVE_READER_NATIVE')
                native[key]=reference.records[key]
    for body,payload,_ in base_rows:
        if body.get('kind') in KINDS:intake(body,payload);groups+=1
    base_groups=groups;spec=intent=checkpoint=active=None;contexts={};counts={}
    previous=base_rows[-1][2];last_intake=None;activations=[];observations=[];finishes=[]
    for seq,(body,payload,event) in enumerate(rows[len(base_rows):],len(base_rows)):
        kind=body['kind']
        prior=body['request']['prior_head'] if kind=='repository_kernel_observe' else body['prior_head']
        check(prior==previous,'LIVE_READER_EVENT_ORDER')
        if kind in KINDS:
            check(intent is None,'LIVE_READER_INTAKE_DURING_DERIVATION')
            intake(body,payload);groups+=1;last_intake=dict(count=seq+1,head=event)
        elif kind=='repository_live_enable':
            check(spec is None and payload['spec']==body['spec'],'LIVE_READER_ENABLE')
            spec=body['spec']
        elif kind=='repository_live_begin':
            check(spec is not None and intent is None and payload['previous']==checkpoint and
                  payload['groups']==groups and payload['source']['supported_intakes']==groups,
                  'LIVE_READER_BEGIN')
            if groups>base_groups:
                check(payload['source']['event_count']==last_intake['count'] and
                      payload['source']['head']==last_intake['head'],'LIVE_READER_INTAKE_HEAD')
            intent=dict(id=digest(body),**payload)
        elif kind=='repository_kernel_activate':
            chosen=body['spec'];path=Path(chosen['path']);manifest=read(path/'manifest.json')
            check(intent is not None and chosen['source']==intent['source'] and
                  sha((path/'manifest.json').read_bytes())==chosen['pin'] and
                  read(path/'COMMITTED')==dict(manifest=chosen['pin']),'LIVE_READER_ACTIVATION_INTENT')
            check(manifest['source']==chosen['source'] and manifest['excluded_tasks']==spec['excluded'] and
                  manifest['queries']==spec['queries'] and manifest['recall_passes'] and
                  all(r['passes'] and r['recall']>=.95 for r in manifest['calibration'].values()),
                  'LIVE_READER_RECALL')
            expected=[k for k in native if native[k]['task'] not in spec['excluded']]
            check(manifest['ids']==expected and not manifest['withdrawn'] and
                  manifest['slots']=={k:native[k]['receipt']['slots'] for k in expected},'LIVE_READER_COVERAGE')
            parent=None if checkpoint is None else {k:checkpoint['kernel'][k] for k in ('path','pin')}
            check(manifest['parent']==parent,'LIVE_READER_KERNEL_PARENT')
            growth={}
            for name,graph in manifest['graphs'].items():
                def neighbors(value):
                    result={}
                    for a,b,score in value:result.setdefault(a,[]).append([b,score])
                    return result
                prior_edges=neighbors(active['manifest']['graphs'][name]['edges']) if active else {}
                current_edges=neighbors(graph['edges'])
                changed=sum(prior_edges[k]!=current_edges[k] for k in set(prior_edges)&set(current_edges))
                added=set(current_edges)-set(prior_edges)
                windows=sum(native[k]['receipt']['slots'][name]['shape'][0] for k in added)
                check(changed==manifest['work'][name]['changed_old_neighbors'] and
                      windows==manifest['work'][name]['added_windows'],'LIVE_READER_UPDATE_DELTA')
                growth[name]=dict(new_windows=windows,changed_old_neighbors=changed,members=len(graph['members']),
                                  recall=manifest['calibration'][name]['recall'])
            regions=[];echoes={};atom_regions={}
            for key in expected:
                record=native[key]
                if record['side']!='echo' or record['absence']:continue
                region=digest(record['receipt']['slots']);atom_regions[key]=region
                echoes[record['condition']]=key
                if region not in regions:regions.append(region)
            check(payload['active']==chosen and payload['previous']==(None if active is None else active['spec']) and
                  payload['regions']==regions and not payload['removed'] and not payload['dirty'],'LIVE_READER_ACTIVATION')
            if active is not None:
                check(payload['standard']==active['standard'] and payload['parameters']==active['parameters'] and
                      payload['parent_model']==active['parent'],'LIVE_READER_PARENT_DRIFT')
            active=dict(spec=chosen,manifest=manifest,regions=regions,echoes=echoes,atoms=atom_regions,
                standard=payload['standard'],parameters=payload['parameters'],parent=payload['parent_model'],bank=payload['bank'])
            activations.append(dict(event=event,kernel=chosen['pin'],groups=groups,regions=len(regions),
                                    records=len(expected),growth=growth))
        elif kind=='repository_live_finish':
            state=body['state'];path=Path(state['path']);m=read(path/'manifest.json')
            check(intent is not None and payload['intent']==intent['id'] and payload['state']==state and
                  sha((path/'manifest.json').read_bytes())==state['pin'] and
                  read(path/'COMMITTED')==dict(manifest=state['pin']),'LIVE_READER_CHECKPOINT')
            check(m['source']==intent['source']==payload['source']==active['spec']['source'] and
                  m['kernel']=={k:active['spec'][k] for k in ('path','pin')} and
                  payload['kernel']==active['spec'],'LIVE_READER_FINISH_LINEAGE')
            checkpoint=dict(**state,kernel=active['spec'],source=m['source']);intent=None
            finishes.append(dict(event=event,state=state,groups=groups))
        elif kind=='repository_kernel_observe':
            r=body['request'];trace=payload['trace'];routing=trace['routing'];ctx=r['context']
            check(intent is None and checkpoint is not None and checkpoint['source']['supported_intakes']==groups,
                  'LIVE_READER_PENDING_READ')
            check(r['kernel']==active['spec']['pin'] and r['parent_model']==active['parent'] and
                  r['observation']==counts.get(ctx,0) and payload['last_event']==digest(r),'LIVE_READER_MODEL_REQUEST')
            check(native[r['key']]['side']=='input' and trace['native_slots']==native[r['key']]['receipt']['slots'],
                  'LIVE_READER_MODEL_NATIVE')
            check(routing['trace']['kernel']==r['kernel'] and routing['trace']['source']==active['spec']['source'],
                  'LIVE_READER_ROUTE_SOURCE')
            support=set()
            for atom in routing['trace']['sources']:
                check(atom in active['manifest']['ids'],'LIVE_READER_ROUTE_ATOM')
                echo=active['echoes'].get(native[atom]['condition'])
                if echo is not None:support.add(echo)
            regions={active['atoms'][k] for k in support}
            mask=np.asarray([r in regions for r in active['regions']],bool)
            check(routing['echo_support']==sorted(support) and trace['allowed']==mask.tolist() and
                  routing['regions']==[r for r in active['regions'] if r in regions],'LIVE_READER_MASK')
            check(set(payload['region_scores'])==set(native[r['key']]['receipt']['slots']),'LIVE_READER_SCORE_ROSTER')
            for value in payload['region_scores'].values():
                scores=tensor(value)
                check(scores.shape==(1,len(mask)) and np.array_equal(np.isfinite(scores[0]),mask) and
                      np.isneginf(scores[0,~mask]).all(),'LIVE_READER_NUMERICAL_MASK')
            check(payload['predicted_regions'][0] in regions and trace['bank']==active['bank'] and
                  np.isfinite(tensor(payload['hidden'])).all() and payload['sequence']==0,'LIVE_READER_MODEL_STATE')
            if ctx in contexts:check(r['previous_state']==contexts[ctx],'LIVE_READER_CONTEXT')
            contexts[ctx]=payload['hidden']['sha256'];counts[ctx]=counts.get(ctx,0)+1
            observations.append(dict(event=event,visible=int(mask.sum()),total=len(mask),context=ctx,kernel=r['kernel']))
        else:raise ValueError('LIVE_READER_UNKNOWN_EVENT:'+kind)
        previous=event
    check(intent is None and checkpoint is not None and groups==base_groups+2,'LIVE_READER_FINAL')
    return dict(activations=activations,observations=observations,finishes=finishes,contexts=contexts,
                checkpoint=checkpoint,groups=groups,records=len(native))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();start=time.perf_counter();config=read(a.root/'config.json');base=Path(config['base'])
    check(sha((base/'experience.sqlite3').read_bytes())==config['base_pin'],'LIVE_READER_BASE_PIN')
    original=Path(config['source']);pin=read(original/'intake-stage.json')['journal']
    reference=RepositoryPopulation(original/'experience.sqlite3',pin)
    contract,base_rows=chain(base/'experience.sqlite3');actual,rows=chain(a.root/'experience.sqlite3')
    check(actual==contract,'LIVE_READER_CONTRACT')
    result=audit(base_rows,rows,reference)
    final=read(a.root/'snapshot-final.json');resume,cold=(read(a.root/(s+'-stage.json')) for s in ('resume','cold'))
    check(resume['journal']==cold['journal']==sha((a.root/'experience.sqlite3').read_bytes()) and
          resume['head']==cold['head']==final['head']==rows[-1][2] and len(rows)==final['events'] and
          cold['legacy_retained'] and cold['source_unchanged'] and not cold['pending'] and
          cold['latest_snapshot_reads_forbidden_during_restore'] and
          cold['latest_snapshot_required_for_restore'] is False,'LIVE_READER_COLD')
    check(result['checkpoint']==final['repository_updates']['checkpoint'] and
          result['contexts']=={k:v['hidden'] for k,v in final['repository_runtime']['contexts'].items()},
          'LIVE_READER_FINAL_SNAPSHOT')
    pending=read(a.root/'snapshot-pending.json');interrupted=read(a.root/'interrupted-checkpoint.json')
    recovered=read(a.root/'snapshot-recovered.json')
    check(pending['repository_updates']['pending'] and not recovered['repository_updates']['pending'] and
          recovered['repository_updates']['checkpoint']['pin']==interrupted['pin'] and
          recovered['events']==pending['events']+3,'LIVE_READER_RESUMED_WITHOUT_REINTAKE')
    checkpoints={r['state']['pin']:inspect(reference,Path(r['state']['path'])) for r in result['finishes']}
    controls=[];faults=a.root/'reader-faults';check(not faults.exists(),'LIVE_READER_FAULT_EXISTS');faults.mkdir()
    for name,code in (('finish_source','LIVE_READER_FINISH_LINEAGE'),('numerical_mask','LIVE_READER_NUMERICAL_MASK'),
                      ('stale_model_kernel','LIVE_READER_MODEL_REQUEST')):
        seq=next(i for i in range(len(rows)-1,len(base_rows)-1,-1) if rows[i][0]['kind']==
                 ('repository_live_finish' if name=='finish_source' else 'repository_kernel_observe'))
        changed=copy.deepcopy(rows);body,payload,_=changed[seq]
        if name=='finish_source':payload['source']['head']='0'*64
        elif name=='stale_model_kernel':body['request']['kernel']='0'*64
        else:
            import base64
            value=next(iter(payload['region_scores'].values()));array=tensor(value).copy()
            array[0,payload['trace']['allowed'].index(False)]=0.;raw=array.astype('<f4').tobytes()
            value.update(sha256=sha(raw),base64=base64.b64encode(raw).decode())
        path=faults/(name+'.sqlite3');shutil.copyfile(a.root/'experience.sqlite3',path)
        previous=changed[seq-1][2]
        with sqlite3.connect(path) as db:
            for index in range(seq,len(changed)):
                b,v,_=changed[index];b['payload']=digest(v)
                event=digest(dict(seq=index,previous=previous,body=b))
                db.execute('INSERT OR REPLACE INTO payloads VALUES (?,?)',(b['payload'],canonical(v)))
                db.execute('UPDATE events SET previous=?,body=?,hash=? WHERE seq=?',(previous,canonical(b),event,index))
                previous=event
        try:audit(base_rows,chain(path)[1],reference)
        except ValueError as error:check(code in str(error),'LIVE_READER_FAULT_CODE:'+str(error))
        else:raise AssertionError('LIVE_READER_FAULT_ACCEPTED:'+name)
        controls.append(dict(name=name,code=code,journal=sha(path.read_bytes()),rehashed=True))
    reference.fresh()
    report=dict(schema='repository-live-readback-v1',result=result,checkpoints=checkpoints,controls=controls,
        journal=resume['journal'],snapshot=sha((a.root/'snapshot-final.json').read_bytes()),events=len(rows),
        prefix_events=len(base_rows),interruption_recovered=True,cold_replay_equal=True,
        number_status='measured',acceptance='author-checked',scientific_admission=False,learned_capability=False,
        wall_seconds=time.perf_counter()-start)
    pin=write_committed(a.root/'readback.json',report)
    print(json.dumps(dict(pin=pin,events=len(rows),controls=len(controls),seconds=report['wall_seconds'])))


if __name__=='__main__':main()
