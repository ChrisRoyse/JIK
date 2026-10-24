"""Physical source, kernel, numerical mask, context and journal boundary readback."""
import argparse
import base64
import copy
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.repository_memory_readback import chain
from claims.codata_source import check,sha
from claims.episodic_memory import canonical,digest
from claims.process_conditions import write_committed


def read(path):return json.loads(Path(path).read_bytes())


def tensor(value):
    raw=base64.b64decode(value['base64'],validate=True)
    check(sha(raw)==value['sha256'] and value['dtype']=='float32-le','CONNECTION_READER_TENSOR')
    a=np.frombuffer(raw,dtype='<f4')
    check(a.size==int(np.prod(value['shape'])),'CONNECTION_READER_SHAPE')
    return a.reshape(value['shape'])


def audit(source_rows,rows):
    check(rows[:len(source_rows)]==source_rows,'CONNECTION_READER_PREFIX')
    native={}
    for body,payload,_ in source_rows:
        if body.get('kind') not in ('repository_cohort_intake','repository_depth_intake','repository_demonstration_intake'):
            continue
        for condition,binding in payload['bindings'].items():
            for side in ('input','echo'):
                receipt=binding[side]
                key=digest(dict(episode=payload['episodes'][condition],receipt=receipt))
                native[key]=dict(condition=condition,side=side,receipt=receipt,episode=payload['episodes'][condition])
    current=None;contexts={};dirty=set();observations=[];activations=[];reset=[];counts={}
    previous=source_rows[-1][2]
    for body,payload,pin in rows[len(source_rows):]:
        kind=body['kind']
        if kind=='repository_kernel_activate':
            spec=body['spec'];path=Path(spec['path'])/'manifest.json'
            check(sha(path.read_bytes())==spec['pin'],'CONNECTION_READER_KERNEL_PIN')
            manifest=read(path)
            check(manifest['source']==spec['source'] and manifest['recall_passes'] and
                all(r['passes'] and r['recall']>=.95 for r in manifest['calibration'].values()),
                'CONNECTION_READER_KERNEL_RECALL')
            regions=[];echoes={};atom_regions={}
            for key in manifest['ids']:
                check(key in native and native[key]['receipt']['slots']==manifest['slots'][key],'CONNECTION_READER_NATIVE')
                record=native[key]
                if record['side']!='echo' or all(s['absent'] for s in record['receipt']['slots'].values()):continue
                region=digest(record['receipt']['slots']);atom_regions[key]=region
                echoes[record['condition']]=key
                if region not in regions:regions.append(region)
            removed=sorted(set(current['manifest']['ids'])-set(manifest['ids'])) if current else []
            check(payload['regions']==regions and payload['active']==spec and payload['removed']==removed
                and payload['previous']==(None if current is None else current['spec'])
                and payload['dirty']==(sorted(contexts) if removed else []),'CONNECTION_READER_ACTIVATION')
            if removed:dirty.update(contexts)
            check(body['prior_head']==previous,'CONNECTION_READER_EVENT_ORDER')
            current=dict(spec=spec,manifest=manifest,regions=regions,echoes=echoes,atoms=atom_regions,
                         bank=payload['bank'],parent=payload['parent_model'])
            activations.append(dict(pin=pin,kernel=spec['pin'],regions=len(regions),removed=removed))
        elif kind=='repository_kernel_observe':
            r=body['request'];trace=payload['trace'];routing=trace['routing'];ctx=r['context']
            check(current is not None and r['kernel']==current['spec']['pin'] and ctx not in dirty,
                  'CONNECTION_READER_STALE_CONTEXT')
            check(r['parent_model']==current['parent'] and r['prior_head']==previous and
                  r['observation']==counts.get(ctx,0) and payload['last_event']==digest(r),
                  'CONNECTION_READER_EVENT_ORDER')
            check(native[r['key']]['side']=='input' and trace['native_slots']==native[r['key']]['receipt']['slots'],
                  'CONNECTION_READER_INPUT')
            check(routing['trace']['kernel']==r['kernel'] and routing['trace']['source']==current['spec']['source'],
                  'CONNECTION_READER_ROUTING')
            support=set()
            for atom in routing['trace']['sources']:
                check(atom in current['manifest']['ids'],'CONNECTION_READER_ROUTED_SOURCE')
                echo=current['echoes'].get(native[atom]['condition'])
                if echo is not None:support.add(echo)
            regions={current['atoms'][k] for k in support}
            mask=np.asarray([reg in regions for reg in current['regions']],dtype=bool)
            check(routing['echo_support']==sorted(support) and trace['allowed']==mask.tolist()
                and routing['regions']==[reg for reg in current['regions'] if reg in regions],
                'CONNECTION_READER_MASK')
            check(set(payload['region_scores'])==set(native[r['key']]['receipt']['slots']),'CONNECTION_READER_SCORE_ROSTER')
            for name,value in payload['region_scores'].items():
                scores=tensor(value)
                check(scores.shape==(1,len(mask)) and np.array_equal(np.isfinite(scores[0]),mask)
                      and np.isneginf(scores[0,~mask]).all(),'CONNECTION_READER_NUMERICAL_MASK')
            check(payload['predicted_regions'][0] in regions and trace['bank']==current['bank'],
                  'CONNECTION_READER_SELECTED_REGION')
            check(np.isfinite(tensor(payload['hidden'])).all() and payload['sequence']==0,'CONNECTION_READER_STATE')
            if ctx in contexts:check(r['previous_state']==contexts[ctx],'CONNECTION_READER_CONTEXT_ISOLATION')
            contexts[ctx]=payload['hidden']['sha256']
            counts[ctx]=counts.get(ctx,0)+1
            observations.append(dict(event=pin,context=ctx,kernel=r['kernel'],input=r['key'],
                visible=int(mask.sum()),total=len(mask),hidden=contexts[ctx],output=payload['output']))
        elif kind=='repository_kernel_reset':
            ctx=body['context']
            check(ctx in dirty and payload['previous']==contexts[ctx] and payload['kernel']==current['spec']['pin'],
                  'CONNECTION_READER_RESET')
            check(body['prior_head']==previous,'CONNECTION_READER_EVENT_ORDER')
            contexts[ctx]=payload['reset'];counts[ctx]=0;dirty.remove(ctx);reset.append(ctx)
        else:raise AssertionError('MNLM_CONNECTION_READER_UNKNOWN_EVENT:'+kind)
        previous=pin
    return dict(activations=activations,observations=observations,contexts=contexts,dirty=sorted(dirty),resets=reset)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();started=time.perf_counter()
    config=read(a.root/'config.json');source=Path(config['source'])
    check(sha((source/'experience.sqlite3').read_bytes())==config['source_pin'],'CONNECTION_READER_SOURCE_PIN')
    old_contract,old_rows=chain(source/'experience.sqlite3')
    contract,rows=chain(a.root/'experience.sqlite3')
    check(contract==old_contract,'CONNECTION_READER_CONTRACT')
    result=audit(old_rows,rows)
    final=read(a.root/'snapshot-final.json');old=read(source/'snapshot-final.json')
    check(len(rows)==final['events'] and rows[-1][2]==final['head'] and
          all(final[k]==v for k,v in old.items() if k not in ('head','events')),'CONNECTION_READER_OLD_STATE')
    runtime=final['repository_runtime']
    check(result['contexts']=={c:s['hidden'] for c,s in runtime['contexts'].items()} and result['dirty']==runtime['dirty'],
          'CONNECTION_READER_FINAL_CONTEXTS')
    warm,cold=read(a.root/'warm-stage.json'),read(a.root/'cold-stage.json')
    check(warm['head']==cold['head']==final['head'] and warm['snapshot']==cold['snapshot']==sha((a.root/'snapshot-final.json').read_bytes())
        and warm['journal']==cold['journal']==sha((a.root/'experience.sqlite3').read_bytes())
        and cold['cold_replay_equal'] and cold['legacy_retained'],'CONNECTION_READER_COLD')
    for row in (warm,cold):check(abs(sum(row['phases'].values())-row['wall_seconds'])<1e-8,'CONNECTION_READER_TIMINGS')
    controls=[]
    with tempfile.TemporaryDirectory(prefix='repository-connection-faults-',dir=a.root.parent) as temporary:
        for name,code in (('mask','CONNECTION_READER_MASK'),('scores','CONNECTION_READER_NUMERICAL_MASK'),
                          ('kernel','CONNECTION_READER_STALE_CONTEXT'),
                          ('native','CONNECTION_READER_INPUT')):
            path=Path(temporary)/(name+'.sqlite3');shutil.copyfile(a.root/'experience.sqlite3',path)
            body,payload,_=copy.deepcopy(rows[-1])
            if name=='mask':payload['trace']['allowed'][0]=not payload['trace']['allowed'][0]
            elif name=='scores':
                value=next(iter(payload['region_scores'].values()));scores=tensor(value).copy()
                scores[0,payload['trace']['allowed'].index(False)]=0.
                raw=scores.astype('<f4').tobytes();value.update(sha256=sha(raw),base64=base64.b64encode(raw).decode())
            elif name=='kernel':body['request']['kernel']='0'*64
            else:next(iter(payload['trace']['native_slots'].values()))['shape'][0]+=1
            body['payload']=digest(payload);seq=len(rows)-1;previous=rows[-2][2]
            pin=digest(dict(seq=seq,previous=previous,body=body))
            with sqlite3.connect(path) as db:
                db.execute('INSERT OR REPLACE INTO payloads VALUES (?,?)',(body['payload'],canonical(payload)))
                db.execute('UPDATE events SET body=?,hash=? WHERE seq=?',(canonical(body),pin,seq))
            try:audit(old_rows,chain(path)[1])
            except ValueError as error:check(code in str(error),'CONNECTION_READER_CONTROL_CODE')
            else:raise AssertionError('MNLM_CONNECTION_READER_FAULT_ACCEPTED:'+name)
            controls.append(dict(name=name,code=code,semantic_defect_rehashed=True))
    report=dict(schema='repository-kernel-connection-readback-v1',result=result,controls=controls,
        source=config['source_pin'],journal=warm['journal'],snapshot=warm['snapshot'],events=len(rows),
        source_events=len(old_rows),wall_seconds=time.perf_counter()-started,number_status='measured',
        acceptance='author-checked',scientific_admission=False,learned_capability=False,serving=False)
    pin=write_committed(a.output,report)
    print(json.dumps(dict(pin=pin,events=len(rows),controls=len(controls),wall_seconds=report['wall_seconds'])),flush=True)


if __name__=='__main__':main()
