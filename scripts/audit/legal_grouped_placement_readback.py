"""Read saved proposal bytes against their immutable native targets and metrics."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch

from audit import legal_grouped_placement_training as producer
from claims.process_conditions import write_committed
from model import legal_placement_comparison as measure
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha, need


def check_row(payload,item,labels):
    row=payload['record'];i=row['offset'];future=item['frames'][i+1]
    need(fingerprint(future)==fingerprint(payload['target']),'PLACEMENT_READBACK_NATIVE_SOURCE')
    need(torch.equal(row['target'],item['codes'][i]) and row['target_count']=={n:int(v[i]) for n,v in item['counts'].items()},
         'PLACEMENT_READBACK_TARGET_LABELS')
    score=measure.numpy_energy(payload['conditional'][0],future,payload['variances'])
    need(abs(score['loss']-row['conditional_energy']['loss'])<=1e-10,'PLACEMENT_READBACK_ENERGY')
    discrete=[];brier=[]
    for level,p in enumerate(payload['probabilities']):
        values=p.numpy();target=int(row['target'][level])
        need(abs(values.sum()-1)<=1e-12 and np.isfinite(values).all() and values[target]>0,'PLACEMENT_READBACK_PROBABILITY')
        discrete.append(-np.log(values[target]));brier.append(np.square(values).sum()-2*values[target]+1)
    need(np.allclose(discrete,row['ce'].numpy(),rtol=0,atol=1e-12)
         and np.allclose(brier,row['brier'].numpy(),rtol=0,atol=1e-12),'PLACEMENT_READBACK_CATEGORICAL')
    codes,counts=labels.encode((payload['auto'],),reference=True,canonical=True)
    need(np.array_equal(codes[0],row['measured_codes'].numpy()),'PLACEMENT_READBACK_ROUNDTRIP')
    for lane,n in enumerate(labels.roster):
        need(int(counts[0,lane])==row['predicted_count'][n],'PLACEMENT_READBACK_AUTONOMOUS_SHAPE')
        d=measure.distance(future[n],payload['auto'][n])
        need(abs(d-row['native_distance'][n]['model'])<=1e-12,'PLACEMENT_READBACK_NATIVE_DISTANCE')
    if payload['wrong'] is not None:
        wrong=measure.numpy_energy(payload['wrong'][0],future,payload['variances'])
        need(abs(wrong['loss']-row['shuffled']['energy']['loss'])<=1e-10,'PLACEMENT_READBACK_SHUFFLED_ENERGY')
    return row


def run(a):
    began=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    report=producer.pinned(a.input,a.input_pin);need(report['complete'],'PLACEMENT_READBACK_INCOMPLETE')
    producer.physical(a.input,report)
    source,bundle=producer.source_bundle(a.root)
    need(report['binding']['code']==producer.code(source),'PLACEMENT_READBACK_CODE_CHANGED')
    scope=report['binding']['scope'];data=producer.load_data(argparse.Namespace(scope=scope,root=a.root),bundle)
    lookup={r['document']:r for r in data['items']};labels=data['labels']
    plan=json.loads((a.input/'donors.json').read_bytes());plan['metadata']={int(k):v for k,v in plan['metadata'].items()}
    measure.validate_donors(plan['metadata'],plan['donors'])
    need(measure.metadata(data['items'])==plan['metadata'],'PLACEMENT_READBACK_SOURCE_METADATA')
    totals={};count=0;checkpoints=0
    for name,result in report['results'].items():
        for stage in ('initial','final'):
            rows=[];identities=set()
            # Match producer order explicitly; filesystem order is never a source of truth.
            for item in data['items']:
                directory=a.input/name/stage
                for lo in range(0,len(item['codes']),8):
                    stop=min(lo+8,len(item['codes']))
                    proposals=tuple(labels.roster) if name=='whole_proposals' else ('assembled',)
                    for proposal in proposals:
                        path=directory/f"document-{item['document']}-rows-{lo}-{stop}-{proposal}.pt"
                        values=torch.load(path,map_location='cpu',weights_only=True)
                        need(len(values)==stop-lo,'PLACEMENT_READBACK_TILE_LENGTH')
                        for index,payload in enumerate(values,lo):
                            row=check_row(payload,item,labels);key=(row['document'],row['offset'],row['proposal'])
                            need(key==(item['document'],index,proposal) and key not in identities,'PLACEMENT_READBACK_ROW_IDENTITY')
                            need(row['donor']==plan['donors'][str(row['document'])][index],'PLACEMENT_READBACK_DONOR')
                            identities.add(key);rows.append(row);count+=1
            need(fingerprint(rows)==result[stage+'_rows'],'PLACEMENT_READBACK_RECORD_STATE')
            summary=measure.summary(rows,labels.roster)
            need(summary==result[stage],'PLACEMENT_READBACK_SUMMARY')
            # Separate NumPy counts check the reported accuracy denominators.
            for key,cell in summary.items():
                side,collection,proposal=key.split('|')
                selected=[r for r in rows if (r['side'],r['source'],r['proposal'])==(side,collection,proposal)]
                equal=np.stack([r['predicted'].numpy()==r['target'].numpy() for r in selected])
                need(len(selected)==cell['rows'] and np.allclose(equal.mean(0),cell['code_accuracy'],rtol=0,atol=1e-15),
                     'PLACEMENT_READBACK_DENOMINATOR')
            totals[name+'|'+stage]=dict(rows=len(rows),paired=sum(r['donor']!=-1 for r in rows),state=fingerprint(rows))
        ledger=[json.loads(line) for line in (a.input/name/'steps.jsonl').read_text().splitlines()]
        need(len(ledger)==result['updates'],'PLACEMENT_READBACK_UPDATE_COUNT')
        from senses.contract import digest
        parent=digest(dict(report['binding'],architecture=name));targets=0
        for step,entry in enumerate(ledger):
            value=dict(entry);actual=value.pop('hash')
            need(entry['cursor']==step and entry['parent']==parent and digest(value)==actual,'PLACEMENT_READBACK_LEDGER')
            parent=actual;targets+=entry['update']['targets']
        need(parent==result['ledger'] and targets==result['rows'],'PLACEMENT_READBACK_COVERAGE')
        for cursor,reference in result['checkpoints'].items():
            path=a.input/name/f'checkpoint-{cursor}.pt';saved=torch.load(path,map_location='cpu',weights_only=True)
            need(file_sha(path)==reference['pin'] and fingerprint(saved)==reference['state'],'PLACEMENT_READBACK_CHECKPOINT')
            state=saved['optimizer']['state'];parameters=sum(len(g['params']) for g in saved['optimizer']['param_groups'])
            need(saved['step']==int(cursor) and len(state)==(parameters if int(cursor) else 0),'PLACEMENT_READBACK_OPTIMIZER_COUNT')
            need(all(int(v['step'])==int(cursor) and torch.isfinite(v['exp_avg']).all()
                     and torch.isfinite(v['exp_avg_sq']).all() for v in state.values()),'PLACEMENT_READBACK_OPTIMIZER_MOMENTS')
            checkpoints+=1
    data['close']();need(not torch.cuda.is_initialized(),'PLACEMENT_READBACK_UNEXPECTED_GPU')
    r=dict(complete=True,input_pin=a.input_pin,rows=count,checkpoints=checkpoints,totals=totals,seconds=time.perf_counter()-began,
           peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
           code=file_sha(__file__),number_status='measured',acceptance='author-checked',scientific_admission=False)
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',r),rows=count)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    for n in ('input','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--input-pin',required=True);a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error)))
        raise
