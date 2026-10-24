"""Cold readback of all persisted discrimination/byte metrics and sampled forwards."""
import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import time
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from torch.nn import functional as F
from audit.capability_conditioning import fingerprint
from audit.capability_conditioning_readback import refused
from audit.capability_region_diagnostic import load
from audit.capability_rescore import install,forward,source_raw
from audit.fsv_identity_loop import configure
from claims.capability_population import need,sha
from claims.process_conditions import write_committed
from lens.lenscommon import take_lease,run_cli
from model.capability_kernel_learning import pinned
from model.capability_structure_input import StructuredContexts
from model.stream_surface import batches,position_batch


def roster(rows,plan):
    need([(r['arm'],r['task']) for r in rows]==[(a,k) for a in plan['source']['checkpoints'] for k in plan['query']],
         'RESCORE_READER_POPULATION')


def row_identity(row,corpus,plan):
    key=row['task'];task=corpus.tasks[key];raw=source_raw(corpus,key)
    need(row['source_sha256']==sha(raw) and row['source_bytes']==len(raw)
         and row['operation']==task['kind'] and row['group']==task['split_group'],'RESCORE_READER_SOURCE')
    candidates=[d for d in corpus.train if corpus.tasks[d]['kind']==task['kind']
                and corpus.tasks[d]['split_group']!=task['split_group']]
    donor=min(candidates,key=lambda d:sha((key+':'+d).encode()))
    need(row['donor']==donor==plan['donors'][key] and row['candidates']==candidates==plan['alternatives'][key],
         'RESCORE_READER_DONOR')
    need(row['identical_surfaces']==[source_raw(corpus,d)==raw for d in candidates],'RESCORE_READER_ALIASES')
    return raw


def delta(left,right):
    need(left.ndim==right.ndim==1 and len(left)==len(right)>1,'RESCORE_READER_POSITIONS')
    ds=[float(b)-float(a) for a,b in zip(left,right)]
    return dict(positions=len(ds),bytes=len(ds)-1,mean_byte_delta=math.fsum(ds[:-1])/(len(ds)-1),
        eos_delta=ds[-1],mean_position_delta=math.fsum(ds)/len(ds),positive=sum(d>0 for d in ds),
        negative=sum(d<0 for d in ds),ties=sum(d==0 for d in ds))


def trace_proof(row,arrays):
    for name in ('correct','shuffled','oracle_correct','oracle_shuffled'):
        values=arrays[name]
        need(values.dtype==np.float32 and values.shape==(row['source_bytes']+1,) and np.isfinite(values).all(),
             'RESCORE_READER_POSITIONS')
        need(row['traces'][name]['positions']==len(values),'RESCORE_READER_POSITIONS')
    need(row['deltas']==dict(predicted=delta(arrays['correct'],arrays['shuffled']),
                            oracle=delta(arrays['oracle_correct'],arrays['oracle_shuffled'])),'RESCORE_READER_DELTA')


def direct_similarity(left,right):
    """Independent untiled path where physically admitted; source arrays stay native."""
    need(left.ndim==right.ndim==2 and left.shape[1]==right.shape[1],'RESCORE_READER_NATIVE')
    need(len(left)*len(right)<=16_000_000,'RESCORE_READER_MATRIX_MEMORY')
    left=left/torch.linalg.vector_norm(left,dim=1,keepdim=True).clamp_min(1e-12)
    right=right/torch.linalg.vector_norm(right,dim=1,keepdim=True).clamp_min(1e-12)
    scores=torch.mm(left,right.T)
    return float((torch.amax(scores,dim=0).mean()+torch.amax(scores,dim=1).mean())*.5)


def lane_proof(lane,expected):
    observed=[lane['target'],*lane['distractors']]
    need(len(observed)==len(expected) and max(abs(a-b) for a,b in zip(observed,expected))<=2e-6,
         'RESCORE_READER_SCORE')
    t=lane['target'];scores=lane['distractors']
    need((lane['wins'],lane['ties'],lane['losses'])==(sum(t>s for s in scores),sum(t==s for s in scores),sum(t<s for s in scores)),
         'RESCORE_READER_COUNTS')
    return max(abs(a-b) for a,b in zip(observed,expected))


def aggregate_proof(rows,summary):
    expected={(r['arm'],r['operation'],g) for r in rows for g in ('all',r['group'])}
    need({(c['arm'],c['operation'],c['group']) for c in summary}==expected and len(summary)==len(expected),
         'RESCORE_READER_CELLS')
    for cell in summary:
        values=[r for r in rows if r['arm']==cell['arm'] and r['operation']==cell['operation']
                and (cell['group']=='all' or cell['group']==r['group'])]
        need(cell['tasks']==len(values) and cell['groups']==len({r['group'] for r in values}),'RESCORE_READER_CELLS')
        for name,result in cell['lanes'].items():
            pairs=[(r['lanes'][name]['target'],x,same) for r in values
                   for x,same in zip(r['lanes'][name]['distractors'],r['identical_surfaces'])]
            wins=sum(a>b for a,b,_ in pairs);ties=sum(a==b for a,b,_ in pairs)
            need(result==dict(wins=wins,ties=ties,losses=len(pairs)-wins-ties,pairs=len(pairs),
                strict_accuracy=wins/len(pairs),tie_half_accuracy=(wins+.5*ties)/len(pairs),
                nonidentical_pairs=sum(not s for _,_,s in pairs),nonidentical_wins=sum(a>b and not s for a,b,s in pairs),
                nonidentical_ties=sum(a==b and not s for a,b,s in pairs)),'RESCORE_READER_AGGREGATE')
        for kind,result in cell['decoder'].items():
            ds=[r['deltas'][kind]['mean_byte_delta'] for r in values];count=sum(r['source_bytes'] for r in values)
            need(result==dict(mean_task_delta=math.fsum(ds)/len(ds),
                mean_byte_delta=math.fsum(d*r['source_bytes'] for d,r in zip(ds,values))/count,bytes=count,
                better=sum(d>0 for d in ds),worse=sum(d<0 for d in ds),ties=sum(d==0 for d in ds)),
                'RESCORE_READER_AGGREGATE')


def explicit_trace(decoder,raw,native,size):
    result=[];c=decoder.config
    for prefix,labels,offsets,slots,items in batches(raw,native,c,batch_size=size):
        memories,_=decoder.memories(slots);h=decoder.tokens(prefix)+position_batch(offsets,prefix.shape[1],c.width)
        for block in decoder.blocks:
            q=block.norm(h);h=h+block.self_attention(q,q,causal=True)
            for spec in decoder.specs:h=block.branches[spec.name](h,memories[spec.name],slots[spec.name])
            h=h+block.ff(block.ff_norm(h))
        logprobs=F.log_softmax(decoder.output(h),dim=-1)
        for i,item in enumerate(items):
            active=labels[i]!=-100
            result.extend((-logprobs[i,active].gather(1,labels[i,active,None]).squeeze(1)).cpu().tolist())
    return np.asarray(result,dtype=np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();need(not a.output.exists(),'RESCORE_READER_EXISTS')
    start=time.perf_counter();ref=dict(path=str(a.report),pin=a.pin);report=pinned(ref);root=a.report.parent
    for name,pin in report['files'].items():need(sha((root/name).read_bytes())==pin,'RESCORE_READER_FILE')
    for name,pin in report['source']['code'].items():need(sha(Path(__file__).with_name(name).read_bytes())==pin,'RESCORE_READER_CODE')
    plan=json.loads((root/'plan.json').read_bytes());rows=list(map(json.loads,(root/'scores.jsonl').read_bytes().splitlines()))
    need(plan['source']==report['source'] and report['rows']==len(rows) and report['model_updates']==0,'RESCORE_READER_SCOPE')
    roster(rows,plan);aggregate_proof(rows,report['summary'])
    gpu=subprocess.check_output(['nvidia-smi'],text=True);take_lease('29','capability-rescore-readback');configure()
    model,corpus,_,_=load(report['source']['checkpoints']['phase12']);contexts=StructuredContexts(corpus)
    targets={k:corpus.target(k,model['world'].config,'cuda',fitting=k in corpus.train)[1] for k in (*corpus.train,*corpus.query)}
    maximum=0.;pairs=0;boundaries=[];rank_changes=0;near_ties=0
    cases={next(k for k in corpus.query if corpus.tasks[k]['kind']==kind) for kind in {corpus.tasks[k]['kind'] for k in corpus.query}}
    with torch.no_grad():
        for arm,reference in report['source']['checkpoints'].items():
            old,moments=install(model,reference);before=fingerprint(model.state_dict())
            need(report['identities'][arm]==dict(model=before,optimizer=moments,decoder=fingerprint(model['decoder'].state_dict())),
                 'RESCORE_READER_MODEL')
            for row in [r for r in rows if r['arm']==arm]:
                raw=row_identity(row,corpus,plan);key=row['task'];donor=row['donor']
                with np.load(root/arm/(key+'.npz'),allow_pickle=False) as arrays:
                    trace_proof(row,arrays)
                    for name,lane in row['lanes'].items():
                        prediction=torch.from_numpy(arrays['predicted__'+name]).to('cuda')
                        truth=targets[key][name];t=truth.values[0,~truth.padding[0]]
                        expected=[direct_similarity(prediction,t)]
                        for other in row['candidates']:
                            slot=targets[other][name];expected.append(direct_similarity(prediction,slot.values[0,~slot.padding[0]]))
                        maximum=max(maximum,lane_proof(lane,expected));pairs+=len(expected)-1
                        for actual,recomputed in zip(lane['distractors'],expected[1:]):
                            near_ties+=abs(lane['target']-actual)<=4e-6
                            rank_changes+=int(np.sign(lane['target']-actual))!=int(np.sign(expected[0]-recomputed))
                    if key in cases:
                        correct=forward(model,corpus,contexts,key,old)['predicted_constellation'].slots
                        shuffled=forward(model,corpus,contexts,donor,old)['predicted_constellation'].slots
                        for tag,slots in [('predicted',correct),('wrong',shuffled)]:
                            for name,slot in slots.items():
                                need(np.array_equal(slot.values[0,~slot.padding[0]].cpu().numpy(),arrays[tag+'__'+name]),
                                     'RESCORE_READER_FORWARD')
                        conditions=dict(correct=correct,shuffled=shuffled,oracle_correct=targets[key],oracle_shuffled=targets[donor])
                        for name,slots in conditions.items():
                            check=explicit_trace(model['decoder'],raw,slots,row['traces'][name]['physical_batch'])
                            need(np.array_equal(check,arrays[name]),'RESCORE_READER_DECODER')
                        boundaries.append(dict(arm=arm,task=key,operation=row['operation'],native_paths=12,decoder_paths=4))
            need(fingerprint(model.state_dict())==before,'RESCORE_READER_STATE_CHANGED')
    first=rows[0];controls=[]
    controls.append(refused('RESCORE_READER_POPULATION',lambda:roster(rows[:-1],plan)))
    changed=copy.deepcopy(first);changed['source_sha256']='0'*64
    controls.append(refused('RESCORE_READER_SOURCE',lambda:row_identity(changed,corpus,plan)))
    changed=copy.deepcopy(first);changed['donor']=first['task']
    controls.append(refused('RESCORE_READER_DONOR',lambda:row_identity(changed,corpus,plan)))
    name=next(iter(first['lanes']));lane=copy.deepcopy(first['lanes'][name]);expected=[lane['target'],*lane['distractors']];lane['target']+=1
    controls.append(refused('RESCORE_READER_SCORE',lambda:lane_proof(lane,expected)))
    with np.load(root/first['arm']/(first['task']+'.npz'),allow_pickle=False) as saved:
        arrays=dict(saved);arrays['correct']=arrays['correct'][:-1]
        controls.append(refused('RESCORE_READER_POSITIONS',lambda:trace_proof(first,arrays)))
    need(sha(Path(corpus.config['journal']['path']).read_bytes())==corpus.config['journal']['pin'],'RESCORE_READER_JOURNAL')
    for reference in report['source']['checkpoints'].values():
        original=pinned(reference);need(sha((Path(reference['path']).parent/'final.pt').read_bytes())==original['final_checkpoint'],
                                       'RESCORE_READER_CHECKPOINT_CHANGED')
    result=dict(schema='capability-rescore-readback-v1',source=ref,rows=len(rows),native_comparisons=pairs,
        max_direct_similarity_difference=maximum,direct_rank_changes=rank_changes,numerically_close_pairs=near_ties,
        forward_boundaries=boundaries,controls=controls,gpu_before=gpu,
        seconds=time.perf_counter()-start,reader_sha256=sha(Path(__file__).read_bytes()),number_status='measured',
        acceptance='author-checked',scientific_admission=False,learned_capability=False,serving=False)
    print(json.dumps(dict(pin=write_committed(a.output,result),rows=len(rows),boundaries=len(boundaries),max_error=maximum)),flush=True)


if __name__=='__main__':run_cli(main)
