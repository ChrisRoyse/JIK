"""Exercise the complete teaching bank and compare bounded scoring to retained code."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from audit.fsv_identity_loop import configure
from audit.repository_demonstration_memory import authenticate
from audit.repository_demonstration_open import open_memory
from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import take_lease,run_cli
from model.constellation_learning import identity_pin
from model.loop_training_checkpoint import tensor_fingerprint
from model.repository_packed_world import PackedRepositoryWorld
from model.repository_relation_checkpoint import start
from model.repository_streaming_data import StreamingTeachingData
from model.tool_context_regions import attach
from senses.contract import SenseError
from senses.trajectory_population import read_pinned


def read(p):return json.loads(p.read_bytes())


def run(view,e,allowed,withheld=False):
    batch=e['baseline'] if withheld else e['batch']
    return view(batch,(view.start('repository:'+e['task']),),allowed=allowed,phase=0,
                event_ids=[batch.pin],evidence_origin=['source_observation'],advance_phase=False)


def loss_from_scores(scores,stage,e,key):
    support=sum(F.log_softmax(v,dim=-1) for v in scores.values())
    lanes=dict(scores,common=support/len(scores));terms=[]
    truth=F.one_hot(torch.tensor(e['outcome'],device=stage['labels'].device),3).float()
    for name,score in lanes.items():
        p=F.softmax(score[0],dim=-1)
        anchor=torch.stack([p[stage['labels']==i].sum() for i in range(3)])
        term=(anchor-truth).square().sum()
        if not e['absent']:
            target=torch.tensor([stage['regions'].index(e['region'])],device=p.device)
            term=term+F.cross_entropy(score,target)
            if name!='common':term=term+(p*stage['costs'][key][name]).sum()
        terms.append(term)
    return torch.stack(terms).mean()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--proof',type=Path,required=True)
    p.add_argument('--proof-pin',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();tick=time.perf_counter();proof=read_pinned(a.proof,a.proof_pin)
    check(not a.output.exists(),'STREAMING_OUTPUT_EXISTS')
    config=read(a.source/'config.json');saved=read(a.source/'snapshot-final.json')
    check(sha((a.source/'experience.sqlite3').read_bytes())==proof['journal'] and
          sha((a.source/'snapshot-final.json').read_bytes())==proof['snapshot'] and
          proof['cold_replay_equal'],'STREAMING_SOURCE')
    recipes=authenticate(config);gpu=subprocess.check_output(['nvidia-smi'],text=True)
    take_lease('28','repository-teaching-streaming-readback');configure()
    session,sessions=open_memory(recipes,a.source,saved)
    check(session.snapshot()==saved,'STREAMING_REPLAY')
    restored=time.perf_counter();print(json.dumps(dict(stage='restored',events=saved['events'])),flush=True)
    data=StreamingTeachingData(session,config['populations']);model,_=start(data)
    before=tensor_fingerprint(dict(model.named_parameters()))
    stage=data.stage(data.order[-1],model);view=attach(stage['view'],model)
    representatives=[next(data.examples[k] for k in stage['keys'] if data.examples[k]['region']==r)
                     for r in stage['regions']]
    blocks=[];size=min(view.block_sizes.values())
    for begin in range(0,len(representatives),size):
        ids=list(range(begin,min(begin+size,len(representatives))))
        old=PackedRepositoryWorld(model,[stage['regions'][i] for i in ids],
                                  [representatives[i] for i in ids],world_source_id=digest(stage['description']))
        blocks.append((ids,attach(old,model)))
    check(len(blocks)>1,'STREAMING_NOT_EXERCISED')
    materialized=time.perf_counter()
    print(json.dumps(dict(stage='materialized',examples=len(stage['keys']),regions=len(stage['regions']),
                         scratch=view.scratch_elements,blocks=[len(i) for i,_ in blocks])),flush=True)
    def reference(e,allowed,withheld=False):
        scores={s.name:[] for s in data.specs};outputs=[]
        for ids,old in blocks:
            mask=allowed[ids]
            check(bool(mask.any()),'STREAMING_AUDIT_EMPTY_BLOCK')
            out=run(old,e,mask,withheld);outputs.append(out)
            for s in data.specs:scores[s.name].append(out['regions'][s.name])
        return {n:torch.cat(v,dim=-1) for n,v in scores.items()},outputs
    keys=[max(stage['keys'],key=lambda k:sum(v.values.numel() for v in data.examples[k]['batch'].values()))]
    keys+= [next(k for k,e in data.examples.items() if e['action_origin']=='supplied_reference_repair'
                 and e['partition']==group) for group in data.order[-2:]]
    keys=list(dict.fromkeys(keys));comparisons=[];gradients=[]
    for key in keys:
        e=data.examples[key]
        for mode in ('full','withheld','restricted'):
            allowed=stage['allowed'].clone()
            if mode=='restricted':allowed[1::2]=False
            with torch.no_grad():
                out=run(view,e,allowed,mode=='withheld');expected,old=reference(e,allowed,mode=='withheld')
            check(all(identity_pin(x)==identity_pin(out) for x in old),'STREAMING_IDENTITY')
            differences={}
            for s in data.specs:
                x,y=out['regions'][s.name],expected[s.name]
                check(torch.equal(torch.isfinite(x),torch.isfinite(y)),'STREAMING_ALLOWED')
                differences[s.name]=float((x[:,allowed]-y[:,allowed]).abs().max())
                check(differences[s.name]<=1e-5,'STREAMING_SCORE_PARITY')
            support=sum(F.log_softmax(v,dim=-1) for v in expected.values())
            selected=int(support.argmax(-1));region=stage['regions'][selected]
            check(out['predicted_constellation'].regions==(region,),'STREAMING_REGION')
            ids,owner=next((ids,b) for ids,b in blocks if selected in ids)
            wanted=owner._placement(out['placement_hidden'],torch.tensor([ids.index(selected)],device=data.device),[e['batch'].pin])
            for s in data.specs:
                x,y=out['predicted_constellation'].slots[s.name],wanted.slots[s.name]
                check(torch.equal(x.padding,y.padding) and torch.allclose(x.values,y.values,rtol=0,atol=1e-6),
                      'STREAMING_PLACEMENT')
            comparisons.append(dict(key=key,mode=mode,score_differences=differences,region=region))
        parameters=[p for p in model.parameters() if p.requires_grad]
        left=data.losses(model,key,data.order[-1]);ga=torch.autograd.grad(left,parameters)
        expected,_=reference(e,stage['allowed']);right=loss_from_scores(expected,stage,e,key)
        gb=torch.autograd.grad(right,parameters)
        check(abs(float((left-right).detach()))<=1e-5,'STREAMING_LOSS_PARITY')
        check(all(torch.allclose(x,y,rtol=1e-4,atol=1e-6) for x,y in zip(ga,gb,strict=True)),
              'STREAMING_GRADIENT_PARITY')
        gradients.append(dict(key=key,loss=float(left.detach()),loss_difference=abs(float((left-right).detach())),
            maximum_gradient_difference=max(float((x-y).abs().max()) for x,y in zip(ga,gb,strict=True))))
    results=data.evaluate(model,data.order[-1])
    check(set(results)==set(data.examples),'STREAMING_COMPLETE_ROSTER')
    controls=[]
    def refuse(name,code,fn):
        try:fn()
        except (ValueError,SenseError) as exc:check(code in str(exc),'STREAMING_CONTROL_CODE:'+str(exc))
        else:raise AssertionError('STREAMING_CONTROL_ACCEPTED:'+name)
        controls.append(dict(name=name,code=code))
    refuse('old_full_bank_over_budget','PACKED_LANE_BUDGET',lambda:PackedRepositoryWorld(
        model,stage['regions'],representatives,world_source_id=digest(stage['description'])))
    refuse('empty_allowed','CONSTELLATION_ALLOWED',lambda:run(view,e,torch.zeros_like(stage['allowed'])))
    original=view.block_sizes;view.block_sizes={k:v+1 for k,v in original.items()}
    refuse('changed_schedule','STREAMED_SCHEDULE_CHANGED',view.check_reference);view.block_sizes=original
    broken=copy.deepcopy(view);broken.bank_seal=broken.stamps();getattr(broken,'values_0_0')[0,0,0]+=1
    refuse('changed_native_bank','PACKED_BANK_CHANGED',broken.check_reference)
    data.fresh();check(tensor_fingerprint(dict(model.named_parameters()))==before,'STREAMING_MODEL_MUTATED')
    check(session.snapshot()==saved,'STREAMING_MEMORY_MUTATED')
    for s in sessions:s.close()
    check(sha((a.source/'experience.sqlite3').read_bytes())==proof['journal'],'STREAMING_JOURNAL_MUTATED')
    value=dict(schema='repository-streaming-readback-v1',source=str(a.source),proof=dict(path=str(a.proof),pin=a.proof_pin),
        binding=data.binding,examples=len(results),regions=len(stage['regions']),results=results,comparisons=comparisons,
        gradients=gradients,controls=controls,bank_elements=view.elements,scratch_elements=view.scratch_elements,
        limit=model.config.max_native_elements,block_sizes=view.block_sizes,gpu_before=gpu,
        model_unchanged=True,journal=proof['journal'],snapshot=proof['snapshot'],number_status='measured',
        acceptance='author-checked',learned_capability=False,scientific_admission=False,serving=False,
        phases=dict(replay=restored-tick,materialization=materialized-restored,scoring=time.perf_counter()-materialized),
        wall_seconds=time.perf_counter()-tick,driver_sha256=sha(Path(__file__).read_bytes()))
    pin=write_committed(a.output,value)
    print(json.dumps(dict(pin=pin,examples=len(results),regions=len(stage['regions']),controls=len(controls),
                         scratch=view.scratch_elements,wall_seconds=value['wall_seconds'])),flush=True)


if __name__=='__main__':run_cli(main)
