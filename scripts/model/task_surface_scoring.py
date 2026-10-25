"""Post-generation surface likelihood; reference bytes never drive a rollout."""
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from claims.process_conditions import write_committed as write
from model.answer_surface_input import predicted_atom
from model.complete_answer_trace import replay
from model.native_population_training import fingerprint
from model.stream_surface import batches, EOS
from text.constellation_corpus import digest, file_sha, need


def native_pin(slots):
    return fingerprint({n:dict(identity=v.identity,rows=v.rows,values=v.values,
        padding=v.padding,absent=v.absent,indices=v.indices) for n,v in slots.items()})


def atom_score(model, slots, raw):
    """Keep the original eight-row decoder arithmetic and score every byte/EOS."""
    need(isinstance(raw, bytes) and raw, 'TASK_SURFACE_REFERENCE')
    device=next(model.parameters()).device
    slots={n:v.to(device) for n,v in slots.items()}
    required=(len(raw)+1)*model.config.vocabulary*4
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                       if x.startswith('MemAvailable:')))*1024
    need(required*4<=available//4, 'TASK_SURFACE_SCORE_PHYSICAL_MEMORY')
    scores=torch.empty((len(raw)+1,model.config.vocabulary),dtype=torch.float32)
    covered=torch.zeros(len(raw)+1,dtype=torch.bool);calls=0
    with torch.no_grad():
        for prefix,labels,offsets,native,items in batches(raw,slots,model.config):
            logits=model(prefix,native,offsets=offsets)['logits'].detach().cpu();calls+=1
            for i,item in enumerate(items):
                a,b=item['scored'];left=a-item['offset'];right=b-item['offset']
                need(not bool(covered[a:b].any()), 'TASK_SURFACE_DUPLICATE_POSITION')
                scores[a:b]=logits[i,left:right];covered[a:b]=True
    need(bool(covered.all()) and bool(scores.isfinite().all()), 'TASK_SURFACE_MISSING_POSITION')
    targets=torch.tensor([*raw,EOS],dtype=torch.int64)
    losses=F.cross_entropy(scores.double(),targets,reduction='none')
    return dict(logits=scores,targets=targets,losses=losses,correct=scores.argmax(-1)==targets,
                raw_sha256=digest(raw),raw_bytes=len(raw),calls=calls,native=native_pin(slots))


def score_trace(path,pin,model,references,*,output,region_pin,region_sizes,expected_request):
    """All events are read even when more atoms were generated than are referenced."""
    path=Path(path);output=Path(output);output.mkdir(exist_ok=False)
    need(file_sha(path/'request.json')==expected_request, 'TASK_SURFACE_TRACE_REQUEST')
    source=json.loads((path/'request.json').read_bytes())
    for spec in model.specs:spec.validate()
    need({s.name:json.loads(s.definition_json)['instrument_lens_id'] for s in model.specs}==source['authority']['lens_ids'],
         'TASK_SURFACE_LENS_IDENTITY')
    need(references and all(digest(v['raw'])==v['id'] for v in references), 'TASK_SURFACE_REFERENCE_PIN')
    atoms=[];generated=events=0;ended=False
    for event in replay(path,pin):
        events+=1
        if event['kind']=='end':ended=True;continue
        atom=predicted_atom(event,model.specs,model.config,region_pin=region_pin,
            region_sizes=region_sizes,trace_pin=pin,position=generated)
        if generated<len(references):
            reference=references[generated];result=atom_score(model,atom.slots,reference['raw'])
            dest=output/f'atom-{generated:06d}.pt';torch.save(result,dest)
            atoms.append(dict(index=generated,path=dest.name,pin=file_sha(dest),reference=reference['id'],
                positions=len(result['targets']),correct=int(result['correct'].sum()),
                nll=float(result['losses'].sum()),native=result['native']))
        generated+=1
    original=json.loads((path/'report.json').read_bytes())
    need(generated==original['generated_atoms'] and ended==original['model_ended'], 'TASK_SURFACE_TRACE_TERMINAL')
    total=sum(len(v['raw'])+1 for v in references);matched=sum(a['positions'] for a in atoms)
    r=dict(schema='task-surface-likelihood-v1',complete=True,trace=pin,trace_request=expected_request,
        region_pin=region_pin,region_sizes=list(region_sizes),atoms=atoms,generated_atoms=generated,events=events,
        model_ended=ended,native_status=original['status'],native_refusal=original['refusal'],
        reference_atoms=len(references),reference_positions=total,matched_positions=matched,
        missing_atoms=max(0,len(references)-generated),missing_positions=total-matched,
        extra_atoms=max(0,generated-len(references)),matched_nll_sum=sum(a['nll'] for a in atoms),
        full_reference_accuracy=sum(a['correct'] for a in atoms)/total,
        teacher_forced=True,free_generation_tested=False,semantic_validity_established=False,
        source_status='previously_exposed_development',reference_read_after_generation=True)
    return r,write(output/'report.json',r)


def paired(left,right,root):
    """Compare the same reference positions; disclose every excluded position."""
    common=min(len(left['atoms']),len(right['atoms']));total=0;delta=0.
    for a,b in zip(left['atoms'][:common],right['atoms'][:common],strict=True):
        need(a['index']==b['index'] and a['reference']==b['reference'] and a['positions']==b['positions'],
             'TASK_SURFACE_PAIR_REFERENCE')
        values=[]
        for parent,record in ((left,a),(right,b)):
            p=Path(root)/parent['path']/record['path'];need(file_sha(p)==record['pin'],'TASK_SURFACE_PAIR_BYTES')
            values.append(torch.load(p,map_location='cpu',weights_only=True)['losses'])
        total+=a['positions'];delta+=float((values[1]-values[0]).sum())
    need(left['reference_positions']==right['reference_positions'], 'TASK_SURFACE_PAIR_DENOMINATOR')
    return dict(common_positions=total,reference_positions=left['reference_positions'],
        excluded_positions=left['reference_positions']-total,control_minus_correct_nll_sum=delta,
        mean_control_minus_correct_nll=None if not total else delta/total)
