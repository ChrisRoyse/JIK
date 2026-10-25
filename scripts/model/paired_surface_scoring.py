"""Paired decoder diagnostics with one parent forward and one native trace read."""
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F
from claims.process_conditions import write_committed as write
from model.answer_surface_input import predicted_atom
from model.answer_surface_stream import native_versions
from model.complete_answer_trace import replay
from model.stream_surface import batches,EOS
from model.task_surface_scoring import native_pin
from text.constellation_corpus import digest,file_sha,need

ARMS=('parent','corrected')
PHYSICAL='MNLM_TEXT_CORPUS_PAIRED_SURFACE_PHYSICAL_MEMORY'


def atom_scores(model,slots,raw):
    """Original prefix arithmetic; the protected sum reuses its own parent logits."""
    model.fresh();need(not model.training and isinstance(raw,bytes) and raw,'PAIRED_SURFACE_INPUT')
    device=next(model.parameters()).device;slots={n:v.to(device) for n,v in slots.items()}
    required=2*(len(raw)+1)*model.config.vocabulary*4
    available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    need(required*4<=available//4,'PAIRED_SURFACE_PHYSICAL_MEMORY')
    scores={key:torch.empty((len(raw)+1,model.config.vocabulary),dtype=torch.float32) for key in ARMS}
    covered=torch.zeros(len(raw)+1,dtype=torch.bool);calls=0;initial=native_versions(slots)
    with torch.no_grad():
        for prefix,labels,offsets,native,items in batches(raw,slots,model.config):
            model.fresh();before=native_versions(native)
            parent=model.parent(prefix,native,offsets=offsets)['logits']
            correction=model.correction(prefix,native,offsets=offsets)['logits']
            values=dict(parent=parent.detach().cpu(),corrected=(parent+correction).detach().cpu());calls+=1
            need(before==native_versions(native),'PAIRED_SURFACE_NATIVE_CHANGED')
            for i,item in enumerate(items):
                lo,hi=item['scored'];left=lo-item['offset']
                need(not bool(covered[lo:hi].any()),'PAIRED_SURFACE_DUPLICATE_POSITION')
                for key in ARMS:scores[key][lo:hi]=values[key][i,left:left+hi-lo]
                covered[lo:hi]=True
            del parent,correction,values
    need(bool(covered.all()) and all(bool(v.isfinite().all()) for v in scores.values()),'PAIRED_SURFACE_MISSING_POSITION')
    need(initial==native_versions(slots),'PAIRED_SURFACE_NATIVE_CHANGED');model.fresh()
    targets=torch.tensor([*raw,EOS],dtype=torch.int64);result={};identity=native_pin(slots)
    for key,v in scores.items():
        result[key]=dict(logits=v,targets=targets,losses=F.cross_entropy(v.double(),targets,reduction='none'),
            correct=v.argmax(-1)==targets,raw_sha256=digest(raw),raw_bytes=len(raw),calls=calls,native=identity)
    return result


def score_trace(path,pin,model,references,*,output,region_pin,region_sizes,expected_request):
    path=Path(path);output=Path(output);output.mkdir(exist_ok=False)
    need(file_sha(path/'request.json')==expected_request,'PAIRED_SURFACE_REQUEST')
    source=json.loads((path/'request.json').read_bytes())
    for spec in model.specs:spec.validate()
    need({s.name:json.loads(s.definition_json)['instrument_lens_id'] for s in model.specs}==source['authority']['lens_ids'],
         'PAIRED_SURFACE_INSTRUMENTS')
    need(references and all(digest(v['raw'])==v['id'] for v in references),'PAIRED_SURFACE_REFERENCE')
    atoms=[];generated=events=0;ended=False;delta=0.
    for event in replay(path,pin):
        events+=1
        if event['kind']=='end':ended=True;continue
        atom=predicted_atom(event,model.specs,model.config,region_pin=region_pin,
            region_sizes=region_sizes,trace_pin=pin,position=generated)
        if generated<len(references):
            reference=references[generated];entry=dict(index=generated,reference=reference['id'],positions=len(reference['raw'])+1)
            try:result=atom_scores(model,atom.slots,reference['raw'])
            except ValueError as exc:
                if str(exc)!=PHYSICAL:raise
                entry.update(status='physical_refusal',refusal=str(exc));atoms.append(entry);generated+=1;continue
            dest=output/f'atom-{generated:06d}.pt'
            with dest.open('xb') as f:torch.save(result,f);f.flush();os.fsync(f.fileno())
            stats={key:dict(correct=int(result[key]['correct'].sum()),nll=float(result[key]['losses'].sum())) for key in ARMS}
            difference=float((result['parent']['losses']-result['corrected']['losses']).sum());delta+=difference
            entry.update(status='scored',path=dest.name,pin=file_sha(dest),scores=stats,
                native=result['parent']['native'],parent_minus_corrected_nll_sum=difference,calls=result['parent']['calls'])
            atoms.append(entry)
        generated+=1
    original=json.loads((path/'report.json').read_bytes())
    need(generated==original['generated_atoms'] and ended==original['model_ended'],'PAIRED_SURFACE_TERMINAL')
    total=sum(len(v['raw'])+1 for v in references);scored=[v for v in atoms if v['status']=='scored']
    matched=sum(v['positions'] for v in scored);missing=sum(len(v['raw'])+1 for v in references[generated:])
    refused=sum(v['positions'] for v in atoms if v['status']=='physical_refusal')
    need(matched+missing+refused==total,'PAIRED_SURFACE_DENOMINATOR')
    counts={key:dict(matched_nll_sum=sum(v['scores'][key]['nll'] for v in scored),
        correct=sum(v['scores'][key]['correct'] for v in scored),
        full_reference_accuracy=sum(v['scores'][key]['correct'] for v in scored)/total) for key in ARMS}
    value=dict(schema='paired-corpus-surface-v1',complete=True,trace=pin,trace_request=expected_request,
        region_pin=region_pin,region_sizes=list(region_sizes),atoms=atoms,generated_atoms=generated,events=events,
        model_ended=ended,native_status=original['status'],native_refusal=original['refusal'],
        reference_atoms=len(references),reference_positions=total,matched_positions=matched,missing_positions=missing,
        refused_positions=refused,missing_atoms=max(0,len(references)-generated),extra_atoms=max(0,generated-len(references)),
        scores=counts,parent_minus_corrected_nll_sum=delta,mean_parent_minus_corrected_nll=None if not matched else delta/matched,
        teacher_forced=True,free_generation_tested=False,semantic_validity_established=False,
        source_status='previously_exposed_development',reference_read_after_generation=True)
    return value,write(output/'report.json',value)


def contrast(left,right,root):
    """Conditioning contrasts only on identical scored positions; exclusions remain."""
    other={v['index']:v for v in right['atoms'] if v['status']=='scored'}
    counts=0;delta={key:0. for key in ARMS}
    for a in left['atoms']:
        if a['status']!='scored' or a['index'] not in other:continue
        b=other[a['index']]
        need(a['reference']==b['reference'] and a['positions']==b['positions'],'PAIRED_SURFACE_CONTRAST_REFERENCE')
        values=[]
        for trace,record in ((left,a),(right,b)):
            path=Path(root)/trace['path']/record['path'];need(file_sha(path)==record['pin'],'PAIRED_SURFACE_CONTRAST_PIN')
            values.append(torch.load(path,map_location='cpu',weights_only=True))
        for key in ARMS:delta[key]+=float((values[1][key]['losses']-values[0][key]['losses']).sum())
        counts+=a['positions']
    need(left['reference_positions']==right['reference_positions'],'PAIRED_SURFACE_CONTRAST_DENOMINATOR')
    return dict(common_positions=counts,reference_positions=left['reference_positions'],
        excluded_positions=left['reference_positions']-counts,
        scores={key:dict(control_minus_correct_nll_sum=v,mean_control_minus_correct_nll=None if not counts else v/counts)
                for key,v in delta.items()})
