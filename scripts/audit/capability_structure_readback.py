"""Independent field/clock/window feature reconstruction and context attention readback."""
import argparse
import copy
from dataclasses import replace
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit.capability_conditioning import fingerprint
from audit.capability_conditioning_readback import refused, independent_score
from audit.capability_order_diagnostic import snapshot
from audit.capability_recognition_learning import prepare, frozen
from audit.fsv_identity_loop import configure
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import take_lease, run_cli
from model.capability_kernel_learning import pinned, predict
from model.capability_structure import attach_context, predict_structured
from model.capability_structure_input import StructuredContexts

ROLE_ORDER=('attested_speaker_act','base_api_source','base_documentation','candidate_source','original_source',
            'public_base_test','recorded_patch','stdout','task_report','problem','recorded_step','source_passage','source_question','source_answer')


def logarithm(x):return math.copysign(math.log1p(abs(float(x))),float(x))


def reconstruct(corpus,descriptor,feature):
    key=descriptor['task'];task=corpus.tasks[key];pop=corpus.sources[task['population']].population
    need([f['id'] for f in descriptor['fields']]==[f['id'] for f in task['selections']['context']['fields']],'STRUCTURE_READER_CONTEXT')
    need(descriptor['operation']==task['kind'] and descriptor['population']==task['population']
         and descriptor['binding']==corpus.sources[task['population']].pin
         and descriptor['source_identifiers_are_features'] is False and descriptor['target_cutoff_is_feature'] is False,'STRUCTURE_READER_SCOPE')
    intervals=task['kind'] in ('respond_to_attested_act','continue_attested_dialogue')
    origin=max(Fraction(pop.fields[f['id']]['clock']['end']) for f in descriptor['fields']) if intervals else None
    need(descriptor['latest_input_end']==(str(origin) if intervals else None),'STRUCTURE_READER_ORIGIN')
    metadata=[];windows={s.name:[] for s in corpus.specs}
    for index,saved in enumerate(descriptor['fields']):
        field=pop.fields[saved['id']];clock=field['clock']
        need(all(saved[k]==field[k] for k in ('id','role','source','clock','text_sha256')) and saved['bytes']==len(field['text'].encode()),
             'STRUCTURE_READER_SOURCE_FIELD')
        position=index if task['kind']=='continue_recorded_steps' else None
        revision=('original' if index<3 else 'revised') if task['kind']=='revise_with_source_evidence' else None
        time_basis='annotation_interval' if intervals else 'unknown' if clock is None else 'receipt_only' if 'available' in clock else 'base_before_patch'
        speaker=field['source']['speaker']['role'] if intervals else None
        need(saved['document_position']==position and saved['revision']==revision and saved['time_basis']==time_basis and saved['speaker_role']==speaker,
             'STRUCTURE_READER_SEMANTICS')
        coords=[float(position is not None),logarithm(position or 0),float(revision=='original'),float(revision=='revised'),
                0.,0.,0.,0.,0.,float(time_basis=='receipt_only'),float(time_basis=='base_before_patch')]
        if intervals:
            lo,hi=Fraction(clock['start']),Fraction(clock['end']);need(lo<=hi<=origin<=Fraction(pop.tasks[key]['provenance']['cutoff']),'STRUCTURE_READER_FUTURE')
            coords[4:9]=[1.,logarithm(lo-origin),logarithm(hi-origin),logarithm(hi-lo),1.]
        metadata.append([float(field['role']==r) for r in ROLE_ORDER]+[float(speaker==r) for r in ('ID','ME','PM','UI')]+coords)
        for spec in corpus.specs:
            data,native_desc=corpus.sources[task['population']].field(field['id'],spec.name)
            need(saved['readings'][spec.name]==dict(native_desc,sha256=sha(data.astype('<f4',copy=False).tobytes()),instrument=spec.identity),
                 'STRUCTURE_READER_NATIVE')
            cells=[]
            for part in native_desc['parts']:
                lo,hi=part['field_window_span'];start,end=part['source_byte_span']
                for offset in range(lo,hi):cells.append(list(map(logarithm,(offset,len(data),start,end,len(field['text'].encode()),offset-lo))))
            windows[spec.name].append(np.asarray(cells,dtype='<f4').tolist())
    expected=dict(field=np.asarray(metadata,dtype='<f4').tolist(),windows=windows)
    need(expected==feature,'STRUCTURE_READER_FEATURES')


def independent_attention(layer,projector,q,values,positions,positioner):
    heads=layer.heads;depth=layer.width//heads
    query=layer.q(q).reshape(1,1,heads,depth).transpose(1,2);combined=torch.zeros_like(query);mass=None
    for lo in range(0,values.shape[1],256):
        memory=projector(values[:,lo:lo+256])+positioner(positions[:,lo:lo+256]);n=memory.shape[1]
        keys=layer.k(memory).reshape(1,n,heads,depth).transpose(1,2)
        vals=layer.v(memory).reshape(1,n,heads,depth).transpose(1,2)
        scores=(query@keys.transpose(-2,-1))/math.sqrt(depth);part_mass=torch.logsumexp(scores,-1);part=scores.softmax(-1)@vals
        weight=torch.zeros_like(part_mass) if mass is None else torch.sigmoid(mass-part_mass)
        combined=weight[...,None]*combined+(1-weight[...,None])*part
        mass=part_mass if mass is None else torch.logaddexp(mass,part_mass)
    return layer.out(combined.transpose(1,2).contiguous().reshape(1,1,layer.width))


def independent_context(model,item,operation,mode):
    world=model['world'];extension=model['context'];state=model['request'](world.start('capability:'+item.task),operation=operation)
    meta=item.field_metadata
    if mode=='boundaries':meta=torch.zeros_like(meta)
    elif mode=='mismatched':meta=meta.roll(1,0)
    derived=[]
    for lane,spec in enumerate(world.specs):
        name=spec.name;slot=item.native[name];q=state.hidden[lane:lane+1,None,:];fields=[]
        for index,(lo,hi) in enumerate(item.spans[name]):
            positions=item.windows[name][index]
            if mode=='boundaries':positions=torch.zeros_like(positions)
            elif mode=='mismatched':positions=positions.flip(0)
            fields.append(independent_attention(extension.windows[name],world.core.projectors[name],q,slot.values[:,lo:hi],
                                                positions.to('cuda')[None],extension.position[name]))
        memory=torch.cat(fields,dim=1)+extension.metadata[name](meta.to('cuda'))[None]
        derived.append(extension.out[name](extension.fields[name](q,memory))[:,0])
    residual=torch.stack(derived,dim=1);state=replace(state,hidden=state.hidden+residual[0])
    output=world(item.native,(state,),allowed=torch.ones(len(corpus_rows(world)),dtype=torch.bool,device='cuda'),phase=0,
        event_ids=[digest(dict(key=item.task,operation=operation))],evidence_origin=['source_observation'],advance_phase=False)
    return output,residual


def corpus_rows(world):return world.reference_rows[world.specs[0].name]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();need(not a.output.exists(),'STRUCTURE_READER_EXISTS')
    ref=dict(path=str(a.root/'report.json'),pin=a.pin);report=pinned(ref);plan=report['plan'];source=plan['source']
    need(plan==json.loads((a.root/'plan.json').read_bytes()),'STRUCTURE_READER_PLAN')
    for name,pin in source['code'].items():need(sha((Path(__file__).resolve().parents[1]/name).read_bytes())==pin,'STRUCTURE_READER_CODE')
    for name,pin in report['files'].items():need(sha((a.root/name).read_bytes())==pin,'STRUCTURE_READER_FILE')
    gpu=subprocess.check_output(['nvidia-smi'],text=True);take_lease('29','capability-structure-readback');configure()
    model,optimizer,corpus,prior,parent,saved=prepare(source['diagnostic'],source['readback'])
    need(fingerprint(model.state_dict())==plan['initial_parent'] and fingerprint(optimizer.state_dict())==plan['initial_optimizer'],'STRUCTURE_READER_PARENT')
    extension=attach_context(model,optimizer,corpus.config['seed']);initial=torch.load(a.root/'adapter-initial.pt',weights_only=True)
    need(fingerprint(extension.state_dict())==fingerprint(initial),'STRUCTURE_READER_INITIAL');protected=frozen(model,optimizer)
    descriptors=[json.loads(x) for x in (a.root/'descriptors.jsonl').read_text().splitlines()]
    features={x['task']:x['features'] for x in map(json.loads,(a.root/'features.jsonl').read_text().splitlines())}
    need([d['task'] for d in descriptors]==[*corpus.train,*corpus.query] and set(features)=={d['task'] for d in descriptors},'STRUCTURE_READER_ROSTER')
    for descriptor in descriptors:reconstruct(corpus,descriptor,features[descriptor['task']])
    contexts=StructuredContexts(corpus);need(contexts.identity==plan['catalogue'] and contexts.census()==plan['census'],'STRUCTURE_READER_CATALOGUE')
    tensors=torch.load(a.root/'outputs.pt',weights_only=True);rows=[json.loads(x) for x in (a.root/'tasks.jsonl').read_text().splitlines()]
    need([r['task'] for r in rows]==plan['selected'],'STRUCTURE_READER_SELECTION');cases=[];controls=[]
    with torch.no_grad():
        for row in rows:
            key=row['task'];task=corpus.tasks[key];item=contexts.get(key,model['world'].config,'cuda')
            raw,_=corpus.target(key,model['world'].config,'cuda',fitting=key in corpus.train)
            for mode in ('parent','initial','boundaries','structured','mismatched'):
                extension.load_state_dict(initial,strict=True)
                if mode=='parent':output=predict(model['world'],model['request'],item.native,key=key,operation=task['kind'])
                else:
                    if mode!='initial':
                        for layer in extension.out.values():layer.weight.copy_(torch.eye(layer.out_features,device='cuda'))
                    actual=predict_structured(model,item,task['kind'],mode='structured' if mode=='initial' else mode)
                    output,residual=independent_context(model,item,task['kind'],'structured' if mode=='initial' else mode)
                    need(torch.equal(actual['structural_context'],residual),'STRUCTURE_READER_ATTENTION')
                    if mode!='initial':need(fingerprint(residual)==row['residual_hashes'][mode],'STRUCTURE_READER_RESIDUAL')
                need(fingerprint(snapshot(output))==row['output_hashes'][mode]==fingerprint(tensors[key][mode]),'STRUCTURE_READER_PREDICTION')
                need(independent_score(model['decoder'],raw,output['predicted_constellation'].slots,skipped=False,
                    size=row['scores'][mode]['physical_batch'])==row['scores'][mode]['loss'],'STRUCTURE_READER_SURFACE')
                cases.append(dict(task=key,mode=mode,context_prediction_surface_exact=True))
        extension.load_state_dict(initial,strict=True)
    descriptor=next(d for d in descriptors if d['operation']=='continue_recorded_steps');key=descriptor['task']
    for mode,code in (('role','SOURCE_FIELD'),('clock','SOURCE_FIELD'),('target','CONTEXT'),('native','NATIVE'),('features','FEATURES')):
        damaged=copy.deepcopy(descriptor);bad_features=copy.deepcopy(features[key])
        if mode=='role':damaged['fields'][0]['role']='source_answer'
        elif mode=='clock':damaged['fields'][0]['clock']={'start':'0','end':'1','id':'invented'}
        elif mode=='target':damaged['fields'][0]['id']=corpus.tasks[key]['selections']['targets']['fields'][0]['id']
        elif mode=='native':next(iter(damaged['fields'][0]['readings'].values()))['windows']+=1
        else:bad_features['field'][0][0]+=1
        controls.append(refused('STRUCTURE_READER_'+code,lambda:reconstruct(corpus,damaged,bad_features)))
    need(frozen(model,optimizer)==protected==report['protected_parent'],'STRUCTURE_READER_DRIFT');model['world'].check_reference()
    result=dict(schema='capability-structure-readback-v1',source=ref,source_tasks=len(descriptors),source_fields=sum(len(d['fields']) for d in descriptors),
        cases=cases,controls=controls,protected_parent_exact=True,gpu_before=gpu,model_updates=0,
        scientific_admission=False,learned_capability=False,serving=False,number_status='measured',acceptance='author-checked')
    print(json.dumps(dict(pin=write_committed(a.output,result))),flush=True)


if __name__=='__main__':run_cli(main)
