"""Real persisted evidence-scope readback and restored corrupt-copy controls."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.multirc_evidence_readback import read,need
from text.multirc_evidence_scope import load,parents
from text.evaluation_multiscale import file_sha
from senses.trajectory_population import publish

def main(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();phases={}
    data,pins,cold=load(a.population,a.quality,a.raw)
    before={p.name:file_sha(p) for p in a.result.iterdir() if p.is_file()}
    publish(a.output/'cold.json',read(a.result,data,pins));publish(a.output/'parent-readback.json',cold)
    phases['cold_sources_and_independent_reconstruction']=time.perf_counter()-start;tick=time.perf_counter()
    shadow=a.output/'owned-shadow';shutil.copytree(a.result,shadow)
    original={p.name:p.read_bytes() for p in shadow.iterdir()};controls=[]
    def control(name,file,mutate,code):
        changed=json.loads(original[file]);mutate(changed)
        (shadow/file).write_text(json.dumps(changed,sort_keys=True,separators=(',',':')),encoding='utf-8')
        if file!='report.json':
            header=json.loads(original['report.json']);header['files'][file]=file_sha(shadow/file)
            (shadow/'report.json').write_text(json.dumps(header,sort_keys=True),encoding='utf-8')
        try:read(shadow,data,pins)
        except ValueError as exc:
            need(str(exc)=='MNLM_EVIDENCE_READBACK_'+code,'CONTROL_WRONG_REFUSAL')
            controls.append(dict(name=name,expected=code,observed=str(exc),refused=True))
        else:raise ValueError('MNLM_EVIDENCE_READBACK_CONTROL_ACCEPTED')
        finally:
            for n,b in original.items():(shadow/n).write_bytes(b)
    control('annotation shifted by one','private-plans.json',lambda v:v[0]['source_evidence_indices'].__setitem__(0,v[0]['source_evidence_indices'][0]+1),'EVIDENCE_ADDRESS')
    control('annotation reference borrowed','private-plans.json',lambda v:v[0]['evidence_references'].__setitem__(0,v[1]['case']),'EVIDENCE_ADDRESS')
    def retain_containing_view(v):
        rid=v[0]['withheld_references'].pop();v[0]['retained_references'].append(rid)
    control('paragraph evidence leaked','private-plans.json',retain_containing_view,'CONTAINMENT_CLOSURE')
    def omit_pair(v):
        refs=data['references'];row=next(r for r in v if any(refs[x]['role']=='sentence_pair' for x in r['withheld_references']))
        rid=next(x for x in row['withheld_references'] if refs[x]['role']=='sentence_pair');row['withheld_references'].remove(rid)
    control('paired view omitted','private-plans.json',omit_pair,'CONTAINMENT_CLOSURE')
    control('case omitted','private-plans.json',lambda v:v.pop(),'COVERAGE')
    control('empty-context flag invented','private-plans.json',lambda v:v[0].__setitem__('empty_remaining_context',not v[0]['empty_remaining_context']),'EMPTY_CONTEXT')
    control('native alias invented','private-plans.json',lambda v:v[0]['shared_native_units'].append(-1),'NATIVE_ALIASES')
    control('denominator inflated','summary.json',lambda v:v['all'].__setitem__('questions',5132),'SUMMARY')
    control('contrast count invented','contrasts.json',lambda v:v['exact']['clean/evaluation'].__setitem__('opposite_label_cross_document_pairs',v['exact']['clean/evaluation']['opposite_label_cross_document_pairs']+1),'CONTRAST_CENSUS')
    control('world grounding asserted','report.json',lambda v:v['claims'].__setitem__('world_truth_assessed',True),'CLAIMS')
    control('new altered truth asserted','report.json',lambda v:v['claims'].__setitem__('altered_input_truth_assessed',True),'CLAIMS')
    control('private selection made model input','report.json',lambda v:v['claims'].__setitem__('model_input',True),'CLAIMS')
    phases['owned_copy_controls_and_restoration']=time.perf_counter()-tick;tick=time.perf_counter()
    restored=read(shadow,data,pins);need(before=={p.name:file_sha(p) for p in a.result.iterdir() if p.is_file()},'ORIGINAL_DRIFT')
    need(before=={p.name:file_sha(p) for p in shadow.iterdir() if p.is_file()},'SHADOW_DRIFT')
    need(pins==parents(a.population,a.quality,a.raw),'PARENT_DRIFT');publish(a.output/'restored.json',restored)
    phases['restored_readback_and_final_hashes']=time.perf_counter()-tick;wall=time.perf_counter()-start
    report=dict(source_report_sha256=before['report.json'],controls=controls,originals_unchanged=True,shadow_restored=True,
        parents_unchanged=True,code_sha256=file_sha(Path(__file__)),acceptance='author-checked',served=False,
        phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(a.output/'report.json',report)
    print(json.dumps(dict(report_sha256=file_sha(a.output/'report.json'),controls=len(controls),wall_seconds=wall)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('population','quality','raw','result','output'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
