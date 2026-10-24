"""Source, parse and result boundaries on owned shadows; originals remain pinned."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.connected_census_readback import readback,digest,check
from text.connected_training_census import load_sources,multirc_census,qasc_census
from senses.trajectory_population import publish


def main():
    p=argparse.ArgumentParser()
    for k in ('multirc','qasc','parent','census','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();phases={};tick=time.perf_counter();records=[]
    roots=dict(multirc=a.multirc,qasc=a.qasc,parent=a.parent,census=a.census)
    original={k:{v.name:digest(v) for v in root.iterdir() if v.is_file()} for k,root in roots.items()}
    baseline=readback(a.census,a.multirc,a.qasc,a.parent);publish(a.output/'cold-readback.json',baseline)
    shadow=a.output/'shadow';shadow.mkdir()
    for k,root in roots.items():shutil.copytree(root,shadow/k)
    def read():return readback(shadow/'census',shadow/'multirc',shadow/'qasc',shadow/'parent')
    check(read()==baseline,'SHADOW_POSITIVE')
    phases['positive_cold_read_and_shadow']=time.perf_counter()-tick;tick=time.perf_counter()
    def mutate(name,path,transform,action,code,rebind=False):
        check(path.resolve().is_relative_to(shadow.resolve()),'SHADOW_BOUNDARY');before=path.read_bytes();old=digest(path)
        report=shadow/'census/report.json';saved=report.read_bytes();changed=transform(before);check(changed!=before,'MUTATION_NOOP')
        path.write_bytes(changed)
        if rebind:
            r=json.loads(saved);r['files'][path.name]=dict(bytes=path.stat().st_size,sha256=digest(path))
            report.write_text(json.dumps(r),encoding='utf-8')
        after=digest(path);error=None
        try:action()
        except ValueError as exc:error=str(exc)
        finally:
            path.write_bytes(before)
            if rebind:report.write_bytes(saved)
        check(error==code,'NEGATIVE_'+name);check(digest(path)==old,'RESTORE_'+name)
        records.append(dict(boundary=name,before_sha256=old,action_sha256=after,error=error,after_sha256=digest(path),restored=True))
    for name,path in [('source_bytes',shadow/'multirc/train.json'),('licence_bytes',shadow/'multirc/LICENSE'),('archive_bytes',shadow/'parent/mutlirc-v2.zip')]:
        mutate(name,path,lambda b:b[:-1],read,'MNLM_CONNECTED_READ_SOURCE_BYTES')
    phases['source_boundary_controls']=time.perf_counter()-tick;tick=time.perf_counter()
    rawm,rawq=load_sources(a.multirc,a.qasc,a.parent)
    for name,obj,parser,change,code in [
        ('evidence_range',rawm,multirc_census,lambda x:x['data'][0]['paragraph']['questions'][0].update(sentences_used=[9999]),'EVIDENCE_INDEX'),
        ('duplicate_document',rawm,multirc_census,lambda x:x['data'][1].update(id=x['data'][0]['id']),'DOCUMENT_ID'),
        ('integer_answer_truth',rawm,multirc_census,lambda x:x['data'][0]['paragraph']['questions'][0]['answers'][0].update(isAnswer=1),'ANSWER_SCHEMA'),
        ('lost_source_text',rawm,multirc_census,lambda x:x['data'][0]['paragraph'].update(text=x['data'][0]['paragraph']['text']+'UNMAPPED'),'SENTENCE_COVERAGE'),
        ('invalid_answer_key',rawq,qasc_census,lambda x:x[0].update(answerKey='Z'),'ANSWER_KEY'),
        ('duplicate_choice',rawq,qasc_census,lambda x:x[0]['choices']['label'].__setitem__(1,'A'),'CHOICES')]:
        path=shadow/(name+'.json');publish(path,obj)
        def transform(b,change=change):
            x=json.loads(b);change(x);return json.dumps(x).encode('utf-8')
        mutate(name,path,transform,lambda path=path,parser=parser:parser(json.loads(path.read_bytes())),'MNLM_CONNECTED_CENSUS_'+code)
    empty=copy.deepcopy(rawm);empty['data'][0]['paragraph']['questions'][0]['answers'][0]['text']=''
    docs,summary=multirc_census(empty)
    check(docs[0]['questions'][0]['quarantine_reasons']==['EMPTY_ANSWER'] and summary['quarantined_questions']==4,'EMPTY_QUARANTINE')
    records.append(dict(boundary='empty_answer_quarantined',passed=True,quarantined_questions=4))
    phases['parse_boundary_controls']=time.perf_counter()-tick;tick=time.perf_counter()
    def modify(fn):
        def transform(b):
            x=json.loads(b);fn(x);return json.dumps(x).encode('utf-8')
        return transform
    blankdoc=next(i for i,d in enumerate(json.loads((shadow/'census/multirc-documents.json').read_bytes())) if any(q['quarantine_reasons'] for q in d['questions']))
    def unquarantine(x):
        next(q for q in x[blankdoc]['questions'] if q['quarantine_reasons'])['quarantine_reasons']=[]
    changes=[
        ('atom_offset','multirc-documents.json',lambda x:x[0]['atoms'][0]['codepoint_span'].__setitem__(0,1),'ATOM_ADDRESSES'),
        ('display_reference','multirc-documents.json',lambda x:x[0]['questions'][0]['display_evidence_ids'].__setitem__(0,0),'QUESTION_BINDING'),
        ('quarantine_removed','multirc-documents.json',unquarantine,'QUESTION_BINDING'),
        ('component_false','qasc-cases.json',lambda x:x[0].update(shared_fact_component='unrelated'),'SHARED_FACT_COMPONENT'),
        ('target_false','qasc-cases.json',lambda x:x[0]['private_target'].update(answerKey='A'),'QASC_BINDING'),
        ('count_false','census.json',lambda x:x['multirc'].update(sentences=1),'MULTIRC_COUNTS_OR_SCOPE'),
        ('admission_false','report.json',lambda x:x.update(scientific_admission=True),'CLAIM_BOUNDARY'),
        ('heldout_false','report.json',lambda x:x.update(heldout_labels_extracted=True),'CLAIM_BOUNDARY'),
        ('source_binding_false','report.json',lambda x:x['source_pins'].update(qasc='0'*64),'RESULT_SOURCE_BINDING')]
    for name,file,fn,code in changes:mutate(name,shadow/'census'/file,modify(fn),read,'MNLM_CONNECTED_READ_'+code,rebind=file!='report.json')
    phases['persisted_result_boundary_controls']=time.perf_counter()-tick;tick=time.perf_counter()
    check(read()==baseline,'SHADOW_FINAL')
    final={k:{v.name:digest(v) for v in root.iterdir() if v.is_file()} for k,root in roots.items()};check(final==original,'ORIGINALS_UNCHANGED')
    check(readback(a.census,a.multirc,a.qasc,a.parent)==baseline,'ORIGINAL_FINAL')
    code={n:digest(Path(__file__).resolve().parents[1]/n) for n in ('text/connected_training_census.py','audit/connected_census_readback.py','audit/fsv_connected_census.py')}
    phases['final_original_and_shadow_readback']=time.perf_counter()-tick;wall=time.perf_counter()-start
    report=dict(schema='connected-census-fsv-v1',passed=True,checks=records,before=original,after=final,code_sha256=code,
                cold_readback=baseline,phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,
                measurements='author-checked',scientific_admission=False,training_ready=False)
    publish(a.output/'report.json',report);print(json.dumps(dict(passed=True,checks=len(records),report_sha256=digest(a.output/'report.json'),seconds=wall)))


if __name__=='__main__':main()
