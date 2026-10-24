"""Reuse pinned whole-field Multi readings for a separate execution-outcome head."""
import json
from pathlib import Path
import zipfile

import torch

from audit import complete_supervision_source as original
from claims.episodic_memory import digest
from model.capability_context_input import ContextStore,ContextNative
from model.complete_answer_outcome import ROLES,PRIOR,OutcomeInputs,routes
from model.native_population_training import fingerprint
from text.capability_answer_source import stamp
from text.constellation_corpus import file_sha,need,digest as bytes_sha

ROOT=original.ROOT
REF=('complete-answer-supervision-v3-20261024','1a2b2e23c0c818db7f8c20c8e330d4da1fdf18216e18804b13498ce5ebb32041')
COLD=('complete-answer-supervision-cold-v3-20261024','a6018d7ccde0f24672ba1182d91c3f744291ddedaa69d85c3b7dfcc43bf4a83c')
NEW=('model/complete_answer_outcome.py','audit/complete_outcome_source.py',
     'audit/complete_outcome_learning.py','audit/complete_outcome_readback.py')


def load(root):
    need(all(file_sha(root/n/'report.json')==h for n,h in (REF,COLD)), 'OUTCOME_SOURCE_PIN')
    first=json.loads((root/REF[0]/'report.json').read_bytes()); second=json.loads((root/COLD[0]/'report.json').read_bytes())
    need(second['source']==dict(path=str(root/REF[0]),pin=REF[1]) and second['all_purpose_and_scope_checks_exact'],
         'OUTCOME_SUPERVISION_READBACK')
    for n,h in first['files'].items():need(file_sha(root/REF[0]/n)==h,'OUTCOME_SUPERVISION_PAYLOAD')
    s=original.load(root); supervisor=s['model']; source=supervisor.view.source; plan=routes(supervisor)
    evidence=json.loads((root/REF[0]/'evidence.json').read_bytes())
    need(all(supervisor.evidence(k)==v for k,v in evidence.items()), 'OUTCOME_ORIGINAL_EVIDENCE')
    coding=[k for k in source.train+source.query if evidence[k]['family']=='coding']
    pins={evidence[k]['original_evidence']['population'] for k in coding}; need(len(pins)==1,'OUTCOME_POPULATION')
    pin=next(iter(pins)); spec=source.specs[pin]
    available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
    plan_root=Path(spec['plan']);need(file_sha(plan_root/'report.json')==spec['plan_pin'],'OUTCOME_PLAN_PIN')
    native_plan=json.loads((plan_root/'report.json').read_bytes())
    need(file_sha(plan_root/'mapping.json')==native_plan['mapping_sha256'],'OUTCOME_MAPPING_PIN')
    mapping=json.loads((plan_root/'mapping.json').read_bytes());uncompressed=0
    for bank in sorted({spec['native'],*mapping['banks']}):
        manifest=json.loads((Path(bank)/'report.json').read_bytes())
        for name in manifest['lenses']:
            with zipfile.ZipFile(Path(bank)/name/'slots.npz') as z:uncompressed+=sum(x.file_size for x in z.infolist())
    need(4*(uncompressed+42018816)<available,'OUTCOME_PHYSICAL_MEMORY')
    store=ContextStore(**{k:spec[k] for k in ('plan','plan_pin','native','native_pin')})
    native=ContextNative(spec['population'],pin,store=store)
    need(all(native.pin==source.tasks[k]['binding'] for k in coding), 'OUTCOME_NATIVE_BINDING')
    roster=tuple(v.name for v in native.specs); rows={}; fields={}; field_records={}; excluded=[]
    for key in coding:
        e=evidence[key]; pop,task,split=source.selected(key); context=[]
        for f in task['context']:
            role=pop.fields[f]['role'];need(role in (*ROLES[:-1],*PRIOR),'OUTCOME_UNKNOWN_ROLE')
            if role in PRIOR:excluded.append(dict(task=key,field=f,role=role));continue
            context.append(dict(field=f,role=role))
        candidate=[dict(field=e['source_field'],role='candidate')]
        need(pop.fields[e['source_field']]['text_sha256']==e['answer_sha256'], 'OUTCOME_WHOLE_PATCH')
        rows[key]=dict(task=key,split=split,group=e['group'],context=context,candidate=candidate,
            outcome=plan[key]['outcome'],scope=plan[key]['scope'],evidence=plan[key]['evidence'],
            generation=plan[key]['generation'],answer_sha256=e['answer_sha256'])
        for f in [r['field'] for r in context+candidate]:
            if f in fields:continue
            slots={}; records={}
            for instrument in native.specs:
                values,desc=native.field(f,instrument.name)
                receipt=source.snapshot['capability_memory']['fields'][pin+':'+f]['slots'][instrument.name]
                need(list(values.shape)==receipt['shape'] and instrument.identity==receipt['instrument']
                     and bytes_sha(values.astype('<f4',copy=False).tobytes())==receipt['sha256']
                     and desc==receipt['description'], 'OUTCOME_NATIVE_RECEIPT')
                slots[instrument.name]=torch.from_numpy(values.copy()); records[instrument.name]=receipt
            fields[f]=slots;field_records[f]=dict(text_sha256=pop.fields[f]['text_sha256'],slots=records)
    need(len(coding)==47 and sum(r['split']=='train' for r in rows.values())==31 and len(fields)==301,'OUTCOME_CENSUS')
    byte_count=sum(x.numel()*x.element_size() for slots in fields.values() for x in slots.values())
    need(4*byte_count<available,'OUTCOME_PHYSICAL_MEMORY')
    inputs=dict(s['inputs'])
    paths={root/REF[0]/n for n in (*first['files'],'report.json')};paths.add(root/COLD[0]/'report.json')
    paths.update(map(Path,store.manifest_files))
    paths.update(Path(r)/n for r,files in store.files.items() for n in files)
    # Retain every explicit historical lookup dependency, including banks whose text
    # was inspected for reuse but whose vectors were not selected by this population.
    config_path=Path(mapping['config']['path']);paths.add(config_path)
    config=json.loads(config_path.read_bytes());old_path=Path(config['old_config']['path']);paths.add(old_path)
    old=json.loads(old_path.read_bytes())
    for ref in old['populations']:
        pop_root=Path(ref['population']);r=json.loads((pop_root/'report.json').read_bytes())
        paths.update(pop_root/n for n in (*r['files'],'report.json','COMMITTED'))
        if r.get('family')=='coding':continue
        retained=Path(ref['plan']);r=json.loads((retained/'report.json').read_bytes());paths.add(retained/'report.json')
        for family in r['families']:paths.update((retained/family/'summary.json',retained/family/'chunks/atoms.parquet'))
    for ref in config['deltas']:
        root0=Path(ref['plan']);paths.update(root0/n for n in ('report.json','delta/summary.json','delta/chunks/atoms.parquet'))
        paths.add(Path(ref['native'])/'report.json')
    for p in paths:inputs[str(p)]=dict(sha256=file_sha(p),bytes=p.stat().st_size)
    base=Path(__file__).resolve().parents[1]; code=dict(s['code'])
    for n in (*NEW,'model/capability_context_input.py','model/capability_delta_input.py','model/proposal_input.py',
              'model/capability_experience_learning.py','model/native_population_training.py'):
        code[n]=file_sha(base/n)
    stamps={p:stamp(p) for p in inputs}
    def fresh():
        s['fresh']()
        need(all(stamp(p)==v for p,v in stamps.items()),'OUTCOME_SOURCE_CHANGED')
        for p,v in inputs.items():need(file_sha(p)==v['sha256'],'OUTCOME_SOURCE_CHANGED')
        for n,h in code.items():need(file_sha(base/n)==h,'OUTCOME_CODE_CHANGED')
    value=dict(rows=rows,fields=fields,roster=roster,dimensions={v.name:v.dims for v in native.specs},roots={k:str(p) for k,p in store.roots.items()},
        instruments={v.name:v.identity for v in native.specs},field_records=field_records,route=plan,
        native_pin=native.pin,excluded=excluded,source_representation='whole_field_native_Multi_not_answer_atom_readings')
    return dict(value=value,inputs=inputs,code=code,fresh=fresh,supervisor=supervisor,bytes=byte_count,
                free_memory=available,native_banks=len(store.banks),uncompressed_bank_bytes=uncompressed)
