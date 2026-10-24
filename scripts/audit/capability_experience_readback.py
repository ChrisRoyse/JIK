"""Cold source, population, native-target and schedule readback for experience growth."""
import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
from audit.arithmetic_readback import receipt as arithmetic_receipt
from audit.capability_conditioning_readback import refused
from claims.capability_population import need,sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from model.capability_experience import ExperienceCorpus
from model.capability_kernel_learning import pinned
from model.capability_supervision import CapabilitySupervision,PURPOSES
from model.contracts import ModelConfig


def purpose(e,p):
    if p=='recorded_continuation':return True
    if p=='successful_patch':return e['scope']=='retained_execution_test_scope' and e['outcome']==1
    if p=='source_answer':return e['scope']=='question_author_answer_only'
    if p=='annotated_arithmetic':
        return e['scope']=='annotated_equalities_only' and bool(e['checks']) and all(c['status']=='pass' for c in e['checks'])
    return False


def aggregates(rows,summary):
    expected={(split,r['operation']) for r in rows for split in (
        ('query',) if r['split']=='query' else (('all_measured_train','kernel_train') if r['kernel_train'] else ('all_measured_train',)))}
    need({(c['split'],c['operation']) for c in summary}==expected and len(summary)==len(expected),'EXPERIENCE_READER_CELLS')
    for c in summary:
        rs=[r for r in rows if r['operation']==c['operation'] and
            (r['split']=='query' if c['split']=='query' else r['split']=='train') and
            (c['split']!='kernel_train' or r['kernel_train'])]
        aliases=defaultdict(set)
        for r in rs:aliases[r['context_identity']].add(r['target_sha256'])
        expected=dict(split=c['split'],operation=c['operation'],tasks=len(rs),groups=len({r['group'] for r in rs}),
            outcomes=dict(Counter(str(r['outcome']) for r in rs)),target_surfaces=len({r['target_sha256'] for r in rs}),
            target_constellations=len({r['target_native'] for r in rs}),distinct_contexts=len(aliases),
            conflicting_contexts=sum(len(v)>1 for v in aliases.values()),
            purposes={p:dict(tasks=sum(purpose(r,p) for r in rs),groups=len({r['group'] for r in rs if purpose(r,p)})) for p in PURPOSES})
        need(c==expected,'EXPERIENCE_READER_SUMMARY')


def consumer(corpus,case,arrays):
    t=corpus.tasks[case['task']];seen=set()
    for label,name in case['arrays'].items():
        side=label.split('__')[1];selection=t['selections']['context' if side=='context' else 'targets']
        expected=np.concatenate([corpus.field(t['population'],f['id'],name) for f in selection['fields']])
        need(np.array_equal(arrays[label],expected),'EXPERIENCE_READER_NATIVE')
        seen.add((side,name))
    need(seen=={(side,s.name) for side in ('context','target') for s in corpus.specs},'EXPERIENCE_READER_NATIVE_ROSTER')


def raw_stock(source,root,expected):
    units=[json.loads(line) for line in (root/'raw-units.jsonl').read_bytes().splitlines()]
    checked=0;arithmetic=0;stock={}
    for kind,filename in [('gsm8k','train.jsonl'),('condaqa','train.json')]:
        ref=source[kind];capture=pinned(ref);parent=Path(ref['path']).parent
        for name,pin in capture['files'].items():
            data=(parent/name).read_bytes()
            need(len(data)==pin['bytes'] and sha(data)==pin['sha256'],'EXPERIENCE_READER_RAW_PIN')
        data=(parent/filename).read_bytes();original=list(map(json.loads,data.splitlines()))
        rs=[r for r in units if r['kind']==kind]
        need(len(rs)==len(original),'EXPERIENCE_READER_RAW_COUNT')
        for index,(row,r) in enumerate(zip(original,rs)):
            context=row['question'] if kind=='gsm8k' else [row['sentence1'],row['sentence2']]
            target=row['answer'] if kind=='gsm8k' else row['label']
            group='gsm8k-question:'+sha(context.encode()) if kind=='gsm8k' else 'condaqa:'+capture['files'][filename]['sha256']+':passage:'+str(row['PassageID'])
            need(r['row']==index and r['group']==group and r['context']==digest(context)
                 and r['target']==sha(target.encode()) and r['artifact']==capture['files'][filename]['sha256'],
                 'EXPERIENCE_READER_RAW_ADDRESS')
            if kind=='gsm8k':
                arithmetic_receipt(target,r['checks']);arithmetic+=len(r['checks'])
                need(r['steps']==len(target.splitlines()),'EXPERIENCE_READER_RAW_STEPS')
            else:need([r['passage'],r['question'],r['edit']]==[row['PassageID'],row['QuestionID'],row['PassageEditID']],
                      'EXPERIENCE_READER_RAW_VARIANT')
            checked+=1
        c=expected[kind]
        need(c['rows']==len(rs) and c['groups']==len({r['group'] for r in rs}) and c['bytes']==len(data)
             and c['context_aliases']==len(rs)-len({r['context'] for r in rs})
             and c['licence']==capture['licence'] and not c['training_ready'] and not c['measurement_ready'],
             'EXPERIENCE_READER_RAW_SUMMARY')
        if kind=='gsm8k':
            need(c['checks']==dict(Counter(x['status'] for r in rs for x in r['checks'])) and
                 c['all_equalities_pass']==sum(bool(r['checks']) and all(x['status']=='pass' for x in r['checks']) for r in rs),
                 'EXPERIENCE_READER_RAW_ARITHMETIC')
    need(checked==len(units),'EXPERIENCE_READER_RAW_ROSTER')
    ref=source['ami'];path=Path(ref['path'])
    need(path.stat().st_size==ref['bytes'] and sha(path.read_bytes())==ref['pin'],'EXPERIENCE_READER_AMI_PIN')
    meetings=json.loads((root/'ami-metadata.json').read_bytes())
    with zipfile.ZipFile(path) as z:
        entries=z.infolist();names={x.filename for x in entries};xml=z.read('corpusResources/meetings.xml')
        parsed={m.attrib['observation']:m for m in ET.fromstring(xml)}
        need(len(meetings)==len(parsed) and {m['meeting'] for m in meetings}==set(parsed),'EXPERIENCE_READER_AMI_MEETINGS')
        for row in meetings:
            m=parsed[row['meeting']]
            need(row['series']==re.sub('[a-d]$','',row['meeting']) and
                 row['participants']==sorted({s.attrib['global_name'] for s in m}) and
                 row['expected_agents']==sorted(s.attrib['nxt_agent'] for s in m),'EXPERIENCE_READER_AMI_METADATA')
            for kind,folder in [('act','dialogueActs'),('word','words')]:
                need(row[kind+'_members']==sorted(n for n in names if n.startswith(folder+'/'+row['meeting']+'.')
                     and n.endswith('.xml')),'EXPERIENCE_READER_AMI_MEMBERS')
        c=expected['ami']
        need(c['entries']==len(entries) and c['uncompressed_bytes']==sum(x.file_size for x in entries)
             and c['act_files']==sum(n.startswith('dialogueActs/') and n.endswith('.xml') for n in names)
             and c['word_files']==sum(n.startswith('words/') and n.endswith('.xml') for n in names)
             and c['meetings']==len(meetings) and c['series']==len({m['series'] for m in meetings})
             and c['metadata_sha256']==sha(xml) and c['licence_sha256']==sha(z.read('LICENCE.txt')),
             'EXPERIENCE_READER_AMI_SUMMARY')
    return dict(raw_units=checked,annotated_equalities=arithmetic,ami_meetings=len(meetings))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--pin',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    need(not a.output.exists(),'EXPERIENCE_READER_EXISTS');start=time.perf_counter()
    ref=dict(path=str(a.report),pin=a.pin);r=pinned(ref);root=a.report.parent
    for name,pin in r['files'].items():need(sha((root/name).read_bytes())==pin,'EXPERIENCE_READER_FILE')
    for name,pin in r['source']['code'].items():need(sha((Path(__file__).resolve().parents[1]/name).read_bytes())==pin,'EXPERIENCE_READER_CODE')
    config=pinned(r['source']['learning']);corpus=ExperienceCorpus(config,training_scope='all_measured')
    supervisor=CapabilitySupervision(corpus)
    rows=list(map(json.loads,(root/'experience.jsonl').read_bytes().splitlines()))
    need([x['task'] for x in rows]==[*sorted(corpus.kernel['ids']),*corpus.kernel['split']['queries']],
         'EXPERIENCE_READER_ROSTER')
    fields=list(map(json.loads,(root/'fields.jsonl').read_bytes().splitlines()))
    lookup={(f['population'],f['field']):f for f in fields};need(len(lookup)==len(fields),'EXPERIENCE_READER_FIELDS')
    visits=0
    for (pin,key),f in lookup.items():
        original=corpus.sources[pin].population.fields[key]
        need(all(f[n]==original[n] for n in ('role','clock','text_sha256')) and f['bytes']==len(original['text'].encode()),
             'EXPERIENCE_READER_FIELD_ADDRESS')
        need(set(f['lanes'])=={s.name for s in corpus.specs},'EXPERIENCE_READER_LANE_ROSTER')
        for name,slot in f['lanes'].items():
            values=corpus.field(pin,key,name)
            need(slot==dict(shape=list(values.shape),sha256=sha(values.astype('<f4',copy=False).tobytes())),
                 'EXPERIENCE_READER_FIELD_NATIVE');visits+=1
        need(f['native_identity']==digest(f['lanes']),'EXPERIENCE_READER_FIELD_IDENTITY')
    for row in rows:
        key=row['task'];t=corpus.tasks[key];e=supervisor.evidence(key)
        need(all(row[k]==v for k,v in e.items()) and row['kernel_train']==(key in corpus.kernel_train),'EXPERIENCE_READER_EVIDENCE')
        context=[{n:lookup[(t['population'],f['id'])][n] for n in ('role','clock','native_identity')}
                 for f in t['selections']['context']['fields']]
        need(row['context_identity']==digest(dict(operation=t['kind'],ordered_fields=context))
             and row['target_native']==lookup[(t['population'],e['target'])]['native_identity'],'EXPERIENCE_READER_CONTEXT')
        for p in PURPOSES:
            d=row['decisions'][p]
            need(d==supervisor.decide(key,p) and d['allowed']==purpose(e,p),'EXPERIENCE_READER_PURPOSE')
    aggregates(rows,r['summary'])
    bank=json.loads((root/'bank.json').read_bytes())
    need([(b['population'],b['field']) for b in bank]==list(corpus.regions),'EXPERIENCE_READER_BANK')
    for b in bank:
        f=lookup[(b['population'],b['field'])];support=[x for x in rows if x['split']=='train' and x['target_native']==f['native_identity']]
        need(b['native']==f['native_identity'] and b['surface']==f['text_sha256'] and b['exact_native_train_tasks']==len(support)
             and b['independent_train_groups']==len({x['group'] for x in support}),'EXPERIENCE_READER_BANK_SUPPORT')
    need(r['kernel_train']==len(corpus.kernel_train) and r['measured_train']==len(corpus.train) and r['query']==len(corpus.query)
         and r['unique_bank_native_constellations']==len({b['native'] for b in bank})
         and r['bank_group_support']=={str(k):v for k,v in Counter(b['independent_train_groups'] for b in bank).items()},
         'EXPERIENCE_READER_COUNTS')
    schedules=json.loads((root/'schedules.json').read_bytes());boundary=[]
    for plan in schedules:
        corpus.check_schedule(plan)
        for kind in sorted({corpus.tasks[k]['kind'] for k in plan['tasks']}):
            key=next(k for k in plan['tasks'] if corpus.tasks[k]['kind']==kind)
            raw,slots=corpus.target(key,ModelConfig(),'cpu',fitting=True)
            need(sha(raw)==supervisor.evidence(key)['target_sha256'],'EXPERIENCE_READER_SCHEDULE_TARGET')
            boundary.append(dict(fraction=plan['fraction'],operation=kind,task=key,lanes=len(slots)))
    cases=json.loads((root/'consumer.json').read_bytes())
    with np.load(root/'consumer.npz',allow_pickle=False) as arrays:
        for case in cases:consumer(corpus,case,arrays)
        altered={k:arrays[k].copy() for k in arrays.files};first=next(iter(cases[0]['arrays']));altered[first].flat[0]+=1
    controls=[refused('EXPERIENCE_READER_NATIVE',lambda:consumer(corpus,cases[0],altered)),
        refused('SUPERVISION_FIT_QUERY',lambda:supervisor.training_target(corpus.query[0],'recorded_continuation',ModelConfig(),'cpu'))]
    absent=next(k for k in corpus.kernel['split']['train'] if k not in corpus.measured_train_set)
    controls.append(refused('EXPERIENCE_NOT_ACTIVE_TRAIN',lambda:corpus.check_training_task(absent)))
    plan=copy.deepcopy(schedules[-1]);plan['tasks'].pop()
    controls.append(refused('EXPERIENCE_INCOMPLETE_GROUP',lambda:corpus.check_schedule(plan)))
    ep=corpus.tasks[corpus.train[0]]['episode'];state=corpus.snapshot['memory']['states'][ep];saved=state['active'];state['active']=False
    try:controls.append(refused('EXPERIENCE_INACTIVE_EVIDENCE',lambda:corpus.check_training_task(corpus.train[0])))
    finally:state['active']=saved
    failed=next(x['task'] for x in rows if x['split']=='train' and x['outcome']==0)
    controls.append(refused('SUPERVISION_FAILED_ACTION',lambda:supervisor.training_target(failed,'successful_patch',ModelConfig(),'cpu')))
    kernel=ExperienceCorpus(config,training_scope='kernel')
    need(tuple(kernel.train)==tuple(corpus.kernel_train) and kernel.regions==corpus.regions
         and kernel.query==corpus.query,'EXPERIENCE_READER_KERNEL_COMPATIBILITY')
    added=next(k for k in corpus.train if k not in kernel.train)
    controls.append(refused('LEARNING_TARGET_SPLIT',lambda:kernel.target(added,ModelConfig(),'cpu',fitting=True)))
    shared=kernel.train[0];left,ls=kernel.target(shared,ModelConfig(),'cpu',fitting=True)
    right,rs=corpus.target(shared,ModelConfig(),'cpu',fitting=True)
    need(left==right and all(np.array_equal(ls[n].values.numpy(),rs[n].values.numpy())
         and np.array_equal(ls[n].padding.numpy(),rs[n].padding.numpy()) for n in ls),'EXPERIENCE_READER_KERNEL_NATIVE')
    damaged=a.output.with_suffix('.corrupt-source.json');need(not damaged.exists(),'EXPERIENCE_DAMAGE_EXISTS')
    with damaged.open('xb') as f:f.write(a.report.read_bytes()+b'\n')
    controls.append(refused('LEARNING_SOURCE_PIN',lambda:pinned(dict(path=str(damaged),pin=a.pin))))
    stock=raw_stock(r['source']['stock'],root,r['stock'])
    execution_support=[]
    for split in ('kernel_train','all_measured_train','query'):
        selected=[x for x in rows if x['family']=='coding' and
                  (x['split']=='query' if split=='query' else x['split']=='train') and
                  (split!='kernel_train' or x['kernel_train'])]
        execution_support.append(dict(split=split,tasks=len(selected),
            successful_cases=sum(x['outcome']==1 for x in selected),
            final_execution_receipts=len({x['observations'][-1]['receipt'] for x in selected}),
            successful_execution_receipts=len({x['observations'][-1]['receipt'] for x in selected if x['outcome']==1}),
            successful_target_constellations=len({x['target_native'] for x in selected if x['outcome']==1}),
            scope='exact retained receipt identities; not a statistical independence claim'))
    for name,pin in json.loads((root/'before.json').read_bytes()).items():
        need(sha(Path(config[name]['path']).read_bytes())==pin,'EXPERIENCE_READER_UNCHANGED')
    result=dict(schema='capability-experience-readback-v1',source=ref,rows=len(rows),native_field_reads=visits,
        purpose_decisions=len(rows)*len(PURPOSES),consumer_cases=len(cases),schedule_boundaries=boundary,
        controls=controls,stock=stock,execution_support=execution_support,seconds=time.perf_counter()-start,
        kernel_scope_compatibility=dict(train=len(kernel.train),query=len(kernel.query),regions=len(kernel.regions),
                                       shared_native_exact=True,expanded_task_refused=True),
        source_sha256=sha(Path(__file__).read_bytes()),number_status='measured',acceptance='author-checked',
        scientific_admission=False,learned_capability=False,serving=False)
    print(json.dumps(dict(pin=write_committed(a.output,result),rows=len(rows),stock=stock)),flush=True)


if __name__=='__main__':main()
