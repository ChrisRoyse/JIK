"""Experience, outcome and exact-native support census before increasing training."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
from audit.capability_objective_audit import publish_rows
from audit.capability_stock import inventory
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from model.capability_experience import ExperienceCorpus
from model.capability_kernel_learning import pinned
from model.capability_supervision import CapabilitySupervision, PURPOSES
from model.contracts import ModelConfig


def field_receipt(corpus, pin, key):
    f = corpus.sources[pin].population.fields[key]
    lanes = {}
    for spec in corpus.specs:
        a = corpus.field(pin, key, spec.name)
        lanes[spec.name] = dict(shape=list(a.shape), sha256=sha(a.astype('<f4', copy=False).tobytes()))
    return dict(population=pin, field=key, role=f['role'], clock=f['clock'],
                text_sha256=f['text_sha256'], bytes=len(f['text'].encode()), lanes=lanes,
                native_identity=digest(lanes))


def cells(rows):
    result = []
    for split in ('kernel_train', 'all_measured_train', 'query'):
        selected = [r for r in rows if r['split'] == ('query' if split == 'query' else 'train')
                    and (split != 'kernel_train' or r['kernel_train'])]
        for kind in sorted({r['operation'] for r in selected}):
            group = [r for r in selected if r['operation'] == kind]
            aliases = defaultdict(list)
            for r in group: aliases[r['context_identity']].append(r)
            result.append(dict(split=split, operation=kind, tasks=len(group),
                groups=len({r['group'] for r in group}), outcomes=dict(Counter(str(r['outcome']) for r in group)),
                target_surfaces=len({r['target_sha256'] for r in group}),
                target_constellations=len({r['target_native'] for r in group}),
                distinct_contexts=len(aliases), conflicting_contexts=sum(
                    len({r['target_sha256'] for r in bucket}) > 1 for bucket in aliases.values()),
                purposes={p:dict(tasks=sum(r['decisions'][p]['allowed'] for r in group),
                    groups=len({r['group'] for r in group if r['decisions'][p]['allowed']})) for p in PURPOSES}))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    need(not a.output.exists(),'EXPERIENCE_OUTPUT_EXISTS'); a.output.mkdir()
    start=tick=time.perf_counter(); phases={}; source=json.loads(a.config.read_bytes())
    source['config_sha256']=sha(a.config.read_bytes())
    source['code']={n:sha((Path(__file__).resolve().parents[1]/n).read_bytes()) for n in (
        'model/capability_experience.py','audit/capability_stock.py','audit/capability_experience_coverage.py')}
    config=pinned(source['learning']); corpus=ExperienceCorpus(config,training_scope='all_measured')
    supervision=CapabilitySupervision(corpus); cache={}; rows=[]
    kernel_train=set(corpus.kernel_train)
    before={name:ref['pin'] for name,ref in config.items() if isinstance(ref,dict) and 'pin' in ref}
    write_committed(a.output/'before.json',before)
    phases['authenticate']=time.perf_counter()-tick; tick=time.perf_counter()
    for key in (*corpus.train,*corpus.query):
        t=corpus.tasks[key]; pin=t['population']; context=[]
        for side in ('context','targets'):
            for f in t['selections'][side]['fields']:
                address=(pin,f['id'])
                if address not in cache: cache[address]=field_receipt(corpus,*address)
                if side=='context':
                    field=cache[address]
                    context.append({n:field[n] for n in ('role','clock','native_identity')})
        target=cache[(pin,t['selections']['targets']['fields'][0]['id'])]
        evidence=supervision.evidence(key)
        rows.append(dict(**evidence,kernel_train=key in kernel_train,
            context_identity=digest(dict(operation=t['kind'],ordered_fields=context)),
            target_native=target['native_identity'],
            decisions={purpose:supervision.decide(key,purpose) for purpose in PURPOSES}))
    publish_rows(a.output/'experience.jsonl',rows);publish_rows(a.output/'fields.jsonl',cache.values())
    bank=[dict(population=p,field=f,native=cache[(p,f)]['native_identity'],
               surface=cache[(p,f)]['text_sha256']) for p,f in corpus.regions]
    groups=defaultdict(set);support=Counter()
    for r in rows:
        if r['split']=='train': support[r['target_native']]+=1;groups[r['target_native']].add(r['group'])
    for entry in bank:entry.update(exact_native_train_tasks=support[entry['native']],
                                   independent_train_groups=len(groups[entry['native']]))
    write_committed(a.output/'bank.json',bank)
    schedules=[corpus.schedule(1,4),corpus.schedule(1,2),corpus.schedule(1,1)]
    write_committed(a.output/'schedules.json',schedules)
    boundaries=[]; arrays={}
    for purpose in PURPOSES[:-1]:
        keys=[r['task'] for r in rows if r['split']=='train' and r['decisions'][purpose]['allowed']]
        key=next((k for k in keys if k not in corpus.kernel_train),keys[0])
        decision,raw,slots=supervision.training_target(key,purpose,ModelConfig(),'cpu')
        context=corpus.batch(key,'context',ModelConfig(),'cpu');names={}
        for side,values in [('context',context),('target',slots)]:
            for name,slot in values.items():
                label=str(len(boundaries))+'__'+side+'__'+name
                arrays[label]=slot.values[0,~slot.padding[0]].numpy();names[label]=name
        boundaries.append(dict(task=key,purpose=purpose,decision=decision,surface_sha256=sha(raw),arrays=names))
    np.savez(a.output/'consumer.npz',**arrays);write_committed(a.output/'consumer.json',boundaries)
    phases['measured_experience']=time.perf_counter()-tick;tick=time.perf_counter()
    units,meetings,stock=inventory(source['stock'])
    publish_rows(a.output/'raw-units.jsonl',units);write_committed(a.output/'ami-metadata.json',meetings)
    phases['raw_inventory']=time.perf_counter()-tick;tick=time.perf_counter()
    for name,pin in before.items():need(sha(Path(config[name]['path']).read_bytes())==pin,'EXPERIENCE_SOURCE_DRIFT')
    files={n:sha((a.output/n).read_bytes()) for n in ('before.json','experience.jsonl','fields.jsonl','bank.json',
        'schedules.json','consumer.json','consumer.npz','raw-units.jsonl','ami-metadata.json')}
    summary=cells(rows)
    report=dict(schema='capability-experience-coverage-v1',source=source,files=files,summary=summary,stock=stock,
        kernel_train=len(corpus.kernel_train),measured_train=len(corpus.train),query=len(corpus.query),
        kernel_regions=len(bank),unique_bank_native_constellations=len({e['native'] for e in bank}),
        bank_group_support=dict(Counter(e['independent_train_groups'] for e in bank)),
        eligible_groups=len({corpus.tasks[k]['split_group'] for k in corpus.train}),
        excluded_split_train=sorted(set(corpus.kernel['split']['train'])-set(corpus.train)),
        unknown_raw_success_coverage=True,training_identity=corpus.training_identity,
        sources_unchanged=True,model_updates=0,scientific_admission=False,serving=False,
        learned_capability=False,number_status='measured',acceptance='author-checked')
    phases['finalize']=time.perf_counter()-tick;wall=time.perf_counter()-start
    need(sum(phases.values())/wall>=.9,'EXPERIENCE_PHASE_COVERAGE');report.update(phases=phases,seconds=wall)
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),summary=summary,stock=stock)),flush=True)


if __name__=='__main__':main()
