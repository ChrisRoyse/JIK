"""Canonical native-input plan for complete source-bound answers and their context."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
from transformers import AutoTokenizer
from claims.process_conditions import write_committed
from model.capability_answer_targets import AnswerTargets
from model.legal_surface_inputs import instruments
from lens.learned import snapshot_dir, preprocessing_sha256
from text.constellation_corpus import canonical, digest, file_sha, need
from text.legal_native_batch import NativeBatchSession
from text.legal_native_store import NativeBank, durable_directory
from text.legal_reuse_input import ReuseInputs

SCHEMA = 'answer-canonical-input-plan-v1'
P = Path('/zfs/archive/mnlm/evidence/fsv/29')
BANK = dict(path='/zfs/hot/mnlm/readings/legal-source-v1-20261022',
            pin='2dd93334c150ed38dc6be22efa219cdf95528642335edab445ef1ba909fbc35d')
ANSWER = dict(path=str(P/'capability-answer-sequences-v1-20261023'),
              pin='ea03479741a29a5b97cdb0477ccb8f56bac87c9bf23c9f3d57559c87c509bef7')


def read(path, pin):
    need(file_sha(path) == pin, 'ANSWER_PLAN_PIN'); return json.loads(Path(path).read_bytes())


def inventory(view):
    texts = {key:r['text'] for key,r in view.inputs.items()}; tasks = {}
    for key in sorted(view.rows):
        row = view.rows[key]; pop,task,split = view.source.selected(key); context = []
        for f in task['context']:
            field = pop.fields[f]; identity = field['text_sha256']; value = field['text']
            need(identity not in texts or texts[identity] == value, 'ANSWER_PLAN_HASH_COLLISION')
            texts[identity] = value
            context.append(dict(field=f, input=identity, role=field['role'], clock=field['clock']))
        targets = [a['measurement'] for a in row['answer']['atoms']] if row['status'] == 'ready' else []
        tasks[key] = dict(population=pop.pin, split=split, split_group=task['split_group'], context=context,
                          targets=targets, answer_status=row['status'],
                          field_order_is_event_chronology=False, native_windows_are_answer_atoms=False)
    return [dict(id=k,text=texts[k]) for k in sorted(texts)], tasks


def authority(ref):
    bank = NativeBank(ref['path'],ref['pin']); specs, _ = instruments(bank)
    c = bank.config; cold = read(Path(c['cold_readback']['path'])/'report.json', c['cold_readback']['pin'])
    need(cold['complete'] and not cold['errors'] and cold['source']['pin'] == c['runtime_proof']['pin'],
         'ANSWER_PLAN_CANONICAL_COLD')
    contracts = {s.name:json.loads(s.definition_json)['native_contract'] for s in specs}
    for name,contract in contracts.items():
        need(contract['runtime'] == 'canonical-complete-native-windows-cuda-fp32-v1'
             and contract['max_windows'] is None and contract['device'] == 'cuda', 'ANSWER_PLAN_RUNTIME')
        need(cold['instruments'][name]['lens_id'] == c['instruments'][name]['lens_id'], 'ANSWER_PLAN_COLD_IDENTITY')
    return bank, specs, contracts


def tokenizer(contract):
    path = snapshot_dir(contract['repo'])
    need(path.name == contract['revision'] and preprocessing_sha256(path) == contract['preprocessing_sha256'],
         'ANSWER_PLAN_TOKENIZER_IDENTITY')
    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    need(tok.is_fast, 'ANSWER_PLAN_FAST_TOKENIZER'); return tok


def census(contract, records):
    tok = tokenizer(contract); counts = []; tokens = []
    # Bounded CPU tokenizer cohorts. All overflow windows remain; there is no window cap.
    for start in range(0,len(records),32):
        texts = [contract['prefix']+r['text'] for r in records[start:start+32]]
        enc = tok(texts, truncation=True, max_length=contract['max_len'], padding=False,
                  return_overflowing_tokens=True)
        mapping = enc['overflow_to_sample_mapping']; n = np.bincount(mapping,minlength=len(texts))
        need(np.all(n > 0), 'ANSWER_PLAN_TOKEN_COVERAGE'); counts.extend(map(int,n))
        active = np.bincount(mapping, weights=[len(v) for v in enc['input_ids']],minlength=len(texts))
        tokens.extend(map(int,active))
    return dict(windows=counts, active_tokens=tokens, total_windows=sum(counts), total_tokens=sum(tokens),
                max_windows=max(counts), inputs_exceeding_old_eight_window_limit=sum(v>8 for v in counts))


def code():
    base = Path(__file__).resolve().parents[1]
    return {n:file_sha(base/n) for n in ('text/answer_native_plan.py','audit/answer_native_plan_readback.py',
                                       'text/answer_native_measure.py','audit/answer_native_queue.py')}


def build(output):
    output.mkdir(exist_ok=False); started = tick = time.perf_counter(); phases = {}
    try:
        view = AnswerTargets(ANSWER['path'],ANSWER['pin']); records,tasks = inventory(view)
        bank,specs,contracts = authority(BANK)
        source = ReuseInputs(**dict(root=bank.config['source']['path'],pin=bank.config['source']['pin']))
        reader = NativeBatchSession(bank.root,bank.pin,source)
        phases['source_authentication'] = time.perf_counter()-tick; tick = time.perf_counter()
        packed = np.frombuffer(b''.join(bytes.fromhex(r['id']) for r in records),dtype='V32')
        gallery = source.arrays['hashes']; positions = np.searchsorted(gallery,packed); reused = []
        for index,(r,uid) in enumerate(zip(records,positions)):
            if uid < len(gallery) and gallery[uid] == packed[index]:
                need(source.input(int(uid))['text'] == r['text'], 'ANSWER_PLAN_REUSE_TEXT')
                reused.append(dict(index=index,uid=int(uid),id=r['id']))
        reuse_indices = {r['index'] for r in reused}; missing = [i for i in range(len(records)) if i not in reuse_indices]
        legacy = []
        for root,pin in sorted({(v['native'],v['native_pin']) for v in view.source.specs.values()}):
            report = read(Path(root)/'report.json',pin)
            for name,item in sorted(report['lenses'].items()):
                lane = read(Path(root)/name/'report.json',item['report_sha256'])
                need(lane['lens_id'] != bank.config['instruments'][name]['lens_id'], 'ANSWER_PLAN_LEGACY_UNEXPECTED_MATCH')
                legacy.append(dict(bank=root,native_pin=pin,name=name,old_lens_id=lane['lens_id'],
                    canonical_lens_id=bank.config['instruments'][name]['lens_id'],reuse=False,
                    reason='instrument_runtime_identity_differs',old_runtime=lane['contract']['runtime']))
        phases['indexed_lookup_and_legacy_contracts'] = time.perf_counter()-tick; tick = time.perf_counter()
        files = []; native = {}; token_counts = {}
        for name in bank.config['roster']:
            token_counts[name] = census(contracts[name],records)
            if reused:
                batch = reader.read_many(name,[r['uid'] for r in reused]); dest = output/name; dest.mkdir()
                traces = [json.loads(t) for t in batch.traces]
                need([len(t['windows']) for t in traces] == [token_counts[name]['windows'][r['index']] for r in reused],
                     'ANSWER_PLAN_REUSE_WINDOWS')
                np.savez(dest/'reused.npz',vectors=batch.values,offsets=batch.offsets)
                write_committed(dest/'traces.json',traces)
                native[name] = dict(lens_id=batch.lens_id,windows=len(batch.values),
                    shards=[dict(path=str(d['root']),pin=d['pin']) for (n,_),d in reader.descriptors.items() if n==name])
                files += [name+'/reused.npz',name+'/traces.json']
        phases['token_census_and_reused_native'] = time.perf_counter()-tick; tick = time.perf_counter()
        write_committed(output/'inputs.json',records); write_committed(output/'tasks.json',tasks)
        write_committed(output/'tokens.json',token_counts); write_committed(output/'legacy.json',legacy)
        files += ['inputs.json','tasks.json','tokens.json','legacy.json']
        for path in files:
            with (output/path).open('rb') as f: os.fsync(f.fileno())
        view.source.fresh(); source.fresh(); reader.fresh()
        phases['publication'] = time.perf_counter()-tick
        result = dict(schema=SCHEMA,complete=True,source=ANSWER,canonical_bank=BANK,
            specs=[asdict(s) for s in specs],contracts=contracts,roster=bank.config['roster'],reused=reused,missing=missing,
            reuse_native=native,inputs=len(records),input_bytes=sum(len(r['text'].encode()) for r in records),
            missing_bytes=sum(len(records[i]['text'].encode()) for i in missing),tasks=len(tasks),
            answer_atoms=sum(len(t['targets']) for t in tasks.values()),
            context_fields=sum(len(t['context']) for t in tasks.values()),native_reader_stats=reader.stats,
            old_runtime_reused=False,training_ready=False,scientific_admission=False,code=code(),
            files={n:dict(bytes=(output/n).stat().st_size,sha256=file_sha(output/n)) for n in files},
            phases=phases,seconds=time.perf_counter()-started,acceptance='author-checked')
        need(sum(phases.values())/result['seconds']>=.9,'ANSWER_PLAN_PHASE_COVERAGE')
        pin = write_committed(output/'report.json',result)
        with (output/'COMMITTED').open('x') as f: f.write(pin); f.flush(); os.fsync(f.fileno())
        durable_directory(output)
        print(json.dumps(dict(pin=pin,inputs=len(records),reused=len(reused),missing=len(missing),
                             census={n:{k:v for k,v in c.items() if k not in ('windows','active_tokens')} for n,c in token_counts.items()})))
    except BaseException as exc:
        write_committed(output/'failure.json',dict(type=type(exc).__name__,error=str(exc))); raise


class AnswerNativePlan:
    def __init__(self,root,pin):
        self.root = Path(root); self.pin = pin; self.report = r = read(self.root/'report.json',pin)
        need((self.root/'COMMITTED').read_text() == pin and r['schema']==SCHEMA and r['complete']
             and not r['training_ready'] and not r['scientific_admission'] and not r['old_runtime_reused'], 'ANSWER_PLAN_SCOPE')
        for n,ref in r['files'].items():
            need((self.root/n).stat().st_size==ref['bytes'] and file_sha(self.root/n)==ref['sha256'], 'ANSWER_PLAN_FILE')
        self.records = json.loads((self.root/'inputs.json').read_bytes()); self.tasks = json.loads((self.root/'tasks.json').read_bytes())
        self.tokens = json.loads((self.root/'tokens.json').read_bytes())
        need(len(self.records)==r['inputs'] and [v['id'] for v in self.records]==sorted({v['id'] for v in self.records}),
             'ANSWER_PLAN_INPUT_ROSTER')
        need(all(digest(v['text'].encode())==v['id'] for v in self.records), 'ANSWER_PLAN_TEXT_PIN')
        need(len(r['missing'])+len(r['reused'])==len(self.records)
             and set(r['missing']).isdisjoint(x['index'] for x in r['reused'])
             and set(r['missing'])|{x['index'] for x in r['reused']}==set(range(len(self.records))), 'ANSWER_PLAN_COVERAGE')
        _,specs,contracts = authority(r['canonical_bank'])
        need([asdict(s) for s in specs]==r['specs'] and contracts==r['contracts'], 'ANSWER_PLAN_INSTRUMENT_IDENTITY')
        for n in r['roster']:
            counts = self.tokens[n]; need(len(counts['windows'])==len(self.records)
                and sum(counts['windows'])==counts['total_windows'] and min(counts['windows'])>0, 'ANSWER_PLAN_WINDOWS')
        self.seal = digest(canonical(dict(records=self.records,tasks=self.tasks,tokens=self.tokens)))

    def fresh(self):
        need(file_sha(self.root/'report.json')==self.pin and
             digest(canonical(dict(records=self.records,tasks=self.tasks,tokens=self.tokens)))==self.seal,'ANSWER_PLAN_CHANGED')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    build(parser.parse_args().output)
