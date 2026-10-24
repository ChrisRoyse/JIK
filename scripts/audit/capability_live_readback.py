"""Independent SQL/state readback of automatic kernel publication and correction.

Does not instantiate the live memory producer, its population builder, or router.
The frozen source/slot roster comes from the accepted prior kernel; eligibility
is recomputed from independently checked journal episodes and dependencies.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.capability_journal_reader import JournalReader
from audit.repository_memory_readback import chain
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed


def read(path): return json.loads(Path(path).read_bytes())


class LiveReader:
    def __init__(self, config, proofs):
        self.config = config
        parent = Path(config['source'])
        need(sha((parent/'experience.sqlite3').read_bytes()) == config['source_pin'], 'TASK_READER_PARENT')
        self.base = JournalReader(read(parent/'config.json'), proofs).read(parent/'experience.sqlite3')
        self.states = copy.deepcopy(self.base['states'])
        self.contract, self.prefix = chain(parent/'experience.sqlite3')
        self.memory_head = self.base['memory_head']
        self.episodes, self.children = {}, {}
        sequences = []
        for body, payload, _ in self.prefix:
            if body.get('kind') == 'invalidate': sequences.append(body['record']['sequence'])
            elif body.get('kind') == 'capability_source_review':
                sequences.extend(r['sequence'] for r in payload['records'])
            if body.get('kind') != 'capability_intake': continue
            for record in payload['memory_records']:
                sequences.append(record['sequence'])
                if record['event']['op'] != 'intake': continue
                ep = record['event']['episode']; self.episodes[ep['id']] = ep
                for parent in ep['evidence']: self.children.setdefault(parent, set()).add(ep['id'])
        self.memory_sequence = max(sequences)+1
        self.origin = read(Path(config['kernel'])/'shared/expanded/kernel/manifest.json')
        self.intent = self.active = self.spec = None
        self.publications, self.transitions = [], []

    def source(self):
        result = copy.deepcopy(self.origin['source'])
        all_ids = []
        for family, value in result['families'].items():
            statuses, eligible = {}, []
            for key, task in sorted(self.base['tasks'].items()):
                if task['family'] != family: continue
                fields = {f['id'] for side in task['selections'].values() for f in side['fields']}
                episodes = [task['episode']] + [self.base['fields'][task['population']+':'+f]['episode']
                    for f in fields] + task['related_executions']
                states = {ep: self.states[ep] for ep in episodes}
                statuses[key] = digest(states)
                if task['split_group'] in value['groups'] and all(
                        s['active'] and s['grounded_scope'] is not None for s in states.values()): eligible.append(key)
            value['statuses'], value['ids'] = statuses, eligible
            value['connections'] = {key: value['connections'][key] for key in eligible}
            all_ids.extend(eligible)
        result['ids'] = sorted(all_ids)
        return result

    def memory_record(self, record):
        need(record['sequence'] == self.memory_sequence and record['previous'] == self.memory_head and record['sha256'] ==
             digest({k: v for k, v in record.items() if k != 'sha256'}), 'TASK_READER_MEMORY_CHAIN')
        event = record['event']; key = event['episode']
        need(key in self.episodes, 'TASK_READER_MEMORY_SCOPE')
        before = digest(self.states)
        if event['op'] == 'invalidate':
            affected, queue = set(), [key]
            while queue:
                item = queue.pop()
                if item in affected: continue
                affected.add(item); queue.extend(self.children.get(item, ()))
            need(sorted(affected) == event['affected'], 'TASK_READER_DEPENDENT_INVALIDATION')
            for item in affected:
                self.states[item].update(grounded_scope=None, review='needs_review', receipt=None)
        elif event['op'] == 'check_source':
            receipt = event['result']
            need(receipt['id'] == digest({k: v for k, v in receipt.items() if k != 'id'})
                 and receipt['episode'] == key and receipt['source'] == self.episodes[key]['source']
                 and receipt['scope'] == 'pinned_artifact_contents_only'
                 and receipt['world_validity'] == 'unassessed', 'TASK_READER_SOURCE_RECEIPT')
            need(all(self.states[ep]['active'] and self.states[ep]['grounded_scope'] is not None
                     for ep in self.episodes[key]['evidence']), 'TASK_READER_PREMATURE_REVIEW')
            self.states[key].update(receipt=receipt, review='source_checked')
        else:
            need(event['op'] == 'promote' and event['receipt'] == self.states[key]['receipt']['id'],
                 'TASK_READER_PROMOTION')
            self.states[key].update(grounded_scope=self.states[key]['receipt']['scope'], review='source_checked')
        self.memory_head = record['sha256']
        self.memory_sequence += 1
        self.transitions.append(dict(op=event['op'], episode=key, before=before, after=digest(self.states)))

    def publication(self, payload):
        need(self.intent is not None and payload['intent'] == self.intent['id']
             and payload['source'] == self.intent['source'] == self.source(), 'TASK_READER_PUBLICATION_SOURCE')
        state = payload['state']; root = Path(state['path']); raw = (root/'manifest.json').read_bytes()
        need(sha(raw) == state['pin'] and read(root/'COMMITTED') == dict(manifest=state['pin']), 'TASK_READER_STATE_PIN')
        checkpoint = json.loads(raw)
        need(checkpoint['kernel'] == payload['kernel'] and checkpoint['source'] == payload['source'],
             'TASK_READER_STATE_JOIN')
        for name, info in checkpoint['files'].items():
            data = (root/name).read_bytes()
            need(dict(bytes=len(data), sha256=sha(data)) == info, 'TASK_READER_STATE_FILE')
        kernel = Path(payload['kernel']['path']); raw = (kernel/'manifest.json').read_bytes()
        need(sha(raw) == payload['kernel']['pin'] and read(kernel/'COMMITTED') == dict(manifest=payload['kernel']['pin']),
             'TASK_READER_KERNEL_PIN')
        value = json.loads(raw); ids = self.source()['ids']
        need(value['source'] == self.source() and value['ids'] == ids
             and value['slots'] == {k: self.origin['slots'][k] for k in ids}
             and value['connections'] == {k: self.origin['connections'][k] for k in ids}, 'TASK_READER_ELIGIBILITY')
        need(value['recall_passes'] and all(c['passes'] and all(
             min(f['recall'], f['target_recall']) >= .95 for f in c['families'].values())
             for c in value['calibration'].values()), 'TASK_READER_RECALL')
        if self.active is not None: need(value['parent'] == self.active['kernel'], 'TASK_READER_GENERATION_PARENT')
        else: need(state == self.spec['seed'], 'TASK_READER_SEED')
        need(value['scientific_admission'] is False and value['learned_capability'] is False,
             'TASK_READER_FALSE_PROMOTION')
        self.active = dict(**state, kernel=payload['kernel'], source=payload['source'])
        self.publications.append(dict(state=state, kernel=payload['kernel'], tasks=len(ids),
            minimum_recall=min(min(f['recall'], f['target_recall']) for c in value['calibration'].values()
                               for f in c['families'].values()), source=digest(self.source())))
        self.intent = None

    def run(self, journal):
        contract, rows = chain(journal)
        need(contract == self.contract and rows[:len(self.prefix)] == self.prefix, 'TASK_READER_PREFIX')
        previous = self.prefix[-1][2]
        for body, payload, event in rows[len(self.prefix):]:
            kind = body['kind']
            if kind == 'invalidate': self.memory_record(body['record'])
            elif kind == 'capability_source_review':
                for record in payload['records']: self.memory_record(record)
            else:
                need(body['prior_head'] == previous, 'TASK_READER_PRIOR_HEAD')
                if kind == 'task_kernel_enable':
                    need(self.spec is None and 'spec' not in body
                         and payload['memory'] == self.memory_head, 'TASK_READER_ENABLE')
                    self.spec = payload['spec']
                elif kind == 'task_kernel_begin':
                    need(self.spec is not None and self.intent is None and payload['source'] == self.source()
                         and payload['previous'] == self.active and payload['memory'] == self.memory_head,
                         'TASK_READER_BEGIN')
                    self.intent = dict(id=digest(body), **payload)
                elif kind == 'task_kernel_supersede':
                    need(payload['intent'] == self.intent and payload['current_source'] == self.source()
                         and self.intent['source'] != self.source(), 'TASK_READER_SUPERSEDE')
                    self.intent = None
                else:
                    need(kind == 'task_kernel_publish', 'TASK_READER_UNMEASURED_EVENT')
                    self.publication(payload)
            previous = event
        return dict(head=previous, events=len(rows), memory_head=self.memory_head, active=self.active,
                    pending=self.intent, publications=self.publications, transitions=self.transitions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--proofs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); need(not a.output.exists(), 'TASK_READER_EXISTS')
    config = read(a.config); root = Path(config['root'])
    reader = LiveReader(config, read(a.proofs)); result = reader.run(root/'experience.sqlite3')
    saved = read(root/'snapshot-final.json')
    need(result['head'] == saved['head'] and result['events'] == saved['events']
         and result['memory_head'] == saved['memory']['head']
         and result['active'] == saved['task_kernel']['checkpoint']
         and result['pending'] is None and result['transitions'], 'TASK_READER_SNAPSHOT')
    need(len(result['publications']) == 4, 'TASK_READER_BOUNDARIES')
    pending = read(root/'pending.json'); cold = read(root/'cold-report.json')
    need(cold['events'] == pending['events']+1 and cold['head'] == result['head'], 'TASK_READER_RECOVERY')
    report = dict(**result, journal=sha((root/'experience.sqlite3').read_bytes()),
        source_eligibility_independently_reconstructed=True, checkpoint_recovery_readback=True,
        number_status='measured', acceptance='author-checked', scientific_admission=False, learned_capability=False,
        model_consumer_connected=False, metric_mapping=['T2/spec27 kernel recall', 'T5/spec27 continual consistency'])
    pin = write_committed(a.output, report)
    print(json.dumps(dict(pin=pin, publications=len(result['publications']), events=result['events'])), flush=True)


if __name__ == '__main__': main()
