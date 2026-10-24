"""Read back the retained failure and explicit routing-version recovery from SQL and native bytes."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.capability_delta_live_readback import DeltaReader, read
from audit.repository_memory_readback import chain
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed


class GrowthReader(DeltaReader):
    def __init__(self, config, proofs):
        parent = Path(config['source'])
        need(sha((parent / 'experience.sqlite3').read_bytes()) == config['source_pin'], 'GROWTH_READER_PARENT')
        super().__init__(read(parent / 'config.json'), proofs)
        pending = self.run_delta(parent / 'experience.sqlite3')
        need(pending['pending'] is not None and len(pending['intakes']) == 3
             and len(pending['publications']) == 1, 'GROWTH_READER_PENDING_PREFIX')
        self.pending_history = copy.deepcopy(pending)
        self.contract, self.prefix = chain(parent / 'experience.sqlite3')
        self.growth_config = config; self.enabled = False
        prior = read(Path(self.active['kernel']['path']) / 'manifest.json')
        code = copy.deepcopy(prior['code'])
        for name in ('kernel/capability_growth_kernel.py', 'kernel/capability_growth_route.py'):
            code['capability'][name] = sha((self.code_root / name).read_bytes())
        self.growth_code = dict(kernel=dict(kernel=code,
            checkpoint=sha((self.code_root / 'kernel/capability_growth_state.py').read_bytes())),
            live=sha((self.code_root / 'model/capability_growth_memory.py').read_bytes()))

    def run_growth(self, journal):
        contract, rows = chain(journal)
        need(contract == self.contract and rows[:len(self.prefix)] == self.prefix, 'GROWTH_READER_PREFIX')
        previous = self.prefix[-1][2]
        for body, payload, event in rows[len(self.prefix):]:
            need(body['prior_head'] == previous and body['code'] == self.growth_code, 'GROWTH_READER_EVENT')
            kind = body['kind']
            if kind == 'task_growth_enable':
                need(not self.enabled and payload['previous_root'] == self.spec['root']
                     and payload['previous'] == self.active and payload['superseded_intent'] == self.intent
                     and payload['memory'] == self.memory_head
                     and payload['root'] == str(Path(self.growth_config['root']) / 'task-kernels'),
                     'GROWTH_READER_MIGRATION')
                self.spec = dict(self.spec, root=payload['root'])
                self.intent = None; self.enabled = True
            elif kind == 'task_growth_begin':
                need(self.enabled and self.intent is None and payload['source'] == self.source()
                     and payload['previous'] == self.active and payload['memory'] == self.memory_head,
                     'GROWTH_READER_BEGIN')
                self.intent = dict(id=digest(body), **payload)
            else:
                need(self.enabled and kind == 'task_growth_publish', 'GROWTH_READER_EVENT_KIND')
                kernel = read(Path(payload['kernel']['path']) / 'manifest.json')
                state = read(Path(payload['state']['path']) / 'manifest.json')
                need(kernel['schema'] == 'connected-task-growth-kernel-v1'
                     and kernel['code'] == self.growth_code['kernel']['kernel']
                     and state['schema'] == 'capability-growth-updater-state-v1'
                     and state['code'] == self.growth_code['kernel']
                     and kernel['routing_policy'] == dict(max_distance_computations=None,
                         historical_plan_limit_applies=False, seeding='all_kernel_entries_plus_native_window_ann'),
                     'GROWTH_READER_KERNEL_VERSION')
                for name, calibration in kernel['calibration'].items():
                    entries = kernel['graphs'][name]['entries']
                    for row in calibration['rows']:
                        route = row['route']
                        need(route['kernel_entries'] == sorted(entries)
                             and set(route['entries']) == set(entries) | set(route['native_window_entries'])
                             and set(route['entries']) <= set(kernel['ids'])
                             and route['work_limit'] is None
                             and route['method'] == 'kernel-and-native-window-entries/weak-graph-best-first',
                             'GROWTH_READER_ENTRY_CONSERVATION')
                        need(row['recall'] == len(set(row['exact']) & {k for k, _ in route['hits']}) /
                             kernel['plan']['top_k'], 'GROWTH_READER_RECALL_ARITHMETIC')
                self.origin = dict(slots=self.slots, connections=self.links)
                self.publication(payload)
            previous = event
        return dict(head=previous, events=len(rows), memory_head=self.memory_head,
            active=self.active, pending=self.intent, publications=self.publications, intakes=self.intakes,
            association_readbacks=self.association_readbacks, retained_pending_head=self.pending_history['head'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'proofs', 'output'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); need(not a.output.exists(), 'GROWTH_READER_EXISTS')
    started = time.perf_counter(); config = read(a.config); root = Path(config['root'])
    reader = GrowthReader(config, read(a.proofs)); result = reader.run_growth(root / 'experience.sqlite3')
    saved = read(root / 'snapshot-final.json')
    need(result['head'] == saved['head'] and result['events'] == saved['events']
         and result['memory_head'] == saved['memory']['head'] and reader.states == saved['memory']['states']
         and result['active'] == saved['task_kernel']['checkpoint'] and result['pending'] is None
         and saved['growth_routing'] is True and len(result['publications']) == 2,
         'GROWTH_READER_FINAL_STATE')
    for stage in ('update', 'cold'):
        report = read(root / (stage+'-report.json'))
        need(report['complete'] and report['head'] == result['head'] and report['events'] == result['events']
             and report['parameters_unchanged'] and report['optimizer_steps'] == 0, 'GROWTH_READER_STAGE')
    report = dict(**result, journal=sha((root / 'experience.sqlite3').read_bytes()),
        native_field_arrays_read=len(reader.field_arrays), failed_intent_preserved=True,
        source_eligibility_independently_reconstructed=True, native_context_arrays_independently_read=True,
        cold_state_readback=True, elapsed=time.perf_counter()-started,
        number_status='measured', acceptance='author-checked', scientific_admission=False,
        learned_capability=False, generated_surface_guard_complete=False,
        metric_mapping=['T2/spec27 kernel recall', 'T5/spec27 continual consistency'])
    pin = write_committed(a.output, report)
    print(json.dumps(dict(pin=pin, events=result['events'], associations=result['association_readbacks'])), flush=True)


if __name__ == '__main__': main()
