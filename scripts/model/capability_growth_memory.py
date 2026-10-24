"""Ledgered routing-version transition; retain historical kernels and failed intents."""
import copy
from pathlib import Path

from claims.capability_population import need, sha
from claims.episodic_memory import digest
from kernel.capability_growth_kernel import TaskCompanion
from kernel.capability_growth_state import restore, save, producers
from kernel.capability_update_state import restore as restore_historical
from model.capability_delta_memory import DeltaCapabilityMemoryLoop
from model.capability_live_memory import read


def code_pins():
    return dict(kernel=producers(), live=sha(Path(__file__).read_bytes()))


class GrowthCapabilityMemoryLoop(DeltaCapabilityMemoryLoop):
    def __init__(self, *args, **kwargs):
        self.growth_routing = False
        super().__init__(*args, **kwargs)

    def enable_growth_routing(self, root, *, expected=None):
        self._fresh()
        need(not self.growth_routing and self.task_spec is not None
             and Path(root).is_absolute() and Path(root).parent.is_dir()
             and root != self.task_spec['root'], 'GROWTH_ENABLE')
        payload = dict(root=root, previous_root=self.task_spec['root'], previous=self.task_checkpoint,
                       superseded_intent=self.task_intent, memory=self.memory.head)
        body = dict(kind='task_growth_enable', prior_head=self.head, payload=digest(payload), code=code_pins())
        self._echo_publish(body, payload, expected)
        self.task_spec = dict(self.task_spec, root=root)
        self.task_intent = self.task_companion = self.task_runtime = None
        self.growth_routing = True
        return body

    def _task_event(self, kind, payload, expected, **arguments):
        if not self.growth_routing: return super()._task_event(kind, payload, expected, **arguments)
        body = dict(kind='task_growth_'+kind, prior_head=self.head,
                    payload=digest(payload), code=code_pins(), **arguments)
        self._echo_publish(body, payload, expected)
        return body

    def _publish_tasks(self, state, *, expected=None):
        if not self.growth_routing: return super()._publish_tasks(state, expected=expected)
        self._fresh(); p, source = self._task_source(); intent = self.task_intent
        need(intent is not None and source == intent['source'], 'GROWTH_PUBLISH_SOURCE')
        companion, runtime = restore(p, state['path'], state['pin'])
        need(companion.last['source'] == source and companion.last['recall_passes']
             and companion.last['parent'] == intent['previous']['kernel'], 'GROWTH_PUBLISH_RECALL_PARENT')
        payload = dict(intent=intent['id'], state=state, kernel=companion.previous, source=source,
                       scientific_admission=False, learned_capability=False)
        body = self._task_event('publish', payload, expected, state=state)
        self.task_checkpoint = dict(**copy.deepcopy(state), kernel=companion.previous, source=source)
        self.task_checkpoint_population = p
        self.task_companion, self.task_runtime, self.task_intent = companion, runtime, None
        return body

    def _begin_tasks(self, *, expected=None):
        if not self.growth_routing: return super()._begin_tasks(expected=expected)
        self._fresh(); _, source = self._task_source()
        need(self.task_intent is None and self.task_checkpoint is not None, 'GROWTH_BEGIN')
        schema = read(Path(self.task_checkpoint['path']) / 'manifest.json')['schema']
        need(self.task_checkpoint['source'] != source or schema == 'capability-updater-state-v1',
             'GROWTH_UNCHANGED_BEGIN')
        payload = dict(source=source, previous=self.task_checkpoint, memory=self.memory.head)
        body = self._task_event('begin', payload, expected)
        self.task_intent = dict(id=digest(body), **copy.deepcopy(payload))
        return body

    def synchronize_tasks(self, *, stop_after_checkpoint=False):
        if not self.growth_routing:
            return super().synchronize_tasks(stop_after_checkpoint=stop_after_checkpoint)
        self._fresh(); p, source = self._task_source(); p.fresh()
        if self.task_intent is not None and self.task_intent['source'] != source: self._supersede_tasks()
        if self.task_intent is None:
            if self.task_checkpoint is not None and self.task_checkpoint['source'] == source:
                schema = read(Path(self.task_checkpoint['path']) / 'manifest.json')['schema']
                if schema == 'capability-growth-updater-state-v1': return False
            self._begin_tasks()
        intent = self.task_intent
        root = Path(self.task_spec['root']) / intent['id']; root.mkdir(parents=True, exist_ok=True)
        attempts = sorted(root.iterdir())
        need(all(p.is_dir() and p.name == str(i).zfill(4) for i, p in enumerate(attempts)), 'GROWTH_ATTEMPTS')
        committed = [p / 'updater' for p in attempts if (p / 'updater/COMMITTED').is_file()]
        need(len(committed) <= 1, 'GROWTH_MULTIPLE_CHECKPOINTS')
        if committed:
            checkpoint = committed[0]
            state = dict(path=str(checkpoint), pin=read(checkpoint / 'COMMITTED')['manifest'])
        else:
            if self.task_companion is None:
                previous = intent['previous']
                need(previous is not None and self.task_checkpoint_population is not None, 'GROWTH_PRIOR')
                schema = read(Path(previous['path']) / 'manifest.json')['schema']
                if schema == 'capability-growth-updater-state-v1':
                    self.task_companion, _ = restore(self.task_checkpoint_population, previous['path'], previous['pin'])
                else:
                    need(schema == 'capability-updater-state-v1', 'GROWTH_PRIOR_SCHEMA')
                    old, _ = restore_historical(self.task_checkpoint_population, previous['path'], previous['pin'])
                    self.task_companion = TaskCompanion(p)
                    self.task_companion.updates = old.updates
                    self.task_companion.previous, self.task_companion.last = old.previous, None
            self.task_companion.population = p
            attempt = root / str(len(attempts)).zfill(4); attempt.mkdir()
            try:
                self.task_companion.sync(source['groups'], attempt / 'kernel')
                pin, _ = save(self.task_companion, attempt / 'updater')
                state = dict(path=str(attempt / 'updater'), pin=pin)
            except BaseException:
                self.task_companion = None
                raise
        if stop_after_checkpoint: return dict(pending=True, **state)
        self._publish_tasks(state)
        return True

    def _replay(self, body):
        kind = body.get('kind', '')
        if kind.startswith('task_growth_'):
            need(body['code'] == code_pins(), 'GROWTH_REPLAY_CODE')
            payload = self.journal.payload(body['payload'])
            handlers = dict(task_growth_enable=lambda: self.enable_growth_routing(payload['root'], expected=body),
                task_growth_begin=lambda: self._begin_tasks(expected=body),
                task_growth_supersede=lambda: self._supersede_tasks(expected=body),
                task_growth_publish=lambda: self._publish_tasks(body['state'], expected=body),
                task_growth_register=lambda: self.register_task_source(payload['spec'], payload['split'], expected=body))
            need(kind in handlers, 'GROWTH_REPLAY_KIND'); handlers[kind]()
        else:
            super()._replay(body)

    def snapshot(self):
        value = super().snapshot()
        if self.growth_routing: value['growth_routing'] = True
        return value
