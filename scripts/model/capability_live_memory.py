"""Automatic capability-kernel publication, correction and recovery in the intrinsic journal."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path

from claims.capability_population import need, sha
from claims.episodic_memory import digest
from kernel.capability_kernel import TaskCompanion, code_pins as kernel_code
from kernel.capability_split import freeze
from kernel.capability_update_state import save, restore
from model.capability_live_population import population
from model.capability_memory import CapabilityMemoryLoop


def read(path): return json.loads(Path(path).read_bytes())


def code_pins():
    root = Path(__file__).resolve().parents[1]
    return dict(kernel=kernel_code(), live={n: sha((root/n).read_bytes()) for n in
        ('model/capability_live_memory.py', 'model/capability_live_population.py',
         'kernel/capability_update_state.py')})


class LiveCapabilityMemoryLoop(CapabilityMemoryLoop):
    def __init__(self, *args, recover_on_open=True, **kwargs):
        self.task_spec = self.task_intent = self.task_checkpoint = None
        self.task_companion = self.task_runtime = self.task_checkpoint_population = None
        self.task_splits, self.task_views = {}, {}
        self.task_batch_depth = 0
        super().__init__(*args, **kwargs)
        if recover_on_open and self.task_spec is not None: self.synchronize_tasks()

    @contextmanager
    def task_batch(self, *, stop_after_checkpoint=False):
        """One kernel update for a connected correction; all memory events stay durable."""
        self.task_batch_depth += 1
        try:
            yield
        except BaseException:
            self.task_batch_depth -= 1
            raise
        else:
            self.task_batch_depth -= 1
            if self.task_spec is not None and self.task_batch_depth == 0:
                self.synchronize_tasks(stop_after_checkpoint=stop_after_checkpoint)

    def _after_task_change(self, expected):
        if self.task_spec is not None and expected is None and self.task_batch_depth == 0:
            self.synchronize_tasks()

    def _task_population(self):
        key = (self.memory.head, digest(self.task_splits))
        if key not in self.task_views: self.task_views = {key: population(self)}
        return self.task_views[key]

    @staticmethod
    def _groups(p):
        return sorted({p.records[k]['task'] for k in p.records
                       if p.records[k]['task'] not in p.split['excluded']})

    def _task_source(self):
        p = self._task_population()
        return p, p.source(self._groups(p))

    def _task_event(self, kind, payload, expected, **arguments):
        body = dict(kind='task_kernel_'+kind, prior_head=self.head,
                    payload=digest(payload), code=code_pins(), **arguments)
        self._echo_publish(body, payload, expected)
        return body

    def enable_task_updates(self, spec, *, expected=None):
        self._fresh()
        need(self.task_spec is None and set(spec) == {'root', 'splits', 'seed'}, 'TASK_LIVE_ENABLE')
        need(Path(spec['root']).is_absolute() and Path(spec['root']).parent.is_dir(), 'TASK_LIVE_ROOT')
        known = {e.native.pin: e.native.population for e in self.capability_sources.values()}
        need(set(spec['splits']) == set(known) and all(spec['splits'][k] == freeze(p) for k, p in known.items()),
             'TASK_LIVE_SPLIT')
        body = self._task_event('enable', dict(spec=spec, memory=self.memory.head), expected)
        self.task_spec, self.task_splits = copy.deepcopy(spec), copy.deepcopy(spec['splits'])
        if expected is None: self.synchronize_tasks()
        return body

    def register_task_source(self, spec, split, *, expected=None):
        """Register a frozen split before the source's atomic capability intake."""
        self._fresh(); need(self.task_spec is not None, 'TASK_LIVE_NOT_ENABLED')
        evidence = self._capability_source(spec); pin = evidence.native.pin
        need(pin not in self.task_splits and split == freeze(evidence.population), 'TASK_LIVE_REGISTER')
        body = self._task_event('register', dict(binding=pin, spec=spec, split=split), expected)
        self.task_splits[pin] = copy.deepcopy(split)
        return body

    def _begin_tasks(self, *, expected=None):
        self._fresh(); p, source = self._task_source()
        need(self.task_intent is None and (self.task_checkpoint is None or
             self.task_checkpoint['source'] != source), 'TASK_LIVE_BEGIN')
        payload = dict(source=source, previous=self.task_checkpoint, memory=self.memory.head)
        body = self._task_event('begin', payload, expected)
        self.task_intent = dict(id=digest(body), **copy.deepcopy(payload))
        return body

    def _supersede_tasks(self, *, expected=None):
        self._fresh(); _, source = self._task_source()
        need(self.task_intent is not None and source != self.task_intent['source'], 'TASK_LIVE_SUPERSEDE')
        body = self._task_event('supersede', dict(intent=self.task_intent, current_source=source), expected)
        self.task_intent = self.task_companion = None
        return body

    def _publish_tasks(self, state, *, expected=None):
        self._fresh(); p, source = self._task_source(); intent = self.task_intent
        need(intent is not None and source == intent['source'], 'TASK_LIVE_PUBLISH_SOURCE')
        companion, runtime = restore(p, state['path'], state['pin'])
        need(companion.last['source'] == source and companion.last['recall_passes'], 'TASK_LIVE_PUBLISH_RECALL')
        previous = intent['previous']
        if previous is not None:
            need(companion.last['parent'] == previous['kernel'], 'TASK_LIVE_PUBLISH_PARENT')
        else:
            need(self.task_spec['seed'] is None or state == self.task_spec['seed'], 'TASK_LIVE_SEED')
        payload = dict(intent=intent['id'], state=state, kernel=companion.previous, source=source,
                       scientific_admission=False, learned_capability=False)
        body = self._task_event('publish', payload, expected, state=state)
        self.task_checkpoint = dict(**copy.deepcopy(state), kernel=companion.previous, source=source)
        self.task_checkpoint_population = p
        self.task_companion, self.task_runtime, self.task_intent = companion, runtime, None
        return body

    def synchronize_tasks(self, *, stop_after_checkpoint=False):
        self._fresh(); need(self.task_spec is not None, 'TASK_LIVE_NOT_ENABLED')
        p, source = self._task_source()
        p.fresh()
        if self.task_intent is not None and self.task_intent['source'] != source: self._supersede_tasks()
        if self.task_intent is None:
            if self.task_checkpoint is not None and self.task_checkpoint['source'] == source: return False
            self._begin_tasks()
        intent = self.task_intent
        seed = self.task_spec['seed'] if intent['previous'] is None else None
        if seed is not None:
            state = seed
        else:
            root = Path(self.task_spec['root'])/intent['id']; root.mkdir(parents=True, exist_ok=True)
            attempts = sorted(root.iterdir())
            need(all(p.is_dir() and p.name == str(i).zfill(4) for i, p in enumerate(attempts)), 'TASK_LIVE_ATTEMPTS')
            committed = [p/'updater' for p in attempts if (p/'updater/COMMITTED').is_file()]
            need(len(committed) <= 1, 'TASK_LIVE_MULTIPLE_CHECKPOINTS')
            if committed:
                checkpoint = committed[0]
                state = dict(path=str(checkpoint), pin=read(checkpoint/'COMMITTED')['manifest'])
            else:
                if self.task_companion is None:
                    previous = intent['previous']
                    if previous is None: self.task_companion = TaskCompanion(p)
                    else:
                        need(self.task_checkpoint_population is not None, 'TASK_LIVE_PRIOR_POPULATION')
                        self.task_companion, _ = restore(self.task_checkpoint_population,
                            previous['path'], previous['pin'])
                self.task_companion.population = p
                attempt = root/str(len(attempts)).zfill(4); attempt.mkdir()
                try:
                    self.task_companion.sync(source['groups'], attempt/'kernel')
                    pin, _ = save(self.task_companion, attempt/'updater')
                    state = dict(path=str(attempt/'updater'), pin=pin)
                except BaseException:
                    self.task_companion = None
                    raise
        if stop_after_checkpoint: return dict(pending=True, **state)
        self._publish_tasks(state)
        return True

    def read_task(self, key, *, kernel):
        self._fresh(); p, source = self._task_source()
        need(self.task_spec is not None and self.task_intent is None and self.task_checkpoint is not None
             and self.task_checkpoint['source'] == source, 'TASK_LIVE_UPDATE_PENDING')
        need(kernel == self.task_checkpoint['kernel']['pin'], 'TASK_LIVE_STALE_KERNEL')
        p.fresh()
        need(key in self.capability_tasks, 'TASK_LIVE_UNKNOWN_TASK')
        task = self.capability_tasks[key]
        self.capability_selection(key, side='context', binding=task['binding'])
        result = self.task_runtime.route(key, source=source)
        for other in result['targets']:
            record = self.capability_tasks[other]
            self.capability_selection(other, side='targets', binding=record['binding'])
        return result

    def observe_capability(self, spec, groups, *, expected=None):
        if self.task_spec is not None and expected is None:
            need(self._capability_source(spec).native.pin in self.task_splits, 'TASK_LIVE_UNREGISTERED_SOURCE')
        body = super().observe_capability(spec, groups, expected=expected)
        self._after_task_change(expected)
        return body

    def invalidate(self, key, *, reason, expected=None):
        body = super().invalidate(key, reason=reason, expected=expected)
        self._after_task_change(expected)
        return body

    def review_capability(self, episode, *, reason, expected=None):
        body = super().review_capability(episode, reason=reason, expected=expected)
        self._after_task_change(expected)
        return body

    def _replay(self, body):
        kind = body.get('kind', '')
        if kind.startswith('task_kernel_'):
            need(body['code'] == code_pins(), 'TASK_LIVE_PRODUCER_CODE')
            payload = self.journal.payload(body['payload'])
            handlers = {'task_kernel_enable': lambda: self.enable_task_updates(payload['spec'], expected=body),
                'task_kernel_register': lambda: self.register_task_source(payload['spec'], payload['split'], expected=body),
                'task_kernel_begin': lambda: self._begin_tasks(expected=body),
                'task_kernel_supersede': lambda: self._supersede_tasks(expected=body),
                'task_kernel_publish': lambda: self._publish_tasks(body['state'], expected=body)}
            need(kind in handlers, 'TASK_LIVE_EVENT_KIND'); handlers[kind]()
        else: super()._replay(body)

    def snapshot(self):
        result = super().snapshot()
        if self.task_spec is not None:
            _, source = self._task_source()
            result['task_kernel'] = dict(spec=copy.deepcopy(self.task_spec), splits=copy.deepcopy(self.task_splits),
                intent=copy.deepcopy(self.task_intent), checkpoint=copy.deepcopy(self.task_checkpoint),
                current_source=source, pending=self.task_intent is not None or self.task_checkpoint is None
                    or self.task_checkpoint['source'] != source, scientific_admission=False, learned_capability=False)
        return result
