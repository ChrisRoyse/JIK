"""Journal-driven repository intake, durable kernel publication, and recovery."""
import copy
import json
from pathlib import Path

from claims.codata_source import check,sha
from claims.episodic_memory import digest
from kernel.repository_kernel import AssociationCompanion
from kernel.repository_population import KINDS
from kernel.repository_update_state import save,restore
from model.repository_kernel_memory import RepositoryKernelMemoryLoop


def code_pins():
    root=Path(__file__).resolve().parents[1]
    return {n:sha((root/n).read_bytes()) for n in ('model/repository_live_memory.py',
        'kernel/repository_live_population.py','kernel/repository_update_state.py')}


def read(path):return json.loads(Path(path).read_bytes())


class LiveRepositoryMemoryLoop(RepositoryKernelMemoryLoop):
    def __init__(self,*args,**kwargs):
        self.live_spec=self.live_intent=self.live_checkpoint=self.live_companion=None
        super().__init__(*args,**kwargs)

    def _echo_publish(self,body,payload,expected):
        seq,previous=self.journal.count,self.head
        super()._echo_publish(body,payload,expected)
        p=self.repository_population
        if body.get('kind') in KINDS and seq>=p.base['events']:
            try:p.consume(body,payload,seq=seq,previous=previous,
                          event=digest(dict(seq=seq,previous=previous,body=body)))
            except BaseException:
                # The intake is already durable. Reopen its journal; never pretend it was undone.
                self.journal.broken=True
                raise

    def enable_repository_updates(self,spec,*,expected=None):
        self._fresh();p=self.repository_population
        check(self.live_spec is None and set(spec)=={'root','excluded','queries'},'LIVE_ENABLE_SPEC')
        check(Path(spec['root']).is_absolute() and Path(spec['root']).parent.is_dir(),'LIVE_ENABLE_ROOT')
        check(spec['excluded'] and spec['queries'] and len(set(spec['queries']))==len(spec['queries']) and
              all(p.records[k]['task'] in spec['excluded'] for k in spec['queries']),'LIVE_ENABLE_SPLIT')
        payload=dict(spec=spec,base=p.base,source=p.source(len(p.groups)))
        body=dict(kind='repository_live_enable',spec=spec,prior_head=self.head,payload=digest(payload),code=code_pins())
        self._echo_publish(body,payload,expected)
        self.live_spec=copy.deepcopy(spec)
        if expected is None:self.synchronize_repository()
        return body

    def _begin_repository_update(self,*,expected=None):
        self._fresh();p=self.repository_population
        check(self.live_spec is not None and self.live_intent is None,'LIVE_BEGIN_STATE')
        source=p.source(len(p.groups))
        check(self.live_checkpoint is None or self.live_checkpoint['source']!=source,'LIVE_ALREADY_CURRENT')
        payload=dict(source=source,previous=self.live_checkpoint,groups=len(p.groups))
        body=dict(kind='repository_live_begin',prior_head=self.head,payload=digest(payload),code=code_pins())
        self._echo_publish(body,payload,expected)
        self.live_intent=dict(id=digest(body),**copy.deepcopy(payload))
        return body

    def _check_companion(self,companion):
        intent=self.live_intent;value=companion.last
        previous=intent['previous']
        expected_parent=None if previous is None else {k:previous['kernel'][k] for k in ('path','pin')}
        check(value['source']==intent['source'] and value['parent']==expected_parent and
              companion.excluded==self.live_spec['excluded'] and companion.queries==self.live_spec['queries'] and
              value['recall_passes'],'LIVE_CHECKPOINT_LINEAGE')

    def _finish_repository_update(self,state,*,expected=None):
        self._fresh();check(self.live_intent is not None,'LIVE_FINISH_WITHOUT_INTENT')
        if expected is not None:
            self.live_companion,_=restore(self.repository_population,state['path'],state['pin'])
        check(self.live_companion is not None,'LIVE_FINISH_WITHOUT_CHECKPOINT')
        self._check_companion(self.live_companion)
        manifest=read(Path(state['path'])/'manifest.json')
        check(sha((Path(state['path'])/'manifest.json').read_bytes())==state['pin'] and
              read(Path(state['path'])/'COMMITTED')==dict(manifest=state['pin']) and
              manifest['kernel']==self.live_companion.previous and manifest['source']==self.live_intent['source'],
              'LIVE_FINISH_CHECKPOINT')
        kernel=dict(**manifest['kernel'],source=manifest['source'])
        check(self.repository_runtime is not None and self.repository_runtime.spec==kernel,'LIVE_FINISH_ACTIVATION')
        payload=dict(intent=self.live_intent['id'],state=state,kernel=kernel,source=manifest['source'])
        body=dict(kind='repository_live_finish',state=state,prior_head=self.head,payload=digest(payload),code=code_pins())
        self._echo_publish(body,payload,expected)
        self.live_checkpoint=dict(**copy.deepcopy(state),kernel=kernel,source=manifest['source'])
        self.live_intent=None
        return body

    def synchronize_repository(self,*,stop_after_checkpoint=False):
        self._fresh();p=self.repository_population
        check(self.live_spec is not None,'LIVE_NOT_ENABLED')
        current=p.source(len(p.groups))
        if self.live_intent is None:
            if self.live_checkpoint is not None and self.live_checkpoint['source']==current:return False
            self._begin_repository_update()
        intent=self.live_intent
        check(intent['source']==current,'LIVE_DERIVATION_SOURCE_CHANGED')
        root=Path(self.live_spec['root'])/intent['id'];root.mkdir(parents=True,exist_ok=True)
        attempts=sorted(root.iterdir())
        check(all(p.is_dir() and p.name==str(i).zfill(4) for i,p in enumerate(attempts)),'LIVE_ATTEMPT_ROSTER')
        committed=[p for p in attempts if (p/'updater'/'COMMITTED').is_file()]
        check(len(committed)<=1,'LIVE_MULTIPLE_CHECKPOINTS')
        if committed:
            checkpoint=committed[0]/'updater';pin=read(checkpoint/'COMMITTED')['manifest']
            self.live_companion,_=restore(p,checkpoint,pin)
        else:
            previous=intent['previous']
            if self.live_companion is None:
                if previous is None:self.live_companion=AssociationCompanion(p,self.live_spec['excluded'],self.live_spec['queries'])
                else:
                    prefix=p.at(previous['source']['supported_intakes'])
                    self.live_companion,_=restore(prefix,previous['path'],previous['pin'])
                    self.live_companion.population=p
            attempt=root/str(len(attempts)).zfill(4);attempt.mkdir();checkpoint=attempt/'updater'
            try:
                self.live_companion.sync(intent['groups'],attempt/'kernel')
                self._check_companion(self.live_companion)
                pin,_=save(self.live_companion,checkpoint)
            except BaseException:
                self.live_companion=None
                raise
        self._check_companion(self.live_companion)
        if stop_after_checkpoint:return dict(pending=True,path=str(checkpoint),pin=pin)
        spec=dict(**self.live_companion.previous,source=current)
        if self.repository_runtime is None or self.repository_runtime.spec!=spec:
            self.activate_repository_kernel(spec)
        self._finish_repository_update(dict(path=str(checkpoint),pin=pin))
        return True

    def activate_repository_kernel(self,spec,*,expected=None):
        if self.live_spec is not None:
            check(self.live_intent is not None and spec['source']==self.live_intent['source'],
                  'LIVE_ACTIVATION_WITHOUT_INTENT')
        return super().activate_repository_kernel(spec,expected=expected)

    def read_repository(self,*args,**kwargs):
        if self.live_spec is not None:
            p=self.repository_population
            check(self.live_intent is None and self.live_checkpoint is not None and
                  self.live_checkpoint['source']==p.source(len(p.groups)),'LIVE_KERNEL_UPDATE_PENDING')
        return super().read_repository(*args,**kwargs)

    def _observe_live(self,method,spec,expected,synchronize):
        check(self.live_intent is None,'LIVE_DERIVATION_PENDING')
        body=method(spec,expected=expected)
        if expected is None and self.live_spec is not None and synchronize:self.synchronize_repository()
        return body

    def observe_cohort(self,spec,expected=None,*,synchronize=True):
        return self._observe_live(super().observe_cohort,spec,expected,synchronize)

    def observe_depth(self,spec,expected=None,*,synchronize=True):
        return self._observe_live(super().observe_depth,spec,expected,synchronize)

    def observe_demonstration(self,spec,expected=None,*,synchronize=True):
        return self._observe_live(super().observe_demonstration,spec,expected,synchronize)

    def _replay(self,body):
        kind=body.get('kind')
        if kind.startswith('repository_live_'):check(body['code']==code_pins(),'LIVE_PRODUCER_CODE')
        if kind=='repository_live_enable':self.enable_repository_updates(body['spec'],expected=body)
        elif kind=='repository_live_begin':self._begin_repository_update(expected=body)
        elif kind=='repository_live_finish':self._finish_repository_update(body['state'],expected=body)
        else:super()._replay(body)

    def snapshot(self):
        result=super().snapshot()
        if self.live_spec is not None:
            p=self.repository_population
            result['repository_updates']=dict(spec=copy.deepcopy(self.live_spec),intent=copy.deepcopy(self.live_intent),
                checkpoint=copy.deepcopy(self.live_checkpoint),source=p.source(len(p.groups)),
                pending=self.live_intent is not None or self.live_checkpoint is None or
                        self.live_checkpoint['source']!=p.source(len(p.groups)),
                scientific_admission=False,learned_capability=False)
        return result
