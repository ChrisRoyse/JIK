"""Connect typed native world learning to the retained evidence journal."""
import json
from pathlib import Path
import time

import torch

from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from model.constellation_learning import configure, decision, frozen_pin, trainable
from model.loop_training_checkpoint import fingerprint, tensor_fingerprint
from model.native_world import NativeWorld, RULE
from model.tool_rehearsal import RehearsalData, RehearsalToolLoop, termination_gate
from model import native_residual_checkpoint as checkpoint
from model import rehearsal_retention, tool_retention, tool_context_regions


def candidate_path(session, name):
    check(isinstance(name,str) and name not in ('','.','..') and '/' not in name and '\\' not in name,
          'NATIVE_WORLD_CANDIDATE_PATH')
    path = (session.native_root/name).resolve()
    check(path.parent == session.native_root, 'NATIVE_WORLD_CANDIDATE_PATH')
    return path


class NativeData(RehearsalData):
    def __init__(self, session, *, train, evaluate, continuation):
        check(session.decisions and session.decisions[-1]['candidate'] == continuation,
              'NATIVE_WORLD_PREDECESSOR')
        prior = session.decisions[-1]
        native = prior['kind'] == 'native_world_candidate'
        super().__init__(session,train=train,evaluate=evaluate,continuation=None if native else continuation)
        self.native_predecessor = None
        self.plan['world_architecture'] = RULE
        if native:
            path = candidate_path(session,continuation)
            check(sha((path/'manifest.json').read_bytes()) == prior['pin'], 'NATIVE_WORLD_PREDECESSOR_PIN')
            m = json.loads((path/'manifest.json').read_bytes())
            binding = m['binding']
            check(m['step'] == self.plan['steps'] and digest(binding) == prior['binding']
                  and all(binding[k] == self.binding[k] for k in ('contract','standard','world','code'))
                  and {k:v for k,v in binding['plan'].items() if k not in ('train','evaluate')} ==
                      {k:v for k,v in self.plan.items() if k not in ('train','evaluate')},
                  'NATIVE_WORLD_PREDECESSOR_CONTRACT')
            check(all(set(binding['plan'][k]) <= set(self.plan[k]) for k in ('train','evaluate'))
                  and all(self.binding['evidence'].get(k) == v for k,v in binding['evidence'].items()),
                  'NATIVE_WORLD_PREDECESSOR_EVIDENCE')
            self.continuation_name = continuation
            self.binding['continuation'] = dict(candidate=continuation,pin=prior['pin'],
                                                world=binding['world'])
            foundation = binding['native_world']['foundation']
            offsets = {n:v+m['step'] for n,v in m['offsets'].items()}
            self.native_predecessor = (path,prior['pin'],m)
            rehearsal_retention.bind(self)
        else:
            path,m,pin = self.predecessor
            foundation = dict(root=str(path/'model'),pin=m['model'])
            offsets = {n:m['optimizer_steps'] for n in m['names']}
        self.model = NativeWorld.from_world(self.model)
        configure(self.model)
        names = [n for n,_ in trainable(self.model)]
        check(set(offsets) <= set(names), 'NATIVE_WORLD_PREDECESSOR_PARAMETERS')
        offsets = {n:offsets.get(n,0) for n in names}
        self.binding['native_world'] = dict(rule=RULE,code=checkpoint.code_pins(),foundation=foundation,
            offsets=offsets, parameter_scope='query_and_place_only',
            migration='zero_native_contribution_preserve_legacy_moments')
        check(frozen_pin(self.model) == self.frozen, 'NATIVE_WORLD_FROZEN_DRIFT')
        for e in self.examples.values():
            e['view'] = NativeWorld.from_world(e['view'])
        self.fresh()

    def prediction_model(self, model, key):
        check(isinstance(model,NativeWorld), 'NATIVE_WORLD_ARCHITECTURE')
        return tool_context_regions.attach(self.examples[key]['view'],model)


class NativeToolLoop(RehearsalToolLoop):
    def __init__(self, *args, native_root, **kwargs):
        self.native_root = Path(native_root).resolve()
        super().__init__(*args,**kwargs)

    def consider_native(self, name, *, pin, train, evaluate, continuation, expected=None):
        self._fresh()
        check(all(r['candidate'] != name for r in self.decisions), 'TOOL_DUPLICATE_CANDIDATE')
        data = NativeData(self,train=train,evaluate=evaluate,continuation=continuation)
        parent = self.decisions[-1]['pin']
        model,_,m = checkpoint.load(candidate_path(self,name),data,expected=pin)
        check(m['step'] == data.plan['steps'] and m['parent'] == parent, 'NATIVE_WORLD_CANDIDATE_INCOMPLETE')
        baseline_model,_,_ = checkpoint.start(data)
        baseline,candidate = data.evaluate(baseline_model),data.evaluate(model)
        result = decision(data,baseline,candidate,model)
        result = tool_retention.assess(data,result,candidate)
        result = termination_gate(data,baseline,candidate,result)
        result = rehearsal_retention.assess(data,candidate,result)
        payload = dict(baseline=baseline,candidate=candidate,result=result,
                       labels={k:e['region'] for k,e in data.examples.items()})
        body = dict(kind='native_world_candidate',candidate=name,pin=pin,parent=parent,
            train=train,evaluate=evaluate,continuation=continuation,binding=digest(data.binding),
            code=checkpoint.code_pins(),result=result,prior_head=self.head,payload=sha(canonical(payload)))
        if expected is None:
            self.journal.append(body,payload)
        else:
            check(body == expected and payload == self.journal.payload(expected['payload']),
                  'NATIVE_WORLD_DECISION_REPLAY')
        self.decisions.append(body)
        return body

    def _replay(self, body):
        if body.get('kind') == 'native_world_candidate':
            check(body['code'] == checkpoint.code_pins(), 'NATIVE_WORLD_CODE_PIN')
            self.consider_native(body['candidate'],pin=body['pin'],train=body['train'],evaluate=body['evaluate'],
                                 continuation=body['continuation'],expected=body)
        else:
            super()._replay(body)


def increment(session, root, *, name, train, evaluate, continuation):
    started = time.perf_counter()
    data = NativeData(session,train=train,evaluate=evaluate,continuation=continuation)
    model,opt,migration = checkpoint.start(data)
    parent = session.decisions[-1]['pin']
    before = data.evaluate(model)
    previous = session.journal.payload(session.decisions[-1]['payload'])['candidate']
    check(all(before[k] == previous[k] for k in data.keys), 'NATIVE_WORLD_MIGRATION_BEHAVIOR')
    original = tensor_fingerprint(dict(model.named_parameters()))
    checkpoint.save(root/(name+'-start'),model,opt,data,step=0,parent=parent)
    phases = dict(bind_migrate_and_initial_readback=time.perf_counter()-started)

    def step(model,opt):
        opt.zero_grad(set_to_none=True)
        protected = rehearsal_retention.prepare(data,model)
        loss,details = data.loss(model)
        check(bool(torch.isfinite(loss)), 'NATIVE_WORLD_NONFINITE_LOSS')
        loss.backward()
        gradients = [n for n,p in trainable(model) if p.grad is not None and bool((p.grad != 0).any())]
        check(gradients and all(n.startswith(('query.','place.')) for n in gradients), 'NATIVE_WORLD_GRADIENT_SCOPE')
        torch.nn.utils.clip_grad_norm_([p for _,p in trainable(model)],1.)
        before = {n:p.detach().clone() for n,p in trainable(model)}
        opt.step()
        retained = rehearsal_retention.apply(data,model,before,protected)
        return dict(loss=float(loss.detach()),details=details,gradients=gradients,update=retained)

    tick = time.perf_counter()
    losses = []
    for cursor in range(1,data.plan['steps']+1):
        losses.append(step(model,opt))
        if cursor == 4:
            mid = checkpoint.save(root/(name+'-mid'),model,opt,data,step=cursor,parent=parent)
    phases['fixed_training'] = time.perf_counter()-tick
    tick = time.perf_counter()
    recovered,recovered_opt,_ = checkpoint.load(root/(name+'-mid'),data,expected=mid)
    for cursor in range(4,data.plan['steps']):
        check(step(recovered,recovered_opt) == losses[cursor], 'NATIVE_WORLD_OPTIMIZER_REPLAY')
    check(fingerprint(model,opt) == fingerprint(recovered,recovered_opt), 'NATIVE_WORLD_OPTIMIZER_RECOVERY')
    phases['optimizer_recovery'] = time.perf_counter()-tick
    tick = time.perf_counter()
    after = data.evaluate(model)
    now = tensor_fingerprint(dict(model.named_parameters()))
    changed = [n for n in original if original[n] != now[n]]
    check(all(n.startswith(('query.','place.')) for n in changed), 'NATIVE_WORLD_FROZEN_DRIFT')
    pin = checkpoint.save(candidate_path(session,name),model,opt,data,step=data.plan['steps'],parent=parent)
    selected = session.consider_native(name,pin=pin,train=train,evaluate=evaluate,continuation=continuation)
    phases['candidate_readback_and_decision'] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    return dict(complete=True,binding=data.binding,before=before,after=after,losses=losses,
        migration=migration,changed=changed,decision=selected,exact_optimizer_recovery=True,
        parameter_count=sum(p.numel() for p in model.parameters()),
        trainable_count=sum(p.numel() for _,p in trainable(model)),
        phases=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
