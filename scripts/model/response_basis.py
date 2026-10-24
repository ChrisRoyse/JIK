"""Pinned retained activations, separately typed by instrument and neural branch."""
from contextlib import AbstractContextManager
import json
from pathlib import Path

import numpy as np
import torch

from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from model.loop_training_checkpoint import tensor_fingerprint
from model import rehearsal_retention

RULE = 'retained_branch_nullspace_v1'


def modules(model):
    return {f'query.{branch}.{s.name}.up':model.query[branch][s.name].up
            for branch in ('_native', '_fields') for s in model.specs}


def invariant(model, names):
    return digest(tensor_fingerprint({n:p for n,p in model.named_parameters() if n not in names}))


class Capture(AbstractContextManager):
    """Observe the actual scoring path; never substitute responses into the model."""
    def __init__(self, data, model):
        self.data, self.model = data, model
        self.rows = {n:dict(h=[], y=[], calls=[]) for n in modules(model)}
        self.handles, self.key = [], None

    def __enter__(self):
        self.original = self.data.forward
        self.override = self.data.__dict__.get('forward')
        def forward(model, key):
            check(key in self.data.retained_keys, 'RESPONSE_CAPTURE_SPLIT')
            self.key = key
            try:
                return self.original(model, key)
            finally:
                self.key = None
        self.data.forward = forward
        for name, module in modules(self.model).items():
            def hook(_module, inputs, output, *, name=name):
                check(self.key is not None and len(inputs) == 1, 'RESPONSE_CAPTURE_CONTEXT')
                h, y = inputs[0].detach(), output.detach()
                check(h.shape[:-1] == y.shape[:-1], 'RESPONSE_CAPTURE_SHAPE')
                row = self.rows[name]
                start = sum(len(v) for v in row['h'])
                h = h.reshape(-1, h.shape[-1]).cpu().numpy().copy()
                y = y.reshape(-1, y.shape[-1]).cpu().numpy().copy()
                row['calls'].append(dict(key=self.key, shape=list(inputs[0].shape),
                                         start=start, stop=start+len(h)))
                row['h'].append(h)
                row['y'].append(y)
            self.handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, *_args):
        for handle in self.handles:
            handle.remove()
        if self.override is None:
            del self.data.__dict__['forward']
        else:
            self.data.forward = self.override

    def collected(self):
        return {n:dict(h=np.concatenate(v['h']), y=np.concatenate(v['y']), calls=v['calls'])
                for n,v in self.rows.items()}


def capture_scores(data, model):
    with Capture(data, model) as capture:
        scores = rehearsal_retention.scores(data, model)
    return scores, capture.collected()


def validate(manifest, arrays):
    check(manifest['rule'] == RULE, 'RESPONSE_BASIS_RULE')
    expected = set()
    for i, row in enumerate(manifest['branches']):
        names = [f'{prefix}_{i}' for prefix in ('h','y','p','s')]
        expected.update(names)
        h, y, p, s = (arrays[n] for n in names)
        rows, width = row['rows'], row['shape'][1]
        check(h.shape == (rows,width) and y.shape == (rows,row['shape'][0])
              and p.shape == (width,width) and s.shape == (min(rows,width),), 'RESPONSE_BASIS_ROWS')
        check(all(np.isfinite(v).all() for v in (h,y,p,s)), 'RESPONSE_BASIS_FINITE')
        check(h.dtype == y.dtype == np.float32 and p.dtype == s.dtype == np.float64,
              'RESPONSE_BASIS_DTYPE')
        offset, keys = 0, []
        for call in row['calls']:
            check(call['start'] == offset and call['stop'] > offset
                  and np.prod(call['shape'][:-1]) == call['stop']-offset
                  and call['shape'][-1] == width, 'RESPONSE_BASIS_COVERAGE')
            offset = call['stop']
            keys.append(call['key'])
        check(offset == rows and set(keys) == set(manifest['retained_keys']), 'RESPONSE_BASIS_COVERAGE')
        tolerance = 64*np.finfo(np.float64).eps*max(rows,width)*max(1., np.linalg.norm(h,ord=2))
        rank_tolerance = max(rows,width)*np.finfo(np.float64).eps*s[0]
        check(row['rank'] == int(np.sum(s > rank_tolerance)) and row['rank_tolerance'] == rank_tolerance,
              'RESPONSE_BASIS_RANK')
        check(np.max(np.abs(p-p.T)) <= tolerance and np.max(np.abs(p@p-p)) <= tolerance
              and np.max(np.abs(h.astype(np.float64)@p)) <= tolerance
              and abs(np.trace(p)-(width-row['rank'])) <= tolerance, 'RESPONSE_BASIS_PROJECTOR')
    check(set(arrays) == expected, 'RESPONSE_BASIS_ARRAY_ROSTER')


def build(root, data, model, parent):
    root = Path(root)
    check(not root.exists(), 'RESPONSE_BASIS_EXISTS')
    scores, captured = capture_scores(data, model)
    check(all(r['retained'] for r in rehearsal_retention.comparisons(data,scores)), 'RESPONSE_INITIAL_RETENTION')
    branches, arrays = [], {}
    for i,(name,row) in enumerate(captured.items()):
        h = row['h'].astype(np.float64)
        _, s, vh = np.linalg.svd(h, full_matrices=False)
        tolerance = max(h.shape)*np.finfo(np.float64).eps*s[0]
        rank = int(np.sum(s > tolerance))
        p = np.eye(h.shape[1])-vh[:rank].T@vh[:rank]
        weight = modules(model)[name].weight
        branches.append(dict(name=name, parameter=name+'.weight', shape=list(weight.shape),
            rows=len(h), rank=rank, rank_tolerance=tolerance, calls=row['calls']))
        arrays.update({f'h_{i}':row['h'], f'y_{i}':row['y'], f'p_{i}':p, f's_{i}':s})
    eligible = [r['parameter'] for r in branches]
    manifest = dict(rule=RULE, parent=parent, parent_binding=digest(data.binding),
        retained_keys=data.retained_keys, branches=branches, noneligible=invariant(model,eligible),
        scores=scores, scope='training_activations_not_sensory_pooling', scientific_admission=False)
    validate(manifest, arrays)
    root.mkdir()
    np.savez(root/'arrays.npz', **arrays)
    raw = (root/'arrays.npz').read_bytes()
    manifest['arrays'] = dict(bytes=len(raw), sha256=sha(raw))
    write_committed(root/'manifest.json', manifest)
    return dict(root=str(root), pin=sha((root/'manifest.json').read_bytes()))


class Basis:
    def __init__(self, spec, parent, binding):
        root = Path(spec['root'])
        raw = (root/'manifest.json').read_bytes()
        check(sha(raw) == spec['pin'], 'RESPONSE_BASIS_PIN')
        self.manifest = json.loads(raw)
        check(self.manifest['parent'] == parent and self.manifest['parent_binding'] == binding,
              'RESPONSE_BASIS_PARENT')
        raw = (root/'arrays.npz').read_bytes()
        check(dict(bytes=len(raw),sha256=sha(raw)) == self.manifest['arrays'], 'RESPONSE_BASIS_BYTES')
        with np.load(root/'arrays.npz',allow_pickle=False) as archive:
            self.arrays = {n:archive[n].copy() for n in archive.files}
        validate(self.manifest,self.arrays)
        self.names = [r['parameter'] for r in self.manifest['branches']]
        self.projectors = {}

    def projectors_on(self, device):
        if not self.projectors:
            self.projectors = {r['parameter']:torch.from_numpy(self.arrays[f'p_{i}']).to(device)
                               for i,r in enumerate(self.manifest['branches'])}
        check(all(p.device == device for p in self.projectors.values()), 'RESPONSE_BASIS_DEVICE')
        return self.projectors

    def measure(self, data, model):
        check(invariant(model,self.names) == self.manifest['noneligible'], 'RESPONSE_NONELIGIBLE_DRIFT')
        scores, captured = capture_scores(data,model)
        drift = {}
        for i,row in enumerate(self.manifest['branches']):
            actual = captured[row['name']]
            check(actual['calls'] == row['calls'] and np.array_equal(actual['h'],self.arrays[f'h_{i}']),
                  'RESPONSE_FEATURE_DRIFT')
            delta = actual['y'].astype(np.float64)-self.arrays[f'y_{i}']
            drift[row['name']] = dict(max_absolute=float(np.max(np.abs(delta))),
                l2=float(np.linalg.norm(delta)), equal=bool(np.array_equal(actual['y'],self.arrays[f'y_{i}'])))
        return scores, drift
