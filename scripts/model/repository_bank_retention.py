"""Retain prior examples under both frozen and enlarged native reference banks."""
import torch
from torch.nn import functional as F

from claims.codata_source import check
from claims.episodic_memory import digest
from model.repository_retention import Retention as BaseRetention, compare, measurements
from model.repository_retention_curvature import POLICY as PREVIOUS_POLICY, update

POLICY = dict(PREVIOUS_POLICY, rule='repository_bank_retention_v1',
              retained_views=['learn_1', 'learn_2'], migration='compare_enlarged_to_original_predecessor')
VIEWS = ('learn_1', 'learn_2')


def joined(views):
    check(set(views) == set(VIEWS), 'BANK_RETENTION_VIEW_ROSTER')
    check(views['learn_1'].keys() == views['learn_2'].keys(), 'BANK_RETENTION_EXAMPLE_ROSTER')
    return {view+'|'+key:row for view in VIEWS for key,row in views[view].items()}


class BankData:
    """Only the retention read changes; training still uses the full enlarged bank."""
    def __init__(self, base, keys):
        self.base, self.keys = base, tuple(keys)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def views(self, model):
        return {v:{k:rows[k] for k in self.keys}
                for v in VIEWS for rows in (self.base.evaluate(model, v),)}

    def evaluate(self, model, partition, **kwargs):
        if partition == 'learn_1':
            check(not kwargs, 'BANK_RETENTION_INTERVENTION')
            return joined(self.views(model))
        return self.base.evaluate(model, partition, **kwargs)


class Retention(BaseRetention):
    def __init__(self, data, model, references, predecessor):
        keys = data.stage('learn_1', model)['keys']
        wrapped = BankData(data, keys)
        super().__init__(wrapped, model, joined(references), predecessor)
        self.original_ceiling = measurements(references['learn_1'])
        self.contract.update(policy=POLICY, banks=self.banks(model),
                             original_ceiling=self.original_ceiling)
        self.pin = digest(self.contract)

    def banks(self, model):
        return {v:dict(description=self.data.base.stage(v, model)['description'],
                       labels=self.data.base.stage(v, model)['labels'].tolist()) for v in VIEWS}

    def checked(self):
        super().checked()
        check(self.contract['policy'] == POLICY and self.contract['original_ceiling'] == self.original_ceiling,
              'BANK_RETENTION_CONTRACT')
        check(self.contract['banks'] == self.banks(self.data.parent), 'BANK_RETENTION_BANK_CHANGED')

    def migration(self, model):
        self.checked()
        rows = self.data.base.evaluate(model, 'learn_2')
        scores = measurements({k:rows[k] for k in self.keys})
        regressions = compare(self.original_ceiling, scores)
        return dict(original=self.original_ceiling, enlarged=scores, regressions=regressions,
                    retention_passed=not regressions, serving=False,
                    scope='exposed_prior_checks_only_not_scientific_admission')

    def prepare(self, model, parameters):
        self.checked()
        base = self.data.base
        per_view = len(self.keys)*(len(model.specs)+1)+sum(not base.examples[k]['absent']
                    for k in self.keys)*len(model.specs)
        count, coordinates = len(VIEWS)*per_view, sum(p.numel() for p in parameters)
        free, _ = torch.cuda.mem_get_info(model.core.initial.device)
        check(count*coordinates*8*4 < free, 'RETENTION_GRADIENT_MEMORY')
        matrix = torch.empty((count, coordinates), device=base.device, dtype=torch.float64)
        order = []
        for view in VIEWS:
            stage = base.stage(view, model)
            for key in self.keys:
                output = base.forward(model, key, view)
                e = base.examples[key]
                truth = torch.tensor([float(i == e['outcome']) for i in range(3)], device=base.device)
                lanes = dict(output['regions'], common=output['constellation_regions']/len(model.specs))
                terms = []
                for name, score in lanes.items():
                    p = F.softmax(score[0], dim=-1)
                    mass = torch.stack([p[stage['labels'] == i].sum() for i in range(3)])
                    terms.append((name, 'brier', (mass-truth).square().sum()))
                    if name != 'common' and not e['absent']:
                        terms.append((name, 'native_expected_loss', (p*stage['costs'][key][name]).sum()))
                for j, (name, metric, term) in enumerate(terms):
                    gradients = torch.autograd.grad(term, parameters, retain_graph=j+1 < len(terms),
                                                    allow_unused=True)
                    offset = 0
                    for parameter, gradient in zip(parameters, gradients, strict=True):
                        matrix[len(order), offset:offset+parameter.numel()] = (
                            0 if gradient is None else gradient.detach().reshape(-1))
                        offset += parameter.numel()
                    order.append(view+'|'+key+'|'+name+'|'+metric)
        check(len(order) == count and len(set(order)) == count and bool(torch.isfinite(matrix).all()),
              'BANK_RETENTION_GRADIENT_ROSTER')
        return matrix, order
