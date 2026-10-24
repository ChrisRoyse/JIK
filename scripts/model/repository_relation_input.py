"""Source-bound repository roles; every native window and file boundary survives."""
import copy

import torch

from claims.codata_source import check, sha
from claims.episodic_memory import digest
from model.native import NativeInput

ROLES = ('task_report', 'original_source', 'candidate_source', 'recorded_patch')


class RepositoryBatch(dict):
    def __init__(self, slots, description):
        super().__init__(slots)
        self.description = copy.deepcopy(description)
        self.pin = digest(self.description)
        self.seal = self.stamps()

    def stamps(self):
        return {n:tuple((id(t),t.data_ptr(),t._version) for t in (s.values,s.padding,s.absent))
                for n,s in self.items()}

    def checked(self, specs):
        d = self.description
        check(self.stamps() == self.seal, 'RELATION_INPUT_TENSOR_CHANGED')
        check(digest(d) == self.pin and d['source']['side'] == 'input', 'RELATION_INPUT_BINDING')
        check(d['source']['scientific_admission'] is False and
              d['source']['numerical_identifiers'] is False, 'RELATION_INPUT_SCOPE')
        check(set(d['fields']) == set(self) == {s.name for s in specs}, 'RELATION_INPUT_PANEL')
        permitted = ROLES[:2] if d['intervention'] == 'withhold_action_and_candidate' else ROLES
        check(d['intervention'] in (None, 'withhold_action_and_candidate'), 'RELATION_INPUT_INTERVENTION')
        for spec in specs:
            slot, cursor = self[spec.name], 0
            present = {f['role'] for f in d['source']['fields'][spec.name]}
            roles = set(permitted)&present
            check(d['absent_roles'][spec.name] == [r for r in ROLES if r not in present] and
                  'task_report' in present and 'recorded_patch' in present, 'RELATION_INPUT_ROLE_ABSENCE')
            seen = set()
            for row in d['fields'][spec.name]:
                check(row['role'] in roles and row['start'] == cursor <= row['end'], 'RELATION_INPUT_SPAN')
                cursor = row['end']
                seen.add(row['role'])
            check(seen == set(roles) and cursor == slot.values.shape[1] and
                  slot.values.shape[0] == 1 and not bool(slot.padding.any()) and
                  slot.rows == (self.pin,), 'RELATION_INPUT_COVERAGE')
        return self

    def role(self, name, role):
        spans = [self[name].values[:,r['start']:r['end']] for r in self.description['fields'][name]
                 if r['role'] == role]
        check(role in ROLES, 'RELATION_INPUT_ROLE')
        return torch.cat(spans, dim=1) if spans else self[name].values[:,:0,:]


def bind(provider, key, model, expected):
    source = provider.population.fields[key]
    native, desc = provider.observed(key, 'input', source['input_available_by'],
                                    model.config, model.core.initial.device, model.specs)
    check(desc == expected['description'], 'RELATION_INPUT_SOURCE_JOIN')
    for spec in model.specs:
        slot = native.slots[spec.name]
        raw = slot.values[0, ~slot.padding[0]].detach().cpu().contiguous().numpy().astype('<f4')
        check(dict(instrument=spec.identity, shape=list(raw.shape), sha256=sha(raw.tobytes()),
                   dtype='float32-le', absent=False) == expected['slots'][spec.name],
              'RELATION_INPUT_NATIVE_JOIN')
    return assemble(native.slots,desc,source['missing']).checked(model.specs)


def assemble(slots,desc,gaps):
    fields,absent = {},{}
    for name in slots:
        rows,cursor = [],0
        for f in desc['fields'][name]:
            start = cursor
            for w in f['windows']:
                check(w['window_start'] == cursor < w['window_end'], 'RELATION_INPUT_WINDOW_JOIN')
                cursor = w['window_end']
            rows.append(dict(role=f['role'], start=start, end=cursor, source=f))
        fields[name] = rows
        absent[name] = [r for r in ROLES if r not in {f['role'] for f in rows}]
    description = dict(schema='repository-relation-input-v1', source=desc, fields=fields,
                       absent_roles=absent, source_gaps=copy.deepcopy(gaps), intervention=None)
    return make(slots, description)


def make(slots, description):
    pin = digest(description)
    rebound = {n:NativeInput(s.identity, (pin,), s.values, s.padding, s.absent) for n,s in slots.items()}
    return RepositoryBatch(rebound, description)


def withhold_action(batch, specs):
    """An explicit evaluation intervention, never an attested replacement input."""
    batch.checked(specs)
    check(batch.description['intervention'] is None, 'RELATION_INPUT_DOUBLE_INTERVENTION')
    desc, slots = copy.deepcopy(batch.description), {}
    desc['intervention'] = 'withhold_action_and_candidate'
    for spec in specs:
        rows, pieces, cursor = [], [], 0
        for row in batch.description['fields'][spec.name]:
            if row['role'] not in ROLES[:2]:
                continue
            piece = batch[spec.name].values[:,row['start']:row['end']]
            rows.append(dict(row, start=cursor, end=cursor+piece.shape[1]))
            pieces.append(piece)
            cursor += piece.shape[1]
        desc['fields'][spec.name] = rows
        values = torch.cat(pieces, dim=1)
        slots[spec.name] = NativeInput(spec.identity, ('intervention',), values,
            torch.zeros(values.shape[:2], dtype=torch.bool, device=values.device),
            torch.zeros(1, dtype=torch.bool, device=values.device))
    return make(slots, desc).checked(specs)
