"""Bind retained field provenance to native windows without creating new senses."""
import torch

from claims.codata_source import check
from claims.episodic_memory import canonical, digest

ROLES = ('action', 'program', 'wheel_filename')


class FieldBatch(dict):
    def __init__(self, slots, description):
        super().__init__(slots)
        self.description = description
        self.pin = digest(description)

    def checked(self, specs):
        check(digest(self.description) == self.pin, 'FIELD_NATIVE_BINDING')
        fields = self.description['fields']
        check(set(fields) == {s.name for s in specs}, 'FIELD_NATIVE_INSTRUMENTS')
        for spec in specs:
            cursor = 0
            counts = dict.fromkeys(ROLES, 0)
            for row in fields[spec.name]:
                role, start, end = row['role'], row['start'], row['end']
                check(role in ROLES and type(start) is int and type(end) is int
                      and start == cursor < end, 'FIELD_NATIVE_SPAN')
                counts[role] += 1
                cursor = end
            slot = self[spec.name]
            check(slot.values.shape[0] == 1 and not bool(slot.padding.any())
                  and cursor == slot.values.shape[1] and counts['action'] == counts['program'] == 1
                  and counts['wheel_filename'] > 0, 'FIELD_NATIVE_COVERAGE')
        return fields

    def role(self, name, role):
        return torch.cat([self[name].values[:,r['start']:r['end']] for r in self.description['fields'][name]
                          if r['role'] == role],dim=1)


def bind(data, key):
    session, e = data.session, data.examples[key]
    request = e['request']
    check('execution_conditions' in request, 'FIELD_NATIVE_CONDITIONS_REQUIRED')
    binding = request['execution_conditions']['native']
    provider = session.condition_inputs.for_condition(binding['condition'])
    check(binding == provider.checked(binding['condition']), 'FIELD_NATIVE_SOURCE')
    action = session.inputs.batch([request['reference']['unit']],data.model.config,session.device)
    extra = provider.batch([binding['unit']],data.model.config,session.device)
    fields = {}
    for spec in data.model.specs:
        a, b, native = action[spec.name], extra[spec.name], e['native'][spec.name]
        av, bv = a.values[:,~a.padding[0]], b.values[:,~b.padding[0]]
        check(torch.equal(native.values,torch.cat((av,bv),dim=1)), 'FIELD_NATIVE_VALUE_JOIN')
        count = av.shape[1]
        rows = [dict(role='action',start=0,end=count,text_sha256=request['reference']['text_sha256'])]
        for r in binding['fields'][spec.name]:
            check(r['field'] in ROLES[1:], 'FIELD_NATIVE_ROLE')
            rows.append(dict(role=r['field'],start=count+r['window_start'],end=count+r['window_end'],
                             text_sha256=r['text_sha256']))
        fields[spec.name] = rows
    description = dict(schema='attested-native-fields-v1',request=digest(request),condition=binding,
                       fields=fields,role_order=list(ROLES),numerical_identifiers=False)
    batch = FieldBatch(e['native'],description)
    batch.checked(data.model.specs)
    # Retain an immutable canonical descriptor in the training binding; no source I/O per update.
    check(canonical(description) == canonical(batch.description), 'FIELD_NATIVE_BINDING')
    return batch,description
