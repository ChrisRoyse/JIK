"""Diagnostic-only resource substitutions; never a recorded occurrence or training input."""
import copy
import torch

from claims.codata_source import check
from claims.episodic_memory import digest
from model.field_native import FieldBatch
from model.native import NativeInput
from model.loop_training_checkpoint import tensor_fingerprint
from model import process_anchor


def substitution(data,model,keys):
    before = digest(tensor_fingerprint(model.state_dict()))
    result = {}
    with torch.no_grad():
        for key,donor in zip(keys,reversed(keys)):
            original,other = data.examples[key]['native'],data.examples[donor]['native']
            slots,fields = {},{}
            for spec in model.specs:
                name = spec.name
                parts = [original.role(name,'action'),original.role(name,'program'),other.role(name,'wheel_filename')]
                values = torch.cat(parts,dim=1)
                slot = original[name]
                slots[name] = NativeInput(slot.identity,slot.rows,values,
                    torch.zeros(values.shape[:2],device=values.device,dtype=torch.bool),slot.absent)
                cursor,rows = 0,[]
                for source,roles in ((original,('action','program')),(other,('wheel_filename',))):
                    for row in source.description['fields'][name]:
                        if row['role'] in roles:
                            count = row['end']-row['start']
                            rows.append(dict(row,start=cursor,end=cursor+count))
                            cursor += count
                fields[name] = rows
            description = copy.deepcopy(original.description)
            description.update(fields=fields,diagnostic_only=True,resource_donor=donor)
            changed = FieldBatch(slots,description)
            changed.checked(model.specs)
            e = data.examples[key]
            view = data.prediction_model(model,key)
            request = e['request']
            output = view(changed,(view.bind_state(e['state']),),allowed=e['allowed'],phase=2,
                event_ids=[digest(request)],evidence_origin=[request['origin']],history=view.context_history)
            result[key] = dict(donor=donor,description=description,
                prediction=process_anchor.predict(output,view.execution_anchor_bank,e['allowed']))
    check(digest(tensor_fingerprint(model.state_dict())) == before,'FIELD_CONTROL_MUTATION')
    return dict(scope='diagnostic_counterfactual_not_observed_or_training',rows=result,parameters_unchanged=True)
