"""Bind the frozen-parent representation to its measured zero-update migration."""
import copy
import json
from pathlib import Path

from claims.codata_source import check,sha
from claims.episodic_memory import digest
from model.successor_box_data import BoxData
from model import successor_residual


def code_pins():
    return {n:sha((Path(__file__).parent/n).read_bytes()) for n in
            ('successor_residual.py','successor_residual_data.py','successor_residual_checkpoint.py',
             'successor_residual_training.py','successor_residual_session.py')}


class ResidualData(BoxData):
    def __init__(self,session,base,spec):
        recipe = copy.deepcopy(spec)
        migration = recipe.pop('residual')
        super().__init__(session,base,recipe)
        raw = Path(migration['path']).read_bytes()
        check(sha(raw) == migration['pin'],'SUCCESSOR_RESIDUAL_MIGRATION_PIN')
        measured = json.loads(raw)
        check(measured['rule'] == successor_residual.RULE and measured['code'] == code_pins()
              and measured['parent'] == self.parent_pin and measured['basis'] == recipe['basis']
              and measured['parent_binding'] == digest(self.binding)
              and measured['before'] == measured['after'] and measured['parameter_identity_retained']
              and all(v['equal'] for v in measured['response_drift'].values()),
              'SUCCESSOR_RESIDUAL_MIGRATION')
        self.binding['successor_residual'] = dict(source=migration,code=code_pins(),
            rule=successor_residual.RULE,static_buffers=measured['static_buffers'],
            parent_binding=digest(self.binding),zero_update_behaviour=digest(measured['before']))
        self.binding_pin = digest(self.binding)
        self.fresh()
