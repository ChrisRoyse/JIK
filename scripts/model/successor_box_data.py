"""Bind the unchanged learning corpus to an independently read numerical repair."""
import copy
import json
from pathlib import Path

from claims.codata_source import check,sha
from claims.episodic_memory import digest
from model.successor_learning import SuccessorData


def code_pins():
    return {n:sha((Path(__file__).parent/n).read_bytes()) for n in
            ('successor_box_data.py','successor_box_session.py','box_correction.py',
             'box_region_solve.py','rowspace_region_solve.py')}


class BoxData(SuccessorData):
    def __init__(self,session,base,spec):
        recipe = copy.deepcopy(spec)
        numeric = recipe.pop('numerical')
        super().__init__(session,base,recipe)
        root = Path(numeric['root'])
        raw,reader_raw = (root/'report.json').read_bytes(),(root/'readback.json').read_bytes()
        check(sha(raw) == numeric['pin'] and sha(reader_raw) == numeric['readback_pin'],'BOX_DATA_PIN')
        report,reader = json.loads(raw),json.loads(reader_raw)
        source = Path(report['source'])/'learning.json'
        source_raw = source.read_bytes()
        check(report['all_certified'] and reader['all_certified'] and
              reader['report'] == dict(bytes=len(raw),sha256=sha(raw)) and
              report['learning'] == dict(bytes=len(source_raw),sha256=sha(source_raw)) and
              json.loads(source_raw)['binding'] == self.binding,'BOX_DATA_SOURCE')
        for name,pin in report['code'].items():
            check(sha((Path(__file__).parents[1]/name).read_bytes()) == pin,'BOX_DATA_CODE')
        check(all(row['passed'] for row in reader['lanes'].values()) and
              set(reader['lanes']) == {s.name for s in self.model.specs},'BOX_DATA_CERTIFICATES')
        self.binding['successor_numerical_repair'] = dict(source=numeric,code=code_pins(),
            unchanged_corpus_binding=digest(self.binding),tolerances_unchanged=True,
            scope='numerical_repair_not_capability_or_sensory_admission')
        self.binding_pin = digest(self.binding)
        self.fresh()
