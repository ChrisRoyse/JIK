"""Fixed per-instrument retention for newly grounded feedback and earlier checks."""
import copy
import json
from pathlib import Path

from claims.codata_source import check, sha
from claims.episodic_memory import digest
from model.controlled_learning import ControlledData, code_pins as previous_code
from model.identity_loop import reference_hash
from model import rehearsal_retention, retained_update

RULE = 'grounded_correction_all_training_retention_and_objective_descent_v1'


def code_pins():
    return dict(previous_code(), **{'model/'+n:sha((Path(__file__).parent/n).read_bytes()) for n in
        ('corrective_learning.py','corrective_session.py','corrective_update.py','corrective_step.py',
         'corrective_projection.py')})


class CorrectiveData(ControlledData):
    def __init__(self, session, spec):
        check(not session.controlled_candidates, 'CORRECTION_FAILED_BRANCH_NOT_ROLLED_BACK')
        super().__init__(session, spec['base'])
        self.correction_spec = copy.deepcopy(spec)
        source = Path(spec['source']['root'])
        raw = (source/'seal.json').read_bytes()
        check(sha(raw) == spec['source']['seal'], 'CORRECTION_SOURCE_SEAL')
        seal = json.loads(raw)
        learning_raw = (source/'learning.json').read_bytes()
        check(sha(learning_raw) == seal['files']['learning.json']['sha256'], 'CORRECTION_BASELINE_BYTES')
        saved = json.loads(learning_raw)
        equivalent = copy.deepcopy(self.binding)
        if session.corrective_candidates:
            # An appended decision changes provenance IDs, not the reference readings.
            check(equivalent['head'] == session.head, 'CORRECTION_CURRENT_HEAD')
            equivalent['head'] = saved['binding']['head']
            for key, view in equivalent['training_views'].items():
                prior_view = saved['binding']['training_views'][key]
                check(view['definition']['rehearsal']['head'] == session.head, 'CORRECTION_VIEW_HEAD')
                view['definition']['rehearsal']['head'] = prior_view['definition']['rehearsal']['head']
                native = self.examples[key]['view']
                check(reference_hash(native.specs,native.references(),digest(view['definition'])) == prior_view['world'],
                      'CORRECTION_REFERENCE_CONTINUITY')
                view['world'] = prior_view['world']
        check(saved['decision']['spec'] == spec['base'] and saved['binding'] == equivalent,
              'CORRECTION_BASELINE_BINDING')
        check(not saved['decision']['result']['selected'], 'CORRECTION_BRANCH_REASON')
        path = source/'candidates/controlled-1-start'
        raw = (path/'manifest.json').read_bytes()
        check(sha(raw) == saved['start_pin'], 'CORRECTION_ROLLBACK_PIN')
        self.field_predecessor = (path, saved['start_pin'], json.loads(raw))
        self.parent_pin = saved['start_pin']
        original = copy.deepcopy(self.binding['grounded_retention'])
        ceilings = copy.deepcopy(original['ceilings'])
        for key in self.controlled:
            check(key in self.plan['train'] and key not in self.plan['evaluate'], 'CORRECTION_FRESH_SPLIT')
            ceilings[key] = rehearsal_retention.measurements(saved['before'][key])
        self.baseline_scores = saved['before']
        predecessor = None
        if session.corrective_candidates:
            prior = next(reversed(session.corrective_candidates.values()))
            check(prior['spec'] == spec and prior['code'] == code_pins(), 'CORRECTION_PREDECESSOR')
            path = session.controlled_root/prior['candidate']
            raw = (path/'manifest.json').read_bytes()
            check(sha(raw) == prior['pin'], 'CORRECTION_PREDECESSOR_PIN')
            manifest = json.loads(raw)
            previous = session.journal.payload(prior['payload'])['candidate']
            for key in ceilings:
                values = rehearsal_retention.measurements(previous[key])
                ceilings[key] = {kind:{n:min(v,values[kind][n]) for n,v in row.items()}
                                 for kind,row in ceilings[key].items()}
            self.baseline_scores = previous
            self.field_predecessor = (path, prior['pin'], manifest)
            self.parent_pin = prior['pin']
            self.binding['native_world']['offsets'] = {n:v+manifest['step'] for n,v in manifest['offsets'].items()}
            predecessor = dict(candidate=prior['candidate'], pin=prior['pin'], payload=prior['payload'])
        self.retained_keys = list(ceilings)
        check(len(self.retained_keys) <= retained_update.MAX_RETAINED, 'CORRECTION_RETENTION_BUDGET')
        self.retained_ceilings = {k:v['native'] for k,v in ceilings.items()}
        self.binding['training_retention'].update(keys=self.retained_keys, ceilings=self.retained_ceilings,
                                                 scope='earlier_and_fresh_grounded_training_only')
        self.binding['grounded_retention'].update(keys=self.retained_keys, ceilings=ceilings,
                                                  scope='earlier_and_fresh_grounded_training_only')
        self.grounded_retention_pin = digest(self.binding['grounded_retention'])
        self.binding['corrective_feedback'] = dict(rule=RULE, spec=spec, code=code_pins(),
            rollback=saved['start_pin'], rejected_candidate=saved['candidate_pin'],
            original_retention=original, fixed_fresh_baseline=digest(saved['before']),
            predecessor=predecessor, scope='known_training_observations_only')
        self.correction_pin = digest(dict(retention=self.binding['grounded_retention'],
                                         correction=self.binding['corrective_feedback']))
        self.fresh()

    def fresh(self):
        super().fresh()
        if 'correction_pin' in self.__dict__:
            check(digest(dict(retention=self.binding['grounded_retention'],
                              correction=self.binding['corrective_feedback'])) == self.correction_pin,
                  'CORRECTION_FIXED_CEILINGS_CHANGED')
            rehearsal_retention.validate(self)
