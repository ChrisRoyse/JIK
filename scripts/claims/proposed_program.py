"""A recorded proposal is a source for a new run, never a historical outcome."""
import json
from pathlib import Path

from claims.action_transfer_edit_source import code_pins, syntax
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest


FILES = {'policy.json', 'syntax-exclusions.json', 'candidates.json',
         'comparisons.json', 'selected.json', 'action.txt', 'program.py'}


def validate(row, action, program):
    check(isinstance(action, bytes) and isinstance(program, bytes), 'PROPOSAL_BYTES')
    span = row['utf8_span']
    check(isinstance(span, list) and len(span) == 2
          and all(type(i) is int for i in span) and 0 <= span[0] < span[1] <= len(action)
          and action[span[0]:span[1]] == program, 'PROPOSAL_SPAN')
    check(len(action) == row['action_bytes'] and sha(action) == row['action_sha256']
          and len(program) == row['program_bytes'] and sha(program) == row['program_sha256']
          and syntax(program) == row['syntax_sha256'], 'PROPOSAL_CONTENT')
    check(row['role'] == 'generated_proposal'
          and row['historical_occurrence_established'] is False
          and row['historical_file_state_established'] is False, 'PROPOSAL_SCOPE')


class ProposedProgram:
    def __init__(self, root, pin):
        self.root, self.pin = Path(root), pin
        raw = (self.root/'report.json').read_bytes()
        check(sha(raw) == pin == (self.root/'COMMITTED').read_text(), 'PROPOSAL_PIN')
        report = json.loads(raw)
        check(raw == canonical(report)+b'\n' and report['schema'] == 'action-transfer-edit-source-v1'
              and report['status'] == 'selected' and report['code'] == code_pins(), 'PROPOSAL_CONTRACT')
        check(all(report[k] is False for k in ('scientific_admission', 'serving', 'executed',
              'forecast_present', 'historical_evaluation_untouched', 'dependency_closure_exposure_complete')),
              'PROPOSAL_SCOPE')
        check(set(report['files']) == FILES, 'PROPOSAL_FILES')
        payloads = {}
        for name, item in report['files'].items():
            path = self.root/name
            check(not path.is_symlink(), 'PROPOSAL_SYMLINK')
            value = path.read_bytes()
            check(len(value) == item['bytes'] and sha(value) == item['sha256'], 'PROPOSAL_FILE_PIN')
            payloads[name] = value
        row = json.loads(payloads['selected.json'])
        check(row == report['selected'], 'PROPOSAL_SELECTION')
        validate(row, payloads['action.txt'], payloads['program.py'])
        self.report, self.row = report, row
        self.action, self.program = payloads['action.txt'], payloads['program.py']

    def reference(self):
        body = dict(kind='recorded_program_proposal', source=self.pin,
            revision=self.row['revision'], shard=self.row['shard'], row=self.row['row'],
            turn=self.row['turn'], task=self.row['task'], raw_shard=self.row['source_sha256'],
            action_sha256=sha(self.action), program_sha256=sha(self.program),
            program_utf8_span=self.row['utf8_span'], historical_occurrence_established=False)
        return dict(id=digest(body), **body)
