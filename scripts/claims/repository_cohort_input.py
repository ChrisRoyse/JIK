"""Fail-closed reader for reserved project partitions and original recorded patches."""
import copy
import json
from pathlib import Path

from claims.codata_source import check,sha
from claims.repository_cohort import POLICY


def partition_check(cases,exposure):
    check(len(cases) == POLICY['projects'] and
          [c['index'] for c in cases] == list(range(len(cases))) and
          [c['partition'] for c in cases] == POLICY['roles'],'COHORT_PARTITION_ORDER')
    projects = [c['repo'].casefold() for c in cases]
    check(len(set(projects)) == len(projects) and
          not set(projects)&set(exposure['projects']),'COHORT_PROJECT_SEPARATION')
    check(len({c['task'] for c in cases}) == len(cases),'COHORT_TASK_SEPARATION')


class RepositoryCohort:
    def __init__(self,root,pin):
        self.root,self.pin = Path(root),pin
        raw = (self.root/'report.json').read_bytes()
        r = json.loads(raw)
        check(sha(raw) == pin == (self.root/'COMMITTED').read_text() and
              r['schema'] == 'repository-learning-cohort-v1' and r['policy'] == POLICY and
              r['code'] == sha((Path(__file__).parent/'repository_cohort.py').read_bytes()),
              'COHORT_PIN')
        for name,entry in r['files'].items():
            check(not Path(name).is_absolute() and '..' not in Path(name).parts and '\\' not in name,
                  'COHORT_PATH')
            p = self.root/name
            check(not p.is_symlink() and p.resolve().is_relative_to(self.root.resolve()),'COHORT_PATH')
            value = p.read_bytes()
            check(dict(bytes=len(value),sha256=sha(value)) == entry,'COHORT_FILE')
        self.cases = json.loads((self.root/'cases.json').read_bytes())
        self.exposure = json.loads((self.root/'exposure.json').read_bytes())
        self.config = json.loads((self.root/'config.json').read_bytes())
        partition_check(self.cases,self.exposure)
        check(set(r['files']) == {'cases.json','config.json','exposure.json'}|
              {a['path'] for c in self.cases for a in c['attempts']},'COHORT_FILE_ROSTER')
        self.report = r

    def training_cases(self,partition):
        check(partition in ('learn_1','learn_2'),'COHORT_TRAINING_PARTITION')
        return copy.deepcopy([c for c in self.cases if c['partition'] == partition])

    def evaluation_cases(self):
        check(False,'COHORT_EVALUATION_EXPOSURE_AUDIT_REQUIRED')
