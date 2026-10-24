"""Independent source-byte and source-clock reconstruction for the experience delta."""
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

from audit.capability_source_readback import SourceReadback, NS
from claims.capability_population import need
from text.evaluation_multiscale import file_sha


class RevisionReadback(SourceReadback):
    def __init__(self, population):
        source = population.report['source']
        self.is_ami = source['source_kind'] == 'ami-chronological-continuation-v1'
        if self.is_ami:
            super().__init__(population, Path('/zfs/archive/mnlm/raw'))
            self.clock_index = {}
            for meeting, _, _ in source['windows']:
                values = []
                for speaker in self.meetings[meeting]:
                    member = 'dialogueActs/' + meeting + '.' + speaker.attrib['nxt_agent'] + '.dialog-act.xml'
                    for act in self.root(member):
                        words = [w for c in act.findall(NS + 'child')
                                 for w in self.linked(c.attrib['href'], 'words') if w.tag == 'w' and w.text]
                        if not words or any('starttime' not in w.attrib or 'endtime' not in w.attrib for w in words):
                            continue
                        values.append((act.attrib[NS + 'id'],
                            min(Fraction(w.attrib['starttime']) for w in words),
                            max(Fraction(w.attrib['endtime']) for w in words)))
                self.clock_index[meeting] = sorted(values, key=lambda v: (v[2], v[1], v[0]))
        else:
            self.pop, self.pins = population, {}
            root = Path(source['root'])
            for name, pin in source['files'].items():
                need(file_sha(root / name) == pin, 'READBACK_REVISION_FILE')
                self.pins[str(root / name)] = pin
            self.data = [json.loads(line) for line in (root / 'train.json').read_bytes().splitlines()]

    def field(self, f):
        if self.is_ami:
            return super().field(f)
        s = f['source']
        need(0 <= s['row'] < len(self.data), 'READBACK_REVISION_ROW')
        row = self.data[s['row']]
        need(row[s['field']] == f['text'], 'READBACK_SOURCE_TEXT')
        need(s['sample_id'] == row['SampleID'] and s['passage_id'] == row['PassageID']
             and s['question_id'] == row['QuestionID'] and s['edit_id'] == row['PassageEditID'],
             'READBACK_REVISION_ADDRESS')

    def task(self, t):
        p = t['provenance']
        fields = self.pop.fields
        context, targets = ([fields[k] for k in t[name]] for name in ('context', 'targets'))
        if self.is_ami:
            cutoff = Fraction(p['cutoff'])
            target = targets[0]['source']['element']
            rows = self.clock_index[p['meeting']]
            ended = [v for v in rows if v[2] <= cutoff]
            expected = [v for v in ended[-8:] if v[0] != target]
            expected.sort(key=lambda v: (v[1], v[2], v[0]))
            need([f['source']['element'] for f in context] == [v[0] for v in expected],
                 'READBACK_CONTINUATION_PREFIX')
            lo, hi = next((lo, hi) for m, lo, hi in self.pop.report['source']['windows'] if m == p['meeting'])
            need(lo <= cutoff < hi and p['previous_complete_acts_available'] == len(ended),
                 'READBACK_CONTINUATION_CUTOFF')
            need(t['split_group'] == 'ami-series:' + re.sub('[a-d]$', '', p['meeting']), 'READBACK_GROUP')
        else:
            old, new = self.data[p['original_row']], self.data[p['row']]
            need(old['PassageEditID'] == 0 and old['PassageID'] == new['PassageID']
                 and old['QuestionID'] == new['QuestionID'] and old['sentence2'] == new['sentence2'],
                 'READBACK_COUNTERFACTUAL_PAIR')
            expected = [new['sentence1'], new['sentence2']]
            if t['kind'] == 'revise_with_source_evidence':
                expected = [old['sentence1'], old['sentence2'], old['label'], new['sentence1']]
                need(new['PassageEditID'] in (1, 2, 3), 'READBACK_COUNTERFACTUAL_DIRECTION')
            need([f['text'] for f in context] == expected and [f['text'] for f in targets] == [new['label']],
                 'READBACK_SOURCE_TASK')
            need(t['split_group'] == 'condaqa:' + self.pop.report['source']['files']['train.json']
                 + ':passage:' + str(new['PassageID']), 'READBACK_GROUP')

    def run(self):
        result = super().run()
        source = self.pop.report['source']
        if self.is_ami:
            expected = {(meeting, v[0]) for meeting, lo, hi in source['windows']
                        for v in self.clock_index[meeting] if lo <= v[1] < hi}
            actual = {(t['provenance']['meeting'], t['provenance']['target_acts'][0])
                      for t in self.pop.tasks.values()}
            need(actual == expected, 'READBACK_TARGET_CENSUS')
            result['target_census'] = len(actual)
        else:
            rows = [(i, r) for i, r in enumerate(self.data) if r['PassageID'] in source['selected_passages']]
            if self.pop.family == 'reasoning':
                rows = [(i, r) for i, r in rows if r['PassageEditID'] != 0]
            need({t['provenance']['row'] for t in self.pop.tasks.values()} == {i for i, _ in rows},
                 'READBACK_TARGET_CENSUS')
            result['target_census'] = len(rows)
            result['target_labels'] = dict(Counter(r['label'] for _, r in rows))
        result.update(independent_entailment_checked=False, learned_capability=False)
        return result
