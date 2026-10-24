"""Reconstruct capability surfaces from their original records, independently of adapters."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

from claims.capability_population import need
from claims.repository_cohort_population import CohortPopulation
from claims.repository_depth_population import DepthPopulation
from claims.repository_demonstration_population import DemonstrationPopulation
from text.evaluation_multiscale import file_sha
from audit.arithmetic_readback import receipt

NS = '{http://nite.sourceforge.net/}'


class SourceReadback:
    def __init__(self, population, raw_root):
        self.pop, self.pins = population, {}
        self.data, self.members, self.xml, self.indices, self.coding = None, {}, {}, {}, {}
        source = population.report['source']
        if population.family == 'reasoning':
            root = raw_root / 'gsm8k-human-training-v1'
            for name, pin in source['files'].items():
                need(file_sha(root / name) == pin, 'READBACK_MATH_FILE')
                self.pins[str(root / name)] = pin
            self.data = [json.loads(line) for line in (root / 'train.jsonl').read_bytes().splitlines()]
        elif population.family == 'conversation':
            path = Path(source['archive'])
            need(file_sha(path) == source['sha256'], 'READBACK_AMI_ARCHIVE')
            self.pins[str(path)] = source['sha256']
            with zipfile.ZipFile(path) as archive:
                for name, pin in source['members_read'].items():
                    raw = archive.read(name)
                    need(hashlib.sha256(raw).hexdigest() == pin, 'READBACK_AMI_MEMBER')
                    self.members[name] = raw
                need(hashlib.sha256(archive.read('LICENCE.txt')).hexdigest() == source['licence_sha256'],
                     'READBACK_AMI_LICENCE')
            self.meetings = {e.attrib['observation']: e for e in self.root('corpusResources/meetings.xml')}
        else:
            need(file_sha(Path(source['config'])) == source['config_sha256'], 'READBACK_CODE_CONFIG')
            self.pins[source['config']] = source['config_sha256']
            kinds = {'repository_cohort_intake': CohortPopulation,
                     'repository_depth_intake': DepthPopulation,
                     'repository_demonstration_intake': DemonstrationPopulation}
            for item in source['populations']:
                native = item['native']
                obj = kinds[item['kind']](native['population'], native['population_pin'])
                self.coding[native['population_pin']] = obj
                self.pins[str(obj.root / 'report.json')] = native['population_pin']

    def root(self, name):
        if name not in self.xml:
            self.xml[name] = ET.fromstring(self.members[name])
            self.indices[name] = {e.attrib[NS + 'id']: e for e in self.xml[name].iter()
                                  if NS + 'id' in e.attrib}
        return self.xml[name]

    def linked(self, href, folder):
        filename, fragment = href.split('#')
        ids = re.findall(r'id\(([^()]+)\)', fragment)
        name = folder + '/' + filename
        root = self.root(name)
        need(len(ids) in (1, 2), 'READBACK_HREF')
        if len(ids) == 1: return [self.indices[name][ids[0]]]
        elements = list(root)
        positions = {e.attrib[NS + 'id']: i for i, e in enumerate(elements)}
        lo, hi = positions[ids[0]], positions[ids[1]]
        need(lo <= hi, 'READBACK_HREF_ORDER')
        return elements[lo:hi + 1]

    def field(self, f):
        s = f['source']
        if self.pop.family == 'reasoning':
            need(s['artifact_sha256'] == self.pop.report['source']['files']['train.jsonl'], 'READBACK_MATH_PIN')
            raw = self.data[s['row']][s['field']]
            if 'codepoint_span' in s:
                lo, hi = s['codepoint_span']
                need(0 <= lo < hi <= len(raw), 'READBACK_MATH_SPAN')
                raw = raw[lo:hi]
            need(raw == f['text'], 'READBACK_SOURCE_TEXT')
        elif self.pop.family == 'conversation':
            self.root(s['member'])
            act = self.indices[s['member']][s['element']]
            words = [w for c in act.findall(NS + 'child') for w in self.linked(c.attrib['href'], 'words')]
            lexical = [w for w in words if w.tag == 'w' and w.text is not None]
            need(' '.join(w.text for w in lexical) == f['text'], 'READBACK_SOURCE_TEXT')
            start = min(Fraction(w.attrib['starttime']) for w in lexical)
            end = max(Fraction(w.attrib['endtime']) for w in lexical)
            need(str(start) == f['clock']['start'] and str(end) == f['clock']['end'], 'READBACK_AMI_CLOCK')
            meeting, agent = Path(s['member']).name.split('.')[:2]
            speaker = next(e for e in self.meetings[meeting] if e.attrib['nxt_agent'] == agent)
            need(s['speaker'] == dict(speaker.attrib), 'READBACK_AMI_SPEAKER')
            need(s['member_sha256'] == hashlib.sha256(self.members[s['member']]).hexdigest(), 'READBACK_ACT_PIN')
        else:
            pop = self.coding[s['population_pin']]
            field = pop.fields[s['condition']][s['side']][s['field_index']]
            raw = b''.join(pop.rows[p['unit']]['text'].encode() for p in field['parts'])
            need(f['text'].encode() == raw and f['role'] == field['role'], 'READBACK_SOURCE_TEXT')
            need(s['original_source'] == field['source'] and s['original_span'] == field['source_span']
                 and s['native_parts'] == field['parts'], 'READBACK_CODE_ADDRESS')

    def task(self, t):
        p = t['provenance']
        if self.pop.family == 'reasoning':
            row = self.data[p['row']]
            need(t['split_group'].endswith(hashlib.sha256(row['question'].encode()).hexdigest()),
                 'READBACK_MATH_GROUP')
            context = [self.pop.fields[k]['text'] for k in t['context']]
            targets = [self.pop.fields[k]['text'] for k in t['targets']]
            if t['kind'] == 'solve_recorded_problem':
                need(context == [row['question']] and targets == [row['answer']], 'READBACK_MATH_TASK')
            else:
                lines = row['answer'].splitlines()
                pos = p['position']
                need(context == [row['question'], *lines[:pos]] and targets == [lines[pos]], 'READBACK_MATH_TASK')
            receipt(targets[0], t['checks'])
        elif self.pop.family == 'conversation':
            self.root(p['member'])
            pair = self.indices[p['member']][p['element']]
            links = {'source': [], 'target': []}
            categories = []
            for pointer in pair:
                role = pointer.attrib['role']
                if role in links:
                    links[role].extend(e.attrib[NS + 'id'] for e in self.linked(pointer.attrib['href'], 'dialogueActs'))
                elif role == 'type':
                    categories.extend(e.attrib['name'] for e in self.linked(pointer.attrib['href'], 'ontologies'))
            need(p['source_acts'] == links['source'] and p['target_acts'] == links['target']
                 and p['source_annotation_types'] == categories, 'READBACK_AMI_LINK')
        else:
            entries = [p] if t['kind'] == 'reproduce_recorded_patch' else [p['failure'], p['repair']]
            for value in entries:
                pop = self.coding[value['population']]
                entry = pop.observations.entries[value['condition']]
                need(value['execution_evidence'] == entry and value['observed_outcome'] == entry['outcome'],
                     'READBACK_EXECUTION')

    def run(self):
        for field in self.pop.fields.values(): self.field(field)
        for task in self.pop.tasks.values(): self.task(task)
        return dict(fields=len(self.pop.fields), tasks=len(self.pop.tasks), source_pins=self.pins,
                    method='original field/annotation/execution readback', author_checked=True)
