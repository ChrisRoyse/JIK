"""Authenticated, paired final evaluation records; no model or native-store load."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from text.constellation_corpus import file_sha, need

ROOT = Path('/zfs/archive/mnlm/evidence/fsv/29')
REFS = {
    'breadth': ('legal-efficient-predictor-breadth-v1-20261023',
                '4921af55a18381dc978e25d593f946534a6e519d2d0b1a4623f640c0993c13f4'),
    'repeated': ('legal-efficient-predictor-repeated-v1-20261023',
                 '9b4ff857a10dcf44d3d18586597325138f3197d2421663b027357a9c40dfd6f9'),
    'schedule': ('legal-breadth-schedule-publication-v1-20261023',
                 'eada66a4f4e29e95ddecc0e41e2e20fcc9ca3c206d0088eb608c3408314c8b3a'),
    'schedule_cold': ('legal-breadth-schedule-cold-v1-20261023',
                      'f8e76e2e91b576d34dec76daea299e1b54f9d2093c6f1890e06dc82def24c39c')}
SUBSETS = ('all', 'target_absent_from_train', 'both_inputs_absent_from_train')
NEW = ('audit/legal_breadth_comparison_source.py', 'audit/legal_breadth_comparison_metrics.py',
       'audit/legal_breadth_comparison.py', 'audit/legal_breadth_comparison_readback.py')


def raw(path, pin):
    value = path.read_bytes()
    need(hashlib.sha256(value).hexdigest() == pin, 'BREADTH_COMPARE_BYTES')
    return value


def code():
    base = Path(__file__).resolve().parents[1]
    return {n:file_sha(base/n) for n in NEW}


class Source:
    def __init__(self, root):
        self.root = root; self.refs = {}; self.files = {}; self.plans = {}; self.stamps = {}
        for k,(name,pin) in REFS.items():
            self.refs[k] = json.loads(self.read(root/name/'report.json', pin))
            need(self.refs[k]['complete'], 'BREADTH_COMPARE_INCOMPLETE')
        a,b = (self.refs[k] for k in ('breadth','repeated'))
        need(a['arm']=='breadth' and b['arm']=='repeated' and a['target_state']==b['target_state'],
             'BREADTH_COMPARE_ARMS')
        for k in a['binding']:
            if k not in ('arm','selection'): need(a['binding'][k]==b['binding'][k], 'BREADTH_COMPARE_BINDING')
        need(a['execution']['code']==b['execution']['code'], 'BREADTH_COMPARE_EXECUTION')
        need(a['updates']==b['updates']==91540 and a['target_presentations']==b['target_presentations']==6455636,
             'BREADTH_COMPARE_WORK')
        for k in ('breadth','repeated'):
            report = self.refs[k]; folder = root/REFS[k][0]
            self.plans[k] = json.loads(self.read(folder/'selection.json', report['files']['selection.json']))
            need(report['binding']['selection']==report['files']['selection.json'], 'BREADTH_COMPARE_SELECTION_PIN')
            summary = json.loads(self.read(folder/'final-evaluation/summary.json',
                                          report['files']['final-evaluation/summary.json']))
            need(summary['summary']==report['evaluations']['final']['summary'], 'BREADTH_COMPARE_PARENT_SUMMARY')
        self.selection = self.plans['breadth']['evaluations']
        need(self.selection==self.plans['repeated']['evaluations'], 'BREADTH_COMPARE_SUPPORT')
        self.validate_selection(self.selection)
        s,c = self.refs['schedule'],self.refs['schedule_cold']
        need(c['previous_pin']==REFS['schedule'][1] and s['summary']==c['summary'], 'BREADTH_COMPARE_SCHEDULE_COLD')
        spec = s['files']['schedule.npy']; pin = spec if isinstance(spec,str) else spec['sha256']
        schedule = np.load(io.BytesIO(self.read(root/REFS['schedule'][0]/'schedule.npy', pin)), allow_pickle=False)
        for arm,field in [('breadth','full'),('repeated','repeated')]:
            ids = np.asarray(self.plans[arm]['train_documents'], dtype=np.int64)
            need(np.array_equal(ids,schedule[field]) and len(set(ids.tolist()))==s['summary']['distinct_documents'][field],
                 'BREADTH_COMPARE_SCHEDULE')
        need(s['summary']['every_step_shape_exact'] and s['summary']['both_arms_share_target_pretraining'],
             'BREADTH_COMPARE_INTERVENTION')
        self.roster = tuple(a['binding']['cache']['roster'])

    @staticmethod
    def validate_selection(rows):
        need(len(rows)==192 and len({r['document'] for r in rows})==192
             and len({r['case_component'] for r in rows})==192, 'BREADTH_COMPARE_CASES')

    def read(self, path, pin):
        value = raw(path,pin); st = path.stat(); self.stamps[str(path)] = (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
        self.files[str(path)] = dict(sha256=pin,bytes=len(value)); return value

    def pair(self, selected):
        rows = []
        for arm in ('breadth','repeated'):
            name = f"final-evaluation/document-{selected['document']}.pt"
            row = torch.load(io.BytesIO(self.read(self.root/REFS[arm][0]/name,self.refs[arm]['files'][name])),
                             map_location='cpu',weights_only=True)
            need(row['document']==selected['document'] and row['side']==selected['side']
                 and len(row['uids'])==selected['stop']-selected['start'], 'BREADTH_COMPARE_DOCUMENT')
            rows.append(row)
        validate_pair(*rows, self.roster)
        return rows

    def fresh(self):
        for name,expected in self.stamps.items():
            st = Path(name).stat()
            need((st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==expected, 'BREADTH_COMPARE_SOURCE_CHANGED')


def validate_pair(a,b,roster):
    for k in ('document','side','source_collection','native_pin','label_pin','exposure'):
        need(a[k]==b[k], 'BREADTH_COMPARE_ROW_BINDING')
    for k in ('uids','target','permutation','modal_successor_uids'):
        need(torch.equal(a[k],b[k]), 'BREADTH_COMPARE_TARGET_ORDER')
    need(set(a['masks'])==set(b['masks']), 'BREADTH_COMPARE_MASKS')
    for k in a['masks']: need(torch.equal(a['masks'][k],b['masks'][k]), 'BREADTH_COMPARE_MASKS')
    for k in a['predictions']:
        if k != 'model': need(torch.equal(a['predictions'][k],b['predictions'][k]), 'BREADTH_COMPARE_BASELINE')
    need(set(a['native'])==set(b['native'])==set(roster), 'BREADTH_COMPARE_ROSTER')
    for k in roster:
        need(torch.equal(a['native'][k]['target_count'],b['native'][k]['target_count'])
             and torch.equal(a['native'][k]['ordered_distances'][:,1:],b['native'][k]['ordered_distances'][:,1:]),
             'BREADTH_COMPARE_NATIVE_BASELINE')
    n = len(a['target'])
    need(n>0 and torch.equal(a['permutation'],torch.roll(torch.arange(n),1))
         and bool(a['masks']['all'].all()), 'BREADTH_COMPARE_PERMUTATION')
    need(torch.equal(a['masks']['permuted_target_distinct'],a['uids'][1:][a['permutation']]!=a['uids'][1:]),
         'BREADTH_COMPARE_PERMUTATION_MASK')
