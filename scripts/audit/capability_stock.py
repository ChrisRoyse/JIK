"""Finite retained-source inventory. Raw availability is separate from learned capability."""
from collections import Counter
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

from claims.capability_population import need, sha
from claims.episodic_memory import digest
from model.capability_kernel_learning import pinned
from text.arithmetic_tasks import equalities


def capture(ref):
    report = pinned(ref); root = Path(ref['path']).parent; payloads = {}
    for name, entry in report['files'].items():
        raw = (root/name).read_bytes()
        need(len(raw) == entry['bytes'] and sha(raw) == entry['sha256'], 'STOCK_SOURCE_BYTES')
        payloads[name] = raw
    return report, payloads


def inventory(config):
    units = []; summary = {}
    for kind, filename in [('gsm8k', 'train.jsonl'), ('condaqa', 'train.json')]:
        report, payloads = capture(config[kind]); raw = payloads[filename]
        artifact = report['files'][filename]['sha256']
        rows = [json.loads(line) for line in raw.splitlines()]
        before = len(units)
        for index, row in enumerate(rows):
            if kind == 'gsm8k':
                context, target = row['question'], row['answer']
                group = 'gsm8k-question:' + sha(context.encode())
                extra = dict(checks=equalities(target), steps=len(target.splitlines()),
                             scope='annotated_equalities_only; interpretation unassessed')
            else:
                context, target = [row['sentence1'], row['sentence2']], row['label']
                group = 'condaqa:' + artifact + ':passage:' + str(row['PassageID'])
                extra = dict(passage=row['PassageID'], question=row['QuestionID'], edit=row['PassageEditID'],
                             scope='question-author answer; not world validity')
            units.append(dict(kind=kind, row=index, group=group, artifact=report['files'][filename]['sha256'],
                              context=digest(context), target=sha(target.encode()), **extra))
        current = units[before:]
        summary[kind] = dict(rows=len(current), groups=len({r['group'] for r in current}),
            context_aliases=len(current)-len({r['context'] for r in current}), bytes=len(raw),
            licence=report['licence'], measurement_ready=False, training_ready=False)
        if kind == 'gsm8k':
            summary[kind]['checks'] = dict(Counter(x['status'] for r in current for x in r['checks']))
            summary[kind]['all_equalities_pass'] = sum(bool(r['checks']) and
                all(x['status']=='pass' for x in r['checks']) for r in current)
    ref = config['ami']; path = Path(ref['path']); raw = path.read_bytes()
    need(len(raw) == ref['bytes'] and sha(raw) == ref['pin'], 'STOCK_AMI_BYTES')
    with zipfile.ZipFile(path) as z:
        entries = z.infolist(); names = {i.filename for i in entries}
        need(len(names) == len(entries), 'STOCK_AMI_DUPLICATE_MEMBER')
        acts = sorted(n for n in names if n.startswith('dialogueActs/') and n.endswith('.xml'))
        words = sorted(n for n in names if n.startswith('words/') and n.endswith('.xml'))
        by_meeting = {}
        for kind, members in [('act',acts),('word',words)]:
            for name in members:
                by_meeting.setdefault((kind,name.split('/')[1].split('.')[0]),[]).append(name)
        metadata = z.read('corpusResources/meetings.xml'); licence = z.read('LICENCE.txt')
        need(b'Attribution 4.0' in licence, 'STOCK_AMI_LICENCE')
        meetings = []
        for meeting in ET.fromstring(metadata):
            name = meeting.attrib['observation']
            speakers = [dict(s.attrib) for s in meeting]
            meetings.append(dict(meeting=name, series=re.sub('[a-d]$', '', name),
                participants=sorted({s['global_name'] for s in speakers}),
                expected_agents=sorted(s['nxt_agent'] for s in speakers),
                act_members=by_meeting.get(('act',name),[]),
                word_members=by_meeting.get(('word',name),[])))
        summary['ami'] = dict(archive_bytes=len(raw), entries=len(entries),
            uncompressed_bytes=sum(i.file_size for i in entries), act_files=len(acts), word_files=len(words),
            meetings=len(meetings), series=len({m['series'] for m in meetings}),
            metadata_sha256=sha(metadata), licence_sha256=sha(licence),
            scope='metadata availability only; dialogue acts and response validity not parsed',
            native_media_coverage_established=False, measurement_ready=False, training_ready=False)
    return units, meetings, summary
