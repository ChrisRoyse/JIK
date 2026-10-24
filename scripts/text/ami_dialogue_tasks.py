"""AMI source-attested response links, prior completed acts and private responses."""
import argparse
from bisect import bisect_right
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text.capability_tasks import TaskBuilder, check
from text.evaluation_multiscale import file_sha

ZIP_PIN = 'b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d'
MEETINGS = ('ES2002a', 'IS1000a', 'TS3005a')
NS = '{http://nite.sourceforge.net/}'


class Annotation:
    def __init__(self, archive):
        self.archive, self.roots, self.index, self.pins = archive, {}, {}, {}
        self.names, self.elements = set(archive.namelist()), {}

    def read(self, name):
        if name not in self.roots:
            check(name in self.names, 'AMI_MEMBER_MISSING')
            raw = self.archive.read(name)
            root = ET.fromstring(raw)
            self.pins[name] = hashlib.sha256(raw).hexdigest()
            self.roots[name] = root
            items = list(root.iter()) if name.startswith('ontologies/') else list(root)
            self.elements[name] = items
            ids = [e.attrib.get(NS + 'id') for e in items]
            check(all(ids) and len(ids) == len(set(ids)), 'AMI_ELEMENT_IDS')
            self.index[name] = {key: i for i, key in enumerate(ids)}
        return self.roots[name]

    def resolve(self, href, folder):
        match = re.fullmatch(r'([^/#]+\.xml)#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?', href)
        check(match is not None, 'AMI_HREF')
        name = folder + '/' + match[1]
        self.read(name)
        elements = self.elements[name]
        first, last = self.index[name][match[2]], self.index[name][match[3] or match[2]]
        check(first <= last, 'AMI_RANGE')
        return name, elements[first:last + 1]


def act_records(reader, meeting, speakers):
    acts = {}
    for agent in sorted(speakers):
        name = f'dialogueActs/{meeting}.{agent}.dialog-act.xml'
        for act in reader.read(name):
            tokens, addresses, types = [], [], []
            for child in act:
                if child.tag == NS + 'child':
                    member, words = reader.resolve(child.attrib['href'], 'words')
                    for w in words:
                        word = dict(id=w.attrib[NS + 'id'], kind=w.tag, text=w.text,
                                    attributes=dict(w.attrib))
                        tokens.append(word)
                    addresses.append(dict(member=member, sha256=reader.pins[member],
                                          href=child.attrib['href']))
                elif child.tag == NS + 'pointer':
                    check(child.attrib['role'] == 'da-aspect', 'AMI_ACT_POINTER')
                    _, ontology = reader.resolve(child.attrib['href'], 'ontologies')
                    types.extend(e.attrib['name'] for e in ontology)
                else:
                    raise ValueError('MNLM_CAPABILITY_TASK_AMI_ACT_CHILD')
            lexical = [w for w in tokens if w['kind'] == 'w' and w['text'] is not None]
            timed = [w for w in lexical if 'starttime' in w['attributes']
                     and 'endtime' in w['attributes']]
            text = ' '.join(w['text'] for w in lexical)
            complete = bool(lexical) and len(timed) == len(lexical)
            start = min((Fraction(w['attributes']['starttime']) for w in timed), default=None)
            end = max((Fraction(w['attributes']['endtime']) for w in timed), default=None)
            check(not complete or start <= end, 'AMI_WORD_CLOCK')
            key = act.attrib[NS + 'id']
            check(key not in acts, 'AMI_DUPLICATE_ACT')
            acts[key] = dict(id=key, agent=agent, text=text, tokens=tokens, types=types,
                complete=complete, start=start, end=end,
                address=dict(archive_sha256=ZIP_PIN, member=name, member_sha256=reader.pins[name],
                             element=key, word_sources=addresses, speaker=speakers[agent],
                             rendition='single-space join of unchanged lexical word nodes',
                             nonlexical_events=[w for w in tokens if w['kind'] != 'w']))
    return acts


def build(archive, output):
    tick = time.perf_counter()
    check(file_sha(archive) == ZIP_PIN, 'AMI_ARCHIVE_PIN')
    source = dict(archive=str(archive), sha256=ZIP_PIN, meetings=list(MEETINGS),
                  licence='CC-BY-4.0', attribution='AMI Meeting Corpus; AMI Project',
                  consent_scope='public corpus release; participant consent not independently audited',
                  producer_sha256=file_sha(Path(__file__)),
                  encoder_exposure_certified=False, previous_local_exposure=['ES2002a'])
    out = TaskBuilder('conversation', source)
    coverage, act_types, pair_types = {}, Counter(), Counter()
    with zipfile.ZipFile(archive) as z:
        check('Attribution 4.0' in z.read('LICENCE.txt').decode(), 'AMI_LICENCE')
        reader = Annotation(z)
        meetings = {e.attrib['observation']: e for e in reader.read('corpusResources/meetings.xml')}
        for meeting in MEETINGS:
            metadata = meetings[meeting]
            speakers = {s.attrib['nxt_agent']: dict(s.attrib) for s in metadata}
            acts = act_records(reader, meeting, speakers)
            valid = sorted((a for a in acts.values() if a['complete'] and a['text'].strip()),
                           key=lambda a: (a['end'], a['start'], a['id']))
            end_times = [a['end'] for a in valid]
            fields = {}

            def field(act):
                if act['id'] not in fields:
                    fields[act['id']] = out.field(act['text'], 'attested_speaker_act', act['address'],
                        clock=dict(id='ami-annotation:' + meeting, start=str(act['start']),
                                   end=str(act['end']), physical_sync_calibrated=False))
                return fields[act['id']]

            member = f'dialogueActs/{meeting}.adjacency-pairs.xml'
            pairs = reader.read(member)
            before, unavailable = len(out.tasks), Counter()
            for pair in pairs:
                links, categories = {'source': [], 'target': []}, []
                for ptr in pair:
                    role = ptr.attrib['role']
                    if role in links:
                        _, elements = reader.resolve(ptr.attrib['href'], 'dialogueActs')
                        links[role].extend(acts[e.attrib[NS + 'id']] for e in elements)
                    elif role == 'type':
                        _, elements = reader.resolve(ptr.attrib['href'], 'ontologies')
                        categories.extend(e.attrib['name'] for e in elements)
                    else:
                        raise ValueError('MNLM_CAPABILITY_TASK_AMI_PAIR_POINTER')
                provenance = dict(archive_sha256=ZIP_PIN, member=member,
                    member_sha256=reader.pins[member], element=pair.attrib[NS + 'id'],
                    source_acts=[a['id'] for a in links['source']],
                    target_acts=[a['id'] for a in links['target']],
                    source_annotation_types=categories)
                reason = None
                if not all(links.values()): reason = 'missing_source_or_target'
                elif any(not a['complete'] or not a['text'].strip()
                         for a in links['source'] + links['target']): reason = 'incomplete_lexical_clock'
                else:
                    cutoff = min(a['start'] for a in links['target'])
                    if any(a['end'] > cutoff for a in links['source']):
                        reason = 'source_not_complete_before_response'
                if reason:
                    unavailable[reason] += 1
                    out.gaps.append(dict(meeting=meeting, reason=reason, source=provenance))
                    continue
                # Search a once-sorted endpoint index, never scan the meeting for every pair.
                end = bisect_right(end_times, cutoff)
                context_acts = {a['id']: a for a in valid[max(0, end - 8):end]}
                context_acts.update({a['id']: a for a in links['source']})
                target_ids = {a['id'] for a in links['target']}
                if target_ids & context_acts.keys():
                    out.gaps.append(dict(meeting=meeting, reason='zero_duration_target_in_context',
                                         source=provenance))
                    unavailable['zero_duration_target_in_context'] += 1
                    continue
                context = sorted(context_acts.values(), key=lambda a: (a['start'], a['end'], a['id']))
                for a in links['source']:
                    for b in links['target']:
                        out.edge(field(a), field(b), 'annotated_response', provenance)
                provenance.update(cutoff=str(cutoff), previous_complete_acts_available=end,
                    bounded_context_rule='last-eight-complete-acts-plus-explicit-source-acts',
                    target_act_types=[a['types'] for a in links['target']],
                    participants=sorted(s['global_name'] for s in speakers.values()),
                    meeting_metadata=dict(metadata.attrib))
                out.task(context=[field(a) for a in context],
                    targets=list(dict.fromkeys(field(a) for a in links['target'])), consequences=[],
                    group='ami-series:' + re.sub('[a-d]$', '', meeting),
                    kind='respond_to_attested_act', provenance=provenance,
                    checks=[dict(scope='annotation_link_and_pre_response_cutoff', status='pass',
                                 response_appropriateness_checked=False, world_claims_checked=False)])
                pair_types.update(categories)
                act_types.update(t for a in links['target'] for t in a['types'])
            coverage[meeting] = dict(source_pairs=len(pairs), retained_tasks=len(out.tasks) - before,
                exclusions=dict(unavailable), source_acts=len(acts), complete_acts=len(valid),
                participants=sorted(s['global_name'] for s in speakers.values()))
        source['members_read'] = reader.pins
        source['licence_sha256'] = hashlib.sha256(z.read('LICENCE.txt')).hexdigest()
    result = out.publish(output, scope='transcribed human speech and annotation links; not audiovisual event proof',
        coverage=dict(meetings=coverage, response_act_types=dict(act_types), pair_types=dict(pair_types),
                      correction_and_abstention_labels='not inferred; require source-specific review',
                      native_audio_and_vision_acquired=False))
    print(json.dumps(dict(result, coverage=coverage, wall_seconds=time.perf_counter() - tick)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.archive, args.output)
