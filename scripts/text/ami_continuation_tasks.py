"""Source-clocked continuations around inspected clarification and self-repair exchanges."""
import argparse
from bisect import bisect_right
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text.ami_dialogue_tasks import Annotation, ZIP_PIN, act_records
from text.capability_tasks import TaskBuilder, check
from text.evaluation_multiscale import file_sha


WINDOWS = [('ES2002b', 295, 327), ('IS1000b', 500, 540), ('TS3005b', 1390, 1420)]
WITNESSES = {
    'clarification_request': 'IS1000b.C.dialog-act.vkaraisk.8',
    'clarification_expansion': 'IS1000b.A.dialog-act.vkaraisk.53',
    'clarification_acknowledgment': 'IS1000b.C.dialog-act.vkaraisk.9',
    'self_correction': 'ES2002b.C.dialog-act.dharshi.22',
}


def build(archive, output):
    start = tick = time.perf_counter()
    check(file_sha(archive) == ZIP_PIN, 'AMI_ARCHIVE_PIN')
    source = dict(source_kind='ami-chronological-continuation-v1', archive=str(archive), sha256=ZIP_PIN,
        windows=WINDOWS, licence='CC-BY-4.0', attribution='AMI Meeting Corpus; AMI Project',
        consent_scope='public corpus release; participant consent not independently audited',
        producer_sha256=file_sha(Path(__file__)),
        parser_sha256=file_sha(Path(__file__).with_name('ami_dialogue_tasks.py')),
        encoder_exposure_certified=False, previous_local_exposure=['ES2002a', 'IS1000a', 'TS3005a'])
    out = TaskBuilder('conversation', source)
    phases = dict(source_pin=time.perf_counter() - tick)
    tick = time.perf_counter()
    coverage, witnesses = {}, {}
    with zipfile.ZipFile(archive) as z:
        check('Attribution 4.0' in z.read('LICENCE.txt').decode(), 'AMI_LICENCE')
        reader = Annotation(z)
        meetings = {e.attrib['observation']: e for e in reader.read('corpusResources/meetings.xml')}
        for meeting, lo, hi in WINDOWS:
            speakers = {s.attrib['nxt_agent']: dict(s.attrib) for s in meetings[meeting]}
            acts = act_records(reader, meeting, speakers)
            valid = sorted((a for a in acts.values() if a['complete'] and a['text'].strip()),
                           key=lambda a: (a['end'], a['start'], a['id']))
            ends = [a['end'] for a in valid]
            targets = sorted((a for a in valid if lo <= a['start'] < hi),
                             key=lambda a: (a['start'], a['end'], a['id']))
            fields = {}

            def field(a):
                if a['id'] not in fields:
                    fields[a['id']] = out.field(a['text'], 'attested_speaker_act', a['address'],
                        clock=dict(id='ami-annotation:' + meeting, start=str(a['start']),
                                   end=str(a['end']), physical_sync_calibrated=False))
                return fields[a['id']]

            before = len(out.tasks)
            for target in targets:
                cutoff = target['start']
                stop = bisect_right(ends, cutoff)
                context = [a for a in valid[max(0, stop - 8):stop] if a['id'] != target['id']]
                context.sort(key=lambda a: (a['start'], a['end'], a['id']))
                if not context:
                    out.gaps.append(dict(meeting=meeting, target=target['id'], reason='no_completed_context'))
                    continue
                provenance = dict(meeting=meeting, cutoff=str(cutoff),
                    target_acts=[target['id']], context_acts=[a['id'] for a in context],
                    selection_window=[lo, hi], context_rule='last-eight-complete-acts',
                    previous_complete_acts_available=stop, annotation_response_link_asserted=False,
                    response_appropriateness_checked=False, world_claims_checked=False)
                for a in context:
                    out.edge(field(a), field(target), 'completed_before_utterance', provenance)
                key = out.task(context=[field(a) for a in context], targets=[field(target)],
                    consequences=[], group='ami-series:' + re.sub('[a-d]$', '', meeting),
                    kind='continue_attested_dialogue', provenance=provenance,
                    checks=[dict(scope='source_words_roles_and_completed_act_cutoff', status='pass',
                                 causal_or_response_link_inferred=False)])
                for name, element in WITNESSES.items():
                    if target['id'] == element:
                        witnesses[name] = dict(task=key, element=element, text=target['text'],
                            basis='author-inspected source wording; coverage metadata only',
                            external_fact_checked=False)
            coverage[meeting] = dict(source_acts=len(acts), complete_acts=len(valid),
                candidate_targets=len(targets), tasks=len(out.tasks) - before,
                participants=sorted(s['global_name'] for s in speakers.values()))
        source['members_read'] = reader.pins
        source['licence_sha256'] = hashlib.sha256(z.read('LICENCE.txt')).hexdigest()
    phases['parse_and_connected_tasks'] = time.perf_counter() - tick
    check(set(witnesses) == set(WITNESSES), 'AMI_COVERAGE_WITNESS')
    tick = time.perf_counter()
    result = out.publish(output, scope='recorded text continuations; no audiovisual reality certification',
        coverage=dict(meetings=coverage, witnesses=witnesses, source_series_separation_required=True,
            supported_abstention_established=False, semantic_labels_invented=False,
            native_audio_and_vision_acquired=False))
    phases['publication'] = time.perf_counter() - tick
    wall = time.perf_counter() - start
    return dict(population=result, phases_seconds=phases, wall_seconds=wall,
                phase_coverage=sum(phases.values()) / wall)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.archive, args.output)))
