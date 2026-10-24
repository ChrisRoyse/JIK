"""Source-field receipts for the pinned identity profile, never world truth."""
import hashlib
import json

from claims.episodic_memory import KINDS, check, digest


class IdentitySourceChecker:
    def __init__(self, inputs, source):
        # LoopInputs has already independently audited raw, population and slots.
        raw=(source/'records.jsonl').read_bytes()
        check(hashlib.sha256(raw).hexdigest()==inputs.source_readback['records_sha256'], 'SOURCE_PIN')
        self.records={r['id']:r for r in map(json.loads,raw.splitlines())}
        self.refs={r['unit']:r for r in inputs.refs if r['role']!='probe_only'}

    def episode(self, unit, kind='source_statement', evidence=()):
        check(type(unit) is int and unit in self.refs, 'SOURCE_ADDRESS')
        ref=self.refs[unit]
        body=dict(claim_kind=kind, content=self.records[ref['record']][ref['field']],
                  source=ref, evidence=list(evidence), event_time=None)
        return dict(id=digest(body), **body)

    def validate_episode(self, episode):
        check(type(episode) is dict and set(episode)=={'id','claim_kind','content','source','evidence','event_time'}, 'EPISODE_SCHEMA')
        check(episode['claim_kind'] in KINDS and episode['event_time'] is None, 'KIND_OR_UNSUPPORTED_TIME')
        ref=episode['source']
        check(type(ref) is dict and type(ref.get('unit')) is int and ref['unit'] in self.refs, 'SOURCE_ADDRESS')
        evidence=episode['evidence']
        check(type(evidence) is list and all(type(x) is str for x in evidence) and len(set(evidence))==len(evidence), 'EVIDENCE_SCHEMA')
        check(episode==self.episode(ref['unit'],episode['claim_kind'],evidence), 'SOURCE_CONTENT')

    def assess(self, episode):
        self.validate_episode(episode)
        check(episode['claim_kind']=='source_statement', 'NO_APPLICABLE_GROUNDING_CHECKER')
        value=dict(checker='identity-source-field-v1',episode=episode['id'],claim_kind='source_statement',
                   scope='pinned_artifact_contents_only',source=episode['source'],world_validity='unassessed')
        return dict(id=digest(value),**value)
