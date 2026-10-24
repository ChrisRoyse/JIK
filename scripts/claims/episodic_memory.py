"""Research memory ledger: source observation is separate from its asserted content.

Grounding is delegated to the supplied executable checker and keeps its scope.
The source-field checker establishes artifact contents only. An observed-event
checker must supply its own validation; the model cannot supply its receipt.
Kinds are supplied metadata, not a claim of learned classification. No deletion.
"""
from collections import deque
import hashlib
import json
import pyarrow.parquet as pq
from claims.qasc_grounding import SOURCE_SHA256,text_sha
from text.evaluation_multiscale import file_sha

KINDS=('source_statement','event_occurrence','measurement','derived_relation',
       'causal_explanation','motive_attribution','prediction','interpretation','unclassified')
ZERO='0'*64

def check(ok,code):
    if not ok:raise ValueError('MNLM_MEMORY_'+code)

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()

class SourceFieldChecker:
    def __init__(self,path):
        check(file_sha(path)==SOURCE_SHA256,'SOURCE_PIN')
        self.rows=pq.read_table(path).to_pylist();check(len(self.rows)==8134,'SOURCE_ROWS')

    def reference(self,row,field):
        check(type(row)==int and 0<=row<len(self.rows) and field in ('fact1','fact2','combinedfact'),'SOURCE_ADDRESS')
        return dict(artifact_sha256=SOURCE_SHA256,row=row,field=field,text_sha256=text_sha(self.rows[row][field]))

    def validate_episode(self,episode):
        check(set(episode)=={'id','claim_kind','content','source','evidence','event_time'},'EPISODE_SCHEMA')
        check(episode['claim_kind'] in KINDS and episode['event_time'] is None,'KIND_OR_UNSUPPORTED_TIME')
        check(type(episode['content'])==str and episode['content']!='','CONTENT')
        ref=episode['source'];check(type(ref)==dict and set(ref)=={'artifact_sha256','row','field','text_sha256'},'SOURCE_SCHEMA')
        check(ref==self.reference(ref['row'],ref['field']) and text_sha(episode['content'])==ref['text_sha256'],'SOURCE_CONTENT')
        check(type(episode['evidence'])==list and all(type(x)==str for x in episode['evidence']) and len(set(episode['evidence']))==len(episode['evidence']),'EVIDENCE_SCHEMA')
        check(episode['id']==digest({k:v for k,v in episode.items() if k!='id'}),'EPISODE_ID')

    def assess(self,episode):
        self.validate_episode(episode)
        check(episode['claim_kind']=='source_statement','NO_APPLICABLE_GROUNDING_CHECKER')
        receipt=dict(checker='pinned-source-field-v1',episode=episode['id'],claim_kind='source_statement',
                     scope='pinned_artifact_contents_only',source=episode['source'],world_validity='unassessed')
        return dict(id=digest(receipt),**receipt)

def episode(checker,row,kind,evidence=()):
    body=dict(claim_kind=kind,content=checker.rows[row]['fact1'],source=checker.reference(row,'fact1'),
              evidence=list(evidence),event_time=None)
    return dict(id=digest(body),**body)

class Memory:
    """Mutable projection of validated events; caller persists its hash-linked log.

    Source tables remain invariant. Adjacency indexes are updated once on intake;
    invalidation/retirement traverse only the affected dependency subgraph.
    A receipt cannot be supplied by the model: assess() executes the checker.
    """
    def __init__(self,checker):
        self.checker=checker;self.episodes={};self.state={};self.children={};self.events=[];self.head=ZERO

    def descendants(self,start):
        seen=set();queue=deque([start])
        while queue:
            item=queue.popleft()
            if item in seen:continue
            seen.add(item);queue.extend(sorted(self.children.get(item,())))
        return sorted(seen)

    def apply(self,command):
        check(type(command)==dict and type(command.get('op'))==str,'COMMAND')
        command=json.loads(canonical(command))
        op=command['op'];expected={'op','episode'} if op in ('intake','check_source','retire','reactivate') else {'op','episode','receipt'} if op=='promote' else {'op','episode','reason'} if op=='invalidate' else set()
        check(expected and set(command)==expected,'COMMAND_SCHEMA')
        payload=dict(command)
        if op=='intake':
            ep=command['episode'];self.checker.validate_episode(ep);key=ep['id']
            check(key not in self.episodes,'DUPLICATE_EPISODE')
            check(all(x in self.episodes for x in ep['evidence']),'MISSING_EVIDENCE')
            # Evidence edges mean provenance, never logical entailment.
            self.episodes[key]=ep;self.state[key]=dict(active=True,grounded_scope=None,review='unassessed',receipt=None)
            for parent in ep['evidence']:self.children.setdefault(parent,set()).add(key)
        else:
            key=command['episode'];check(type(key)==str and key in self.state,'UNKNOWN_EPISODE');state=self.state[key]
            if op=='check_source':
                check(state['active'],'INACTIVE_EPISODE');receipt=self.checker.assess(self.episodes[key])
                payload['result']=receipt;state.update(receipt=receipt,review='source_checked')
            elif op=='promote':
                check(state['active'],'INACTIVE_EPISODE');receipt=state['receipt']
                check(receipt is not None and command['receipt']==receipt['id'],'NO_MATCHING_CHECKED_RECEIPT')
                check(receipt==self.checker.assess(self.episodes[key]),'RECEIPT_SCOPE')
                state.update(grounded_scope=receipt['scope'],review='source_checked')
            elif op=='retire':
                check(state['active'],'ALREADY_RETIRED')
                check(all(self.state[x]['grounded_scope'] is None for x in self.descendants(key)),'ACTIVE_GROUNDED_DEPENDENCY')
                state.update(active=False)
            elif op=='reactivate':
                check(not state['active'],'ALREADY_ACTIVE');state.update(active=True)
            else:
                check(type(command['reason'])==str and command['reason'].strip()!='','INVALIDATION_REASON')
                affected=self.descendants(key);payload['affected']=affected
                for x in affected:self.state[x].update(grounded_scope=None,review='needs_review',receipt=None)
        record=dict(sequence=len(self.events),previous=self.head,event=payload)
        record['sha256']=digest(record);self.head=record['sha256'];self.events.append(record)
        return json.loads(canonical(record))

    def snapshot(self):
        def observed(key,state):
            return (self.episodes[key]['claim_kind']=='event_occurrence'
                    and (state['receipt'] or {}).get('world_validity')=='scoped_observed_event')
        return json.loads(canonical(dict(head=self.head,states={k:self.state[k] for k in sorted(self.state)},
                    active_episodes=sorted(k for k,v in self.state.items() if v['active']),
                    grounded_source_statements=sorted(k for k,v in self.state.items()
                        if v['grounded_scope'] is not None and not observed(k,v)),
                    grounded_world_claims=sorted(k for k,v in self.state.items()
                        if v['grounded_scope'] is not None and observed(k,v)),
                    epistemic_distinction_learned=False)))
