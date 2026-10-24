"""Independent source, transition and hash-chain reconstruction of memory prototype."""
import hashlib,json
from pathlib import Path
import pyarrow.parquet as pq
from text.evaluation_multiscale import file_sha

def need(ok,code):
    if not ok:raise ValueError('MNLM_MEMORY_READBACK_'+code)

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')).hexdigest()
def read_json(path):return json.loads(path.read_bytes())

def read(root,raw):
    header=read_json(root/'report.json');need(header['schema']=='episodic-source-memory-v1' and header['complete'],'STATE')
    need(header['served'] is False and header['epistemic_distinction_learned'] is False and header['grounded_world_claims']==0,'CLAIMS')
    need(file_sha(raw)==header['source_sha256']=='b9a297b5ab55f1605c7682ffbb7042c26d7ecb9ff1e1aa5a820d4e791c8302d1','SOURCE_PIN')
    need({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==set(header['files'])|{'report.json'},'FILES')
    for name,pin in header['files'].items():
        p=(root/name).resolve();need(p.is_relative_to(root.resolve()) and p.stat().st_size==pin['bytes'] and file_sha(p)==pin['sha256'],'FILE_PIN')
    source=pq.read_table(raw).to_pylist();need(len(source)==8134,'SOURCE_ROWS')
    raw_lines=(root/'events.jsonl').read_bytes().splitlines(keepends=True)
    records=[json.loads(l) for l in raw_lines];head='0'*64;states={};episodes={};evidence={};snapshots=[];history={};offset=0
    for seq,r in enumerate(records):
        need(set(r)=={'sequence','previous','event','sha256'} and r['sequence']==seq and r['previous']==head,'CHAIN_ORDER')
        need(r['sha256']==digest({k:v for k,v in r.items() if k!='sha256'}),'CHAIN_DIGEST');head=r['sha256'];e=r['event'];op=e['op']
        allowed={'intake':{'op','episode'},'check_source':{'op','episode','result'},'promote':{'op','episode','receipt'},'retire':{'op','episode'},'reactivate':{'op','episode'},'invalidate':{'op','episode','reason','affected'}}
        need(op in allowed and set(e)==allowed[op],'EVENT_SCHEMA')
        if op=='intake':
            ep=e['episode'];need(set(ep)=={'id','claim_kind','content','source','evidence','event_time'},'EPISODE_SCHEMA');key=ep['id'];ref=ep['source']
            need(key not in episodes and key==digest({k:v for k,v in ep.items() if k!='id'}),'EPISODE_ID')
            need(ep['claim_kind'] in ('source_statement','unclassified') and ep['event_time'] is None,'KIND_TIME')
            need(set(ref)=={'artifact_sha256','row','field','text_sha256'} and type(ref['row'])==int and 0<=ref['row']<3 and ref['field']=='fact1','SOURCE_ADDRESS')
            need(ref['artifact_sha256']==header['source_sha256'] and ep['content']==source[ref['row']]['fact1'] and ref['text_sha256']==hashlib.sha256(ep['content'].encode('utf-8')).hexdigest(),'SOURCE_CONTENT')
            need(type(ep['evidence'])==list and len(set(ep['evidence']))==len(ep['evidence']) and all(x in episodes for x in ep['evidence']),'EVIDENCE')
            episodes[key]=ep;evidence[key]=set(ep['evidence']);states[key]=dict(active=True,grounded_scope=None,review='unassessed',receipt=None)
        else:
            key=e['episode'];need(key in states,'UNKNOWN_EPISODE');s=states[key]
            # This independent proof uses a bounded six-node scan, not the runtime adjacency walk.
            affected={key}
            while True:
                nxt=affected|{node for node,parents in evidence.items() if parents&affected}
                if nxt==affected:break
                affected=nxt
            if op=='check_source':
                need(s['active'] and episodes[key]['claim_kind']=='source_statement','CHECK_SCOPE')
                value=dict(checker='pinned-source-field-v1',episode=key,claim_kind='source_statement',scope='pinned_artifact_contents_only',source=episodes[key]['source'],world_validity='unassessed')
                receipt=dict(id=digest(value),**value);need(e['result']==receipt,'CHECK_RECEIPT');s.update(receipt=receipt,review='source_checked')
            elif op=='promote':
                need(s['active'] and s['receipt'] is not None and e['receipt']==s['receipt']['id'] and episodes[key]['claim_kind']=='source_statement','PROMOTION_SCOPE')
                s.update(grounded_scope='pinned_artifact_contents_only',review='source_checked')
            elif op=='retire':
                need(s['active'] and all(states[x]['grounded_scope'] is None for x in affected),'RETIRE_SUPPORT');s['active']=False
            elif op=='reactivate':need(not s['active'],'REACTIVATE');s['active']=True
            else:
                need(type(e['reason'])==str and bool(e['reason'].strip()) and e['affected']==sorted(affected),'INVALIDATION_CLOSURE')
                for x in affected:states[x].update(grounded_scope=None,review='needs_review',receipt=None)
        snapshots.append(digest(states))
        offset+=len(raw_lines[seq])
        state_snapshot=dict(head=head,states={k:states[k] for k in sorted(states)},active_episodes=sorted(k for k,s in states.items() if s['active']),
            grounded_source_statements=sorted(k for k,s in states.items() if s['grounded_scope'] is not None),grounded_world_claims=[],epistemic_distinction_learned=False)
        history[head]=dict(states=json.loads(json.dumps(states)),projection=digest(state_snapshot),bytes=offset)
    need(len(episodes)==6 and len(records)==19,'COUNTS')
    expected=dict(head=head,states={k:states[k] for k in sorted(states)},active_episodes=sorted(k for k,s in states.items() if s['active']),
                  grounded_source_statements=sorted(k for k,s in states.items() if s['grounded_scope'] is not None),grounded_world_claims=[],epistemic_distinction_learned=False)
    need(read_json(root/'snapshot.json')==expected and read_json(root/'state-digests.json')==snapshots,'PROJECTION')
    need(header['events']==len(records) and header['episodes']==6 and header['grounded_source_statements']==3,'COUNTS')
    refusals=read_json(root/'refusals.json');need(len(refusals)==header['refusal_count']==10,'REFUSAL_COUNT')
    for refusal in refusals:
        before=refusal['before'];need(before==refusal['after'] and before['head'] in history,'REFUSAL_HEAD')
        old=history[before['head']];need(before['projection']==old['projection'] and before['bytes']==old['bytes'],'REFUSAL_PERSISTENCE')
        cmd=refusal['command'];state=old['states'][cmd['episode']]
        if cmd['op']=='check_source':need(episodes[cmd['episode']]['claim_kind']!='source_statement','REFUSAL_CHECKER');code='NO_APPLICABLE_GROUNDING_CHECKER'
        elif cmd['op']=='promote':need(state['receipt'] is None or state['receipt']['id']!=cmd['receipt'],'REFUSAL_PROMOTION');code='NO_MATCHING_CHECKED_RECEIPT'
        else:need(cmd['op']=='retire' and state['grounded_scope'] is not None,'REFUSAL_RETIRE');code='ACTIVE_GROUNDED_DEPENDENCY'
        need(refusal['error']=='MNLM_MEMORY_'+code,'REFUSAL_REASON')
    need(sum(header['phases_seconds'].values())/header['wall_seconds']>=.9,'COST_PHASES')
    return dict(report_sha256=file_sha(root/'report.json'),events=len(records),episodes=len(episodes),grounded_source_statements=3,grounded_world_claims=0,
                hash_chain_and_every_transition_reconstructed=True,source_bytes_exact=True,acceptance='author-checked',served=False)
