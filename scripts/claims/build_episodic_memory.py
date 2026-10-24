"""Six real episodes exercise source-only grounding and reversible memory lifecycle."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.episodic_memory import SourceFieldChecker,Memory,episode,canonical,digest,check
from claims.qasc_grounding import SOURCE_SHA256
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    check(not a.output.exists(),'OUTPUT_EXISTS');start=time.perf_counter();phases={};checker=SourceFieldChecker(a.raw);memory=Memory(checker)
    phases['pinned_source_read']=time.perf_counter()-start;tick=time.perf_counter();a.output.mkdir(parents=True);states=[];refusals=[];observations=[];contents=[]
    with (a.output/'events.jsonl').open('xb') as log:
        def apply(command):
            event=memory.apply(command);log.write(canonical(event)+b'\n');log.flush();states.append(digest(memory.state));return event
        def refuse(name,command,code):
            before=dict(head=memory.head,projection=digest(memory.snapshot()),bytes=(a.output/'events.jsonl').stat().st_size)
            try:apply(command)
            except ValueError as e:error=str(e);check(code in error,'WRONG_REFUSAL')
            else:raise ValueError('MNLM_MEMORY_UNEXPECTED_SUCCESS')
            after=dict(head=memory.head,projection=digest(memory.snapshot()),bytes=(a.output/'events.jsonl').stat().st_size);check(before==after,'REFUSAL_MUTATED_STATE')
            refusals.append(dict(name=name,command=command,error=error,before=before,after=after))
        for row in range(3):
            observation=episode(checker,row,'source_statement');content=episode(checker,row,'unclassified',[observation['id']])
            observations.append(observation);contents.append(content)
            apply(dict(op='intake',episode=observation));apply(dict(op='intake',episode=content))
            apply(dict(op='check_source',episode=observation['id']));receipt=memory.state[observation['id']]['receipt']['id']
            apply(dict(op='promote',episode=observation['id'],receipt=receipt))
            refuse('content has no domain checker',dict(op='check_source',episode=content['id']),'NO_APPLICABLE_GROUNDING_CHECKER')
            refuse('source receipt borrowed for content',dict(op='promote',episode=content['id'],receipt=receipt),'NO_MATCHING_CHECKED_RECEIPT')
            refuse('active base evidence retired',dict(op='retire',episode=observation['id']),'ACTIVE_GROUNDED_DEPENDENCY')
        apply(dict(op='retire',episode=contents[1]['id']));apply(dict(op='reactivate',episode=contents[1]['id']))
        apply(dict(op='invalidate',episode=observations[0]['id'],reason='research lifecycle test: reassess source and dependent claims'))
        check(memory.state[contents[0]['id']]['review']=='needs_review','INVALIDATION_DEPENDENT')
        refuse('stale receipt after invalidation',dict(op='promote',episode=observations[0]['id'],receipt=refusals[1]['command']['receipt']),'NO_MATCHING_CHECKED_RECEIPT')
        apply(dict(op='check_source',episode=observations[0]['id']));apply(dict(op='promote',episode=observations[0]['id'],receipt=memory.state[observations[0]['id']]['receipt']['id']))
        # Repeat the retirement/reactivation path after reassessment, keeping audit history.
        apply(dict(op='retire',episode=contents[0]['id']));apply(dict(op='reactivate',episode=contents[0]['id']))
    phases['persisted_transitions_and_fail_closed_requests']=time.perf_counter()-tick;tick=time.perf_counter()
    publish(a.output/'snapshot.json',memory.snapshot());publish(a.output/'state-digests.json',states);publish(a.output/'refusals.json',refusals)
    check(file_sha(a.raw)==SOURCE_SHA256,'SOURCE_DRIFT');files={p.name:dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in a.output.iterdir() if p.is_file()}
    phases['snapshots_and_immutable_source_readback']=time.perf_counter()-tick;wall=time.perf_counter()-start
    publish(a.output/'report.json',dict(schema='episodic-source-memory-v1',complete=True,source_sha256=SOURCE_SHA256,files=files,
        episodes=6,events=len(memory.events),grounded_source_statements=3,grounded_world_claims=0,epistemic_distinction_learned=False,served=False,
        refusal_count=len(refusals),phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall))
    from audit.episodic_memory_readback import read
    print(json.dumps(read(a.output,a.raw)),flush=True)

if __name__=='__main__':main()
