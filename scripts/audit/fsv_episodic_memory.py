"""Persisted source/proof/lifecycle corruptions, including fully rehashed event chains."""
import argparse,json,shutil,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.episodic_memory_readback import read,need,digest,read_json
from claims.episodic_memory import canonical
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha

def main():
    p=argparse.ArgumentParser()
    for name in ('memory','raw','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();phases={};cold=read(a.memory,a.raw)
    publish(a.output/'cold.json',cold);before={str(p):file_sha(p) for p in a.memory.rglob('*') if p.is_file()};before[str(a.raw)]=file_sha(a.raw)
    shadow=a.output/'shadow';shutil.copytree(a.memory,shadow);controls=[];phases['cold_and_shadow']=time.perf_counter()-start;tick=time.perf_counter()
    def probe(name,rel,change,code):
        path=shadow/rel;header=shadow/'report.json';saved={path:path.read_bytes(),header:header.read_bytes()};old=file_sha(path)
        try:
            value=[json.loads(l) for l in path.read_bytes().splitlines()] if rel.endswith('.jsonl') else read_json(path);change(value)
            if rel=='events.jsonl':
                head='0'*64
                for i,r in enumerate(value):
                    r['sequence']=i;r['previous']=head;r['sha256']=digest({k:v for k,v in r.items() if k!='sha256'});head=r['sha256']
                path.write_bytes(b''.join(canonical(r)+b'\n' for r in value))
            else:path.write_bytes(canonical(value)+b'\n')
            action=file_sha(path);need(action!=old,'FSV_NOOP')
            if rel!='report.json':
                r=read_json(header);r['files'][rel]=dict(bytes=path.stat().st_size,sha256=action);header.write_bytes(canonical(r)+b'\n')
            observed=None
            try:read(shadow,a.raw)
            except ValueError as e:observed=str(e)
            need(observed is not None and code in observed,'FSV_'+name)
        finally:
            for f,data in saved.items():f.write_bytes(data)
            need(all(f.read_bytes()==data for f,data in saved.items()),'FSV_RESTORE')
        controls.append(dict(name=name,boundary=rel,before_sha256=old,action_sha256=action,after_sha256=file_sha(path),observed=observed,refused=True))
    def wrong_source(v):
        ep=v[0]['event']['episode'];ep['content']+=' changed';ep['id']=digest({k:x for k,x in ep.items() if k!='id'})
    probe('source bytes changed','events.jsonl',wrong_source,'SOURCE_CONTENT')
    def borrowed_receipt(v):
        e=next(r['event'] for r in v if r['event']['op']=='promote');e['episode']=v[1]['event']['episode']['id']
    probe('source receipt used for content','events.jsonl',borrowed_receipt,'PROMOTION_SCOPE')
    def explanation(v):next(r['event'] for r in v if r['event']['op']=='check_source')['result']['claim_kind']='causal_explanation'
    probe('observation receipt claims causal explanation','events.jsonl',explanation,'CHECK_RECEIPT')
    def retire(v):next(r['event'] for r in v if r['event']['op']=='retire')['episode']=v[0]['event']['episode']['id']
    probe('retire active base support','events.jsonl',retire,'RETIRE_SUPPORT')
    def invalidation(v):next(r['event'] for r in v if r['event']['op']=='invalidate')['affected'].pop()
    probe('omit invalidated dependent','events.jsonl',invalidation,'INVALIDATION_CLOSURE')
    probe('world truth inferred from source','snapshot.json',lambda v:v['grounded_world_claims'].append(v['grounded_source_statements'][0]),'PROJECTION')
    probe('refusal changed memory','refusals.json',lambda v:v[0]['after'].update(head='f'*64),'REFUSAL_HEAD')
    probe('learned understanding claimed','report.json',lambda v:v.update(epistemic_distinction_learned=True),'CLAIMS')
    phases['eight_negative_controls']=time.perf_counter()-tick;tick=time.perf_counter();need(read(shadow,a.raw)==cold,'FSV_FINAL')
    after={p:file_sha(Path(p)) for p in before};need(before==after,'FSV_ORIGINALS');publish(a.output/'before.json',before);publish(a.output/'after.json',after)
    phases['restored_readback_and_sources']=time.perf_counter()-tick;wall=time.perf_counter()-start
    publish(a.output/'report.json',dict(source_report_sha256=cold['report_sha256'],controls=controls,originals_unchanged=True,shadow_restored=True,
        phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,served=False,acceptance='author-checked'))
    print(json.dumps(dict(report_sha256=file_sha(a.output/'report.json'),controls=len(controls),cold=cold)),flush=True)

if __name__=='__main__':main()
