"""Real decode replay, corrupt-copy audits and bounded episode-view selection."""
import argparse
import copy
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.perception_episodes_readback import bind,read,need,sha
from senses.perception_episodes import select
from senses.trajectory_population import publish

def main(a):
    start=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False);phases={}
    bound=bind(a.source,a.episodes,replay=True);publish(a.output/'cold.json',read(a.episodes,bound))
    shadow=a.output/'owned-shadow';shutil.copytree(a.episodes,shadow)
    originals={p.relative_to(shadow).as_posix():p.read_bytes() for p in shadow.rglob('*.json')}
    original_hashes={p.relative_to(a.episodes).as_posix():sha(p) for p in a.episodes.rglob('*') if p.is_file()}
    phases['source_pin_native_decode_replay_and_copy']=time.perf_counter()-start;tick=time.perf_counter();controls=[]
    def change(v,path,new):
        t=v
        for k in path[:-1]:t=t[k]
        t[path[-1]]=new;return v
    def corrupt(name,file,edit,expected):
        value=edit(json.loads(originals[file]));(shadow/file).write_text(json.dumps(value),encoding='utf-8')
        if file!='report.json':
            h=json.loads(originals['report.json']);h['files'][file]=dict(bytes=(shadow/file).stat().st_size,sha256=sha(shadow/file))
            (shadow/'report.json').write_text(json.dumps(h),encoding='utf-8')
        try:read(shadow,bound)
        except ValueError as exc:
            need(str(exc)=='MNLM_PERCEPTION_READBACK_'+expected,'CONTROL_REASON')
            controls.append(dict(name=name,observed=str(exc),refused=True,before='cold accepted',after='original bytes restored'))
        else:raise ValueError('MNLM_PERCEPTION_FSV_CONTROL_ACCEPTED')
        finally:
            for n,b in originals.items():(shadow/n).write_bytes(b)
    clip=sorted(bound['annotations'])[0];pub=f'public/{clip}.json';priv=f'private/{clip}.json';probe=f'probes/{clip}.json'
    corrupt('wrong source clip',pub,lambda v:change(v,['blob_sha256'],'0'*64),'IDENTITY_SPLIT')
    corrupt('validation relabelled as train',f'public/video_1580.json',lambda v:change(v,['provider_split'],'train'),'IDENTITY_SPLIT')
    corrupt('different source clock',pub,lambda v:change(v,['clock','id'],'0'*64),'CLOCK')
    corrupt('25fps substituted for native timebase',pub,lambda v:change(v,['nodes',0,'time_base'],[1,25]),'NATIVE_COORDINATES')
    corrupt('early native availability',pub,lambda v:change(v,['nodes',0,'available_after'],[0,1]),'NATIVE_COORDINATES')
    corrupt('omitted last native block',pub,lambda v:change(v,['nodes'],v['nodes'][:-1]),'NATIVE_COORDINATES')
    corrupt('parent bypasses child future cutoff',pub,lambda v:change(v,['contexts',0,'available_after'],[0,1]),'CONTEXT_CLOSURE')
    corrupt('context omits native member',pub,lambda v:change(v,['contexts',0,'members'],v['contexts'][0]['members'][:-1]),'CONTEXT_CLOSURE')
    corrupt('answer in public payload',pub,lambda v:{**v,'answer_id':0},'PUBLIC_SCHEMA')
    corrupt('annotation called timed transcript',pub,lambda v:change(v,['language'],dict(state='present',text='invented')),'LANGUAGE')
    corrupt('native audio replaced by text',pub,lambda v:change(v,['native_media'],['image','text']),'LANGUAGE')
    corrupt('rehashed native PTS altered',probe,lambda v:change(v,['frames',0,'pts'],1),'NATIVE_REPLAY')
    corrupt('provider answers authorized as model input',priv,lambda v:change(v,['qa_training_authorized'],True),'PRIVATE_SCOPE')
    corrupt('unapplied QA cutoff claimed',priv,lambda v:change(v,['qa_cutoff_applied'],True),'PRIVATE_SCOPE')
    corrupt('world truth claimed',pub,lambda v:change(v,['flags','world_facts_established'],True),'FLAGS')
    corrupt('counts invented','report.json',lambda v:change(v,['episodes',0,'audio_samples'],1),'COUNTS')
    # Source boundary: owned hardlinks are immutable; replace only the small README.
    raw=a.output/'owned-source';raw.mkdir()
    for name in bound['pins']:
        if name=='README.md':(raw/name).write_bytes((a.source/name).read_bytes()+b'\nchanged')
        else:os.link(a.source/name,raw/name)
    try:bind(raw,a.episodes,replay=False)
    except ValueError as exc:
        need(str(exc)=='MNLM_PERCEPTION_READBACK_SOURCE_PIN','SOURCE_CONTROL')
        controls.append(dict(name='changed source licence bytes',observed=str(exc),refused=True,before='cold accepted',after='owned source restored'))
    else:raise ValueError('MNLM_PERCEPTION_FSV_SOURCE_ACCEPTED')
    finally:(raw/'README.md').write_bytes((a.source/'README.md').read_bytes())
    phases['corrupt_copies_and_restoration']=time.perf_counter()-tick;tick=time.perf_counter()
    selections=[]
    for clip in sorted(bound['annotations']):
        g=json.loads((a.episodes/'public'/f'{clip}.json').read_bytes())
        for cutoff in ([1,1],[10,1],g['contexts'][-1]['available_after']):
            got=select(g,clock_id=g['clock']['id'],split_group=g['split_group'],through=cutoff,required_media=['audio','image'])
            end=Fraction(*cutoff)
            need(got['nodes']==[n for n in g['nodes'] if Fraction(*n['span'][1])<=end],'SELECTION_NATIVE')
            need(all(Fraction(*c['span'][1])<=end for c in got['contexts']),'SELECTION_CONTEXT')
            selections.append(dict(clip=clip,through=cutoff,native=len(got['nodes']),contexts=len(got['contexts'])))
    g=json.loads((a.episodes/pub).read_bytes());base=dict(clock_id=g['clock']['id'],split_group=g['split_group'],through=[1,1],required_media=['audio','image'])
    for name,kwargs,code in [('wrong clock selection',dict(clock_id='0'*64),'CLOCK'),('wrong split selection',dict(split_group='0'*64),'SPLIT'),
                            ('absent text selection',dict(required_media=['text']),'MISSING_MODALITY'),('unavailable media at origin',dict(through=[0,1]),'UNAVAILABLE_MODALITY')]:
        try:select(g,**{**base,**kwargs})
        except ValueError as exc:
            need(str(exc)=='MNLM_PERCEPTION_'+code,'SELECTION_CONTROL')
            controls.append(dict(name=name,observed=str(exc),refused=True,before='valid selector accepted',after='input unchanged'))
        else:raise ValueError('MNLM_PERCEPTION_FSV_SELECTION_ACCEPTED')
    publish(a.output/'selections.json',selections)
    phases['real_cutoff_selections']=time.perf_counter()-tick;tick=time.perf_counter()
    publish(a.output/'restored.json',read(shadow,bound))
    for root in (a.episodes,shadow):need(original_hashes=={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()},'ORIGINAL_DRIFT')
    for n,(_,digest) in bound['pins'].items():need(sha(a.source/n)==sha(raw/n)==digest,'SOURCE_DRIFT')
    phases['cold_restoration_and_original_hashes']=time.perf_counter()-tick;wall=time.perf_counter()-start
    report=dict(controls=controls,selections=len(selections),source_report_sha256=sha(a.episodes/'report.json'),native_replay=True,
        originals_unchanged=True,shadow_restored=True,phases_seconds=phases,wall_seconds=wall,
        phase_coverage=sum(phases.values())/wall,acceptance='author-checked',meaning_asserted=False)
    publish(a.output/'report.json',report)
    print(json.dumps(dict(controls=len(controls),wall_seconds=wall,report_sha256=sha(a.output/'report.json'))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','episodes','output'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
