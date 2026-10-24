"""Real container-clock episodes; no encoder, transcript inference or truth promotion.

PC-07/11/13: one archive pass, one native decoder probe per clip, linear views.
One-second contexts are declared presentation windows, not discovered events.
"""
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha

REVISION = '3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf'
PINS = {
    'sample_videos.zip': (225370639, '68e5cca8c3064859f273ec7a52c15d5eb4a337032d12e6d2f379079825a72537'),
    'sample_annotations.zip': (2305618, '2d919a4b7154bf66a71ea3964be4b413c1995e6c90de54d3da9422f3ebfdfaea'),
    'README.md': (30386, '9f739a07616e3f60af0723800d813ca87953496ba9e288fc62b4bfd7578afa85'),
    'cut_frame_mapping_train.json': (39498, 'cbea6c721fd150f861ec6fe8e7ff005792f754df1359b1364349b5abcaab6cc4')}
FLAGS = dict(scientific_admission=False,training_ready=False,world_facts_established=False,
             physical_sync_calibrated=False,transcript_available=False,sealed_evaluation=False,
             decoded_media_cutoff_enforced=False,meaning_asserted=False)

def check(ok, code):
    if not ok: raise ValueError('MNLM_PERCEPTION_'+code)

def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def rational(x): return [x.numerator,x.denominator]

def source(root):
    for name,(size,sha) in PINS.items():
        p=root/name
        check(p.is_file() and p.stat().st_size==size and file_sha(p)==sha,'SOURCE_PIN')
    with zipfile.ZipFile(root/'sample_annotations.zip') as z:
        check(z.namelist()==['sample.json'] and z.getinfo('sample.json').file_size==19931707,'ANNOTATION_ARCHIVE')
        annotations=json.loads(z.read('sample.json'))
    check(len(annotations)==8,'SOURCE_COUNT')
    cuts=json.loads((root/'cut_frame_mapping_train.json').read_bytes())
    return annotations,cuts

def command(executable,path):
    return [executable,'-v','error','-threads','0','-show_streams','-show_frames',
            '-show_entries','stream=index,codec_type,codec_name,time_base,sample_rate,channels,width,height,start_pts,duration_ts,nb_frames:frame=media_type,stream_index,pts,duration,pkt_duration,nb_samples',
            '-of','json',str(path)]

def timeline(clip,blob,probe,split):
    streams={s['index']:s for s in probe['streams']}
    check(len(streams)==2 and sorted(s['codec_type'] for s in streams.values())==['audio','video'],'NATIVE_MEDIA')
    clock=identity(['container-presentation-clock',blob]);nodes=[];counts=defaultdict(int);last={}
    for f in probe['frames']:
        stream=streams[f['stream_index']];si=stream['index'];kind=stream['codec_type']
        check(f['media_type']==kind and type(f.get('pts')) is int,'FRAME_COORDINATE')
        tb=Fraction(stream['time_base']);start=f['pts']*tb
        duration=Fraction(f['nb_samples'],int(stream['sample_rate'])) if kind=='audio' else f.get('duration',f.get('pkt_duration',0))*tb
        check(duration>0 and start>=0 and (si not in last or start>=last[si]),'FRAME_ORDER')
        end=start+duration;last[si]=start;ordinal=counts[si];counts[si]+=1
        body=dict(stream=si,ordinal=ordinal,modality='audio' if kind=='audio' else 'image',
                  pts=f['pts'],time_base=rational(tb),span=[rational(start),rational(end)],
                  available_after=rational(end),sample_count=f['nb_samples'] if kind=='audio' else None)
        nodes.append(dict(id=identity([blob,body]),**body))
    check(all(counts[s]>0 for s in streams),'EMPTY_STREAM')
    nodes.sort(key=lambda n:(Fraction(*n['span'][0]),n['stream'],n['ordinal']))
    bins=defaultdict(list)
    for n in nodes:
        lo,hi=(Fraction(*v) for v in n['span'])
        # Each native block participates in every intersecting one-second view.
        for index in range(lo.numerator//lo.denominator,-(-hi.numerator//hi.denominator)):
            bins[index].append(n)
    contexts=[]
    for index,members in sorted(bins.items()):
        span=[rational(min(Fraction(*n['span'][0]) for n in members)),
              rational(max(Fraction(*n['span'][1]) for n in members))]
        contexts.append(dict(id=identity([blob,'one-second',index]),level='one_second_presentation_context',
            nominal_span=[[index,1],[index+1,1]],span=span,available_after=span[1],members=[n['id'] for n in members]))
    all_span=[rational(min(Fraction(*n['span'][0]) for n in nodes)),rational(max(Fraction(*n['span'][1]) for n in nodes))]
    contexts.append(dict(id=identity([blob,'whole-episode']),level='episode',nominal_span=all_span,
        span=all_span,available_after=all_span[1],members=[n['id'] for n in nodes]))
    return dict(schema='perception-container-episode-v1',clip=clip,blob_sha256=blob,
        split_group=identity(['perception-source-clip',blob]),provider_split=split,exposure='development_exposed',
        clock=dict(id=clock,unit='rational_seconds',origin='container_presentation_timestamps',
                   physical_sync_calibrated=False),native_media=['audio','image'],
        language=dict(state='absent',reason='no_attested_timed_transcript; annotations_are_private'),
        nodes=nodes,contexts=contexts,flags=FLAGS)

def select(graph, *, clock_id, split_group, through, required_media):
    """Select references only. Decoder-level enforcement is a separate required gate."""
    check(clock_id==graph['clock']['id'],'CLOCK')
    check(split_group==graph['split_group'],'SPLIT')
    check(type(through)==list and len(through)==2 and all(type(v)==int for v in through) and through[1]>0,'CUTOFF')
    cutoff=Fraction(*through);check(cutoff>=0,'CUTOFF')
    check(type(required_media)==list and required_media and set(required_media)<=set(graph['native_media']),'MISSING_MODALITY')
    nodes=[n for n in graph['nodes'] if Fraction(*n['available_after'])<=cutoff and n['modality'] in required_media]
    check(set(required_media)=={n['modality'] for n in nodes},'UNAVAILABLE_MODALITY')
    return dict(clock_id=clock_id,split_group=split_group,through=through,nodes=nodes,
        contexts=[c for c in graph['contexts'] if Fraction(*c['available_after'])<=cutoff],
        reference_only=True,decoded_media_cutoff_enforced=False,meaning_asserted=False)

def main(a):
    start=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False);phases={};errors=[];episodes=[]
    annotations,cuts=source(a.source)
    phases['source_pin_and_annotation_parse']=time.perf_counter()-start;tick=time.perf_counter()
    executable=shutil.which(a.ffprobe);check(executable is not None,'FFPROBE_MISSING')
    runtime=dict(executable_sha256=file_sha(Path(executable)),version=subprocess.run([executable,'-version'],capture_output=True,check=True).stdout.decode())
    publish(a.output/'runtime.json',runtime)
    for d in ('media','probes','public','private'):(a.output/d).mkdir()
    phases['runtime_and_output_setup']=time.perf_counter()-tick
    with zipfile.ZipFile(a.source/'sample_videos.zip') as z:
        check(set(z.namelist())=={f'videos/{k}.mp4' for k in annotations} and len(z.infolist())==8,'MEDIA_ARCHIVE')
        for clip in sorted(annotations):
            tick=time.perf_counter()
            try:
                member=z.getinfo(f'videos/{clip}.mp4');check(not member.flag_bits&1 and 0<member.file_size<=225370639,'MEDIA_MEMBER')
                path=a.output/'media'/f'{clip}.mp4'
                with z.open(member) as inp,path.open('xb') as out:shutil.copyfileobj(inp,out,1024**2)
                blob=file_sha(path);cmd=command(executable,path)
                result=subprocess.run(cmd,capture_output=True,check=True)
                check(not result.stderr.strip(),'PROBE_STDERR')
                (a.output/'probes'/f'{clip}.json').write_bytes(result.stdout)
                probe=json.loads(result.stdout);ann=annotations[clip];meta=ann['metadata']
                check(meta['video_id']==clip and meta['split'] in ('train','valid'),'PROVIDER_ID')
                graph=timeline(clip,blob,probe,meta['split'])
                publish(a.output/'public'/f'{clip}.json',graph)
                publish(a.output/'private'/f'{clip}.json',dict(annotations=ann,annotation_source_sha256=PINS['sample_annotations.zip'][1],
                    train_cut_frame=cuts.get(clip),label_role='provider_annotation_not_world_fact_or_sensory_input',
                    qa_training_authorized=False,qa_cutoff_applied=False))
                episodes.append(dict(clip=clip,blob_sha256=blob,provider_split=meta['split'],
                    image_frames=sum(n['modality']=='image' for n in graph['nodes']),
                    audio_blocks=sum(n['modality']=='audio' for n in graph['nodes']),
                    audio_samples=sum(n['sample_count'] or 0 for n in graph['nodes']),
                    contexts=len(graph['contexts']),command=cmd))
            except (OSError,ValueError,KeyError,TypeError,subprocess.CalledProcessError) as exc:
                errors.append(dict(clip=clip,error=str(exc),type=type(exc).__name__))
            phases[clip]=time.perf_counter()-tick
    tick=time.perf_counter();files={p.relative_to(a.output).as_posix():dict(bytes=p.stat().st_size,sha256=file_sha(p))
                                  for p in sorted(a.output.rglob('*')) if p.is_file()}
    phases['cold_artifact_hashes']=time.perf_counter()-tick;wall=time.perf_counter()-start
    report=dict(schema='perception-episodes-build-v1',complete=not errors and len(episodes)==8,errors=errors,
        source_pins={n:dict(bytes=size,sha256=sha) for n,(size,sha) in PINS.items()},revision=REVISION,
        episodes=episodes,files=files,flags=FLAGS,code_sha256=file_sha(Path(__file__)),
        phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,acceptance='author-checked')
    publish(a.output/'report.json',report)
    print(json.dumps(dict(complete=report['complete'],errors=errors,episodes=[{k:v for k,v in e.items() if k!='command'} for e in episodes],
        wall_seconds=wall,report_sha256=file_sha(a.output/'report.json'))))
    return 0 if report['complete'] else 2

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--ffprobe',default='ffprobe');raise SystemExit(main(p.parse_args()))
