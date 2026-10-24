"""Independent source-byte, native-clock, containment and private-target readback."""
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

def need(ok,code):
    if not ok:raise ValueError('MNLM_PERCEPTION_READBACK_'+code)

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def ident(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def pair(v):return [v.numerator,v.denominator]

def bind(source,root,*,replay=True):
    pins={'sample_videos.zip':(225370639,'68e5cca8c3064859f273ec7a52c15d5eb4a337032d12e6d2f379079825a72537'),
          'sample_annotations.zip':(2305618,'2d919a4b7154bf66a71ea3964be4b413c1995e6c90de54d3da9422f3ebfdfaea'),
          'README.md':(30386,'9f739a07616e3f60af0723800d813ca87953496ba9e288fc62b4bfd7578afa85'),
          'cut_frame_mapping_train.json':(39498,'cbea6c721fd150f861ec6fe8e7ff005792f754df1359b1364349b5abcaab6cc4')}
    for n,(size,h) in pins.items():need((source/n).stat().st_size==size and sha(source/n)==h,'SOURCE_PIN')
    with zipfile.ZipFile(source/'sample_annotations.zip') as z:annotations=json.loads(z.read('sample.json'))
    cuts=json.loads((source/'cut_frame_mapping_train.json').read_bytes());blobs={};probes={}
    with zipfile.ZipFile(source/'sample_videos.zip') as z:
        need(set(z.namelist())=={'videos/'+k+'.mp4' for k in annotations} and len(z.infolist())==8,'SOURCE_MEMBERS')
        for k in annotations:
            with z.open('videos/'+k+'.mp4') as f:blobs[k]=hashlib.file_digest(f,'sha256').hexdigest()
            need(sha(root/'media'/f'{k}.mp4')==blobs[k],'EXTRACTED_MEDIA')
            probes[k]=json.loads((root/'probes'/f'{k}.json').read_bytes())
    if replay:
        executable=shutil.which('ffprobe');need(executable is not None,'RUNTIME')
        runtime=json.loads((root/'runtime.json').read_bytes())
        need(runtime['executable_sha256']==sha(Path(executable)),'RUNTIME')
        for k in sorted(annotations):
            # Independent invocation, not the producer's command builder.
            cmd=[executable,'-v','error','-threads','0','-show_streams','-show_frames','-show_entries',
                 'stream=index,codec_type,codec_name,time_base,sample_rate,channels,width,height,start_pts,duration_ts,nb_frames:frame=media_type,stream_index,pts,duration,pkt_duration,nb_samples',
                 '-of','json',str(root/'media'/f'{k}.mp4')]
            p=subprocess.run(cmd,capture_output=True,check=True)
            need(not p.stderr.strip() and json.loads(p.stdout)==probes[k],'NATIVE_REPLAY')
    return dict(pins=pins,annotations=annotations,cuts=cuts,blobs=blobs,probes=probes,replayed=replay)

def graph_read(g,probe,clip,blob,split,*,exposure='development_exposed'):
    flags={k:False for k in ('scientific_admission','training_ready','world_facts_established','physical_sync_calibrated',
                            'transcript_available','sealed_evaluation','decoded_media_cutoff_enforced','meaning_asserted')}
    need(set(g)=={'schema','clip','blob_sha256','split_group','provider_split','exposure','clock','native_media','language','nodes','contexts','flags'},'PUBLIC_SCHEMA')
    need(g['schema']=='perception-container-episode-v1' and g['flags']==flags,'FLAGS')
    need(g['clip']==clip and g['blob_sha256']==blob and g['split_group']==ident(['perception-source-clip',blob]) and
         g['provider_split']==split and g['exposure']==exposure
         and exposure in ('development_exposed','native_timing_only'),'IDENTITY_SPLIT')
    need(g['clock']==dict(id=ident(['container-presentation-clock',blob]),unit='rational_seconds',
         origin='container_presentation_timestamps',physical_sync_calibrated=False),'CLOCK')
    need(g['native_media']==['audio','image'] and g['language']==dict(state='absent',reason='no_attested_timed_transcript; annotations_are_private'),'LANGUAGE')
    expected=[];streams={s['index']:s for s in probe['streams']};index=defaultdict(int)
    need(len(streams)==2 and {s['codec_type'] for s in streams.values()}=={'audio','video'},'NATIVE_MEDIA')
    for f in probe['frames']:
        s=streams[f['stream_index']];audio=s['codec_type']=='audio';tb=Fraction(s['time_base']);begin=f['pts']*tb
        end=begin+(Fraction(f['nb_samples'],int(s['sample_rate'])) if audio else f.get('duration',f.get('pkt_duration',0))*tb)
        n=dict(stream=s['index'],ordinal=index[s['index']],modality='audio' if audio else 'image',pts=f['pts'],
               time_base=pair(tb),span=[pair(begin),pair(end)],available_after=pair(end),sample_count=f['nb_samples'] if audio else None)
        index[s['index']]+=1;expected.append(dict(id=ident([blob,n]),**n))
    expected.sort(key=lambda n:(Fraction(*n['span'][0]),n['stream'],n['ordinal']))
    need(g['nodes']==expected,'NATIVE_COORDINATES')
    # Endpoint sweep reconstructs window membership independently of producer's per-node loop.
    starts=defaultdict(list);ends=defaultdict(list)
    for i,n in enumerate(expected):
        a,b=(Fraction(*x) for x in n['span']);starts[a.numerator//a.denominator].append(i)
        ends[-(-b.numerator//b.denominator)].append(i)
    active=set();contexts=[]
    for second in range(min(starts),max(ends)):
        active.difference_update(ends.get(second,[]));active.update(starts.get(second,[]))
        if not active:continue
        members=[expected[i] for i in sorted(active)]
        span=[pair(min(Fraction(*n['span'][0]) for n in members)),pair(max(Fraction(*n['span'][1]) for n in members))]
        contexts.append(dict(id=ident([blob,'one-second',second]),level='one_second_presentation_context',
            nominal_span=[[second,1],[second+1,1]],span=span,available_after=span[1],members=[n['id'] for n in members]))
    span=[pair(min(Fraction(*n['span'][0]) for n in expected)),pair(max(Fraction(*n['span'][1]) for n in expected))]
    contexts.append(dict(id=ident([blob,'whole-episode']),level='episode',nominal_span=span,span=span,available_after=span[1],members=[n['id'] for n in expected]))
    need(g['contexts']==contexts,'CONTEXT_CLOSURE')
    return dict(clip=clip,blob_sha256=blob,provider_split=split,image_frames=sum(n['modality']=='image' for n in expected),
        audio_blocks=sum(n['modality']=='audio' for n in expected),audio_samples=sum(n['sample_count'] or 0 for n in expected),contexts=len(contexts))

def read(root,bound):
    h=json.loads((root/'report.json').read_bytes())
    need(h['schema']=='perception-episodes-build-v1' and h['complete'] is True and h['errors']==[],'STATE')
    need(h['source_pins']=={n:dict(bytes=s,sha256=b) for n,(s,b) in bound['pins'].items()} and
         h['revision']=='3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf','SOURCE_PINS')
    expected_names={'runtime.json'}|{f'{d}/{k}.{ext}' for k in bound['annotations'] for d,ext in (('media','mp4'),('probes','json'),('public','json'),('private','json'))}
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    need(actual==expected_names|{'report.json'} and set(h['files'])==expected_names,'FILES')
    for n in expected_names:need(h['files'][n]==dict(bytes=(root/n).stat().st_size,sha256=sha(root/n)),'DIGEST')
    need(h['code_sha256']==sha(Path(__file__).resolve().parents[1]/'senses/perception_episodes.py'),'CODE')
    need(h['wall_seconds']>0 and .9<=h['phase_coverage']<=1 and all(t>=0 for t in h['phases_seconds'].values()) and
         abs(sum(h['phases_seconds'].values())/h['wall_seconds']-h['phase_coverage'])<1e-12,'COST')
    counts=[];discrepancies=[]
    for clip,ann in sorted(bound['annotations'].items()):
        need(sha(root/'media'/f'{clip}.mp4')==bound['blobs'][clip],'EXTRACTED_MEDIA')
        probe=json.loads((root/'probes'/f'{clip}.json').read_bytes());need(probe==bound['probes'][clip],'NATIVE_REPLAY')
        g=json.loads((root/'public'/f'{clip}.json').read_bytes());need(h['flags']==g['flags'],'FLAGS')
        row=graph_read(g,probe,clip,bound['blobs'][clip],ann['metadata']['split']);counts.append(row)
        private=json.loads((root/'private'/f'{clip}.json').read_bytes())
        need(private==dict(annotations=ann,annotation_source_sha256=bound['pins']['sample_annotations.zip'][1],train_cut_frame=bound['cuts'].get(clip),
             label_role='provider_annotation_not_world_fact_or_sensory_input',qa_training_authorized=False,qa_cutoff_applied=False),'PRIVATE_SCOPE')
        discrepancies.append(dict(clip=clip,provider_audio_samples=ann['metadata']['audio_samples'],decoded_audio_samples=row['audio_samples'],
            delta=row['audio_samples']-ann['metadata']['audio_samples'],provider_frames=ann['metadata']['num_frames'],decoded_frames=row['image_frames']))
    need([{k:v for k,v in e.items() if k!='command'} for e in h['episodes']]==counts,'COUNTS')
    return dict(report_sha256=sha(root/'report.json'),episodes=counts,provider_native_differences=discrepancies,
        native_probe_replayed=bound['replayed'],private_labels_excluded=True,acceptance='author-checked',meaning_asserted=False)
