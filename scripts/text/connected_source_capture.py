"""Bounded original-provider archive capture; no corpus payload parsing or training."""
import argparse
from pathlib import Path,PurePosixPath
import stat
import sys
import tarfile
import time
import urllib.request
import urllib.parse
import zipfile
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha

REVISION="c44a320c2bb682d614a675d2ba4be072150db4eb"
SOURCES={
    "multirc":dict(metadata="https://cogcomp.seas.upenn.edu/multirc/",archive="https://cogcomp.seas.upenn.edu/multirc/data/mutlirc-v2.zip",
                   metadata_name="provider.html",archive_name="mutlirc-v2.zip",revision=None),
    "qasc":dict(metadata=f"https://raw.githubusercontent.com/allenai/qasc/{REVISION}/README.md",archive="https://data.allenai.org/downloads/qasc/qasc_dataset.tar.gz",
                metadata_name="README.md",archive_name="qasc_dataset.tar.gz",revision=REVISION)}
LIMITS=dict(archive_bytes=16*1024**2,metadata_bytes=128*1024,uncompressed_bytes=128*1024**2,entries=256,socket_seconds=30,capture_seconds=180)


def check(ok,code):
    if not ok:raise ValueError("MNLM_CONNECTED_SOURCE_"+code)


def capture(url,path,limit):
    tick=time.perf_counter();request=urllib.request.Request(url,headers={"Accept-Encoding":"identity"})
    with urllib.request.urlopen(request,timeout=LIMITS["socket_seconds"]) as r:
        check(urllib.parse.urlparse(r.url).scheme=="https","HTTPS_REDIRECT");length=r.headers.get("Content-Length")
        expected=int(length) if length is not None else None;check(expected is None or 0<expected<=limit,"HTTP_SIZE")
        count=0
        with path.open('xb') as out:
            while block:=r.read(min(128*1024,limit+1-count)):
                count+=len(block);check(count<=limit and time.perf_counter()-tick<=LIMITS["capture_seconds"],"CAPTURE_BOUND");out.write(block)
        check(count>0 and (expected is None or count==expected),"HTTP_TRUNCATED")
        final=urllib.parse.urlsplit(r.url)._replace(query='',fragment='')
        receipt=dict(url=url,final_url=urllib.parse.urlunsplit(final),bytes=count,content_length=expected,content_type=r.headers.get('Content-Type'),
                     etag=r.headers.get('ETag'),modified=r.headers.get('Last-Modified'))
    return {**receipt,"sha256":file_sha(path)}


def validate_inventory(entries):
    check(len(entries)<=LIMITS["entries"] and sum(e['bytes'] for e in entries)<=LIMITS['uncompressed_bytes'],"ARCHIVE_BOUND")
    names=[e['name'] for e in entries];check(len(names)==len(set(names)),"ARCHIVE_DUPLICATE")
    for e in entries:
        p=PurePosixPath(e['name']);check(not p.is_absolute() and '..' not in p.parts and '\\' not in e['name'] and ':' not in e['name']
            and e['kind'] in ('file','directory') and e['bytes']>=0,"ARCHIVE_MEMBER")
    return entries


def inventory(path):
    entries=[]
    if path.suffix=='.zip':
        with zipfile.ZipFile(path) as z:
            for m in z.infolist():
                check(not stat.S_ISLNK(m.external_attr>>16) and not (m.flag_bits&1),"ARCHIVE_LINK_OR_ENCRYPTED")
                entries.append(dict(name=m.filename,bytes=m.file_size,compressed_bytes=m.compress_size,crc32=m.CRC,
                                    kind='directory' if m.is_dir() else 'file'))
    else:
        with tarfile.open(path,'r:gz') as tf:
            for m in tf:
                entries.append(dict(name=m.name,bytes=m.size,kind='file' if m.isfile() else 'directory' if m.isdir() else 'other'))
                check(len(entries)<=LIMITS['entries'] and sum(e['bytes'] for e in entries)<=LIMITS['uncompressed_bytes'],"ARCHIVE_BOUND")
    return validate_inventory(entries)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',choices=SOURCES,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();start=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False);config=SOURCES[a.source];phases={};receipts={};errors=[];entries=[]
    for kind in ('metadata','archive'):
        tick=time.perf_counter()
        try:receipts[kind]=capture(config[kind],a.output/config[kind+'_name'],LIMITS[kind+'_bytes'])
        except (ValueError,OSError) as exc:errors.append(dict(phase=kind,error=str(exc)))
        phases[kind+'_capture']=time.perf_counter()-tick
    tick=time.perf_counter()
    if not errors:
        try:
            meta=(a.output/config['metadata_name']).read_text(encoding='utf-8')
            check(config['archive'].split('/')[-1] in meta,'PROVIDER_LINK')
            if a.source=='qasc':check('https://creativecommons.org/licenses/by/4.0/' in meta,'LICENCE_LINK')
            entries=inventory(a.output/config['archive_name']);publish(a.output/'archive-inventory.json',entries)
        except (ValueError,OSError,tarfile.TarError,zipfile.BadZipFile) as exc:errors.append(dict(phase='inventory',error=str(exc)))
    phases['provider_binding_and_inventory']=time.perf_counter()-tick;tick=time.perf_counter()
    publish(a.output/'capture.json',dict(source=a.source,revision=config['revision'],files=receipts,limits=LIMITS,credentials_used=False))
    files={p.name:dict(sha256=file_sha(p),bytes=p.stat().st_size) for p in a.output.iterdir() if p.is_file()}
    phases['cold_pins']=time.perf_counter()-tick;wall=time.perf_counter()-start
    rep=dict(schema='connected-grounding-capture-v1',source=a.source,revision=config['revision'],files=files,errors=errors,complete=not errors,
        archive_obtained='archive' in receipts,payload_rows_read=0,training_ready=False,heldout_labels_extracted=False,scientific_admission=False,served=False,
        phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(a.output/'report.json',rep);print(json.dumps(dict(source=a.source,complete=rep['complete'],errors=errors,inventory=entries,report_sha256=file_sha(a.output/'report.json'))))
    return 2 if errors else 0


if __name__=='__main__':raise SystemExit(main())
