"""Bounded official MuSiQue capture; provider code and held-out labels never run."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import time
import urllib.parse
import urllib.request
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha

COMMIT='922ac98f19a201998dbdae6d7f2887a5258dbdeb'
OBJECT='1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h'
SELECTED=('musique_ans_v1.0_train.jsonl','musique_full_v1.0_train.jsonl')
LIMIT=320*1024**2


def check(ok,code):
    if not ok:raise ValueError('MNLM_MUSIQUE_'+code)


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__();self.active=False;self.action=None;self.fields={}

    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='form' and a.get('id')=='download-form':
            check(self.action is None,'DUPLICATE_FORM');self.active=True;self.action=a.get('action')
        if tag=='input' and self.active and a.get('type')=='hidden':
            check(a.get('name') not in self.fields,'DUPLICATE_FORM_FIELD');self.fields[a.get('name')]=a.get('value')

    def handle_endtag(self,tag):
        if tag=='form':self.active=False


def metadata(url,path):
    with urllib.request.urlopen(url,timeout=30) as r:
        raw=r.read(128*1024+1);check(0<len(raw)<=128*1024,'METADATA_BOUND')
    with path.open('xb') as f:f.write(raw)
    return dict(url=url,bytes=len(raw),sha256=file_sha(path))


def capture(root,receipts):
    for name in ('README.md','LICENSE','download_data.sh'):
        receipts[name]=metadata(f'https://raw.githubusercontent.com/StonyBrookNLP/musique/{COMMIT}/{name}',root/name)
    check(OBJECT in (root/'download_data.sh').read_text(encoding='utf-8'),'OFFICIAL_OBJECT')
    check('CC BY 4.0' in (root/'README.md').read_text(encoding='utf-8'),'LICENCE')
    url='https://drive.google.com/uc?'+urllib.parse.urlencode(dict(export='download',id=OBJECT))
    with urllib.request.urlopen(url,timeout=30) as r:
        raw=r.read(128*1024+1);check(len(raw)<=128*1024,'FORM_BOUND')
    form=DownloadForm();form.feed(raw.decode('utf-8'))
    check(form.action=='https://drive.usercontent.google.com/download' and form.fields.get('id')==OBJECT
          and form.fields.get('export')=='download' and form.fields.get('confirm')=='t'
          and set(form.fields)<= {'id','export','confirm','uuid'},'OFFICIAL_FORM')
    request=urllib.request.Request(form.action+'?'+urllib.parse.urlencode(form.fields),headers={'Accept-Encoding':'identity'})
    tick=time.perf_counter();count=0;digest=hashlib.sha256();path=root/'musique_data_v1.0.zip'
    with urllib.request.urlopen(request,timeout=30) as r:
        if 'html' in r.headers.get('Content-Type','').lower():
            raw=r.read(128*1024+1);check(len(raw)<=128*1024,'REFUSAL_BOUND')
            with (root/'http-refusal.html').open('xb') as f:f.write(raw)
            title=re.search(r'<title>(.*?)</title>',raw.decode('utf-8',errors='replace'),re.I|re.S)
            quota=title is not None and title.group(1).strip()=='Google Drive - Quota exceeded'
            check(False,'SOURCE_QUOTA_EXCEEDED' if quota else 'ARCHIVE_HTTP_TYPE')
        length=r.headers.get('Content-Length');expected=int(length) if length is not None else None
        check(expected is None or 0<expected<=LIMIT,'ARCHIVE_HTTP_SIZE')
        with path.open('xb') as out:
            while block:=r.read(1024*1024):
                count+=len(block);check(count<=LIMIT and time.perf_counter()-tick<=180,'ARCHIVE_CAPTURE_BOUND')
                out.write(block);digest.update(block)
        check(count>0 and (expected is None or count==expected),'ARCHIVE_HTTP_TRUNCATED')
        receipts[path.name]=dict(url='https://drive.google.com/file/d/'+OBJECT+'/view',bytes=count,sha256=digest.hexdigest(),
                                content_length=expected,content_type=r.headers.get('Content-Type'),etag=r.headers.get('ETag'))
    check(file_sha(path)==digest.hexdigest(),'ARCHIVE_COLD_HASH')
    return receipts


def extract(root):
    inventory=[];selected={}
    with zipfile.ZipFile(root/'musique_data_v1.0.zip') as z:
        members=z.infolist();names=[m.filename for m in members]
        check(len(names)==len(set(names)),'ARCHIVE_DUPLICATE')
        check(sum(m.file_size for m in members)<=1024**3,'ARCHIVE_INVENTORY_BOUND')
        for m in members:
            p=PurePosixPath(m.filename)
            check(not p.is_absolute() and '..' not in p.parts and '\\' not in m.filename
                  and not stat.S_ISLNK(m.external_attr>>16) and not (m.flag_bits&1),'ARCHIVE_MEMBER')
            inventory.append(dict(name=m.filename,bytes=m.file_size,compressed_bytes=m.compress_size,crc32=m.CRC,directory=m.is_dir()))
        for name in SELECTED:
            found=[m for m in members if PurePosixPath(m.filename).name==name and not m.is_dir()]
            check(len(found)==1 and 0<found[0].file_size<=512*1024**2,'TRAIN_MEMBER')
            m=found[0];count=0
            with z.open(m) as src,(root/name).open('xb') as out:
                while block:=src.read(1024*1024):count+=len(block);out.write(block)
            check(count==m.file_size,'TRAIN_MEMBER_TRUNCATED')
            selected[name]=dict(archive_member=m.filename,bytes=count,sha256=file_sha(root/name),crc32=m.CRC)
    publish(root/'archive-inventory.json',inventory)
    return selected


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();phases={};errors=[];receipts={};selected={}
    tick=time.perf_counter()
    try:
        capture(a.output,receipts);phases['official_bounded_capture']=time.perf_counter()-tick;tick=time.perf_counter()
        selected=extract(a.output);phases['exact_named_training_members']=time.perf_counter()-tick;tick=time.perf_counter()
    except (OSError,ValueError,zipfile.BadZipFile) as exc:
        errors.append(dict(error=str(exc)));phases['failed_phase']=time.perf_counter()-tick;tick=time.perf_counter()
    publish(a.output/'capture.json',dict(provider_commit=COMMIT,files=receipts,credentials_used=False,
                                       archive_obtained='musique_data_v1.0.zip' in receipts))
    files={f.name:dict(bytes=f.stat().st_size,sha256=file_sha(f)) for f in sorted(a.output.iterdir()) if f.is_file()}
    phases['cold_source_pins']=time.perf_counter()-tick;wall=time.perf_counter()-started
    report=dict(schema='musique-source-v1',provider_commit=COMMIT,selected=selected,files=files,errors=errors,complete=not errors,
                development_labels_extracted=False,test_labels_extracted=False,scientific_admission=False,training_ready=False,served=False,
                source_truth='provider-annotated-composed-questions-and-context-support',world_truth_adjudicated=False,
                attribution='Trivedi, Balasubramanian, Khot and Sabharwal 2022, MuSiQue: Multihop Questions via Single-hop Question Composition, TACL; CC BY4.0',
                phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(a.output/'report.json',report);print(json.dumps({k:report[k] for k in ('complete','selected','errors','wall_seconds','phase_coverage')}))
    return 0 if not errors else 2


if __name__=='__main__':raise SystemExit(main())
