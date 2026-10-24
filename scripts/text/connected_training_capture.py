"""Training-only bytes and pinned provider provenance, with explicit licence scope."""
import argparse
import json
from pathlib import Path
import sys
import time
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from text.connected_source_capture import capture,check,LIMITS
from text.evaluation_multiscale import file_sha
from senses.trajectory_population import publish

MULTIRC_REV="0e52e6f98f9d2970c90bb86502ed7488eb720fc5"
MULTIRC_PARENT="fe72f0c1ba8a334dc3684c6fc59fe7e9546f22e68f253ebd8c6e4f3047bbdcd9"
MEMBER="splitv2/train_456-fixedIds.json"
HF_REV="a34ba204eb9a33b919c10cc08f4f1c8dae5ec070"
HF_MEMBER="data/train-00000-of-00001.parquet"
HF_PIN="b9a297b5ab55f1605c7682ffbb7042c26d7ecb9ff1e1aa5a820d4e791c8302d1"


def multirc(root,parent):
    check(parent is not None and file_sha(parent/'report.json')==MULTIRC_PARENT,"PARENT_PIN")
    p=json.loads((parent/'report.json').read_bytes());check(p['complete'] is True and p['source']=='multirc',"PARENT_STATE")
    for n,v in p['files'].items():check(file_sha(parent/n)==v['sha256'] and (parent/n).stat().st_size==v['bytes'],"PARENT_FILE")
    receipts={};base=f'https://raw.githubusercontent.com/CogComp/multirc/{MULTIRC_REV}/'
    for name,url in [('provider-resource.json','https://cogcomp.seas.upenn.edu/page/resource_view/view-backend.php?id=116'),('LICENSE',base+'LICENSE'),('README.md',base+'README.md')]:
        receipts[name]=capture(url,root/name,LIMITS['metadata_bytes'])
    resource=json.loads((root/'provider-resource.json').read_bytes())
    check('https://github.com/CogComp/multirc/blob/master/LICENSE' in resource['basic']['description'],'DATASET_LICENCE_BINDING')
    licence=(root/'LICENSE').read_text(encoding='utf-8');check('Research and Academic Use License' in licence and 'academic and research purposes' in licence,'LICENCE_CONTENT')
    with zipfile.ZipFile(parent/'mutlirc-v2.zip') as z:
        m=z.getinfo(MEMBER);check(m.file_size==3434340 and m.file_size<=LIMITS['archive_bytes'],'TRAINING_SIZE')
        raw=z.read(m);check(len(raw)==m.file_size,'TRAINING_TRUNCATED')
    with (root/'train.json').open('xb') as f:f.write(raw)
    receipts['train.json']=dict(archive_sha256=file_sha(parent/'mutlirc-v2.zip'),archive_member=MEMBER,bytes=len(raw),sha256=file_sha(root/'train.json'))
    return dict(files=receipts,parent_sha256=MULTIRC_PARENT,revision=MULTIRC_REV,representation='original-provider-json-training-member',
                licence='Research and Academic Use License',use_scope='research-only; no production/export admission')


def qasc(root):
    base=f'https://huggingface.co/datasets/allenai/qasc/resolve/{HF_REV}/';receipts={}
    for name,url in [('provider-tree.json',f'https://huggingface.co/api/datasets/allenai/qasc/tree/{HF_REV}/data'),('README.md',base+'README.md')]:
        receipts[name]=capture(url,root/name,LIMITS['metadata_bytes'])
    tree=json.loads((root/'provider-tree.json').read_bytes());entry=[e for e in tree if e['path']==HF_MEMBER]
    check(len(entry)==1 and entry[0]['size']==1967904 and entry[0]['lfs']['oid']==HF_PIN,'PROVIDER_FILE_PIN')
    check('cc-by-4.0' in (root/'README.md').read_text(encoding='utf-8'),'LICENCE_CONTENT')
    receipts['train.parquet']=capture(base+HF_MEMBER,root/'train.parquet',LIMITS['archive_bytes'])
    check(receipts['train.parquet']['bytes']==1967904 and receipts['train.parquet']['sha256']==HF_PIN,'TRAINING_FILE_PIN')
    return dict(files=receipts,parent_sha256=None,revision=HF_REV,representation='provider-Hugging-Face-Parquet; original-archive-equivalence-unproven',
                licence='CC-BY-4.0',use_scope='source inspection; population and scientific admission pending')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',choices=('multirc','qasc'),required=True);p.add_argument('--parent',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();phases={};errors=[];provenance={};tick=time.perf_counter()
    try:provenance=multirc(a.output,a.parent) if a.source=='multirc' else qasc(a.output)
    except (ValueError,OSError,zipfile.BadZipFile,KeyError) as exc:errors.append(dict(error=str(exc)))
    phases['provider_metadata_and_exact_training_payload']=time.perf_counter()-tick;tick=time.perf_counter()
    publish(a.output/'capture.json',provenance)
    files={p.name:dict(sha256=file_sha(p),bytes=p.stat().st_size) for p in a.output.iterdir() if p.is_file()}
    phases['cold_file_pins']=time.perf_counter()-tick;wall=time.perf_counter()-start
    rep=dict(schema='connected-training-capture-v1',source=a.source,files=files,provenance=provenance,errors=errors,complete=not errors,
             payload_rows_read=0,training_ready=False,heldout_labels_extracted=False,scientific_admission=False,served=False,
             phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(a.output/'report.json',rep);print(json.dumps(dict(source=a.source,complete=rep['complete'],errors=errors,report_sha256=file_sha(a.output/'report.json'),seconds=wall)))
    return 2 if errors else 0


if __name__=='__main__':raise SystemExit(main())
