"""Lossless legal atoms with bounded columnar groups and source-document addresses."""
import argparse
from bisect import bisect_right
import json
import os
from pathlib import Path
import resource
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canon.training_scope import POLICY_SHA256, require_training_source
from claims.process_conditions import write_committed
from text.constellation_corpus import ATOM_SCHEMA, FLAGS, SEGMENTER_ID, canonical, digest, document, file_sha, need
from text.constellation_input import stamp, validate_rows
from text.legal_stock import EXPERIMENT, SOURCES

SCHEMA = 'legal-lossless-constellation-corpus-v1'
MAX_ROWS, MAX_TEXT_BYTES = 8192, 8 << 20


def build(source_root, source_pin, output):
    output.mkdir(parents=True, exist_ok=False)
    start = tick = time.perf_counter(); cpu = time.process_time()
    raw=(source_root/'report.json').read_bytes(); stock=json.loads(raw)
    need(digest(raw)==source_pin and stock['complete'], 'LEGAL_STOCK')
    source_stamps={n:stamp(source_root/n) for n in stock['files']}
    for name,ref in stock['files'].items():
        need((source_root/name).stat().st_size==ref['bytes'] and file_sha(source_root/name)==ref['sha256'], 'LEGAL_FILE')
    phases=dict(authenticate=time.perf_counter()-tick,segment=0.,persist=0.,source=0.)
    counts=dict(documents=0,atoms=0,transitions=0,characters=0,bytes=0,short_spans=0,long_spans=0,
                beyond_old_document_cap=0,layout=0)
    sources=[];groups=[];pending=[];pending_bytes=0;published=0
    try:
        with pq.ParquetWriter(output/'atoms.parquet',ATOM_SCHEMA,compression='zstd') as writer, \
             (output/'documents.jsonl').open('xb') as docs:
            def flush():
                nonlocal pending,pending_bytes,published
                if not pending:return
                tick=time.perf_counter()
                groups.append(dict(start=published,rows=len(pending),text_bytes=pending_bytes))
                published+=len(pending)
                writer.write_table(pa.Table.from_pylist(pending,schema=ATOM_SCHEMA),row_group_size=len(pending))
                pending=[];pending_bytes=0;phases['persist']+=time.perf_counter()-tick
            for source_id,name,expected,_ in SOURCES:
                require_training_source(source_id,experiment=EXPERIMENT)
                source=dict(domain='legal',sha256=stock['files']['raw/'+name]['sha256'],path=name,
                    repo='CourtListener retained bulk export',revision=stock['files']['raw/'+name]['sha256'],
                    licence=next(iter(stock['counts'][name]['licences'])))
                sources.append(dict(source_id=source_id,**source));ordinal=0
                with (source_root/'raw'/name).open('rb') as inp:
                    while True:
                        tick=time.perf_counter();line=inp.readline()
                        if not line:phases['source']+=time.perf_counter()-tick;break
                        value=json.loads(line);phases['source']+=time.perf_counter()-tick;tick=time.perf_counter()
                        rows,meta=document(source,ordinal,dict(text=value['text']))
                        need(rows,'LEGAL_EMPTY_DOC')
                        meta.update(source_record=digest(canonical(dict(source=source_id+':'+name,ordinal=ordinal))),
                                    atom_start=counts['atoms'],atom_end=counts['atoms']+len(rows))
                        validate_rows(rows,meta)
                        counts['documents']+=1;counts['atoms']+=len(rows);counts['transitions']+=len(rows)-1
                        counts['characters']+=len(value['text']);counts['bytes']+=len(value['text'].encode())
                        counts['short_spans']+=sum(len(r['text'].strip())<24 for r in rows)
                        counts['long_spans']+=sum(len(r['text'].strip())>600 for r in rows)
                        counts['layout']+=sum(r['kind']=='layout' for r in rows)
                        counts['beyond_old_document_cap']+=max(0,len(rows)-48)
                        phases['segment']+=time.perf_counter()-tick;tick=time.perf_counter()
                        docs.write(canonical(meta)+b'\n');phases['persist']+=time.perf_counter()-tick
                        for r in rows:
                            size=r['byte_end']-r['byte_start']
                            if pending and (len(pending)>=MAX_ROWS or pending_bytes+size>MAX_TEXT_BYTES):flush()
                            pending.append(r);pending_bytes+=size
                        ordinal+=1
                        if counts['documents']%10000==0:
                            print(json.dumps(dict(documents=counts['documents'],atoms=counts['atoms'],
                                                  seconds=time.perf_counter()-start)),flush=True)
                need(ordinal==expected,'LEGAL_DOCUMENT_COUNT')
            flush();docs.flush();os.fsync(docs.fileno())
        tick=time.perf_counter()
        need(counts['documents']==stock['records'] and counts['bytes']==sum(c['text_bytes'] for c in stock['counts'].values()),
             'LEGAL_BYTE_CONSERVATION')
        need(all(stamp(source_root/n)==s for n,s in source_stamps.items()),'LEGAL_SOURCE_CHANGED')
        write_committed(output/'row-groups.json',groups)
        files={n:dict(bytes=(output/n).stat().st_size,sha256=file_sha(output/n))
               for n in ('atoms.parquet','documents.jsonl','row-groups.json')}
        phases['finalize']=time.perf_counter()-tick;seconds=time.perf_counter()-start
        need(sum(phases.values())/seconds>=.9,'LEGAL_TIMING')
        report=dict(schema=SCHEMA,complete=True,errors=[],flags=FLAGS,counts=counts,files=files,sources=sources,
            stock=dict(path=str(source_root),pin=source_pin),experiment=EXPERIMENT,policy_sha256=POLICY_SHA256,
            segmenter=SEGMENTER_ID,contiguous=True,full_fields=True,atom_cap=None,character_filter=None,
            row_group_limits=dict(atoms=MAX_ROWS,text_bytes=MAX_TEXT_BYTES,oversize_single_atom='preserve'),
            splits_frozen=False,training_ready=False,scientific_admission=False,model_updates=0,
            source_metadata_enters_inputs=False,clock='source-order only; not event chronology',
            files_source_unchanged=True,phases_seconds=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            code_sha256=file_sha(Path(__file__)),definition_sha256=file_sha(Path(__file__).with_name('constellation_corpus.py')),
            number_status='measured',acceptance='author-checked')
        pin=write_committed(output/'report.json',report)
        write_committed(output/'COMMITTED',dict(report_sha256=pin))
        print(json.dumps(dict(pin=pin,counts=counts,seconds=seconds)),flush=True)
    except BaseException as error:
        write_committed(output/'failure.json',dict(error=str(error),counts=counts,phases=phases))
        raise


class LegalCorpus:
    """Same document/batch surface as Corpus; no one-row-group-per-document assumption."""
    def __init__(self,root,pin):
        self.root,self.pin=Path(root),pin
        need(file_sha(self.root/'report.json')==pin,'LEGAL_CORPUS_PIN')
        self.report=r=json.loads((self.root/'report.json').read_bytes())
        need(r['schema']==SCHEMA and r['complete'] and not r['errors'] and r['flags']==FLAGS,'LEGAL_CORPUS_SCOPE')
        need(json.loads((self.root/'COMMITTED').read_bytes())==dict(report_sha256=pin),'LEGAL_COMMITTED')
        self.stamps={self.root/'report.json':stamp(self.root/'report.json')}
        for name,ref in r['files'].items():
            path=self.root/name
            need(path.stat().st_size==ref['bytes'] and file_sha(path)==ref['sha256'],'LEGAL_CORPUS_FILE')
            self.stamps[path]=stamp(path)
        self.table=pq.ParquetFile(self.root/'atoms.parquet')
        self.groups=json.loads((self.root/'row-groups.json').read_bytes())
        self.starts=[g['start'] for g in self.groups];cursor=0
        for i,g in enumerate(self.groups):
            need(g['start']==cursor and self.table.metadata.row_group(i).num_rows==g['rows'],'LEGAL_ROW_GROUPS')
            cursor+=g['rows']
        need(cursor==self.table.metadata.num_rows==r['counts']['atoms'] and len(self.groups)==self.table.num_row_groups,
             'LEGAL_ATOM_COUNT')
        with (self.root/'documents.jsonl').open('rb') as inp:self.documents=[json.loads(line) for line in inp]
        cursor=0
        for d in self.documents:
            need(d['atom_start']==cursor and d['atom_end']-cursor==sum(f['atoms'] for f in d['fields']),'LEGAL_DOC_RANGE')
            cursor=d['atom_end']
        need(cursor==r['counts']['atoms'] and len(self.documents)==r['counts']['documents'],'LEGAL_DOC_COUNT')
    def fresh(self):
        need(all(stamp(p)==v for p,v in self.stamps.items()),'LEGAL_CORPUS_CHANGED')
    def document(self,index):
        need(type(index) is int and 0<=index<len(self.documents),'LEGAL_DOCUMENT_INDEX');self.fresh()
        d=self.documents[index];start,end=d['atom_start'],d['atom_end'];g=bisect_right(self.starts,start)-1;rows=[]
        while g<len(self.groups) and self.starts[g]<end:
            offset=self.starts[g];table=self.table.read_row_group(g)
            lo=max(start-offset,0);hi=min(end-offset,len(table))
            rows.extend(table.slice(lo,hi-lo).to_pylist());g+=1
        validate_rows(rows,d);self.fresh();return rows
    def batches(self,count):
        need(type(count) is int and count>0,'LEGAL_BATCH_SIZE');self.fresh()
        for batch in self.table.iter_batches(batch_size=count):yield batch.to_pylist()
        self.fresh()
    def training_authority(self):
        raise ValueError('MNLM_TEXT_CORPUS_ADMISSION_REQUIRED')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--pin',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.source,a.pin,a.output)
