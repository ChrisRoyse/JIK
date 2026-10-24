"""Every legal atom must reconstruct its original source and its successor topology."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from text.constellation_corpus import canonical,digest,file_sha,need
from text.constellation_input import validate_rows
from text.legal_corpus import LegalCorpus


def pair(rows,context,target):
    need(type(context) is int and type(target) is int and 0<=context<target<len(rows),'LEGAL_PAIR_INDEX')
    left,right=rows[context],rows[target]
    need(target==context+1 and left['next']==right['atom_id'] and right['previous']==left['atom_id']
         and left['sequence']==right['sequence'] and left['byte_end']==right['byte_start'],'LEGAL_PAIR_FUTURE')
    return dict(input_atom=left['atom_id'],target_atom=right['atom_id'],
                input_sha256=left['text_sha256'],target_sha256=right['text_sha256'],
                input_end=left['byte_end'],target_start=right['byte_start'])


def run(root,pin,output):
    start=tick=time.perf_counter();cpu=time.process_time();output.mkdir(parents=True,exist_ok=False)
    corpus=LegalCorpus(root,pin);report=corpus.report
    source=Path(report['stock']['path']);stock_raw=(source/'report.json').read_bytes()
    need(digest(stock_raw)==report['stock']['pin'],'LEGAL_STOCK_PIN');stock=json.loads(stock_raw)
    phases=dict(authenticate=time.perf_counter()-tick);tick=time.perf_counter()
    iterator=(r for batch in corpus.batches(8192) for r in batch)
    docs=iter(corpus.documents);counts=dict(documents=0,atoms=0,transitions=0,characters=0,bytes=0,
        short_spans=0,long_spans=0,beyond_old_document_cap=0,layout=0)
    boundaries=[];first_case=None;longest=(-1,-1);max_atoms=(-1,-1)
    for s in report['sources']:
        raw_path=source/'raw'/s['path'];need(file_sha(raw_path)==s['sha256'],'LEGAL_RAW_HASH')
        for ordinal,line in enumerate(raw_path.open('rb')):
            value=json.loads(line);meta=next(docs);text=value['text'];size=meta['atom_end']-meta['atom_start']
            rows=[next(iterator) for _ in range(size)]
            validate_rows(rows,meta)
            expected=digest(canonical(dict(source=s['source_id']+':'+s['path'],ordinal=ordinal)))
            need(meta['source_record']==expected and meta['source_sha256']==s['sha256']
                 and meta['ordinal']==ordinal and meta['source_path']==s['path'],'LEGAL_DOC_SOURCE')
            need(''.join(r['text'] for r in rows)==text,'LEGAL_FULL_TEXT_CONSERVATION')
            need(meta['provider_metadata']=={},'LEGAL_METADATA_LEAK')
            counts['documents']+=1;counts['atoms']+=len(rows);counts['transitions']+=len(rows)-1
            counts['characters']+=len(text);counts['bytes']+=len(text.encode())
            counts['short_spans']+=sum(len(r['text'].strip())<24 for r in rows)
            counts['long_spans']+=sum(len(r['text'].strip())>600 for r in rows)
            counts['beyond_old_document_cap']+=max(0,len(rows)-48)
            counts['layout']+=sum(r['kind']=='layout' for r in rows)
            index=counts['documents']-1
            longest=max(longest,(len(text),index));max_atoms=max(max_atoms,(len(rows),index))
            if ordinal==0:boundaries.append(index)
            if first_case is None and len(rows)>=3:first_case=(deepcopy(rows),deepcopy(meta))
        boundaries.append(counts['documents']-1)
    need(next(docs,None) is None and next(iterator,None) is None,'LEGAL_TRAILING_CONTENT')
    need(counts==report['counts'],'LEGAL_RECOUNT')
    phases['all_source_rows_atoms_and_topology']=time.perf_counter()-tick;tick=time.perf_counter()
    selected=sorted(set(boundaries+[longest[1],max_atoms[1]]+
                        sorted(range(len(corpus.documents)),key=lambda i:corpus.documents[i]['id'])[:32]))
    selections=[]
    for i in selected:
        rows=corpus.document(i);meta=corpus.documents[i]
        need(digest(''.join(r['text'] for r in rows).encode())==meta['fields'][0]['sha256'],'LEGAL_INDEX_SELECTION')
        pairs=[pair(rows,j,j+1) for j in sorted({0,len(rows)//2,len(rows)-2}) if 0<=j<len(rows)-1]
        selections.append(dict(document=i,id=meta['id'],atoms=len(rows),source_record=meta['source_record'],pairs=pairs))
    rows,meta=first_case
    bad_link=deepcopy(rows);bad_link[0]['next']=rows[-1]['atom_id']
    bad_span=deepcopy(rows);bad_span[0]['byte_end']+=1
    controls=dict(wrong_successor=rejected(lambda:validate_rows(bad_link,meta)),
        byte_span_gap=rejected(lambda:validate_rows(bad_span,meta)),
        missing_atom=rejected(lambda:validate_rows(rows[1:],meta)),
        reversed_order=rejected(lambda:validate_rows(list(reversed(rows)),meta)),
        future_skip=rejected(lambda:pair(rows,0,2)),
        target_as_input=rejected(lambda:pair(rows,1,1)),
        no_implicit_training_admission=rejected(corpus.training_authority))
    write_committed(output/'consumer-selections.json',selections)
    phases['indexed_consumer_and_controls']=time.perf_counter()-tick;tick=time.perf_counter()
    corpus.fresh();need(digest((source/'report.json').read_bytes())==report['stock']['pin'],'LEGAL_FINAL_SOURCE')
    files={'consumer-selections.json':file_sha(output/'consumer-selections.json')}
    phases['finalize']=time.perf_counter()-tick;seconds=time.perf_counter()-start
    need(sum(phases.values())/seconds>=.9,'LEGAL_READBACK_TIMING')
    result=dict(schema='legal-corpus-readback-v1',complete=True,source=dict(path=str(root),pin=pin),
        counts=counts,all_source_text_reconstructed=True,all_successor_links_checked=True,
        all_spans_checked=True,all_documents_preserved=True,indexed_selections=len(selected),
        controls=controls,files=files,largest_document_characters=longest[0],largest_document_atoms=max_atoms[0],
        row_groups=len(corpus.groups),phases_seconds=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        model_updates=0,scientific_admission=False,training_ready=False,
        acceptance='author-checked',number_status='measured',code_sha256=file_sha(Path(__file__)))
    receipt=write_committed(output/'report.json',result)
    print(json.dumps(dict(pin=receipt,counts=counts,consumer_cases=len(selected),controls=controls,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--pin',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.pin,a.output)
