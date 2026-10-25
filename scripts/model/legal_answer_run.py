"""Durable complete-answer epochs: bounded update blocks, atomic publication, exact resume.

Unpublished attempts are retained. Only contiguous published blocks advance the
authoritative cursor; an interrupted block is recomputed from the last checkpoint.
"""
import fcntl
import json
import os
from pathlib import Path
import re

import torch

from model.legal_answer_epoch import persist
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha, need


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()


def sync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def write(path,value):
    with path.open('xb') as f:f.write(encoded(value));f.flush();os.fsync(f.fileno())


def read(path):
    need(not path.is_symlink(),'ANSWER_RUN_MANIFEST_SYMLINK')
    raw=path.read_bytes();value=json.loads(raw)
    need(encoded(value)==raw,'ANSWER_RUN_NONCANONICAL_MANIFEST');return value


def stamp(path):
    s=path.stat();return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns


class AnswerRun:
    def __init__(self,root,epoch,*,checkpoint_documents=256,fault=None):
        need(type(checkpoint_documents) is int and checkpoint_documents>0,'ANSWER_RUN_CHECKPOINT_POLICY')
        need(epoch.cursor==0 and epoch.chain==epoch.binding_pin,'ANSWER_RUN_FRESH_EPOCH_REQUIRED')
        self.root=Path(root);self.root.mkdir(exist_ok=True);need(not self.root.is_symlink(),'ANSWER_RUN_ROOT_SYMLINK')
        self.epoch=epoch;self.interval=checkpoint_documents
        self.fault=fault or (lambda stage:None);self.lock=None;self.stream=None;self.stage=None;self.poisoned=False
        self.binding=json.loads(encoded(dict(schema='complete-answer-run-v1',epoch=epoch.binding,
            checkpoint_documents=checkpoint_documents,publication='atomic_immutable_block',partial='retained_uncommitted')))
        try:
            self.lock=(self.root/'.writer.lock').open('a+b');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            names=[p.name for p in self.root.iterdir()];self.attempts={}
            for name in names:
                match=re.fullmatch(r'\.(initial|block-\d{6})-(\d{6})',name)
                if match:self.attempts[match[1]]=max(self.attempts.get(match[1],0),int(match[2])+1)
            initial=self.root/'initial.json'
            if not initial.exists():
                need(not any(n.startswith('commit-') for n in names),'ANSWER_RUN_MISSING_INITIAL')
                directory=self.attempt('initial');snapshot=persist(directory/'state.pt',epoch.snapshot())
                ref=self.relative(snapshot);value=dict(binding=self.binding,checkpoint=ref)
                write(directory/'receipt.json',value);sync_directory(directory)
                os.link(directory/'receipt.json',initial);sync_directory(self.root)
            value=read(initial);need(value['binding']==self.binding,'ANSWER_RUN_BINDING')
            self.restore(value['checkpoint'],0,epoch.binding_pin)
            self.parent=file_sha(initial);self.chain=epoch.chain;self.model=fingerprint(epoch.model.state_dict())
            self.optimizer=fingerprint(epoch.optimizer.state_dict());self.commits=[];used={Path(value['checkpoint']['path']).parent.name}
            commits=sorted(n for n in names if n.startswith('commit-'))
            for index,name in enumerate(commits,1):
                need(name==f'commit-{index:06d}.json','ANSWER_RUN_COMMIT_GAP')
                path=self.root/name;commit=read(path)
                self.check_block(commit,index,epoch.cursor,self.chain,self.model,self.optimizer)
                # Historical checkpoint payloads are authenticated lazily if selected for rollback.
                # Recovery reads all compact rows and the latest checkpoint only.
                epoch.cursor=commit['stop'];epoch.chain=commit['chain']
                self.chain=commit['chain'];self.model=commit['model'];self.optimizer=commit['optimizer']
                self.parent=file_sha(path);self.commits.append(dict(path=name,pin=self.parent,stop=commit['stop']))
                used.add(Path(commit['checkpoint']['path']).parent.name)
            if commits:
                last=read(self.root/commits[-1]);self.restore(last['checkpoint'],last['stop'],last['chain'])
                need(fingerprint(epoch.model.state_dict())==self.model
                     and fingerprint(epoch.optimizer.state_dict())==self.optimizer,'ANSWER_RUN_LATEST_STATE')
            self.committed_cursor=epoch.cursor;self.block_chain=self.chain;self.block_model=self.model;self.block_optimizer=self.optimizer
            self.uncommitted=sorted(n for n in names if re.fullmatch(r'\.(initial|block-\d{6})-\d{6}',n) and n not in used)
            self.watch(initial, self.root/commits[-1] if commits else initial,
                       self.path(last['checkpoint']['path']) if commits else self.path(value['checkpoint']['path']))
            self.fresh()
        except BaseException:self.close();raise

    def attempt(self,label):
        number=self.attempts.get(label,0);self.attempts[label]=number+1
        path=self.root/f'.{label}-{number:06d}';path.mkdir();sync_directory(self.root);return path

    def path(self,name):
        relative=Path(name)
        need(not relative.is_absolute() and len(relative.parts)==2
             and re.fullmatch(r'\.(initial|block-\d{6})-\d{6}',relative.parts[0]) is not None
             and relative.parts[1] in ('state.pt','rows.jsonl'), 'ANSWER_RUN_PATH')
        path=self.root/relative
        need(not path.is_symlink() and not path.parent.is_symlink(),'ANSWER_RUN_SYMLINK');return path

    def relative(self,receipt):
        path=Path(receipt['path'])
        return dict(path=str(path.relative_to(self.root)),pin=receipt['pin'],state=receipt['state'],bytes=path.stat().st_size)

    def restore(self,ref,cursor,chain):
        path=self.path(ref['path']);need(path.stat().st_size==ref['bytes'],'ANSWER_RUN_CHECKPOINT_SIZE')
        self.epoch.restore(path,ref,cursor=cursor,chain=chain)

    def watch(self,*paths):self.watched={p:stamp(p) for p in paths}

    def fresh(self):
        need(self.lock is not None and not self.lock.closed and not self.poisoned,'ANSWER_RUN_CLOSED_OR_FAILED')
        need(all(stamp(p)==s for p,s in self.watched.items()),'ANSWER_RUN_COMMITTED_CHANGED')
        need(self.epoch.cursor>=self.committed_cursor and self.epoch.chain==self.chain,'ANSWER_RUN_LIVE_CURSOR')
        self.epoch.fresh()

    def check_rows(self,path,start,stop,chain,model,optimizer):
        cursor=start
        with path.open('rb') as f:
            for raw in f:
                need(raw.endswith(b'\n'),'ANSWER_RUN_TORN_COMMITTED_ROW');row=json.loads(raw)
                need(encoded(row)+b'\n'==raw,'ANSWER_RUN_ROW_CANONICAL');entry=row['entry'];update=entry['update']
                need(cursor<stop and entry['cursor']==cursor and entry['document']==self.epoch.documents[cursor]
                     and entry['parent']==chain and update['before']==model
                     and row['chain']==fingerprint(entry) and row['counts']==self.epoch.counts(cursor+1),
                     'ANSWER_RUN_ROW_CHAIN')
                need(update['decisions']==self.epoch.counts(cursor+1)['decisions']-self.epoch.counts(cursor)['decisions']
                     and update['full'] is True,'ANSWER_RUN_ROW_COVERAGE')
                chain=row['chain'];model=update['after'];optimizer=update['optimizer'];cursor+=1
        need(cursor==stop,'ANSWER_RUN_ROW_COUNT');return chain,model,optimizer

    def check_block(self,value,index,start,chain,model,optimizer):
        need(value['schema']=='complete-answer-block-v1' and value['block']==index
             and value['parent']==self.parent and value['start']==start
             and type(value['stop']) is int and start<value['stop']<=min(start+self.interval,len(self.epoch.documents))
             and value['before']==chain,'ANSWER_RUN_BLOCK_BINDING')
        rows=self.path(value['rows']['path']);checkpoint=self.path(value['checkpoint']['path'])
        need(rows.parent==checkpoint.parent and value['rows']['count']==value['stop']-start
             and file_sha(rows)==value['rows']['pin'],'ANSWER_RUN_ROWS_PIN')
        actual=self.check_rows(rows,start,value['stop'],chain,model,optimizer)
        need(actual==(value['chain'],value['model'],value['optimizer']),'ANSWER_RUN_BLOCK_CHAIN')
        need(checkpoint.is_file() and checkpoint.stat().st_size==value['checkpoint']['bytes'],'ANSWER_RUN_CHECKPOINT_SIZE')

    def step(self):
        self.fresh();need(self.epoch.cursor<len(self.epoch.documents),'ANSWER_RUN_COMPLETE')
        try:
            if self.stream is None:
                self.stage=self.attempt(f'block-{len(self.commits)+1:06d}')
                self.stream=(self.stage/'rows.jsonl').open('xb')
            cursor=self.epoch.cursor;value,row=self.epoch.update(self.epoch.documents[cursor])
            need(row['entry']['parent']==self.chain and row['entry']['update']['before']==self.model,'ANSWER_RUN_UPDATE_CHAIN')
            self.chain=row['chain'];self.model=fingerprint(value['model']);self.optimizer=fingerprint(value['optimizer'])
            need(row['entry']['update']['after']==self.model and row['entry']['update']['optimizer']==self.optimizer,'ANSWER_RUN_UPDATE_STATE')
            row['sum_recorded_weighted_loss']=sum(r['weighted_loss'] for r in value['record']['records'])
            self.stream.write(encoded(row)+b'\n');self.stream.flush();os.fsync(self.stream.fileno());self.fault('rows_durable')
            if self.epoch.cursor-self.committed_cursor==self.interval or self.epoch.cursor==len(self.epoch.documents):self.commit()
            return value,row
        except BaseException:self.poisoned=True;raise

    def commit(self):
        self.fresh()
        if self.epoch.cursor==self.committed_cursor:return None
        try:
            self.stream.close();self.stream=None
            ref=self.relative(persist(self.stage/'state.pt',self.epoch.snapshot()));self.fault('checkpoint_durable')
            index=len(self.commits)+1;rows=self.stage/'rows.jsonl'
            value=dict(schema='complete-answer-block-v1',block=index,parent=self.parent,start=self.committed_cursor,
                stop=self.epoch.cursor,before=self.block_chain,chain=self.chain,model=self.model,optimizer=self.optimizer,
                rows=dict(path=str(rows.relative_to(self.root)),pin=file_sha(rows),count=self.epoch.cursor-self.committed_cursor),checkpoint=ref)
            self.check_block(value,index,self.committed_cursor,self.block_chain,self.block_model,self.block_optimizer)
            write(self.stage/'receipt.json',value);sync_directory(self.stage);self.fault('receipt_durable')
            final=self.root/f'commit-{index:06d}.json';os.link(self.stage/'receipt.json',final);self.fault('commit_linked')
            sync_directory(self.root);self.fault('directory_synced')
            need(read(final)==value,'ANSWER_RUN_PUBLISHED_READBACK');self.parent=file_sha(final)
            self.commits.append(dict(path=final.name,pin=self.parent,stop=self.epoch.cursor))
            self.committed_cursor=self.epoch.cursor;self.block_chain=self.chain;self.block_model=self.model;self.block_optimizer=self.optimizer
            self.watch(self.root/'initial.json',final,self.stage/'state.pt');self.stage=None;return value
        except BaseException:self.poisoned=True;raise

    def finish(self):
        self.fresh();need(self.epoch.cursor==len(self.epoch.documents),'ANSWER_RUN_INCOMPLETE')
        self.commit();return dict(binding=self.binding,cursor=self.committed_cursor,chain=self.chain,
            counts=self.epoch.counts(self.committed_cursor),commits=self.commits,uncommitted_attempts=self.uncommitted,
            model=self.model,optimizer=self.optimizer,complete=True)

    def close(self):
        if self.stream is not None:self.stream.close();self.stream=None
        if self.lock is not None:self.lock.close();self.lock=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
