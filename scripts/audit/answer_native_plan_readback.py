"""Independent source-to-input, tokenizer and persisted native reuse readback."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
from claims.process_conditions import write_committed
from model.capability_answer_targets import AnswerTargets
from text.answer_native_plan import AnswerNativePlan, tokenizer
from text.constellation_corpus import digest,file_sha,need
from text.legal_native_store import NativeShard,SHARD_INPUTS,shard_name


def refused(code,fn):
    try: fn()
    except ValueError as exc:
        need(code in str(exc),'ANSWER_PLAN_CONTROL_CODE:'+str(exc)); return dict(code=code,observed=str(exc))
    raise ValueError('ANSWER_PLAN_CONTROL_ACCEPTED:'+code)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    started=tick=time.perf_counter();phases={}
    try:
        plan=AnswerNativePlan(a.plan,a.pin);r=plan.report;view=AnswerTargets(r['source']['path'],r['source']['pin'])
        indexed={v['id']:v['text'] for v in plan.records};used=set();source_records=[]
        for key,mapping in plan.tasks.items():
            pop,task,side=view.source.selected(key);row=view.rows[key]
            need(mapping['split']==side and mapping['split_group']==task['split_group']
                 and mapping['population']==pop.pin and not mapping['field_order_is_event_chronology'], 'ANSWER_RECOUNT_SPLIT')
            need([x['field'] for x in mapping['context']]==task['context'],'ANSWER_RECOUNT_CONTEXT_ROSTER')
            for context in mapping['context']:
                f=pop.fields[context['field']];identity=context['input']
                need(indexed[identity]==f['text'] and digest(f['text'].encode())==identity
                     and context['role']==f['role'] and context['clock']==f['clock'],'ANSWER_RECOUNT_CONTEXT_TEXT')
                used.add(identity)
            expected=[]
            if row['status']=='ready':
                body=pop.fields[row['answer']['source_field']]['text'].encode()
                for atom in row['answer']['atoms']:
                    raw=b''.join(body[lo:hi] for lo,hi in atom['measurement_spans']);identity=digest(raw)
                    need(indexed[identity].encode()==raw,'ANSWER_RECOUNT_ATOM_TEXT');expected.append(identity);used.add(identity)
            need(mapping['targets']==expected and mapping['answer_status']==row['status'],'ANSWER_RECOUNT_TARGET_ORDER')
            source_records.append(dict(task=key,context=len(mapping['context']),answer_atoms=len(expected),split=side))
        need(set(indexed)==used and set(plan.tasks)==set(view.rows),'ANSWER_RECOUNT_COVERAGE')
        phases['source_and_publication']=time.perf_counter()-tick;tick=time.perf_counter()
        tokens={};reused={};controls=[]
        config=json.loads((Path(r['canonical_bank']['path'])/'configuration.json').read_bytes())
        for name in r['roster']:
            contract=r['contracts'][name];tok=tokenizer(contract);counts=[];active=[]
            for record in plan.records:
                enc=tok(contract['prefix']+record['text'],padding=False,truncation=True,
                        max_length=contract['max_len'],return_overflowing_tokens=True)
                counts.append(len(enc['input_ids']));active.append(sum(map(len,enc['input_ids'])))
            need(counts==plan.tokens[name]['windows'] and active==plan.tokens[name]['active_tokens'], 'ANSWER_RECOUNT_TOKENS')
            tokens[name]=dict(windows=sum(counts),tokens=sum(active),maximum_windows=max(counts),over_eight=sum(n>8 for n in counts))
            if r['reused']:
                saved=np.load(a.plan/name/'reused.npz',allow_pickle=False);v=saved['vectors'];o=saved['offsets'];saved.close()
                traces=json.loads((a.plan/name/'traces.json').read_bytes());native=[]
                for i,item in enumerate(r['reused']):
                    uid=item['uid'];start=uid//SHARD_INPUTS*SHARD_INPUTS;end=min(start+SHARD_INPUTS,config['inputs'])
                    shard=NativeShard(Path(r['canonical_bank']['path'])/name/shard_name(start,end),r['canonical_bank']['pin'],
                                      config['instruments'][name]['lens_id'],start,end)
                    try:
                        values,trace=shard.input(uid,item['id']);lo,hi=map(int,o[i:i+2])
                        need(np.array_equal(v[lo:hi],values) and traces[i]==trace and
                             digest(plan.records[item['index']]['text'].encode())==item['id'],'ANSWER_RECOUNT_REUSE_EXACT')
                        native.append(dict(uid=uid,input_index=item['index'],windows=hi-lo,sha256=digest(values.tobytes()),shard_pin=shard.pin))
                        wrong=values.copy();wrong[0,0]=np.nextafter(wrong[0,0],np.float32(np.inf))
                        controls.append(refused('ANSWER_RECOUNT_REUSE_EXACT',lambda:need(np.array_equal(wrong,v[lo:hi]),'ANSWER_RECOUNT_REUSE_EXACT')))
                        controls.append(refused('NATIVE_SHARD_INPUT_BINDING',lambda:shard.input(uid,'0'*64)))
                    finally:
                        shard.table.close();shard.vectors._mmap.close();shard.offsets._mmap.close()
                reused[name]=native
        phases['independent_tokens_and_native_bytes']=time.perf_counter()-tick;tick=time.perf_counter()
        controls.append(refused('ANSWER_PLAN_PIN',lambda:AnswerNativePlan(a.plan,'0'*64)))
        original=plan.records[0]['text'];plan.records[0]['text']=original+' '
        controls.append(refused('ANSWER_PLAN_CHANGED',plan.fresh));plan.records[0]['text']=original
        bad=a.output/'corrupt-plan';shutil.copytree(a.plan,bad)
        with (bad/'inputs.json').open('ab') as f:f.write(b' ')
        controls.append(refused('ANSWER_PLAN_FILE',lambda:AnswerNativePlan(bad,a.pin)))
        legacy=json.loads((a.plan/'legacy.json').read_bytes())
        need(legacy and all(not x['reuse'] and x['old_lens_id']!=x['canonical_lens_id'] for x in legacy),'ANSWER_RECOUNT_LEGACY_REJECTED')
        for entry in legacy:
            controls.append(refused('ANSWER_RECOUNT_LEGACY_IDENTITY',lambda:need(entry['old_lens_id']==entry['canonical_lens_id'],'ANSWER_RECOUNT_LEGACY_IDENTITY')))
        plan.fresh();view.source.fresh();phases['negative_controls']=time.perf_counter()-tick
        for n,value in [('sources',source_records),('tokens',tokens),('reused',reused),('controls',controls)]:
            write_committed(a.output/(n+'.json'),value)
        result=dict(schema='answer-canonical-plan-readback-v1',complete=True,plan=dict(path=str(a.plan),pin=a.pin),
            inputs=len(indexed),tasks=len(source_records),reused_inputs=len(r['reused']),missing_inputs=len(r['missing']),
            source_bytes_exact=True,target_order_exact=True,all_reused_native_exact=True,all_token_counts_exact=True,
            code=r['code'],tokens=tokens,controls=len(controls),phases=phases,seconds=time.perf_counter()-started,
            files={n+'.json':file_sha(a.output/(n+'.json')) for n in ('sources','tokens','reused','controls')},
            gpu_work_started=False,scientific_admission=False,acceptance='author-checked')
        pin=write_committed(a.output/'report.json',result);print(json.dumps(dict(pin=pin,**result)))
    except BaseException as exc:
        write_committed(a.output/'failure.json',dict(type=type(exc).__name__,error=str(exc)));raise


if __name__=='__main__':main()
