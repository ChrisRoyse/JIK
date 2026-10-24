"""Reconstruct state carry, native errors and ledger joins from retained bytes."""
import copy
import io
import json
from pathlib import Path
import sqlite3

import numpy as np
import torch

from audit.next_echo_readback import decode, distance
from claims.codata_source import check, sha
from claims.episodic_memory import ZERO, canonical, digest
from model.attempt_echo_input import AttemptEchoInputs


def read(path):
    return json.loads(Path(path).read_bytes())


def fp(a):
    return dict(shape=list(a.shape),dtype=a.dtype.str,sha256=sha(a.tobytes()))


def chain(path):
    rows,head = [],ZERO
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        for seq,previous,raw,actual in db.execute('SELECT * FROM events ORDER BY seq'):
            body = json.loads(raw)
            check(seq == len(rows) and previous == head and canonical(body) == raw and
                  digest(dict(seq=seq,previous=previous,body=body)) == actual,'CYCLE_READER_CHAIN')
            payload = None
            if body.get('payload') is not None:
                data, = db.execute('SELECT data FROM payloads WHERE hash=?',(body['payload'],)).fetchone()
                payload = json.loads(data)
                check(sha(data) == body['payload'] and canonical(payload) == data,'CYCLE_READER_PAYLOAD')
            rows.append((body,payload,actual));head=actual
    return rows


def transition(record, before, phase, context):
    request,after = record['request'],record['after']
    check(record['before'] == before and request['before'] == digest(before)
          and request['context'] == context and request['phase'] == phase
          and after['context'] == context and after['sequence'] == before['sequence']+1
          and after['standard'] == before['standard'] and after['world'] == before['world']
          and after['last_event'] == digest(request),'CYCLE_READER_TRANSITION')
    old,new = decode(before['hidden']),decode(after['hidden'])
    check(old.shape == new.shape == (6,32) and np.all(np.any(old != new,axis=1)),
          'CYCLE_READER_UNCHANGED_LANE')
    return after,np.linalg.norm(new-old,axis=1).tolist()


def read_states(root, seal, forecasts, observed, feedback):
    config=read(root/'config.json');source=Path(config['source'])
    previous,final=read(source/'snapshot-final.json'),read(root/'snapshot-final.json')
    proof=Path(config['proof'])/'seal.json'
    check(sha(proof.read_bytes()) == config['proof_pin'],'CYCLE_READER_SOURCE_SEAL')
    source_files=read(proof)['files']
    for name in ('snapshot-final.json','experience.sqlite3'):
        raw=(source/name).read_bytes()
        check(dict(bytes=len(raw),sha256=sha(raw)) == source_files['session/'+name],
              'CYCLE_READER_SOURCE_BYTES')
    old_rows,rows=chain(source/'experience.sqlite3'),chain(root/'experience.sqlite3')
    check(rows[:len(old_rows)] == old_rows and len(old_rows) == previous['events'] == 466
          and len(rows) == final['events'] == 468 and rows[-1][2] == final['head']
          and rows[-3][2] == previous['head'] == config['experiment']['source_head']
          and rows[-2][0] == seal['forecast'] and rows[-2][1] == forecasts
          and rows[-1][0]['kind'] == 'state_cycle_feedback' and rows[-1][1] == feedback,
          'CYCLE_READER_JOURNAL_JOIN')
    check(final['native_feedback_contexts'] == previous['native_feedback_contexts'] and
          final['native_feedback_enrollments'] == previous['native_feedback_enrollments'],
          'CYCLE_READER_OLD_CONTEXTS')
    cold=read(root/'cold-stage.json');restart=read(root/'forecast-restart.json')
    check(cold['snapshot'] == digest(final) and cold['head'] == final['head'] and
          cold['journal'] == sha((root/'experience.sqlite3').read_bytes()) and
          restart['snapshot'] == digest(read(root/'snapshot-forecast.json')),
          'CYCLE_READER_COLD_STATE')
    inputs=AttemptEchoInputs(**read(source/'config.json')['inputs'])
    memory_head=previous['memory']['head']
    for record in feedback['memory_records']:
        check(record['previous'] == memory_head and record['sha256'] ==
              digest({k:v for k,v in record.items() if k != 'sha256'}),'CYCLE_READER_MEMORY_CHAIN')
        memory_head=record['sha256']
    check(memory_head == final['memory']['head'] and len(feedback['episodes']) == 2
          and all(final['memory']['states'][k] == v for k,v in previous['memory']['states'].items()),
          'CYCLE_READER_MEMORY_PREFIX')
    old_history=inputs.observations.forecasts['history']
    arrays={}
    for spec in inputs.specs:
        with np.load(inputs.native/spec.name/'slots.npz',allow_pickle=False) as archive:
            values,ptr=archive['vectors'],archive['offsets']
            arrays[spec.name]={}
            for key in inputs.keys:
                units=[r['unit'] for r in inputs.population.refs
                       if r['observation'] == key and r['unit'] is not None]
                check(inputs.population.probe['unit'] not in units,'CYCLE_READER_PROBE')
                arrays[spec.name][key]=np.concatenate([values[ptr[u]:ptr[u+1]] for u in units])
    initials={};checkpoint_files={}
    for variant,name in (('before','brier-1'),('corrected','successor-1')):
        folder=root/'candidates'/name
        raw=(folder/'manifest.json').read_bytes();manifest=json.loads(raw)
        check(sha(raw) == (folder/'COMMITTED').read_text(),'CYCLE_READER_CHECKPOINT')
        for file,entry in manifest['files'].items():
            content=(folder/file).read_bytes()
            check(len(content) == entry['bytes'] and sha(content) == entry['sha256'] and
                  content == (source/'candidates'/name/file).read_bytes(),'CYCLE_READER_WEIGHT_BYTES')
            checkpoint_files[str(folder/file)]=entry
        weights=torch.load(io.BytesIO((folder/'weights.pt').read_bytes()),weights_only=True,map_location='cpu')
        initials[variant]=weights['core.initial'].numpy()
    contexts=forecasts['contexts'];summaries={};comparisons={};transitions=[]
    check(len(contexts) == len(feedback['contexts']) == 8,'CYCLE_READER_CONTEXT_COUNT')
    for context,entry in sorted(contexts.items()):
        original=previous['native_feedback_contexts'][entry['source_context']]
        start=entry['start'];mode=entry['mode'];variant=entry['variant'];condition=entry['condition']
        check(entry['source_state'] == original['state'] and entry['source_observation'] == original['observation']
              and entry['checkpoint'] == original['fork']['candidate'] and original['status'] == 'corrected_state'
              and original['condition'] == condition and original['variant'] == variant,
              'CYCLE_READER_SOURCE_STATE')
        check(start['standard'] == original['state']['standard'] and start['world'] == original['state']['world'],
              'CYCLE_READER_REFERENCE')
        if mode == 'carried':
            check(start == dict(original['state'],context=context) and start['sequence'] == 25,
                  'CYCLE_READER_CARRIED')
        else:
            check(mode == 'reset' and start['sequence'] == 0 and start['last_event'] is None
                  and np.array_equal(decode(start['hidden']),initials[variant]),'CYCLE_READER_RESET')
        history=entry['history'];old_ob=entry['source_observation']
        check(history['context'] == context and history['rows'] == old_history['rows']+[old_ob]
              and len(history['frames']) == 10 and history['frames'][:-1] == old_history['frames'],
              'CYCLE_READER_HISTORY')
        last={}
        for spec in inputs.specs:
            values=arrays[spec.name][old_ob][None]
            last[spec.name]=dict(values=fp(values),padding=fp(np.zeros(values.shape[:2],dtype=bool)),
                                absent=fp(np.zeros(1,dtype=bool)),rows=[old_ob])
            check(sum(f[spec.name]['values']['shape'][1] for f in history['frames']) == 12,
                  'CYCLE_READER_WINDOW_COUNT')
        check(history['frames'][-1] == last,'CYCLE_READER_HISTORY_NATIVE')
        state=start
        for phase,record in enumerate(entry['trace']):
            check(phase < 3 and record['request']['history'] == digest(history)
                  and record['request']['origin'] == 'generated_proposal','CYCLE_READER_PLAN')
            state,delta=transition(record,state,phase,context)
            transitions.append(dict(context=context,phase=phase,delta_by_instrument=dict(zip(arrays,delta))))
        check(len(entry['trace']) == 3 and state == entry['state'],'CYCLE_READER_PLAN_END')
        updated=feedback['contexts'][context];new_ob=updated['observation']
        old_actual,new_actual=inputs.observations.read(old_ob),observed.read(new_ob)
        pre=observed.condition_inputs().episode(condition)
        episode=observed.episode(new_ob,evidence=(pre['id'],))
        receipt=observed.assess(episode)
        memory=final['memory']['states'][episode['id']]
        check(updated['episode'] == episode['id'] and updated['receipt'] == receipt['id']
              and episode['id'] in feedback['episodes'] and episode['id'] in final['memory']['grounded_world_claims']
              and memory['receipt'] == receipt and memory['active'] and memory['grounded_scope'] == episode['scope']
              and episode['why'] is None and episode['why_status'] == 'unresolved'
              and any(r['event'].get('op') == 'intake' and r['event']['episode'] == episode
                      for r in feedback['memory_records']),'CYCLE_READER_EPISODE_RECEIPT')
        check(old_actual['captured'] == new_actual['captured'] and
              old_actual['container'] != new_actual['container'] and
              old_actual['finished'] < seal['committed_at'] < new_actual['started'],
              'CYCLE_READER_REUSE')
        reuse=feedback['cached_readings'][context]
        check(reuse == dict(previous=old_ob,current=new_ob,native=inputs.identity,exact_surface_match=True,
                            values={n:fp(v[old_ob][None]) for n,v in arrays.items()}),'CYCLE_READER_REUSE_KEY')
        prediction=forecasts['predictions'][condition]['models'][entry['prediction']]
        errors={}
        for phase,record in enumerate(feedback['transitions'][context],3):
            request=record['request']
            check(phase <= 4 and request['observation'] == new_ob and request['episode'] == updated['episode']
                  and request['receipt'] == updated['receipt'] and request['origin'] == 'execution_receipt'
                  and request['same_occurrence_reflection'] == (phase == 4), 'CYCLE_READER_FEEDBACK')
            state,delta=transition(record,state,phase,context)
            check(record['observed'] == reuse['values'],'CYCLE_READER_NATIVE_JOIN')
            for spec in inputs.specs:
                packed=prediction['constellation'][spec.name]
                check(packed['instrument'] == spec.identity,'CYCLE_READER_INSTRUMENT')
                pred=decode(packed).astype(np.float64)
                actual=arrays[spec.name][old_ob].astype(np.float64)
                pred/=np.linalg.norm(pred,axis=1,keepdims=True)
                actual/=np.linalg.norm(actual,axis=1,keepdims=True)
                cosine=np.clip(actual@pred.T,-1,1);error=record['error'][spec.name]
                check(np.array_equal(cosine,np.asarray(error['cosines'])) and
                      distance(cosine) == error['ordered_distance'],'CYCLE_READER_NATIVE_ERROR')
                errors[spec.name]=error['ordered_distance']
            transitions.append(dict(context=context,phase=phase,delta_by_instrument=dict(zip(arrays,delta))))
        check(len(feedback['transitions'][context]) == 2 and state == updated['state'] and
              state['sequence'] == (30 if mode == 'carried' else 5) and updated['status'] == 'corrected_state',
              'CYCLE_READER_CORRECTION_END')
        check(final['state_cycle_contexts'][context] == updated,'CYCLE_READER_FINAL_JOIN')
        summaries[context]=dict(source=entry['source_context'],mode=mode,variant=variant,condition=condition,
            sequence_before=start['sequence'],sequence_after=state['sequence'],native_errors=errors,
            before=start['hidden']['sha256'],after=state['hidden']['sha256'])
        comparisons.setdefault(condition,{})[entry['prediction']]=errors
    for source_context in config['experiment']['contexts']:
        pair=[v for v in contexts.values() if v['source_context'] == source_context]
        check(len(pair) == 2 and {v['mode'] for v in pair} == {'carried','reset'}
              and pair[0]['history']['frames'] == pair[1]['history']['frames']
              and pair[0]['history']['rows'] == pair[1]['history']['rows'],'CYCLE_READER_PAIR')
    controls=[]
    first=next(iter(contexts.values()));bad=copy.deepcopy(first['trace'][0]);bad['after']['sequence']+=1
    try:
        transition(bad,first['start'],0,first['start']['context'])
    except ValueError as error:
        check('CYCLE_READER_TRANSITION' in str(error),'CYCLE_READER_CONTROL')
        controls.append(dict(name='forged_phase_counter',code='CYCLE_READER_TRANSITION'))
    else:
        raise AssertionError('CYCLE_READER_CONTROL_ACCEPTED')
    return dict(events=len(rows),head=final['head'],source_head=previous['head'],contexts=summaries,
        native_comparisons=comparisons,transitions=transitions,controls=controls,checkpoint_files=checkpoint_files,
        old_contexts_unchanged=True,cold_snapshot=cold['snapshot'],history_frames=10,history_windows_per_instrument=12,
        memory_head=memory_head,scoped_occurrences=feedback['episodes'],why_status='unresolved')
