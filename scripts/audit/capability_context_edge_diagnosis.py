"""Replay a retained nonpositive-edge failure from canonical indices and native bytes."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.capability_route_diagnosis import sources, read
from audit.repository_memory_readback import chain
from claims.capability_population import need, sha
from claims.process_conditions import write_committed
from kernel.capability_lane import TaskLane
from kernel.capability_updates import TaskUpdates
from kernel.repository_update_state import load_index
from model.capability_context_input import ContextNative, ContextStore
from model.tool_kernel import PLAN


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'old-config', 'delta-config', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--retain-neighbors',action='store_true')
    a = p.parse_args(); need(not a.output.exists(), 'EDGE_DIAG_EXISTS')
    started = time.perf_counter(); config = read(a.config); root = Path(config['root'])
    _, rows = chain(root/'experience.sqlite3')
    need(rows[-1][0]['kind']=='task_growth_begin', 'EDGE_DIAG_PENDING')
    pending = rows[-1][1]; prior = pending['previous']; source = pending['source']
    providers = sources(read(a.delta_config), a.old_config)
    spec = config['native_source']; store = ContextStore(**{k:spec[k] for k in ('plan','plan_pin','native','native_pin')})
    native = ContextNative(spec['population'],spec['population_pin'],store=store)
    providers[native.population.pin] = native
    ids = source['ids']; values = {}; instruments = {}
    for v in providers.values():
        instruments.update({s.name:s.identity for s in v.specs})
        for key,t in v.population.tasks.items():
            if key not in ids: continue
            values[key] = {s.name:np.concatenate([v.field(f,s.name)[0] for f in t['context']]) for s in v.specs}
    need(set(values)==set(ids), 'EDGE_DIAG_SOURCE_COVERAGE')
    loaded = time.perf_counter(); results = {}
    state = Path(prior['path']); kernel = Path(prior['kernel']['path'])
    cache = read(kernel/'cache.json')
    for name in sorted(instruments):
        tick = time.perf_counter(); lane = TaskLane({k:values[k][name] for k in ids})
        cls=TaskUpdates
        if a.retain_neighbors:
            from kernel.capability_retained_updates import TaskUpdates as cls
        m = read(state/(name+'.json')); u = cls(instruments[name], PLAN, cache[name])
        u.index = load_index(state/(name+'.usearch'),lane.unit.shape[1])
        u.keys,u.signatures,u.owners=m['keys'],m['signatures'],dict(m['owners'])
        u.next_key,u.components,u.previous_edges=m['next_key'],m['components'],m['previous_edges']
        try:
            graph, work = u.update(lane)
            result = dict(failed=False,work=work,min_edge=min(e[2] for e in graph['edges']))
        except ValueError as exc:
            need(str(exc)=='MNLM_KERNEL_NONPOSITIVE_EDGE','EDGE_DIAG_OTHER_FAILURE:'+str(exc))
            tb=exc.__traceback__
            while tb and tb.tb_frame.f_code.co_name!='update': tb=tb.tb_next
            need(tb is not None,'EDGE_DIAG_TRACE'); frame=tb.tb_frame.f_locals
            key=frame['key']; q=lane.windows(key)
            exact=sorted(((lane.score(q,k),k) for k in ids if k!=key),key=lambda x:(-x[0],x[1]))
            result=dict(failed=True,atom=key,windows=len(q),candidate_count=len(frame['candidates']),
                selected=frame['neighbors'][key],global_top=exact[:PLAN['neighbors']],
                positive_global=sum(s>0 for s,_ in exact),global_min=min(s for s,_ in exact),
                omitted_positive_top=[k for s,k in exact[:PLAN['neighbors']] if s>0 and k not in frame['candidates']],
                candidates=sorted(frame['candidates']))
        result['seconds']=time.perf_counter()-tick; results[name]=result
        print(json.dumps(dict(instrument=name,**{k:v for k,v in result.items() if k not in ('candidates','work')})),flush=True)
    output=dict(schema='capability-context-edge-diagnosis-v1',source=source,
        journal=sha((root/'experience.sqlite3').read_bytes()),pending=rows[-1][2],results=results,
        phases_seconds=dict(load=loaded-started,diagnosis=time.perf_counter()-loaded),
        atoms=len(ids),retain_neighbors=a.retain_neighbors,reembedded=False,production_changed=False,number_status='measured',
        acceptance='author-checked',producer_sha256=sha(Path(__file__).read_bytes()))
    print(json.dumps(dict(pin=write_committed(a.output,output))),flush=True)


if __name__=='__main__': main()
