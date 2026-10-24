"""Native region margins, bounded by both sealed and expanded-bank Brier scores."""
import json
from pathlib import Path

import torch

from claims.codata_source import check,sha
from claims.episodic_memory import digest
from model.brier_region_targets import bound,RULE


def bind(data,spec):
    raw = Path(spec['path']).read_bytes()
    check(sha(raw) == spec['pin'],'SUCCESSOR_BASELINE_PIN')
    baseline = json.loads(raw)
    check(baseline['binding'] == data.binding and baseline['parent'] == data.parent_pin,
          'SUCCESSOR_BASELINE_BINDING')
    feedback = data.session.next_proposals[data.spec['experiment']]['feedback']
    sealed = data.session.journal.payload(feedback['payload'])['scores']
    rows,bounds = {},{}
    for key,record in data.new.items():
        e = data.examples[key]
        allowed = e['allowed'].nonzero().flatten().tolist()
        before = baseline['scores'][key]['termination']['score']
        original = sealed[record['condition']]['models']['corrected']
        check(before['outcome'] == original['outcome'] == data.outcomes[key] and
              int(data.labels[key][e['region']]) == data.outcomes[key],'SUCCESSOR_OUTCOME')
        scores = dict(common=before['common'],**before['instruments'])
        prior = dict(common=original['common'],**original['instruments'])
        bounds[key] = {n:dict(bound(min(v['brier'],prior[n]['brier']),len(allowed)),
            expanded_brier=v['brier'],sealed_brier=prior[n]['brier']) for n,v in scores.items()}
        for lane in [s.name for s in data.model.specs]:
            values = rows.setdefault(lane,[])
            lane_margin = bounds[key][lane]['margin']
            common_share = bounds[key]['common']['margin']/len(data.model.specs)
            for rival in allowed:
                if rival == e['region']:
                    continue
                cost = float(e['region_costs'][lane][1][rival])
                values.append(dict(kind='native',key=key,target=e['region'],rival=rival,
                    native_cost=cost,lane_margin=lane_margin,common_share=common_share,
                    floor=max(cost,lane_margin,common_share)))
    data.typed_rows,data.typed_row_pin = rows,digest(rows)
    data.typed_floors = {n:torch.tensor([r['floor'] for r in rr],dtype=torch.float64,
                                      device=data.session.device) for n,rr in rows.items()}
    data.typed_native = {n:torch.ones(len(rr),dtype=torch.bool,device=data.session.device)
                        for n,rr in rows.items()}
    return dict(rule=RULE,source=spec,rows=rows,bounds=bounds,
        labels_used_only_as_training_targets=True,forward_unchanged=True,
        sealed_brier_ceilings_never_weakened=True,scope='exposed_feedback_not_transfer')
