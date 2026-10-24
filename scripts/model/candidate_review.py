"""Persist a bounded learning candidate using the shared unchanged admission gates."""
from claims.codata_source import check
from claims.episodic_memory import digest
from model.constellation_learning import decision
from model.tool_rehearsal import termination_gate
from model import field_checkpoint, rehearsal_retention, tool_retention


def review(session, name, pin, spec, *, factory, code, kind, registry, expected=None):
    session._fresh()
    check(not registry, 'CANDIDATE_REVIEW_DUPLICATE')
    check(isinstance(name,str) and name not in ('','.','..') and '/' not in name and '\\' not in name,
          'CANDIDATE_REVIEW_PATH')
    root = session.controlled_root/name
    check(root.resolve().parent == session.controlled_root.resolve(), 'CANDIDATE_REVIEW_PATH')
    data = factory(session,session.learning_base,spec)
    candidate, _, manifest = field_checkpoint.load(root,data,expected=pin)
    check(manifest['step'] == data.plan['steps'] and manifest['parent'] == data.parent_pin, 'CANDIDATE_REVIEW_INCOMPLETE')
    parent, _, _ = field_checkpoint.start(data)
    before, after = data.evaluate(parent),data.evaluate(candidate)
    data.response_basis.measure(data,candidate)
    result = decision(data,before,after,candidate)
    result = tool_retention.assess(data,result,after)
    result = termination_gate(data,before,after,result)
    result = rehearsal_retention.assess(data,after,result)
    payload = dict(before=before,after=after,result=result,binding=digest(data.binding),
        source_prediction_unchanged=True,weight_learning_comparison='same_expanded_reference_bank',
        serving=False,scientific_admission=False)
    body = dict(kind=kind,candidate=name,pin=pin,parent=data.parent_pin,spec=spec,
                prior_head=session.head,payload=digest(payload),code=code,result=result)
    if expected is None:
        session.journal.append(body,payload)
    else:
        check(body == expected and payload == session.journal.payload(expected['payload']), 'CANDIDATE_REVIEW_REPLAY')
    registry[name] = body
    return body
