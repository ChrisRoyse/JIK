"""Purpose-bound, operation-matched experience schedules and typed training targets."""
from collections import Counter, defaultdict

import torch
from torch.nn import functional as F

from claims.capability_population import need, sha
from claims.episodic_memory import digest
from model.capability_objective import native_similarity
from model.capability_structure import predict_structured

PURPOSE = {
    'repository_context_to_recorded_patch': 'successful_patch',
    'continue_recorded_steps': 'annotated_arithmetic',
    'solve_recorded_problem': 'annotated_arithmetic',
    'respond_to_source_question': 'source_answer',
    'revise_with_source_evidence': 'source_answer',
    'continue_attested_dialogue': 'recorded_continuation',
    'respond_to_attested_act': 'recorded_continuation',
}
ARMS = ('full_correct', 'full_shuffled', 'repeat_correct', 'repeat_shuffled')


def schedules(corpus, supervisor):
    decisions = {k: supervisor.decide(k, PURPOSE[corpus.tasks[k]['kind']], fitting=k in corpus.train)
                 for k in (*corpus.train, *corpus.query)}
    full = sorted((k for k in corpus.train if decisions[k]['allowed']),
                  key=lambda k: sha(('experience-growth-v1:' + k).encode()))
    pools = defaultdict(list)
    for key in full:
        if key in corpus.kernel_train: pools[corpus.tasks[key]['kind']].append(key)
    counts = Counter(); repeated = []
    for key in full:
        kind = corpus.tasks[key]['kind']; need(len(pools[kind]) > 1, 'EXPERIENCE_FIT_SUPPORT')
        repeated.append(pools[kind][counts[kind] % len(pools[kind])]); counts[kind] += 1
    arms = {}
    for name, keys in [('full', full), ('repeat', repeated)]:
        indices = defaultdict(list)
        for i, k in enumerate(keys): indices[corpus.tasks[k]['kind']].append(i)
        targets = list(keys)
        for ids in indices.values():
            # Group repeated identities, then rotate by the largest multiplicity.
            # This deranges task identities while preserving exact target frequencies.
            order = sorted(ids, key=lambda i: (keys[i], i))
            shift = max(Counter(keys[i] for i in ids).values())
            need(2 * shift <= len(ids), 'EXPERIENCE_NULL_SUPPORT')
            for j, i in enumerate(order): targets[i] = keys[order[(j + shift) % len(order)]]
        need(Counter(keys) == Counter(targets) and all(k != t for k, t in zip(keys, targets)),
             'EXPERIENCE_NULL_DERANGEMENT')
        arms[name + '_correct'] = [dict(task=k, target=k) for k in keys]
        arms[name + '_shuffled'] = [dict(task=k, target=t) for k, t in zip(keys, targets)]
    donors = {}
    for key in corpus.query:
        task = corpus.tasks[key]
        eligible = [k for k in pools[task['kind']] if corpus.tasks[k]['split_group'] != task['split_group']]
        need(eligible, 'EXPERIENCE_QUERY_DONOR')
        donors[key] = min(eligible, key=lambda k: sha((key + ':' + k).encode()))
    return dict(arms=arms, decisions=decisions, donors=donors,
                full_unique=len(full), repeat_unique=len(set(repeated)), operation_counts=dict(counts))


def check_step(corpus, plan, arm, index, key, target_key):
    need(arm in ARMS and key in corpus.train and target_key in corpus.train, 'EXPERIENCE_FIT_QUERY')
    need(plan['arms'][arm][index] == dict(task=key, target=target_key), 'EXPERIENCE_FIT_SCHEDULE')
    kind = corpus.tasks[key]['kind']
    need(corpus.tasks[target_key]['kind'] == kind, 'EXPERIENCE_FIT_OPERATION')
    for k in (key, target_key):
        d = plan['decisions'][k]
        need(d['allowed'] and d['fitting'] and d['purpose'] == PURPOSE[kind], 'EXPERIENCE_FIT_PURPOSE')


def objective(model, contexts, targets, labels, plan, arm, index, key, target_key):
    corpus = contexts.corpus; check_step(corpus, plan, arm, index, key, target_key)
    world = model['world']; kind = corpus.tasks[key]['kind']
    output = predict_structured(model, contexts.get(key, world.config, 'cuda'), kind)
    target = targets[target_key][1]
    label = torch.tensor([labels[target_key]['label']], device='cuda')
    chosen = world.supervised_placement(output, label,
        allowed=torch.ones(len(corpus.regions), dtype=torch.bool, device='cuda'),
        event_ids=[digest(dict(key=key, operation=kind))])
    regions = []; placements = []; lanes = {}
    for spec in world.specs:
        r = F.cross_entropy(output['regions'][spec.name], label)
        x, y = chosen.slots[spec.name], target[spec.name]
        p = 1 - native_similarity(x.values[0, ~x.padding[0]], y.values[0, ~y.padding[0]])
        regions.append(r); placements.append(p)
        lanes[spec.name] = dict(region=float(r.detach()), placement=float(p.detach()))
    loss = torch.stack(regions).mean() + torch.stack(placements).mean()
    need(bool(torch.isfinite(loss)), 'EXPERIENCE_FIT_LOSS')
    return loss, dict(loss=float(loss.detach()), label=int(label), lanes=lanes,
                      placement_region=chosen.regions[0], purpose=plan['decisions'][target_key])
