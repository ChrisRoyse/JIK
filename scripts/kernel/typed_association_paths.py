"""Stream exact paths in declared instrument order without aggregating lens scores."""
from kernel.published_associations import need


class TypedPaths:
    def __init__(self, lanes):
        need(isinstance(lanes, dict) and lanes and all(n == lane.binding['instrument'] for n, lane in lanes.items()), 'PATH_LANES')
        signatures = {(v.binding['source_pin'], v.binding['bank_pin'], v.population) for v in lanes.values()}
        need(len(signatures) == 1, 'PATH_SOURCE_MISMATCH')
        self.lanes = dict(lanes)

    def paths(self, origin, instruments):
        # This is a physical traversal-stack limit, not a limit on emitted work.
        need(isinstance(instruments, (list, tuple)) and 0 < len(instruments) <= 128, 'PATH_STACK_MEMORY')
        need(all(isinstance(n, str) and n in self.lanes for n in instruments), 'PATH_MISSING_INSTRUMENT')
        steps = tuple(instruments); stack = []; selected = []
        def outgoing(uid, depth):
            lane = self.lanes[steps[depth]]
            return iter(lane.read([uid], lens_id=lane.binding['lens_id'])[0]['edges'])
        try:
            stack.append(outgoing(origin, 0))
            while stack:
                try: edge = next(stack[-1])
                except StopIteration:
                    stack.pop()
                    if selected: selected.pop()
                    continue
                if len(stack) == len(steps):
                    path = [*selected, edge]
                    for n, e in zip(steps, path): self.lanes[n].fresh(e['origin'])
                    yield dict(origin=origin, target=edge['target'], instruments=list(steps), edges=[dict(e) for e in path],
                        conclusion='association_path_only', scientific_admission=False, implicit_training_authority=False,
                        all_pairs_exhaustive=False, cross_instrument_score=None)
                else:
                    selected.append(edge); stack.append(outgoing(edge['target'], len(stack)))
        finally:
            stack.clear(); selected.clear()
