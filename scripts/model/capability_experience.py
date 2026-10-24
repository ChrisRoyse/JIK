"""Explicit training views over active measured experience; routing bank stays frozen."""
from claims.capability_population import need
from claims.episodic_memory import digest
from model.capability_kernel_learning import KernelLearningCorpus


class IndexedTuple(tuple):
    """Keep canonical iteration while avoiding linear membership per training target."""
    def __new__(cls, values):
        result = super().__new__(cls, values)
        result.members = frozenset(result)
        return result

    def __contains__(self, key):
        return key in self.members


class ExperienceCorpus(KernelLearningCorpus):
    def __init__(self, config, *, training_scope):
        need(training_scope in ('kernel', 'all_measured'), 'EXPERIENCE_SCOPE')
        super().__init__(config)
        self.kernel_train = IndexedTuple(self.train)
        self.query = IndexedTuple(self.query)
        self.measured_train = IndexedTuple(sorted(self.kernel['ids']))
        self.measured_train_set = set(self.measured_train)
        self.query_set = set(self.query)
        need(len(self.measured_train) == len(set(self.measured_train))
             and set(self.kernel_train) <= set(self.measured_train)
             <= set(self.kernel['split']['train']), 'EXPERIENCE_TRAIN_ROSTER')
        self.query_groups = {self.tasks[k]['split_group'] for k in self.query}
        for key in self.measured_train: self.check_training_task(key)
        self.train = self.kernel_train if training_scope == 'kernel' else self.measured_train
        self.training_scope = training_scope
        # self.regions remains the original kernel bank. More training contexts
        # must not silently change a frozen recognition target or query split.
        self.training_identity = digest(dict(kernel=config['kernel'], train=self.train,
                                             regions=self.regions, scope=training_scope))

    def check_training_task(self, key):
        need(key in self.measured_train_set and key in self.tasks and key not in self.query_set,
             'EXPERIENCE_NOT_ACTIVE_TRAIN')
        t = self.tasks[key]
        need(t['split_group'] not in self.query_groups, 'EXPERIENCE_GROUP_LEAK')
        need(t['population'] in self.sources, 'EXPERIENCE_UNMEASURED_SOURCE')
        source = self.sources[t['population']]
        need(source.pin == t['binding'] and t['kind'] == source.population.tasks[key]['kind']
             and len(t['selections']['targets']['fields']) == 1, 'EXPERIENCE_BINDING')
        episodes = [t['episode'], *t['related_executions']]
        for side, selection in t['selections'].items():
            need(selection == source.population.selection(key, side=side), 'EXPERIENCE_SELECTION')
            for f in selection['fields']:
                receipt = self.snapshot['capability_memory']['fields'][t['population'] + ':' + f['id']]
                episodes.append(receipt['episode'])
        need(all(self.snapshot['memory']['states'][e]['active'] and
                 self.snapshot['memory']['states'][e]['grounded_scope'] is not None for e in episodes),
             'EXPERIENCE_INACTIVE_EVIDENCE')

    def schedule(self, numerator, denominator):
        need(self.training_scope == 'all_measured' and 0 < numerator <= denominator,
             'EXPERIENCE_SCHEDULE_SCOPE')
        groups = sorted({self.tasks[k]['split_group'] for k in self.train}, key=digest)
        count = (len(groups) * numerator + denominator - 1) // denominator
        chosen = groups[:count]
        selected = set(chosen)
        plan = dict(identity=self.training_identity, fraction=[numerator, denominator], groups=chosen,
                    tasks=[k for k in self.train if self.tasks[k]['split_group'] in selected],
                    policy='hash-ordered independent-group prefix; no outcomes or query targets',
                    exposure='development_exposed; not a fresh evaluation seal')
        self.check_schedule(plan)
        return plan

    def check_schedule(self, plan):
        need(plan['identity'] == self.training_identity, 'EXPERIENCE_SCHEDULE_IDENTITY')
        groups = plan['groups']
        fraction = plan['fraction']
        need(len(fraction) == 2 and all(type(n) is int for n in fraction)
             and 0 < fraction[0] <= fraction[1], 'EXPERIENCE_FRACTION')
        available = sorted({self.tasks[k]['split_group'] for k in self.train}, key=digest)
        count = (len(available)*fraction[0]+fraction[1]-1)//fraction[1]
        need(groups == available[:count], 'EXPERIENCE_GROUP_PREFIX')
        need(groups and len(groups) == len(set(groups)) and not set(groups) & self.query_groups,
             'EXPERIENCE_SCHEDULE_GROUPS')
        need(set(groups) <= {self.tasks[k]['split_group'] for k in self.train}, 'EXPERIENCE_UNKNOWN_GROUP')
        selected = set(groups)
        expected = [k for k in self.train if self.tasks[k]['split_group'] in selected]
        need(plan['tasks'] == expected, 'EXPERIENCE_INCOMPLETE_GROUP')
        for key in expected: self.check_training_task(key)
