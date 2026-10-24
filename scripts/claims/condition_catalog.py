"""Route immutable condition sets without changing their evidence identities."""
from claims.codata_source import check
from claims.episodic_memory import digest


class ConditionCatalog:
    def __init__(self, providers):
        self.providers = tuple(providers)
        check(self.providers, "CONDITION_CATALOG_EMPTY")
        self.inputs, self.lookup = {}, {}
        source = self.providers[0].commitment["source"]
        for provider in self.providers:
            check(provider.commitment["source"] == source, "CONDITION_CATALOG_SOURCE")
            check(not self.inputs.keys() & provider.inputs.keys(), "CONDITION_CATALOG_COLLISION")
            self.inputs.update(provider.inputs)
            self.lookup.update((key, provider) for key in provider.inputs)
        self.pin = digest(dict(schema="condition-catalog-v1", providers=[p.pin for p in self.providers]))

    def for_condition(self, key):
        check(key in self.lookup, "CONDITION_CATALOG_UNKNOWN")
        return self.lookup[key]

    def read(self, key):
        return self.for_condition(key).read(key)

    def episode(self, key, evidence=()):
        return self.for_condition(key).episode(key, evidence)

    def validate_episode(self, episode):
        self.for_condition(episode["source"]["condition"]).validate_episode(episode)

    def assess(self, episode):
        return self.for_condition(episode["source"]["condition"]).assess(episode)
