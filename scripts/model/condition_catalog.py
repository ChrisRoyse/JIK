"""Keep program-specific native condition windows in their original providers."""
from claims.codata_source import check
from claims.condition_catalog import ConditionCatalog


class NativeConditions:
    def __init__(self, providers):
        self.providers = tuple(providers)
        check(self.providers, "CONDITION_NATIVE_CATALOG_EMPTY")
        self.base, self.specs = self.providers[0].base, self.providers[0].specs
        check(all(p.base is self.base and p.specs == self.specs for p in self.providers),
              "CONDITION_NATIVE_CATALOG_SOURCE")
        self.conditions = ConditionCatalog([p.conditions for p in self.providers])
        self.lookup = {k:p for p in self.providers for k in p.conditions.inputs}
        self.identity = dict(schema="condition-native-catalog-v1",
                             providers=[p.identity for p in self.providers])

    def for_condition(self, key):
        check(key in self.lookup, "CONDITION_NATIVE_CATALOG_UNKNOWN")
        return self.lookup[key]

    def checked(self, key):
        return self.for_condition(key).checked(key)

    def merge(self, reference, binding, config, device):
        return self.for_condition(binding["condition"]).merge(reference, binding, config, device)
