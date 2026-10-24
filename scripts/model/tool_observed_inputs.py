"""Append measured occurrences to a typed bank without changing source-report rows."""
import copy

import numpy as np

from claims.codata_source import check
from claims.episodic_memory import digest


def append(inputs, native, observation):
    check(native is not None and native.specs == inputs.specs, "TOOL_GROWTH_NATIVE_REQUIRED")
    catalogue = native
    native = native.for_observation(observation)
    binding = native.checked(observation)
    entries = getattr(inputs, "observed_regions", {})
    check(all(r["observation"] != observation for r in entries.values()),
          "TOOL_GROWTH_ALREADY_PRESENT")
    result = copy.copy(inputs)
    unit = len(inputs.refs)
    result.refs = list(inputs.refs)+[dict(native.refs[binding["unit"]])]
    result.arrays = {}
    for spec in inputs.specs:
        old, offsets = inputs.arrays[spec.name]
        values, ptr = native.arrays[spec.name]
        addition = values[ptr[binding["unit"]]:ptr[binding["unit"]+1]]
        result.arrays[spec.name] = (
            np.concatenate((old, addition)),
            np.concatenate((offsets, np.asarray([offsets[-1]+len(addition)], dtype=np.int64))))
        for array in result.arrays[spec.name]:
            array.flags.writeable = False
    response = dict(kind="captured_process_output", unit=unit, observation=observation,
                    native=binding, target=native.observations.entries[observation]["target"])
    result.observed_regions = {**entries, str(unit): response}
    result.execution_inputs = catalogue
    result.source_ancestry = (*getattr(inputs, "source_ancestry", ()), inputs.source_id)
    result.source_id = digest(dict(parent=inputs.source_id, addition=response))
    return result, response


def validate(inputs, key, response, action):
    native = getattr(inputs, "execution_inputs", None)
    check(native is not None and getattr(inputs, "observed_regions", {}).get(key) == response,
          "TOOL_KERNEL_OBSERVED_SOURCE")
    observation = response["observation"]
    native = native.for_observation(observation)
    check(response["native"] == native.checked(observation)
          and response["target"] == native.observations.entries[observation]["target"]
          and response["target"] == inputs.population.source.reference(action["task"], action["turn"]+1)
          and response["unit"] == int(key)
          and action["partition"] == "train", "TOOL_KERNEL_OBSERVED_BINDING")
    for spec in inputs.specs:
        x, ptr = inputs.arrays[spec.name]
        y, other = native.arrays[spec.name]
        u, v = int(key), response["native"]["unit"]
        check(np.array_equal(x[ptr[u]:ptr[u+1]], y[other[v]:other[v+1]]),
              "TOOL_KERNEL_OBSERVED_WINDOWS")
