"""Independent native-view reconstruction and causal private-region controls."""
import base64
import copy
import hashlib

import numpy as np

from claims.codata_source import check
from claims.episodic_memory import canonical, digest
from audit.tool_native_recognition import scores as recognition_scores, context as world_context


def inspect(inputs, record, bank, episodes, promoted, support, core, extra, hidden):
    _, recognized = recognition_scores(inputs, record, core, extra, hidden)
    check(recognized == int(record["region"].split(":")[0]), "PRIVATE_READ_RECOGNITION")
    req, saved = record["request"], record["region_context"]
    private = []
    check(record["support"] == {str(k): v for k, v in support.items()}
          and set(support) == set(range(1, req["turn"]+1)), "PRIVATE_READ_PREFIX")
    for turn in range(3, req["turn"], 2):
        ref = inputs.population.lookup[req["task"], turn]
        ep = support[turn]
        check(ep in promoted and episodes[ep]["source"] ==
              inputs.population.source.reference(req["task"], turn), "PRIVATE_READ_SOURCE")
        private.append(dict(reference=ref, episode=ep))
    shared = [bank[i] for i in record["routing"]["allowed"]]
    units = list(dict.fromkeys(shared+[p["reference"]["unit"] for p in private]))
    desc = dict(schema="tool-private-regions-v2", source=inputs.source_id,
                context=req["context"], task=req["task"], cutoff=req["turn"],
                kernel=record["routing"]["kernel"], shared_world=record["world"],
                shared=shared, private=private, units=units,
                history=[inputs.population.lookup[req["task"], t] for t in range(1, req["turn"])])
    check(saved["definition"] == desc, "PRIVATE_READ_DEFINITION")
    rows = [f"{u}:{inputs.refs[u]['text_sha256']}" for u in units]
    check(record["region"] in rows and record["baseline_unit"] in units, "PRIVATE_READ_SELECTION")
    # Reconstruct the byte-addressed world independently, from original native arrays.
    h = hashlib.sha256(canonical(dict(source_id=digest(desc), instruments=[s.identity for s in inputs.specs])))
    hidden = world_context(inputs, record, core, extra, hidden)
    selected = int(record["region"].split(":")[0])
    windows, errors = 0, {}
    for lane, spec in enumerate(inputs.specs):
        a, ptr = inputs.arrays[spec.name]
        lengths = [int(ptr[u+1]-ptr[u]) for u in units]
        values = np.zeros((len(units), max(lengths), spec.dims), np.float32)
        padding = np.ones(values.shape[:2], bool)
        for i, unit in enumerate(units):
            values[i, :lengths[i]] = a[ptr[unit]:ptr[unit+1]]
            padding[i, :lengths[i]] = False
        h.update(canonical(dict(instrument=spec.name, rows=rows)))
        for array in (values, padding, np.zeros(len(units), bool)):
            h.update(canonical([list(array.shape), array.dtype.str]))
            h.update(array.tobytes())
        book = a[ptr[selected]:ptr[selected+1]].astype(np.float64)
        name = spec.name
        projected = book@core[f"projectors.{name}.weight"].astype(np.float64).T
        projected += core[f"projectors.{name}.bias"]
        residual = np.tanh((projected+hidden[lane])@extra[f"place.{name}.weight"].astype(np.float64).T
                          + extra[f"place.{name}.bias"])
        residual = .1*residual/np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1.)
        expected = book/np.linalg.norm(book, axis=1, keepdims=True)+residual
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        packed = record["predictions"][name]
        actual = np.frombuffer(base64.b64decode(packed["base64"]), dtype="<f4").reshape(packed["shape"])
        check(actual.shape == expected.shape, "PRIVATE_READ_WINDOW_ROSTER")
        errors[name] = float(np.max(np.abs(actual-expected)))
        check(errors[name] < 1e-6, "PRIVATE_READ_PLACEMENT")
        windows += sum(lengths)
    check(h.hexdigest() == saved["world"], "PRIVATE_READ_WORLD_PIN")
    return dict(context=req["context"], cutoff=req["turn"], regions=len(units),
                private_occurrences=len(private), native_windows=windows, max_error=errors,
                selected=selected, selected_private=selected in [p["reference"]["unit"] for p in private],
                latest_available=True, world=saved["world"])


def controls(session, refuse):
    from model.tool_context_regions import historical
    record = next(v["record"] for v in session.forecasts.values()
                  if v["record"]["request"]["context"] == "withheld:0")
    desc = record["region_context"]["definition"]
    def reject(name, mutate, code="TOOL_PRIVATE_HISTORY"):
        changed = copy.deepcopy(record)
        mutate(changed)
        refuse(name, code, lambda: historical(session, changed))
    reject("foreign_private_context",
           lambda r: r["region_context"]["definition"].update(context="withheld:1"))
    future = session.inputs.population.lookup[record["request"]["task"], record["request"]["turn"]+1]
    reject("future_private_response",
           lambda r: r["region_context"]["definition"]["private"].append(dict(reference=future, episode="0"*64)))
    reject("proposal_as_private_observation",
           lambda r: r["region_context"]["definition"]["private"].append(
               dict(reference=r["request"]["reference"], episode=r["support"][str(r["request"]["turn"])])))
    reject("altered_private_units", lambda r: r["region_context"]["definition"]["units"].reverse())
    reject("reordered_history", lambda r: r["region_context"]["definition"]["history"].reverse())
    reject("missing_history", lambda r: r["region_context"]["definition"]["history"].pop())
    reject("future_history", lambda r: r["region_context"]["definition"]["history"].append(future))
    reject("altered_private_world", lambda r: r["region_context"].update(world="0"*64),
           "TOOL_PRIVATE_VIEW_PIN")
    other = next(v["record"] for v in session.forecasts.values()
                 if v["record"]["request"]["context"] == "withheld:1")
    reject("foreign_private_episode", lambda r: r["support"].update({"3": other["support"]["3"]}),
           "TOOL_PRIVATE_SOURCE")
    reject("missing_private_prefix", lambda r: r["support"].pop("3"), "TOOL_PRIVATE_PREFIX")
    check(desc["private"], "PRIVATE_CONTROL_REAL_PREFIX")
