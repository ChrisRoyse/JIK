"""Source-bound action -> response routing over separate native-window kernels."""
import json
from pathlib import Path

import numpy as np

from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from kernel.derive import derive
from kernel.native_route import NativeLane, Router


PLAN = dict(ann_candidates=64, ann_connectivity=16, ann_expansion_add=128,
    ann_expansion_search=64, ef_search=64, entry_points=1, kernel_fraction=.1,
    max_distance_computations=512, neighbors=16, recall_gate=.95, top_k=10,
    usearch_version="2.26.5")


def code_pins():
    scripts = Path(__file__).resolve().parents[1]
    return {name: sha((scripts/name).read_bytes()) for name in
            ("model/tool_kernel.py", "model/tool_observed_inputs.py",
             "kernel/native_route.py", "kernel/derive.py")}


def pairs(inputs):
    return {str(ref["unit"]): dict(response=ref,
        action=inputs.population.lookup[ref["task"], ref["turn"]-1])
        for ref in inputs.population.initial_bank()}


def windows(inputs, name, unit):
    values, ptr = inputs.arrays[name]
    return values[ptr[unit]:ptr[unit+1]]


def lanes(inputs, pair_map):
    ids = sorted(pair_map)
    result = {}
    for spec in inputs.specs:
        pieces = [windows(inputs, spec.name, pair_map[key]["action"]["unit"]) for key in ids]
        offsets = np.asarray([0]+list(np.cumsum([len(p) for p in pieces])), dtype=np.int64)
        result[spec.name] = NativeLane(ids, np.concatenate(pieces), offsets)
    return result


def calibration(inputs, lane_map, graphs):
    """Actions excluded from graph; no target responses are accessed here."""
    refs = [inputs.population.lookup[row["task"], turn]
            for row in inputs.population.source.episodes
            if row["status"] == "ready" and row["partition"] == "development"
            for turn in (6, 10)]
    check(len(refs) == 32, "TOOL_KERNEL_CALIBRATION_ROSTER")
    result = {}
    for spec in inputs.specs:
        lane = lane_map[spec.name]
        router = Router(lane, graphs[spec.name], PLAN)
        rows = []
        for ref in refs:
            query = windows(inputs, spec.name, ref["unit"])
            # Independent original-window scoring, not Router's normalized cache.
            q = query.astype(np.float64)
            q /= np.sqrt(np.einsum("ij,ij->i", q, q))[:, None]
            direct = {}
            for key in lane.ids:
                row = lane.rows[key]
                x = lane.original[lane.offsets[row]:lane.offsets[row+1]].astype(np.float64)
                x /= np.sqrt(np.einsum("ij,ij->i", x, x))[:, None]
                c = np.einsum("ik,jk->ij", q, x)
                direct[key] = float(min(np.max(c, axis=1).mean(), np.max(c, axis=0).mean()))
            exact = sorted(direct, key=lambda key: (-direct[key], key))[:PLAN["top_k"]]
            trace = router.search(query)
            check(all(abs(score-direct[key]) < 2e-12 for key, score in trace["hits"]),
                  "TOOL_KERNEL_SCORE_READBACK")
            recall = len(set(exact) & {key for key, _ in trace["hits"]})/PLAN["top_k"]
            rows.append(dict(reference=ref, exact=exact, route=trace, recall=recall))
        recall = sum(r["recall"] for r in rows)/len(rows)
        result[spec.name] = dict(rows=rows, recall=recall, passes=recall >= PLAN["recall_gate"])
    return result


def validate_pairs(inputs, pair_map):
    check(set(pair_map) == {str(u) for u in inputs.world_units}, "TOOL_KERNEL_REGION_ROSTER")
    for key, pair in pair_map.items():
        a, b = pair["action"], pair["response"]
        if b.get("kind") == "captured_process_output":
            from model.tool_observed_inputs import validate
            check(canonical(a) == canonical(inputs.population.lookup.get((a["task"], a["turn"])))
                  and a["role"] == "generated_proposal", "TOOL_KERNEL_PAIR_SOURCE")
            validate(inputs, key, b, a)
            continue
        check(canonical(a) == canonical(inputs.population.lookup.get((a["task"], a["turn"])))
              and canonical(b) == canonical(inputs.population.lookup.get((b["task"], b["turn"])))
              and a["task"] == b["task"] and b["turn"] == a["turn"]+1
              and a["role"] == "generated_proposal" and b["role"] == "recorded_response"
              and a["partition"] == b["partition"] == "train" and key == str(b["unit"]),
              "TOOL_KERNEL_PAIR_SOURCE")


def build(inputs, path, *, pair_map=None):
    check(not path.exists(), "TOOL_KERNEL_EXISTS")
    pair_map = pairs(inputs) if pair_map is None else pair_map
    validate_pairs(inputs, pair_map)
    lane_map = lanes(inputs, pair_map)
    graphs, work = {}, {}
    for name, lane in lane_map.items():
        edges, work[name] = lane.weave(PLAN)
        graphs[name] = derive(lane.ids, edges, PLAN)
    measured = calibration(inputs, lane_map, graphs)
    result = dict(schema="tool-paired-kernel-v1", source=inputs.source_id, plan=PLAN,
        code=code_pins(), pairs=pair_map, instruments={s.name: s.identity for s in inputs.specs},
        graphs=graphs, work=work, calibration=measured, scientific_admission=False,
        world_truth_established=False)
    path.mkdir(parents=True)
    raw = canonical(result)
    (path/"manifest.json").write_bytes(raw)
    (path/"COMMITTED").write_text(sha(raw))
    return sha(raw)


class ToolKernel:
    def __init__(self, inputs, path, expected, *, expected_pairs=None):
        raw = (path/"manifest.json").read_bytes()
        check(sha(raw) == expected == (path/"COMMITTED").read_text(), "TOOL_KERNEL_PIN")
        value = json.loads(raw)
        expected_pairs = pairs(inputs) if expected_pairs is None else expected_pairs
        validate_pairs(inputs, expected_pairs)
        check(value["code"] == code_pins() and value["source"] == inputs.source_id
              and value["plan"] == PLAN and value["pairs"] == expected_pairs
              and value["instruments"] == {s.name: s.identity for s in inputs.specs}
              and set(value["graphs"]) == {s.name for s in inputs.specs}
              and value["scientific_admission"] is False
              and value["world_truth_established"] is False, "TOOL_KERNEL_CONTRACT")
        self.inputs, self.pin, self.pairs = inputs, expected, value["pairs"]
        self.lanes = lanes(inputs, self.pairs)
        actual = calibration(inputs, self.lanes, value["graphs"])
        check(canonical(actual) == canonical(value["calibration"]), "TOOL_KERNEL_RECALL_READBACK")
        check(all(row["passes"] for row in actual.values()), "TOOL_KERNEL_RECALL_GATE")
        self.calibration = actual
        self.routers = {name: Router(lane, value["graphs"][name], PLAN)
                        for name, lane in self.lanes.items()}
        self.identity = dict(pin=expected, source=inputs.source_id, code=code_pins())

    def action_references(self):
        return tuple(pair["action"] for pair in self.pairs.values())

    def route(self, reference):
        expected = self.inputs.population.lookup.get((reference.get("task"), reference.get("turn")))
        check(expected is not None and canonical(reference) == canonical(expected)
              and reference["role"] == "generated_proposal", "TOOL_KERNEL_QUERY")
        traces, votes = {}, {}
        for spec in self.inputs.specs:
            trace = self.routers[spec.name].search(windows(self.inputs, spec.name, reference["unit"]))
            traces[spec.name] = trace
            for rank, (key, _) in enumerate(trace["hits"], 1):
                votes[key] = votes.get(key, 0.)+1/(60+rank)
        chosen = sorted(votes, key=lambda key: (-votes[key], key))[:PLAN["top_k"]]
        check(len(chosen) == PLAN["top_k"], "TOOL_KERNEL_SCARCITY")
        selected = {self.pairs[key]["response"]["unit"] for key in chosen}
        allowed = [i for i, unit in enumerate(self.inputs.world_units) if unit in selected]
        check(len(allowed) == len(chosen), "TOOL_KERNEL_RESPONSE_JOIN")
        return dict(kernel=self.pin, query=reference, traces=traces, responses=chosen,
                    fusion="reciprocal_rank_k60", allowed=allowed)
