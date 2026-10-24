"""Independent scalar readback of committed full-constellation forecasts."""
import base64
import json
import math
import sqlite3
import struct

from claims.codata_source import sha
from claims.episodic_memory import canonical, digest, ZERO


def decode(value):
    raw = base64.b64decode(value["base64"], validate=True)
    assert sha(raw) == value["sha256"]
    n, d = value["shape"]
    assert value["dtype"] == "float32-le" and len(raw) == n*d*4
    flat = struct.unpack("<"+"f"*(n*d), raw)
    return [flat[i*d:(i+1)*d] for i in range(n)]


def cosine(a, b):
    return max(-1., min(1., math.fsum(x*y for x, y in zip(a, b)) /
        math.sqrt(math.fsum(x*x for x in a)*math.fsum(y*y for y in b))))


def edit_distance(matrix):
    # Full-table scalar implementation, independent of the production rolling rows.
    n, m = len(matrix), len(matrix[0])
    grid = [[float(i+j) for j in range(m+1)] for i in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, m+1):
            grid[i][j] = min(grid[i-1][j]+1, grid[i][j-1]+1, grid[i-1][j-1]+1-matrix[i-1][j-1])
    return grid[n][m]/max(n, m)


def read_boundary(path, expected, specs):
    raw = (path.parent/"constellation-boundary.json").read_bytes()
    body = json.loads(raw)
    assert canonical(body) == raw and body["standard"] == expected["standard"] and body["world"] == expected["world"]
    assert body["representation_only"] and body["live_snapshot_unchanged"]
    assert body["differentiable_per_instrument"] and not body["reference_gradients"]
    assert set(body["actual_multiwindow_slots"]) == set(specs) and len(body["controls"]) == 10
    for name, slot in body["actual_multiwindow_slots"].items():
        actual, prediction = slot["input"], decode(slot["prediction"])
        assert len(actual) == len(prediction) == slot["windows"] == 2
        assert slot["instrument"] == slot["prediction"]["instrument"] == specs[name].identity
        assert all(len(v) == specs[name].dims for v in actual+prediction)
        matrix = [[cosine(x, y) for y in prediction] for x in actual]
        assert all(abs(a-b) < 1e-12 for x, y in zip(matrix, slot["cosines"]) for a, b in zip(x, y))
        assert abs(edit_distance(matrix)-slot["ordered_loss"]) < 1e-12
        exact = [[cosine(x, y) for y in actual] for x in actual]
        dropped = [[cosine(x, y) for y in actual[:1]] for x in actual]
        reversed_windows = [[cosine(x, y) for y in reversed(actual)] for x in actual]
        for values, name in ((exact, "exact_loss"), (dropped, "dropped_loss"), (reversed_windows, "reversed_loss")):
            assert abs(edit_distance(values)-slot[name]) < 1e-12
        assert slot["dropped_loss"] > slot["exact_loss"] and slot["reversed_loss"] > slot["exact_loss"]
    if (path.parent/"report.json").is_file():
        report = json.loads((path.parent/"report.json").read_bytes())
        assert report["extension"]["sha256"] == sha(raw)
    return dict(sha256=sha(raw), controls=len(body["controls"]), native_windows=12)


def independent(path, inputs, source, expected):
    db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    head, count, forecasts, resolved, contexts = ZERO, 0, {}, {}, {}
    specs = {s.name: s for s in inputs.specs}
    try:
        for seq, previous, raw, pin in db.execute("SELECT seq,previous,body,hash FROM events ORDER BY seq"):
            body = json.loads(raw)
            assert seq == count and previous == head and canonical(body) == raw
            assert pin == digest(dict(seq=seq, previous=previous, body=body))
            kind = body["kind"]
            if kind == "world_init":
                contexts = {c: dict(sequence=0) for c in body["contract"]["contexts"]}
            elif kind == "world_advance":
                r = body["request"]
                assert r["sequence"] == contexts[r["context"]]["sequence"]
                assert len({tuple(v) for v in body["owner"].values()}) == 1
                contexts[r["context"]] = dict(sequence=r["sequence"]+1, last_event=digest(r),
                                              hidden_sha256=body["hidden_sha256"])
            elif kind == "prediction_commit":
                r = body["record"]
                assert body["id"] == digest(r) and r["prior_head"] == head
                assert r["mode"] == "retrospective_temporal_replay" and r["issued_at"][:10] >= r["target_published"]
                assert r["context_sequence"] == contexts[r["context"]]["sequence"]
                assert set(r["predictions"]) == set(specs)
                constellation = r["constellation"]
                assert constellation["schema"] == "typed-native-constellation-v1"
                assert constellation["bank"] == expected["world"]
                assert constellation["ordered_instruments"] == [s.identity for s in inputs.specs]
                selected = int(constellation["region"].split(":")[0])
                assert constellation["region"] == str(selected)+":"+inputs.refs[selected]["text_sha256"]
                assert any(a["unit"] == selected and a["edition"] == r["source_edition"]
                           and a["kind"] == "row" for a in inputs.aliases)
                for name, packed in r["predictions"].items():
                    assert packed["instrument"] == specs[name].identity and packed["shape"][1] == specs[name].dims
                    predicted = decode(packed)
                    vectors, ptr = inputs.arrays[name]
                    region = vectors[ptr[selected]:ptr[selected+1]]
                    assert len(predicted) == len(region)
                    assert all(cosine(a, b) >= math.sqrt(.99)-1e-6 for a, b in zip(predicted, region))
                forecasts[body["id"]] = (seq, r)
            elif kind == "prediction_observe":
                key, request = body["forecast"], body["request"]
                committed, record = forecasts[key]
                assert key not in resolved and committed < seq
                assert request["quantity"] == record["quantity"] and request["context"] == record["context"]
                assert request["checked_receipt"] == source.assess(source.episode(record["target_edition"], record["quantity"]))
                assert request["prior_sequence"] == record["context_sequence"]
                assert body["assessment"]["metric"] == "per_instrument_ordered_cosine_edit_distance_v1"
                for name, value in body["assessment"]["instruments"].items():
                    pred = decode(record["predictions"][name])
                    arrays, offsets = inputs.arrays[name]
                    old = record["baseline_unit"]
                    future = inputs.unit("row", record["target_edition"], record["quantity"])
                    a = [list(map(float, v)) for v in arrays[offsets[old]:offsets[old+1]]]
                    b = [list(map(float, v)) for v in arrays[offsets[future]:offsets[future+1]]]
                    pc = [[cosine(x, y) for y in pred] for x in b]
                    bc = [[cosine(x, y) for y in a] for x in b]
                    assert value["prediction_windows"] == len(pred) and value["target_windows"] == len(b)
                    assert value["baseline_windows"] == len(a) and value["instrument"] == specs[name].identity
                    for actual, saved in ((pc, value["prediction_cosines"]), (bc, value["persistence_cosines"])):
                        assert len(actual) == len(saved)
                        assert all(len(x) == len(y) and all(abs(i-j) < 1e-12 for i, j in zip(x, y))
                                   for x, y in zip(actual, saved))
                    loss, baseline = edit_distance(pc), edit_distance(bc)
                    assert abs(value["prediction_loss"]-loss) < 1e-12
                    assert abs(value["persistence_loss"]-baseline) < 1e-12
                    assert abs(value["delta"]-(loss-baseline)) < 1e-12
                resolved[key] = body["assessment"]
                contexts[request["context"]] = dict(sequence=request["prior_sequence"]+1,
                    last_event=digest(request), hidden_sha256=body["hidden_sha256"])
            head, count = pin, count+1
        assert head == expected["head"] and count == expected["events"]
        assert set(forecasts) == set(resolved) == set(expected["forecasts"])
        for key, result in resolved.items():
            saved = expected["forecasts"][key]
            assert saved["result"] == result and saved["grounding"]["establishes_empirical_truth"] is False
            assert saved["constellation"] == forecasts[key][1]["constellation"]
            assert saved["shapes"] == {n: v["shape"] for n, v in forecasts[key][1]["predictions"].items()}
        for c, state in contexts.items():
            assert all(expected["contexts"][c][k] == v for k, v in state.items())
        boundary = read_boundary(path, expected, specs)
        return dict(events=count, forecasts=len(forecasts), scored=len(resolved),
                    instrument_scores=sum(len(x["instruments"]) for x in resolved.values()),
                    scalar_score_readback=True, common_region_full_windows=True, boundary=boundary, head=head)
    finally:
        db.close()
