"""Scientific source spans -> existing typed native instrument interfaces."""
import json
from pathlib import Path

import numpy as np

from audit.window_readback import readback as window_readback
from claims.codata_source import check, sha
from claims.scientific_population import readback as population_readback
from model.contracts import Instrument
from model.loop_input import LANES, LoopInputs
from senses.contract import canonical, digest


class ScientificInputs:
    def __init__(self, source, population, native, *, population_pin, native_pin, audit_windows=True):
        population, native = Path(population), Path(native)
        self.units, self.aliases, _ = population_readback(population, source, population_pin)
        self.window_readback = window_readback(native, population) if audit_windows else None
        raw = (native/"report.json").read_bytes()
        check(sha(raw) == native_pin, "NATIVE_PIN")
        report = json.loads(raw)
        check(report["complete"] and not report["errors"]
              and report["calibration_report_sha256"] == population_pin
              and tuple(sorted(report["lenses"])) == LANES, "NATIVE_JOIN")
        self.refs = [{"text_sha256": sha(u["text"].encode())} for u in self.units]
        self.lookup = {(r["kind"], r["edition"], r["quantity"]): r for r in self.aliases}
        self.specs, self.arrays = [], {}
        for name in LANES:
            root = native/name
            raw = (root/"report.json").read_bytes()
            check(sha(raw) == report["lenses"][name]["report_sha256"], "LANE_PIN")
            lane = json.loads(raw)
            check(sha((root/"slots.npz").read_bytes()) == lane["files"]["slots.npz"]["sha256"], "SLOT_PIN")
            with np.load(root/"slots.npz", allow_pickle=False) as z:
                x, ptr = z["vectors"].copy(), z["offsets"].copy()
            check(len(ptr) == len(self.units)+1 and ptr[0] == 0 and ptr[-1] == len(x)
                  and np.all(np.diff(ptr) > 0), "NATIVE_ROSTER")
            contract = lane["contract"]
            definition = dict(schema="loop-instrument-v1", name=name, channel="language",
                              layout="multi", dims=x.shape[1], frozen=True,
                              weights_sha256=contract["weights_sha256"],
                              training_lineage={"status": "upstream_not_independently_attested"},
                              measurement_contract=contract, lens_id=lane["lens_id"])
            self.specs.append(Instrument(name, digest(definition), "language", "multi", x.shape[1],
                                         canonical(definition).decode()).validate())
            self.arrays[name] = x, ptr
        self.specs = tuple(self.specs)
        self.source_id = digest(dict(population=population_pin, native=native_pin))
        self.world_units = sorted({r["unit"] for r in self.aliases if r["kind"] == "row"})
        self.world_index = {u: i for i, u in enumerate(self.world_units)}

    batch = LoopInputs.batch

    def unit(self, kind, edition, quantity):
        check((kind, edition, quantity) in self.lookup, "INPUT_ADDRESS")
        return self.lookup[kind, edition, quantity]["unit"]

    def world_bank(self, config, device):
        return self.batch(self.world_units, config, device)

    def allowed(self, science, device, *, historical=False):
        """Current checked publication rows; lineage gates are not learned lenses."""
        import torch
        check(science.edition is not None, "NO_ADOPTED_EDITION")
        # One read of the live head before building this bounded candidate snapshot.
        if historical:
            # The caller replays a checked prefix inside the ledger's read transaction.
            check(science.db.in_transaction and science._replaying, "HISTORICAL_SNAPSHOT")
        else:
            row = science.db.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
            check(row is not None and row[0] == science.head, "STALE_LEDGER")
        valid = set()
        for quantity in science.source.rows[science.edition]:
            ep = science.source.episode(science.edition, quantity)
            state = science.memory.state.get(ep["id"])
            if state and state["active"] and state["grounded_scope"] == "pinned_scientific_reference_contents":
                science._checked(science.edition, quantity)
                valid.add(self.unit("row", science.edition, quantity))
        check(bool(valid), "NO_CURRENT_WORLD_REGIONS")
        return torch.tensor([u in valid for u in self.world_units], dtype=torch.bool, device=device)
