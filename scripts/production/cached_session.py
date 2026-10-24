"""Lazy model ownership; validated native cache hits never acquire the GPU."""
from senses.contract import require
from production.atom_cache import panel_key, selector, input_key, instrument
from production.native_rows import one, joined
from production.session import prepare


class CachedSession:
    def __init__(self, plan, name, reader, blobs, cache, load):
        self.plan, self.name, self.reader, self.blobs, self.cache, self.load = plan, name, reader, blobs, cache, load
        self.item = plan["instruments"][name]
        self.module = self.item["spec"]["module"]
        self.panel, self.selector = panel_key(plan), selector(plan, name)
        self.delegate, self.info = None, None
        self.pending = {}
        self.hits = self.misses = 0

    def _bind(self, info):
        if self.info is not None:
            require(self.info == info, "MNLM_CACHE_CONFLICT", "cached and resident instrument identities differ")
        self.info = info
        self.dim, self.kind, self.lens_id = info["dim"], info["kind"], info["lens_id"]
        self.contract = {**info["identity"], "ucr_generation": self.reader.generation,
                         "policies_sha256": self.reader.policy_hash}
        if self.module in ("audio_ucr", "image_ucr", "vision_ucr"):
            self.contract["blobs_manifest_sha256"] = self.blobs.sha256

    def begin_shard(self):
        self.pending = {}

    @property
    def initial_state(self):
        return self.delegate.initial_state if self.delegate is not None else "cache-only"

    def state_digest(self):
        return self.delegate.state_digest() if self.delegate is not None else "cache-only"

    @property
    def replay_mode(self):
        hits = [hit for _, hit in self.pending.values()]
        return "cache_readback" if all(hits) else "mixed_forward_and_cache" if any(hits) else "forward"

    def batch(self, rows, policy):
        prepared = prepare(self.item, self.reader, rows, policy)
        results, details, missing = [None] * len(rows), [None] * len(rows), []
        for i, p in enumerate(prepared):
            key = input_key(self.panel, self.selector, p)
            found = self.cache.lookup(key)
            state = (key, found is not None)
            atom = key["atom"]
            if atom in self.pending:
                require(self.pending[atom] == state, "MNLM_CACHE_INPUT", "visible input changed during shard measurement")
            self.pending[atom] = state
            if found is None:
                missing.append(i)
            else:
                results[i], detail, info, origin = found
                self._bind(info)
                details[i] = {**detail, "cache_origin": origin}
        if missing:
            if self.delegate is None:
                self.delegate = self.load()
                slot = {"contract": self.delegate.contract, "lens_id": self.delegate.lens_id,
                        "seal": {"dims": self.delegate.dim}}
                self._bind(instrument(slot, self.item))
            arrays, _, extra = self.delegate.batch_prepared([prepared[i] for i in missing])
            for j, i in enumerate(missing):
                results[i], details[i] = one(arrays, j), extra[j]
        return joined(results, self.kind, self.dim), [p["binding"] for p in prepared], details

    def accept(self, rows, path, checkpoint, after_insert=None):
        keys = [self.pending[r["atom"]][0] for r in rows]
        self.hits += sum(self.pending[r["atom"]][1] for r in rows)
        self.misses += sum(not self.pending[r["atom"]][1] for r in rows)
        return self.cache.accept(self.plan, self.name, rows, keys, path, checkpoint, after_insert=after_insert)

    def backfill(self, rows, policy, path, checkpoint):
        prepared = prepare(self.item, self.reader, rows, policy)
        keys = [input_key(self.panel, self.selector, p) for p in prepared]
        return self.cache.accept(self.plan, self.name, rows, keys, path, checkpoint)

    def close(self):
        if self.delegate is not None:
            self.delegate.close()
