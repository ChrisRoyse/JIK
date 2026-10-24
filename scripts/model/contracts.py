"""Versioned sensing machinery, independent of the observations it measures (#25)."""
from dataclasses import asdict, dataclass
import re

from senses.contract import canonical, digest, require

MIN_INSTRUMENTS = 5
HEX = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class Instrument:
    name: str
    identity: str
    channel: str
    layout: str
    dims: int
    definition_json: str

    def validate(self):
        import json
        require(isinstance(self.name, str) and bool(re.fullmatch(r"[a-zA-Z0-9_-]+", self.name)),
                "MNLM_MODEL_INSTRUMENT", "use a nonempty stable instrument name without dots")
        require(self.layout in ("dense", "sparse", "multi") and type(self.dims) is int and self.dims > 0,
                "MNLM_MODEL_LAYOUT", "native layout and positive dimension required")
        require(isinstance(self.channel, str) and bool(self.channel), "MNLM_MODEL_CHANNEL", "declare the input channel")
        definition = json.loads(self.definition_json)
        require(canonical(definition).decode() == self.definition_json and digest(definition) == self.identity,
                "MNLM_MODEL_IDENTITY", "instrument definition changed; register a new version")
        require(all(definition.get(k) == getattr(self, k) for k in ("name", "channel", "layout", "dims"))
                and HEX.fullmatch(definition.get("weights_sha256", ""))
                and definition.get("frozen") is True and bool(definition.get("training_lineage")),
                "MNLM_MODEL_IDENTITY", "pin weights, layout, frozen runtime and training lineage")
        return self

    @classmethod
    def from_slot(cls, name, slot):
        c, seal = slot["contract"], slot["seal"]
        # Generation/policy/row hashes belong to observation provenance, not the
        # identity of the instrument. A new source must not manufacture a lens.
        definition = {"schema": "model-instrument-v1", "name": name, "channel": c["channel"],
                      "layout": seal["kind"], "dims": seal["dims"], "frozen": True,
                      "weights_sha256": c["weights_sha256"], "revision": c.get("revision"),
                      "upstream": c.get("repo"), "instrument_lens_id": c.get("lens_id"),
                      "runtime": c.get("runtime"), "environment": c.get("environment"),
                      "execution_code_sha256": c.get("execution_code_sha256"),
                      "training_lineage": {"declared_corpus_hash": c.get("corpus_hash"),
                                           "status": "algorithmic" if c.get("kind") == "encoder"
                                           else "upstream_training_data_not_attested"}}
        if c.get("training_lineage"):
            definition["training_lineage"] = {"declared_corpus_hash": c.get("corpus_hash"),
                                               "status": "declared_checkpoint_lineage",
                                               "record": c["training_lineage"],
                                               "checkpoint_id": c.get("checkpoint_id")}
        return cls(name, digest(definition), c["channel"], seal["kind"], seal["dims"],
                   canonical(definition).decode()).validate()


def validate_bank(instruments):
    require(len(instruments) >= MIN_INSTRUMENTS, "MNLM_MODEL_MIN_INSTRUMENTS",
            "register at least five encoders/embedders; duplicates do not count")
    for spec in instruments:
        spec.validate()
    require(len({s.name for s in instruments}) == len(instruments)
            and len({s.identity for s in instruments}) == len(instruments),
            "MNLM_MODEL_DUPLICATE", "duplicate instrument name or version")
    # Relabeling the exact frozen machinery does not create another instrument.
    import json
    machinery = []
    for spec in instruments:
        d = json.loads(spec.definition_json)
        d.pop("name")
        machinery.append(digest(d))
    require(len(set(machinery)) == len(machinery), "MNLM_MODEL_DUPLICATE", "renamed duplicate machinery")


@dataclass(frozen=True)
class ModelConfig:
    width: int = 32
    heads: int = 4
    layers: int = 1
    vocabulary: int = 258
    max_prefix: int = 32
    max_native_tokens: int = 8192
    max_native_elements: int = 16_000_000
    max_attention_elements: int = 16_000_000
    max_parameters: int = 8_000_000
    checkpoint_bytes: int = 64 << 20

    def validate(self):
        require(all(type(v) is int and v > 0 for v in asdict(self).values()),
                "MNLM_MODEL_CONFIG", "positive integer configuration required")
        require(self.width % self.heads == 0 and self.width % 2 == 0 and self.vocabulary >= 2,
                "MNLM_MODEL_CONFIG", "even width divisible by attention heads required")


def parameter_count(config, specs):
    """Exact count for network.py; checked before allocation, including sparse tables."""
    h, v = config.width, config.vocabulary
    base = v * h + h * v + v  # token embedding and output head
    layer = 12 * h * h + 13 * h  # self attention, FFN, two layer norms
    branch = 4 * h * h + 6 * h + 1  # cross attention, norm, gain
    projectors = sum(s.dims * h if s.layout == "sparse" else s.dims * h + h for s in specs)
    return base + config.layers * (layer + len(specs) * branch) + projectors
