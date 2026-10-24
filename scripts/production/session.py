"""One frozen instrument resident across bounded shards; source adapters live in UCR."""
import hashlib
import json

import blake3
import numpy as np

from senses.contract import canonical, digest, require
from production.input_plan import environment
from learned_ucr import embed_batch, visible_batch


def prepare(item, reader, rows, policy):
    """One bound view read per row, shared by cache identity and actual forward."""
    module = item["spec"]["module"]
    if module == "learned_ucr":
        texts, bindings = visible_batch(reader, policy, rows)
        return [{"input": t, "context": None, "binding": b} for t, b in zip(texts, bindings)]
    out = []
    for row in rows:
        if module == "universal":
            view, contexts, binding = reader.read(row, policy)
            value, context = view, contexts.get(item["spec"]["channel"])
        elif module in ("image_ucr", "vision_ucr"):
            from image_ucr import visible_image
            value, binding = visible_image(reader, policy, row)
            context = None
        elif module == "audio_ucr":
            from audio_ucr import visible_audio
            value, binding = visible_audio(reader, policy, row)
            context = None
        else:
            require(False, "MNLM_STREAM_INSTRUMENT", "unknown input module")
        require(binding["atom"] == row["atom"], "MNLM_STREAM_INPUT", "input atom drift")
        out.append({"input": value, "context": context, "binding": binding})
    return out


def multi(parts, windows):
    sizes = [len(p) for p in parts]
    return {"tokens": np.concatenate(parts), "offsets": np.r_[0, np.cumsum(sizes, dtype=np.int64)],
            "absent": np.asarray(sizes) == 0, "windows": np.asarray(windows, np.int64)}


class Session:
    def __init__(self, item, reader, code, blobs=None):
        self.item, self.reader, self.blobs, self.lens = item, reader, blobs, None
        self.spec = item["spec"]
        self.module = self.spec["module"]
        if self.module == "universal":
            from universal import specs
            from senses.instruments import DIM
            current = next((s for s in specs() if s["name"] == self.spec["name"]), None)
            require(current == self.spec, "MNLM_STREAM_INSTRUMENT", "algorithmic instrument contract drift")
            identity, self.dim, self.kind = dict(self.spec), DIM, "multi"
        elif self.module == "learned_ucr":
            from learned import LearnedLens, check_gpu
            from catalogue import by_name
            require(by_name(self.spec["name"]) == self.spec, "MNLM_STREAM_INSTRUMENT", "catalogue spec drift")
            runtime = check_gpu()
            self.lens = LearnedLens(self.spec, "cuda")
            identity = {**self.lens.contract(), "inputs": "ucr", "channel": "surface", "runtime": runtime}
            self.dim, self.kind = self.lens.dim, self.lens.kind
        elif self.module == "image_ucr":
            from image_ucr import SPEC
            from senses import pixels
            require(self.spec == SPEC and blobs is not None, "MNLM_STREAM_INSTRUMENT", "image spec/blob manifest required")
            identity, self.dim, self.kind = {**pixels.contract(), **SPEC}, pixels.DIM, "multi"
        elif self.module == "vision_ucr":
            from vision_encoder import VisionEncoder, SPEC
            require(self.spec == SPEC and blobs is not None, "MNLM_STREAM_INSTRUMENT", "vision spec/blob manifest required")
            self.lens = VisionEncoder(self.spec)
            identity, self.dim, self.kind = self.lens.identity, self.lens.dim, "multi"
        elif self.module == "audio_ucr":
            from audio_encoder import AudioEncoder, SPEC
            require(self.spec == SPEC and blobs is not None, "MNLM_STREAM_INSTRUMENT", "audio spec/blob manifest required")
            self.lens = AudioEncoder(self.spec)
            identity, self.dim, self.kind = self.lens.identity, self.lens.dim, "multi"
        else:
            require(False, "MNLM_STREAM_INSTRUMENT", "unknown instrument module")
        require(identity["weights_sha256"] == item["weights_sha256"], "MNLM_LENS_FROZEN_VIOLATION", self.spec["name"])
        if self.module in ("learned_ucr", "audio_ucr", "vision_ucr"):
            require(identity.get("revision") == item.get("revision"), "MNLM_LENS_FROZEN_VIOLATION", "snapshot revision drift")
        identity = {**identity, "environment": environment(), "execution_code_sha256": code}
        self.lens_id = blake3.blake3(canonical(identity)).hexdigest()[:32]
        self.contract = {**identity, "ucr_generation": reader.generation, "policies_sha256": reader.policy_hash}
        if self.module in ("audio_ucr", "image_ucr", "vision_ucr"):
            self.contract["blobs_manifest_sha256"] = blobs.sha256
        self.initial_state = self.state_digest()

    def state_digest(self):
        if self.lens is None:
            return self.item["weights_sha256"]
        tensors = {}
        if hasattr(self.lens, "model"):
            require(not any(p.requires_grad for p in self.lens.model.parameters()),
                    "MNLM_LENS_FROZEN_VIOLATION", "trainable parameter in inference instrument")
            tensors.update(self.lens.model.state_dict())
        if hasattr(self.lens, "table"):
            tensors["table"] = self.lens.table
        if getattr(self.lens, "head", None):
            tensors.update({f"head.{i}": x for i, x in enumerate(self.lens.head) if x is not None})
        h = hashlib.sha256()
        for name, tensor in sorted(tensors.items()):
            h.update(canonical([name, str(tensor.dtype), list(tensor.shape)]))
            array = tensor.detach().contiguous().cpu().numpy()
            h.update(memoryview(array).cast("B"))
        return h.hexdigest()

    def batch(self, rows, policy):
        return self.batch_prepared(prepare(self.item, self.reader, rows, policy))

    def batch_prepared(self, prepared):
        bindings = [p["binding"] for p in prepared]
        if self.module == "learned_ucr":
            texts = [p["input"] for p in prepared]
            arrays = embed_batch(self.lens, texts)
            extra = [{"absence_reason": ("encoded_zero" if self.lens.batch_segments[i]["windows"] else
                                         "surface_unobserved_or_empty") if arrays["absent"][i] else None,
                      "text_windows": self.lens.batch_segments[i]}
                     for i in range(len(prepared))]
        else:
            parts, extra, windows = [], [], []
            for p in prepared:
                if self.module == "universal":
                    from senses.instruments import measure
                    result = measure(p["input"], self.spec["channel"], p["context"])
                    tokens = result["tokens"]
                    detail = {k: v for k, v in result.items() if k != "tokens"}
                    count = int(bool(len(tokens)))
                elif self.module in ("image_ucr", "vision_ucr"):
                    from senses.pixels import measure
                    result = (self.lens.measure if self.module == "vision_ucr" else measure)(p["input"], self.blobs)
                    tokens = result["tokens"]
                    detail = {k: v for k, v in result.items() if k != "tokens"}
                    count = len(result["regions"])
                else:
                    regions = p["input"]
                    tokens, spans = self.lens.embed(regions, self.blobs)
                    detail, count = {"windows": spans, "absence_reason": None if regions else "audio_unobserved"}, len(spans)
                parts.append(tokens)
                extra.append(detail)
                windows.append(count)
            arrays = multi(parts, windows)
        return arrays, bindings, extra

    def close(self):
        try:
            require(self.state_digest() == self.initial_state, "MNLM_LENS_FROZEN_VIOLATION", "resident weights changed")
        finally:
            self.lens = None
            if self.module in ("learned_ucr", "audio_ucr", "vision_ucr"):
                import torch
                torch.cuda.empty_cache()


def batch_digest(result):
    """Layout and exact values, including windows and provenance; no pooled score."""
    from slotio import arrays_digest
    arrays, bindings, extra = result
    layout = {k: [str(a.dtype), list(a.shape)] for k, a in arrays.items()}
    return digest([layout, arrays_digest(arrays), bindings, extra])
