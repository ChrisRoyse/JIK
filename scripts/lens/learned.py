"""Learned lenses of the Stage 3 catalogue: frozen Hugging Face checkpoints read from the local cache with
`trust_remote_code=False`, run in float32 on the GPU with deterministic algorithms, identified by the SHA-256 of their
weight files and the repository commit. Kinds: dense (pooled, unit-normalised), sparse (vocabulary weights,
SPLADE form or the bge-m3 sparse head), multi (unit token vectors through the ColBERT projection), rater (classifier
logits), nll (mean next-token loss in bits), static (mean of a static token table). A text longer than the window is
read in up to MAX_WINDOWS windows and pooled (mean for dense and rater, max for sparse, concatenation for multi, token-
weighted for nll); overflow refuses rather than truncates. The window count is recorded per atom. No CPU fallback: a missing GPU
refuses MNLM_LENS_NO_GPU."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from lenscommon import LensError
import text_windows

MAX_WINDOWS = 8
TOKEN_BUDGET = 16384  # per-atom window tokens per forward; never depends on batch neighbours
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")


def snapshot_dir(repo: str) -> Path:
    """The cached snapshot of the repository's `main` ref, read from the cache layout directly (the hub client refuses a
    snapshot whose README or ONNX files were deliberately not fetched). Nothing is downloaded here."""
    import os

    cache = Path(os.environ.get("HF_HUB_CACHE") or (Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"))
    d = cache / f"models--{repo.replace('/', '--')}"
    ref = d / "refs" / "main"
    if not ref.exists():
        raise LensError("MNLM_LENS_NOT_DOWNLOADED", f"{repo}: no cached snapshot under {d}")
    snap = d / "snapshots" / ref.read_text(encoding="utf-8").strip()
    if not snap.is_dir():
        raise LensError("MNLM_LENS_NOT_DOWNLOADED", f"{repo}: snapshot {snap.name} missing")
    return snap


def weights_sha256(snap: Path) -> tuple[str, list[str]]:
    files = sorted(str(p.relative_to(snap)) for p in snap.rglob("*") if p.is_file() and p.suffix in WEIGHT_SUFFIXES and "onnx" not in p.parts)
    h = hashlib.sha256()
    if not files:
        raise LensError("MNLM_LENS_WEIGHTS_MISSING", str(snap))
    for f in files:
        h.update(f.encode("utf-8") + b"\x00")
        with (snap / f).open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
    return h.hexdigest(), files


def preprocessing_sha256(snap: Path):
    """Once per loaded instrument, bind tokenizer/config assets as well as weights."""
    files = sorted(p for p in snap.rglob("*") if p.is_file() and
                   p.suffix in (".json", ".txt", ".model", ".vocab", ".merges") and
                   not {"onnx", "openvino"}.intersection(p.parts))
    hashes = {}
    for path in files:
        with path.open("rb") as f:
            hashes[str(path.relative_to(snap))] = hashlib.file_digest(f, "sha256").hexdigest()
    if not hashes:
        raise LensError("MNLM_LENS_TOKENIZER_MISSING", str(snap))
    return hashes


def _load_state(snap: Path, name: str) -> dict:
    p = snap / name
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(p))
    import torch

    return torch.load(str(p), map_location="cpu", weights_only=True)


class LearnedLens:
    def __init__(self, spec: dict, device: str):
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer

        self.spec, self.device, self.kind = spec, device, spec["kind"]
        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise LensError("MNLM_LENS_NO_GPU", "learned instruments require their declared CUDA runtime")
        self.snap = snapshot_dir(spec["repo"])
        self.revision = self.snap.name
        self.sha256, self.weight_files = weights_sha256(self.snap)
        self.preprocessing = preprocessing_sha256(self.snap)
        self.windows: list[int] = []
        self.segments: list[dict] = []
        kw = {"trust_remote_code": False}
        if self.kind == "static":
            from tokenizers import Tokenizer

            self.tok = Tokenizer.from_file(str(self.snap / "tokenizer.json"))
            st = _load_state(self.snap, "model.safetensors")
            key = "embeddings" if "embeddings" in st else next(iter(st))
            self.table = st[key].detach().float().to(device)
            self.dim = int(self.table.shape[1])
            return
        self.tok = AutoTokenizer.from_pretrained(str(self.snap), **kw)
        if self.tok.pad_token is None:  # GPT-2 has no pad token; padded positions are masked out of every pooling path
            self.tok.pad_token = self.tok.eos_token
            self.tok.padding_side = "right"
        if self.kind == "sparse" and not spec.get("head"):
            self.model = AutoModelForMaskedLM.from_pretrained(str(self.snap), dtype=torch.float32, **kw)
        elif self.kind == "rater":
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.snap), dtype=torch.float32, **kw)
        elif self.kind == "nll":
            self.model = AutoModelForCausalLM.from_pretrained(str(self.snap), dtype=torch.float32, **kw)
        else:
            self.model = AutoModel.from_pretrained(str(self.snap), dtype=torch.float32, **kw)
        self.model.eval().requires_grad_(False).to(device)
        self.head = None
        if spec.get("head"):
            hp = spec["head"]
            st = _load_state(self.snap, hp) if hp.endswith((".pt", ".safetensors")) else _load_state(self.snap, self._main_weights())
            w = st["weight"] if "weight" in st else st["linear.weight"]
            b = st.get("bias", st.get("linear.bias"))
            self.head = (w.float().to(device), b.float().to(device) if b is not None else None)
        if self.kind == "dense":
            self.dim = int(self.model.config.hidden_size)
        elif self.kind == "multi":
            self.dim = int(self.head[0].shape[0])
        elif self.kind == "rater":
            self.dim = int(self.model.config.num_labels)
        elif self.kind == "nll":
            self.dim = 1
        else:
            self.dim = int(self.tok.vocab_size)

    def _main_weights(self) -> str:
        for f in self.weight_files:
            if f.endswith(("model.safetensors", "pytorch_model.bin")) and "/" not in f:
                return f
        return self.weight_files[0]

    def contract(self) -> dict:
        shape = {"dense": f"Dense({self.dim})", "static": f"Dense({self.dim})", "sparse": f"Sparse({self.dim})", "multi": f"Multi({self.dim})", "rater": f"Dense({self.dim})", "nll": "Dense(1)"}[self.kind]
        return {"name": self.spec["name"], "repo": self.spec["repo"], "revision": self.revision, "weights_sha256": self.sha256, "weight_files": self.weight_files,
                "preprocessing_sha256": self.preprocessing, "device": str(self.device),
                "segmentation": text_windows.SCHEMA,
                "window_reduction": {"dense": "mean", "rater": "mean", "sparse": "max", "multi": "append-native-token-rows",
                                     "nll": "token-weighted-mean", "static": "whole-input-table-mean"}[self.kind],
                "segmentation_code_sha256": hashlib.sha256(Path(text_windows.__file__).read_bytes()).hexdigest(),
                "execution_unit": "atom-local-v1", "forward_token_budget": TOKEN_BUDGET,
                "corpus_hash": self.revision, "shape": shape, "modality": "text", "dtype": "f32", "norm": "unit" if self.kind in ("dense", "static", "multi") else "none",
                "pool": self.spec.get("pool"), "prefix": self.spec.get("prefix", ""), "max_len": self.spec["max_len"], "max_windows": MAX_WINDOWS, "kind": self.kind}

    # ---------------------------------------------------------------- forward
    def _tokenize(self, texts: list[str]):
        if not self.tok.is_fast:
            raise LensError("MNLM_LENS_OFFSETS_UNAVAILABLE", "tokenizer must expose exact character offsets; no count-only fallback")
        prefix = self.spec.get("prefix", "")
        enc = self.tok([prefix + t for t in texts], truncation=True, max_length=self.spec["max_len"],
                       return_overflowing_tokens=True, padding=True, return_tensors="pt",
                       return_offsets_mapping=True, return_special_tokens_mask=True)
        mapping = enc.pop("overflow_to_sample_mapping").tolist()
        from collections import Counter
        counts = Counter(mapping)
        overflow = {i: n for i, n in counts.items() if n > MAX_WINDOWS}
        if overflow:
            raise LensError("MNLM_LENS_WINDOW_BUDGET", f"{self.spec['name']}: full atoms require {overflow} windows; budget={MAX_WINDOWS}; choose an attested longer-context instrument or atom scope; no bytes were silently discarded")
        self.segments = text_windows.build(texts, prefix, enc, mapping, self.kind)
        enc = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        return enc, mapping

    def embed(self, texts: list[str]):
        """Keep each atom's exact forward shape independent of batch neighbours.

        The model stays resident. Native outputs are stacked, never pooled across
        records. Even deterministic CUDA kernels can change rounding when batch
        geometry changes; batching is an I/O boundary, not instrument input.
        """
        if len(texts) <= 1 or self.kind == "static":
            return self._embed(texts)
        parts, segments = [], []
        for text in texts:
            parts.append(self._embed([text]))
            segments.extend(self.segments)
        self.segments = segments
        if self.kind == "multi":
            sizes = [len(p[0]) for p in parts]
            return np.concatenate([p[0] for p in parts]), np.r_[0, np.cumsum(sizes, dtype=np.int64)]
        if self.kind == "sparse":
            from scipy.sparse import vstack
            return vstack(parts).tocsr()
        return np.concatenate(parts)

    def _embed(self, texts: list[str]):
        """Returns per kind: dense/rater/nll/static -> np.float32 [n, d]; sparse -> csr [n, V]; multi -> (tokens [T, d], offsets [n+1]). Records windows.
        Windows are forwarded in chunks of at most TOKEN_BUDGET tokens of one
        atom (static tables already reduce each atom independently)."""
        import torch

        if not texts or (self.kind != "static" and len(texts) != 1):
            raise LensError("MNLM_LENS_ATOM_EXECUTION", "forward requires one observed atom; use embed for I/O batches")
        self.segments = []
        if self.kind == "static":
            if self.tok.truncation is not None or self.tok.padding is not None:
                raise LensError("MNLM_LENS_WINDOW_PROVENANCE", "static table tokenizer must expose the whole unpadded source")
            out = np.zeros((len(texts), self.dim), np.float32)
            for i, t in enumerate(texts):
                prefix = self.spec.get("prefix", "")
                encoded = self.tok.encode(prefix + t, add_special_tokens=False)
                ids = encoded.ids
                self.segments.extend(text_windows.build([t], prefix, {
                    "input_ids": [ids], "attention_mask": [encoded.attention_mask],
                    "offset_mapping": [[list(p) for p in encoded.offsets]],
                    "special_tokens_mask": [encoded.special_tokens_mask]}, [0], self.kind))
                self.windows.append(1)
                if ids:
                    indices = torch.tensor(ids, dtype=torch.long, device=self.device)
                    v = self.table.index_select(0, indices).mean(0)
                    v = torch.nn.functional.normalize(v, dim=0)
                    out[i] = v.detach().cpu().numpy()
            return out
        enc, mapping = self._tokenize(texts)
        W = max(1, TOKEN_BUDGET // int(enc["input_ids"].shape[1]))
        parts = []
        with torch.inference_mode():
            for s in range(0, len(mapping), W):
                e = {k: v[s:s + W].to(self.device) for k, v in enc.items()}
                parts.append(self._windows(e))
        n = len(texts)
        if self.kind == "nll":
            tot = np.concatenate([p[0] for p in parts])
            cnt = np.concatenate([p[1] for p in parts])
            return self._pool_nll(tot, cnt, mapping, n)
        if self.kind == "multi":
            return self._multi([t for p in parts for t in p], mapping, n)
        V = np.concatenate(parts)
        if self.kind == "sparse":
            return self._sparse(V, mapping, n)
        vec = self._pool_windows(V, mapping, n)
        if self.kind == "rater":
            return vec
        nrm = np.linalg.norm(vec, axis=1, keepdims=True)
        return np.where(nrm > 0, vec / np.where(nrm > 0, nrm, 1), 0.0).astype(np.float32)

    def _windows(self, enc):
        """One forward over a chunk of windows; returns per-window outputs on the CPU."""
        import torch

        mask = enc["attention_mask"]
        if self.kind == "nll":
            logits = self.model(**enc).logits.float()
            ids = enc["input_ids"]
            lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            m = mask[:, 1:].float()
            return (-(lp * m).sum(1).cpu().numpy(), m.sum(1).cpu().numpy())
        if self.kind == "rater":
            return self.model(**enc).logits.float().cpu().numpy()
        if self.kind == "sparse" and self.head is None:
            logits = self.model(**enc).logits.float()
            w = torch.log1p(torch.relu(logits)) * mask.unsqueeze(-1)
            return w.max(1).values.cpu().numpy()
        hidden = self.model(**enc).last_hidden_state.float()
        if self.kind == "sparse":
            w, b = self.head
            tw = torch.relu(hidden @ w.T + (b if b is not None else 0)).squeeze(-1) * mask
            tw[:, 0] = 0  # the CLS position carries no term
            V = torch.zeros(hidden.shape[0], self.tok.vocab_size, device=self.device)
            V.scatter_reduce_(1, enc["input_ids"], tw, reduce="amax", include_self=True)
            for sid in (self.tok.cls_token_id, self.tok.eos_token_id, self.tok.pad_token_id, self.tok.unk_token_id):
                if sid is not None:
                    V[:, sid] = 0
            return V.cpu().numpy()
        if self.kind == "multi":
            w, b = self.head
            tv = torch.nn.functional.normalize(hidden @ w.T + (b if b is not None else 0), dim=-1).cpu().numpy()
            mk = mask.cpu().numpy().astype(bool)
            return [row[m] for row, m in zip(tv, mk)]
        pool = self.spec["pool"]
        if pool == "cls":
            vec = hidden[:, 0]
        elif pool == "last":
            if self.tok.padding_side == "left":
                vec = hidden[:, -1]
            else:
                vec = hidden[torch.arange(hidden.shape[0], device=self.device), mask.sum(1) - 1]
        else:
            m = mask.unsqueeze(-1).float()
            vec = (hidden * m).sum(1) / m.sum(1).clamp(min=1)
        return vec.cpu().numpy()

    def _pool_windows(self, vec: np.ndarray, mapping: list[int], n: int) -> np.ndarray:
        out = np.zeros((n, vec.shape[1]), np.float32)
        counts = np.zeros(n, np.int64)
        for row, m in zip(vec, mapping):
            out[m] += row
            counts[m] += 1
        self.windows.extend(counts.tolist())
        return out / np.maximum(counts, 1)[:, None]

    def _sparse(self, V: np.ndarray, mapping: list[int], n: int):
        out = np.zeros((n, V.shape[1]), np.float32)
        counts = np.zeros(n, np.int64)
        for row, m in zip(V, mapping):
            out[m] = np.maximum(out[m], row)
            counts[m] += 1
        self.windows.extend(counts.tolist())
        return sp.csr_matrix(out)

    def _multi(self, per_window: list[np.ndarray], mapping: list[int], n: int):
        per = [[] for _ in range(n)]
        counts = np.zeros(n, np.int64)
        for toks, m in zip(per_window, mapping):
            per[m].append(toks)
            counts[m] += 1
        self.windows.extend(counts.tolist())
        d = per_window[0].shape[-1] if per_window else self.dim
        toks = [np.concatenate(p) if p else np.zeros((0, d), np.float32) for p in per]
        offsets = np.concatenate([[0], np.cumsum([len(t) for t in toks])]).astype(np.int64)
        return np.concatenate(toks).astype(np.float32) if toks else np.zeros((0, d), np.float32), offsets

    def _pool_nll(self, tot: np.ndarray, cnt: np.ndarray, mapping: list[int], n: int) -> np.ndarray:
        out, counts = np.zeros(n), np.zeros(n)
        windows = np.zeros(n, np.int64)
        for t, c, mp in zip(tot, cnt, mapping):
            out[mp] += t
            counts[mp] += c
            windows[mp] += 1
        self.windows.extend(windows.tolist())
        return (out / np.maximum(counts, 1) / np.log(2)).astype(np.float32)[:, None]


def check_gpu() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise LensError("MNLM_LENS_NO_GPU", "learned lenses require CUDA; no CPU fallback (a different runtime is a different lens)")
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0), "deterministic": torch.are_deterministic_algorithms_enabled()}


def dump_json(obj) -> str:
    return json.dumps(obj, sort_keys=True)
