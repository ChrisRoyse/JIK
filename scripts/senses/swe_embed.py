"""Learned (embedded) SWE senses on the GPU, frozen pretrained models from pinned local snapshots (#12).

  surface   S1  microsoft/unixcoder-base over the submitted diff (code embedder; mean over tokens)
  behaviour S4b Qwen3-Embedding-0.6B over the agent's last two environment observations
  relations S5b Qwen3-Embedding-0.6B cosine(issue, diff) and cosine(issue, last observation)
No CPU fallback: a CPU run is a different instrument, so the job refuses without CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lens.lenscommon import run_cli, take_lease
from senses.trajectory_population import publish

HUB = Path.home() / ".cache/huggingface/hub"
UNIX = ("microsoft/unixcoder-base", "5604afdc964f6c53782a6813140ade5216b99006")
QWEN = ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3")
MAX_TOKENS = {"unix": 512, "qwen": 1024}


def snapshot(repo, rev):
    path = HUB / ("models--" + repo.replace("/", "--")) / "snapshots" / rev
    if not path.is_dir():
        raise SystemExit(f"MNLM_EMBED_SNAPSHOT_MISSING {repo}@{rev}")
    return path


def load(repo, rev, **kw):
    path = snapshot(repo, rev)
    return AutoTokenizer.from_pretrained(path, **kw), AutoModel.from_pretrained(path, torch_dtype=torch.float16).cuda().eval()


@torch.no_grad()
def unix_embed(tok, model, texts, batch=32):
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=MAX_TOKENS["unix"], return_tensors="pt").to("cuda")
        h = model(**enc).last_hidden_state.float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def qwen_embed(tok, model, texts, batch=16):
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=MAX_TOKENS["qwen"], return_tensors="pt").to("cuda")
        h = model(**enc).last_hidden_state.float()
        last = enc["attention_mask"].sum(1) - 1 if tok.padding_side == "right" else torch.full((h.shape[0],), h.shape[1] - 1, device="cuda")
        v = h[torch.arange(h.shape[0], device="cuda"), last]
        out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("MNLM_EMBED_NO_CUDA: a CPU run would be a different instrument")
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    take_lease("12", "swe-embed-senses")
    started = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    recs = [json.loads(l) for l in (a.observations / "observations.jsonl").read_text(encoding="utf-8").splitlines()]
    diffs = [r["patch"] for r in recs]
    issues = [r["issue"] for r in recs]
    lastobs = []
    for r in recs:
        env = [t["text"] for t in r["trajectory"] if t["role"] == "user"][1:]
        lastobs.append("\n".join(env[-2:])[-6000:] or "(no observation)")
    tok, model = load(*UNIX)
    S1 = unix_embed(tok, model, diffs)
    del model; torch.cuda.empty_cache()
    tok, model = load(*QWEN, padding_side="left")
    E_issue = qwen_embed(tok, model, issues)
    E_diff = qwen_embed(tok, model, diffs)
    E_obs = qwen_embed(tok, model, lastobs)
    S4b = E_obs
    S5b = np.stack([(E_issue * E_diff).sum(1), (E_issue * E_obs).sum(1), (E_diff * E_obs).sum(1)], 1)
    files = {}
    for name, M in (("surface", S1), ("behaviour_embed", S4b), ("relations_embed", S5b)):
        M = M.astype(np.float32)
        if not np.isfinite(M).all():
            raise SystemExit(f"MNLM_EMBED_NONFINITE {name}")
        np.save(a.output / f"{name}.npy", M)
        files[name] = {"path": f"{name}.npy", "shape": list(M.shape), "sha256": hashlib.sha256((a.output / f"{name}.npy").read_bytes()).hexdigest()}
    order = [r["uid"] for r in recs]
    publish(a.output / "manifest.json", {"schema": "swe-embedded-senses-v1", "senses": files,
        "models": {"surface": UNIX, "behaviour_embed": QWEN, "relations_embed": QWEN}, "max_tokens": MAX_TOKENS,
        "dtype": "float16", "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "uid_order_sha256": hashlib.sha256("\n".join(order).encode()).hexdigest(),
        "observations_sha256": json.loads((a.observations / "manifest.json").read_bytes())["observations"]["sha256"],
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "wall_seconds": time.perf_counter() - started})
    print(json.dumps({k: v["shape"] for k, v in files.items()}))


if __name__ == "__main__":
    run_cli(main)
