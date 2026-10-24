"""S9 learned judge: an instrument trained on repo-disjoint data, then frozen (#12 senses; operator thesis:
data trains the instruments, anchors evaluate them).

train  Pool: every non-empty attempt in the pinned trajectory shards whose repository appears in no
       cohort (the v3 census rows, which include v4, plus every v2 attempt task list). Inputs per attempt, all
       pre-evaluation: the issue (from the first user turn), the submitted diff, the last two environment
       observations. Frozen Qwen3-Embedding-0.6B (pinned snapshot, GPU) embeds the three; the head is
       an L2 logistic over [issue, diff, obs, issue*diff, diff*obs], C chosen by repo-grouped CV on the
       pool. Target: the provider's `target` of POOL attempts only. No observation, label or test of
       any cohort task is read. The head is saved and hash-pinned.
apply  The frozen head reads an observation store's issue/patch/trajectory and writes judge.npy:
       [logit, issue.diff cosine, diff.obs cosine] per observation.
No CPU fallback.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lens.lenscommon import run_cli, take_lease
from senses.trajectory_population import publish

VERSION = "swe-judge-v2"
C_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0)
VARIANTS = ("full", "compact")  # chosen by repo-grouped CV on the pool only
SHARDS = "raw/swe-agent-trajectories/data/*.parquet"
repo_of = lambda iid: iid.rsplit("-", 1)[0]


def issue_text(turns):
    first = next((t["text"] or "" for t in turns if t["role"] == "user"), "")
    m = re.search(r"ISSUE:\n(.*?)\n\s*INSTRUCTIONS:", first, re.S)
    return (m.group(1) if m else first)[:8000] or "(no issue)"


def last_obs(turns):
    env = [t["text"] or "" for t in turns if t["role"] == "user"][1:]
    return "\n".join(env[-2:])[-6000:] or "(no observation)"


def texts(turns, patch):
    turns = [t for t in turns if t["role"] != "system"]
    return issue_text(turns), patch or "(empty)", last_obs(turns)


def embed(issues, diffs, obs):
    import torch
    from senses.swe_embed import QWEN, load, qwen_embed
    if not torch.cuda.is_available():
        raise SystemExit("MNLM_JUDGE_NO_CUDA: a CPU run would be a different instrument")
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    tok, model = load(*QWEN, padding_side="left")
    E = [qwen_embed(tok, model, x) for x in (issues, diffs, obs)]
    return E, torch.cuda.get_device_name(0)


def features(E, variant, diffs):
    i, d, o = E
    if variant == "full":
        return np.concatenate([i, d, o, i * d, d * o], 1).astype(np.float64)
    from lens.vinfo import block
    size = np.array([[np.log1p(len(x)), np.log1p(x.count("\n+")), np.log1p(x.count("\n-")), np.log1p(x.count("diff --git"))] for x in diffs])
    cos = np.stack([(i * d).sum(1), (d * o).sum(1), (i * o).sum(1)], 1)
    return np.concatenate([block(i), block(d), block(o), cos, size], 1).astype(np.float64)


def excluded_repos(root):
    ev = root / "evidence/fsv/12"
    ex = {repo_of(r["task"]) for r in json.loads((ev / "cohort-v3-20261017/census.json").read_bytes())["rows"]}
    for f in sorted((ev / "cohort-v2-20261017").glob("attempts-tasks*.json")):
        ex |= {repo_of(t["task"]) for t in json.loads(f.read_bytes())}
    return ex


def train(a):
    import pyarrow.parquet as pq
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    take_lease("12", "swe-judge-train")
    started = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    ex = excluded_repos(a.root)
    pool, shards = [], {}
    for f in sorted(glob.glob(str(a.root / SHARDS))):
        shards[Path(f).name] = hashlib.sha256(Path(f).read_bytes()).hexdigest()
        t = pq.read_table(f, columns=["instance_id", "target", "generated_patch", "trajectory"])
        for j, r in enumerate(t.to_pylist()):
            if repo_of(r["instance_id"]) in ex or not (r["generated_patch"] or "").strip():
                continue
            pool.append((Path(f).name, j, r["instance_id"], int(bool(r["target"])), *texts(r["trajectory"], r["generated_patch"])))
    y = np.array([p[3] for p in pool]); groups = np.unique([repo_of(p[2]) for p in pool], return_inverse=True)[1]
    rows_sha = hashlib.sha256("\n".join(f"{p[0]}:{p[1]}" for p in pool).encode()).hexdigest()
    if a.cache.exists():
        z = np.load(a.cache)
        if str(z["rows_sha256"]) != rows_sha:
            raise SystemExit("MNLM_JUDGE_CACHE_ROWS: cached pool embeddings are for different rows")
        E, device = [z["issue"], z["diff"], z["obs"]], str(z["device"])
    else:
        E, device = embed([p[4] for p in pool], [p[5] for p in pool], [p[6] for p in pool])
        np.savez(a.cache, issue=E[0], diff=E[1], obs=E[2], rows_sha256=rows_sha, device=device)
    diffs = [p[5] for p in pool]
    grid = {}
    for variant in VARIANTS:
        X = features(E, variant, diffs)
        for C in C_GRID:
            ce = 0.0
            for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X[tr]), y[tr])
                pr = clf.predict_proba(sc.transform(X[te]))[np.arange(len(te)), y[te]]
                ce += -np.log2(np.clip(pr, 1e-6, 1)).sum()
            grid[f"{variant}|{C}"] = ce / len(y)
    best = min(grid, key=grid.get)
    variant, C = best.split("|")[0], float(best.split("|")[1])
    scores = grid
    X = features(E, variant, diffs)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X), y)
    np.savez(a.output / "head.npz", mean=sc.mean_, scale=sc.scale_, coef=clf.coef_[0], intercept=clf.intercept_)
    prior = -(y.mean() * np.log2(y.mean()) + (1 - y.mean()) * np.log2(1 - y.mean()))
    publish(a.output / "manifest.json", {"schema": "swe-judge-head-v1", "version": VERSION, "C": C, "variant": variant,
        "pool_better_than_prior": bool(grid[best] < prior),
        "pool_cv_bits_per_attempt": scores, "pool_prior_entropy_bits": float(prior),
        "pool": {"attempts": len(pool), "tasks": len({p[2] for p in pool}), "repos": int(groups.max() + 1), "positives": int(y.sum()),
                 "rows_sha256": rows_sha},
        "excluded_repositories": len(ex), "shards_sha256": shards, "device": device,
        "head_sha256": hashlib.sha256((a.output / "head.npz").read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "wall_seconds": time.perf_counter() - started})
    print(json.dumps({"pool": len(pool), "variant": variant, "C": C, "cv_bits": scores, "prior_bits": float(prior)}))


def apply(a):
    take_lease("12", "swe-judge-apply")
    m = json.loads((a.head / "manifest.json").read_bytes())
    if hashlib.sha256((a.head / "head.npz").read_bytes()).hexdigest() != m["head_sha256"]:
        raise SystemExit("MNLM_JUDGE_HEAD_HASH: the frozen head differs from its manifest")
    h = np.load(a.head / "head.npz")
    a.output.mkdir(parents=True, exist_ok=False)
    recs = [json.loads(l) for l in (a.observations / "observations.jsonl").read_text(encoding="utf-8").splitlines()]
    ex = excluded_repos(a.root)
    if any(repo_of(r["task"]) not in ex for r in recs):
        raise SystemExit("MNLM_JUDGE_POOL_OVERLAP: an observation's repository is not excluded from the pool")
    T = [texts(r["trajectory"], r["patch"]) for r in recs]
    E, device = embed(*zip(*T))
    if not m.get("pool_better_than_prior"):
        raise SystemExit("MNLM_JUDGE_NO_POOL_SIGNAL: the head does not beat the prior on held-out pool repositories")
    logit = ((features(E, m["variant"], [t[1] for t in T]) - h["mean"]) / h["scale"]) @ h["coef"] + h["intercept"][0]
    M = np.stack([logit, (E[0] * E[1]).sum(1), (E[1] * E[2]).sum(1)], 1).astype(np.float32)
    if not np.isfinite(M).all():
        raise SystemExit("MNLM_JUDGE_NONFINITE")
    np.save(a.output / "judge.npy", M)
    publish(a.output / "manifest.json", {"schema": "swe-embedded-senses-v1", "version": VERSION,
        "senses": {"judge": {"path": "judge.npy", "shape": list(M.shape), "sha256": hashlib.sha256((a.output / "judge.npy").read_bytes()).hexdigest()}},
        "head_sha256": m["head_sha256"], "device": device,
        "uid_order_sha256": hashlib.sha256("\n".join(r["uid"] for r in recs).encode()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps({"shape": list(M.shape)}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--root", type=Path, required=True); t.add_argument("--output", type=Path, required=True)
    t.add_argument("--cache", type=Path, required=True, help="pool embedding cache (.npz), reused across head variants")
    q = sub.add_parser("apply"); q.add_argument("--root", type=Path, required=True); q.add_argument("--head", type=Path, required=True)
    q.add_argument("--observations", type=Path, required=True); q.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    (train if a.cmd == "train" else apply)(a)


if __name__ == "__main__":
    run_cli(main)
