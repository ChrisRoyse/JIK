"""Fetches the learned candidates of the catalogue into the Hugging Face cache (weights, tokenizer, config, heads) and
records for each the repository commit and the files taken, so registration (measure.py) can pin them. Downloads are
untrusted bytes: they stay in the cache directory and are only ever loaded by the lens runtime. No token is needed (all
public); a gated or missing repository is recorded as MNLM_LENS_DOWNLOAD_FAILED, never skipped silently.
    python -I scripts/lens/download.py --out /zfs/hot/mnlm --version p0-v1 [--names a,b]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalogue import LEARNED  # noqa: E402
from lenscommon import HOST, now_iso, write_json  # noqa: E402

PATTERNS = ["*.json", "*.txt", "*.model", "*.safetensors", "*.bin", "*.pt", "tokenizer*", "vocab*", "merges*", "1_Dense/*", "sentencepiece*", "spm*"]
SKIP = ["*onnx*", "*openvino*", "*.msgpack", "*.h5", "*tf_model*", "*flax*", "*.ckpt", "*rust_model*", "*.gguf", "*imgs*", "*images*", "*.png"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="p0-v1")
    ap.add_argument("--names", default="")
    args = ap.parse_args()
    from huggingface_hub import snapshot_download

    want = set(args.names.split(",")) if args.names else None
    repos = sorted({c["repo"] for c in LEARNED if not want or c["name"] in want})
    out = Path(args.out) / "lenses" / args.version / "downloads.json"
    results = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    for repo in repos:
        t0 = time.time()
        try:
            path = snapshot_download(repo, allow_patterns=PATTERNS, ignore_patterns=SKIP)
            files = sorted(str(p.relative_to(path)) for p in Path(path).rglob("*") if p.is_file())
            results[repo] = {"ok": True, "snapshot": path, "revision": Path(path).name, "files": files, "bytes": sum((Path(path) / f).stat().st_size for f in files),
                             "secs": round(time.time() - t0, 1), "ts": now_iso()}
        except Exception as exc:  # noqa: BLE001 - recorded as a typed failure of this repo, the loop continues
            results[repo] = {"ok": False, "code": "MNLM_LENS_DOWNLOAD_FAILED", "error": f"{type(exc).__name__}: {exc}"[:300], "secs": round(time.time() - t0, 1), "ts": now_iso()}
        write_json(out, results)
        print(f"{repo}: {'ok' if results[repo]['ok'] else results[repo]['error']} {results[repo]['secs']} s", flush=True)
    print(json.dumps({"repos": len(repos), "ok": sum(r["ok"] for r in results.values()), "host": HOST}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
