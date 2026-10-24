"""One-shot: adapt the Poly agent cabinet copied into .claude/agents/ for MNLM.

Inserts an MNLM standing-orders block after each agent's title line (it supersedes the Poly-specific
standing orders further down, which are kept as institutional history) and rewrites the project
names, repo and paths that would otherwise send an agent to the wrong repository or host.
Idempotent: a file that already carries the MNLM block is left alone. Run from the repo root:

    python -I scripts/adapt_agents.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / '.claude' / 'agents'
MARK = '## 0. MNLM STANDING ORDERS'

BLOCK = f"""{MARK} (binding; supersede any Poly-specific orders below)

You are working on **MNLM** (`ChrisRoyse/MNLM`, `C:\\code\\MNLM`), the Meaning Base and the
meaning-native language model specified in `docs/MEANING_BASE_SPEC_v1.0.md`. You inherit no project
memory beyond this file and your own `.claude/agent-memory/` notes. Read `AGENTS.md`; when the work
touches doctrine read issues #1 (doctrine), #2 (context), #3 (dataset decision), #5 (pipeline cost)
and #6 (roadmap) in `ChrisRoyse/MNLM`. The Poly-specific catalogues further down in this file are
institutional history from the sibling project: keep their *reasoning*, replace their *paths*.

- **Two hosts, one RTX 5090 each.** Workstation: code `C:\\code\\MNLM`, data `D:\\MNLMData`.
  `aiwonder` (Linux, shared, SSH host, user and key from `AIWONDER_HOST`/`AIWONDER_USER`/`AIWONDER_KEY`): code
  `/home/croyse/mnlm`, data `/zfs/hot/mnlm` and `/zfs/archive/mnlm`. Runbook `docs/AIWONDER.md`.
  Never write into Poly's, Calyx's, kernelweights' or another user's trees; never kill a GPU
  process you did not launch; check `nvidia-smi` before any GPU job and record it in the issue.
- **FSV or it did not happen.** Name the persisted source of truth, run the real path, read the
  state back through a separate path, audit ≥ 3 boundaries with before/action/after evidence and
  negative controls, minimum sufficient proof corpus first. Your own readback is "author-checked";
  "verified" needs an independent session and operator adjudication.
- **Measured or projected.** Every number carries one tag; a measured number carries an evidence
  pointer to persisted bytes. Every claim maps to thesis T1–T5 and to an evaluation-substrate metric.
- **The differentiation contract.** ≥ 0.05 bits conditional on the panel, ≤ 0.6 pairwise
  correlation, ≥ 50 samples, five complementary lenses before a meaning, frozen hash-pinned lenses,
  no-flatten typed slots, embed what is latent / encode what is explicit, sufficiency
  `I(panel;Y) ≥ H(Y)` with the deficit named per slot, kernel gated on measured held-out recall ≥ 0.95.
- **Pipeline cost is correctness (issue #5, PC-nn).** State the cost function against real N and the
  loop invariants before writing a stage; never raise a budget or timeout to fix a cost; never
  delete a fail-closed check to go faster; cite the PC class in the issue and the commit.
- **Fail closed**: typed `{{code, message, remediation}}` refusals; no mocks, no fallbacks, no
  hosted CI, no silent CPU fallback of a GPU lens, no second state store beside the ledger.
- **Secrets never in repo, issue, log or report** (SSH/VPN passwords, HF tokens). Names only.
- **Files < 500 lines**; replace tracked markdown, never grow it; status goes in the owning issue.
"""

REWRITES = [
    (re.compile(r'ChrisRoyse/Poly\b'), 'ChrisRoyse/MNLM'),
    (re.compile(r'C:\\\\code\\\\poly'), r'C:\\code\\MNLM'),
    (re.compile(r'C:\\code\\poly'), r'C:\\code\\MNLM'),
    (re.compile(r'\bPoly/Calyx\b'), 'MNLM/Calyx'),
    (re.compile(r'STANDING ORDERS \(Poly, non-negotiable\)'), 'POLY-ERA STANDING ORDERS (history; superseded by §0 above)'),
]


def adapt(path: Path) -> str:
    text = path.read_text(encoding='utf-8')
    if MARK in text:
        return 'skip'
    for rx, rep in REWRITES:
        text = rx.sub(rep, text)
    m = re.search(r'^# .*$', text, re.M)
    if not m:
        return 'no-title'
    insert_at = m.end()
    text = text[:insert_at] + '\n\n' + BLOCK + text[insert_at:]
    path.write_text(text, encoding='utf-8', newline='\n')
    return 'adapted'


def main() -> int:
    if not AGENTS.is_dir():
        print(f'MNLM_AGENTS_DIR_MISSING: {AGENTS}', file=sys.stderr)
        return 1
    counts = {}
    for f in sorted(AGENTS.glob('*.md')):
        if f.name == 'README.md':
            continue
        r = adapt(f)
        counts[r] = counts.get(r, 0) + 1
        print(f'{r:8s} {f.name}')
    print(counts)
    return 0


if __name__ == '__main__':
    sys.exit(main())
