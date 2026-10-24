# Per-tick loop prompt (MNLM)

Continue one MNLM build increment. The binding rules are `AGENTS.md` and GitHub issues #1 (doctrine) and #5 (pipeline cost) in `ChrisRoyse/MNLM`.

Each tick:

1. Read `AGENTS.md`, issue #2 (context) and the open `scope:direction` issues for current critical-path work; read issue #6 (roadmap) for the stage order.
2. Pick the highest-priority open `stage:*` issue with no assignee, or follow operator direction. Nothing in stages 1–12 starts until the dataset decision (#3) is adjudicated by the operator.
3. Claim the issue (atomic `gh issue edit --add-assignee` plus a CLAIM comment with the cost preflight: real N, per-what, invariants, PC-nn classes).
4. Audit the task against the live tree and the live hosts first; assume the issue description is stale; read the persisted bytes before writing code.
5. Run MNLM compute only on aiwonder; keep hash-addressed FSV evidence under `/zfs/archive/mnlm/evidence/fsv/<issue>/<run>/`, with ≥ 3 boundaries and negative controls. The workstation is reserved for other systems.
6. Every number in the comment is tagged `measured` with an evidence pointer or `projected`.
7. Close out with RESOLVED / PAUSE / BLOCKED comment containing the evidence table. Say "author-checked", never "verified", for your own readback.

Memory = GitHub issues. Agent cabinet memories under `.claude/agent-memory/` hold only MNLM facts.
