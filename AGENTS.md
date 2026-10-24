# MNLM repository operating boundary

MNLM builds the Meaning Base and the meaning-native language model specified in
`docs/MEANING_BASE_SPEC_v1.0.md`; its source paper is `docs/mnlmpaper.txt`
(Royse 2026, 14 October 2026), and why senses work is `docs/senses_of_a_machine.txt`
(Royse, 18 October 2026: P1–P5).
Binding doctrine is GitHub issue #1; onboarding is issue #2.

- The only source repository and GitHub issue tracker for this project is
  `ChrisRoyse/MNLM`. Work from `C:\code\MNLM` on `main`. State is managed in
  MNLM GitHub issues: read #1, #2 and the open `scope:direction` issues before
  work and update the owning task issue with physical FSV evidence before
  closing it. Tracked markdown does not grow: replace text, and put status in
  the owning issue, not in `docs/`.
- Objective (2026-10-19): the minimum connected continuous-learning foundation
  in `docs/END_GOAL.md`. Parallel identity loops are intrinsic model state,
  sharing the fixed Jesus standard (#10); world beliefs remain correctable.
- MNLM compute runs only on `aiwonder` (operator, 2026-10-20); the workstation
  is for other systems. Keep code/SSH on `C:\code\MNLM`, local archives on
  `D:\MNLMData`; never launch local MNLM training, inference or batch jobs.
  aiwonder: Ubuntu 26.04, RTX 5090 32 GB, Ryzen 9 9950X, 121 GB RAM; SSH host, user and key from `AIWONDER_HOST`/`AIWONDER_USER`/`AIWONDER_KEY`. Code: `/home/croyse/mnlm`;
  data: `/zfs/{hot,archive}/mnlm`. Use `scripts/aiwonder.sh`; see `docs/AIWONDER.md`.
- aiwonder is dedicated to MNLM (operator direction 2026-10-15: "fully free up
  all resources on aiwonder and maximally use them all on this project"); the
  kernelweights scheduler was stopped and disabled (#4). Use all 32 threads,
  121 GB and the whole GPU. Before any GPU job still run
  `nvidia-smi` and record what is resident; a reappearing `kw-*` container is
  a relaunch to stop, not a tenant to respect. Never write into `~/calyx`,
  Poly's trees or any ZFS dataset we did not create; Leapable's backup and
  health timers stay untouched unless the operator says otherwise. One MNLM
  GPU job per host at a time; a job holds a lease file, not a hope.
- Every number is tagged `measured` (read back from persisted bytes, with an
  evidence pointer) or `projected`; a projection never becomes a result by
  repetition. Every claim maps to one of T1–T5 and a spec §27 metric.
- Pipeline cost is a correctness property. Before writing, extending or wiring
  any data-processing path, read issue #5 (pipeline cost, the PC-nn catalogue
  inherited from Poly #1148). State the cost function against real N and name
  what is invariant across the loop; `unknown N` is a tracked gap, a guess is
  not. No total-work budgets (operator, 2026-10-22); retain physical-memory
  tiling and leases. Prove the CPU/GPU bottleneck before concurrency, and
  never delete a fail-closed verification to make something fast.
- Full State Verification means: name the persisted source of truth, execute
  the real path, independently read that state back, and audit at least three
  meaningful boundaries with before/action/after evidence and negative
  controls. Tests and return values alone are not acceptance. Choose the
  minimum sufficient proof corpus first (doctrine §3A).
- Say "author-checked" for your own readback; "verified" is reserved for an
  independent session's readback that the operator adjudicated.
- What we build (operator, 2026-10-19, #1/#15; paper §12–15, §25): vision,
  audio and language are three modalities, each read by multiple frozen,
  hash-pinned instruments. A modality is not an admitted lens. Each analyzed
  modality needs at least five admitted native embedders; five is the floor,
  never the ceiling. Captions/transcripts cannot replace native sight/hearing
  or count as independent evidence of their source. Text can report an event
  without establishing that it happened. Hand-built features and per-entity
  lookups are not senses. The MNLM is trained from scratch on constellations.
  Atoms retain attested sentence/pair/paragraph/page containment; media keep
  native clocks, cutoffs and episode lineage. Order never invents synchronization. The
  typed per-instrument readings form the atom's constellation. The model takes
  constellations in through one typed branch per instrument, recognises the
  next atom's region (frozen semantic-ID codebook) before it places the next
  slots inside it, and decodes to surface text through the same slots. Paper
  §8 Table 1 is the design checklist: measure (apprehension), count
  (reproduction), compose (recognition); slots regressed with no region are the
  mean of a multimodal meaning, off the manifold (#25: guard 0 %). The same instruments then re-measure the output and guard it per slot;
  a failed slot is a refusal, not a guess. Tokens exist only inside the surface
  decoder. A pretrained generative LLM is never a sense, a judge or a
  component.
- Each instrument counts as a separate sense only when it passes the
  differentiation contract, which is not negotiable. The contract decides
  redundancy; the medium does not, so two text embedders are two senses if
  they measure differently. A lens is admitted only when it adds at least 0.05
  bits about a grounded anchor conditional on the panel, correlates at or below
  0.6 with every admitted lens, and is frozen and hash-pinned. A meaning is
  asserted only when at least five admitted lenses are present for that atom.
  Anchors evaluate instruments, and data trains them; a lens that re-runs the
  anchor's oracle or reads the label's siblings is not a sense. Thresholds are
  never tuned toward a gate. No-flatten: slots stay typed, never averaged or
  concatenated into one number. Measure senses the way the reference systems do
  (spec §19.3 step 2): read reality in chunks and score each document by its set of
  chunk readings, never one pooled whole-text vector; use the Calyx `calyx bits`
  method (port: `scripts/text/calyx_bits.py`), not a home-made estimator. When a
  measurement disagrees with the operator's papers, assume a method error first,
  Johari-window the likely causes, and fix the method; never report the theory wrong
  on one estimator's say-so.
  The kernel is gated on held-out recall ≥ 0.95, never the DFVS report.
- Fail closed with actionable error codes. No fallback success, no mock data, no
  hosted CI, no silent CPU fallback of a GPU lens (a different runtime is a
  different lens), no workaround that hides broken state.
- Secrets never enter the repo, an issue, a log or a report: not the SSH
  password, not the HF token, not the VPN credentials. Names only.
