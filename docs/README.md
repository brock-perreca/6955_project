# Documentation index

This is the documentation root. The repo is structured for AI-first
navigation: `CLAUDE.md` at the repo root is the orientation hub and
points here as the first stop.

## Doc roles at a glance

Each doc has a single authoritative role. **If two docs disagree on a
fact, fix the one that's straying out of its lane** rather than
duplicating content.

| Doc | Role | Read it for |
|---|---|---|
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | "right now" | What's running today, current best policy, known gaps |
| [`PROJECT_TIMELINE.md`](PROJECT_TIMELINE.md) | "how we got here" | Chronological narrative across phases 0–5 |
| [`RESTART_LOG.md`](RESTART_LOG.md) | "recent batches" | Per-batch setup, observation, render commands since the 2026-04-28 restart |
| [`TIER0_DIAGNOSTICS.md`](TIER0_DIAGNOSTICS.md) | "the morphology-vs-reward verdict" | The 2026-04-29 Tier 0 ledger: A.1 reachability check, A.2 hip-trace probe, C hip-relax retraining. Resolves why every pre-2026-04-29 run had stiff hips. |
| [`ROADMAP.md`](ROADMAP.md) | "what's queued next" | Prioritized future work |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | "where things live" | Directory map, import graph, entry-point commands |
| [`METHODS.md`](METHODS.md) | "how the code works" | Env, reward formula, RSI, BC, optimizer, CLI flags |
| [`REWARD_DESIGN.md`](REWARD_DESIGN.md) | "why each reward term" | Term-by-term rationale + Goodhart's-Law exploit taxonomy |
| [`DATA_SOURCES.md`](DATA_SOURCES.md) | "where the data lives" | Ulrich layout, OpenCap layout, joint sign convention |
| [`LEGACY_TRACKS.md`](LEGACY_TRACKS.md) | "what's frozen and why" | `src/legacy/` contents and re-run requirements |
| [`RUN_LOG.md`](RUN_LOG.md) | "legacy demos" | The four symmetry-pretrain failure-mode demos with reproduce commands |
| [`papers/papers.md`](papers/papers.md) | "primary sources" | Per-paper "what it is + when to read it" index |
| [`reports/`](reports/) | "writeups" | Original proposal PDF + current authoritative `.docx` |

## Quick navigation to non-doc content

- **Active code:** [`../src/walker2d/`](../src/walker2d/) — phase-conditioned
  PPO + AMP/AIRL.
- **Diagnostics:** [`../src/diagnostics/`](../src/diagnostics/) — sanity
  checks, biomech evaluation, the gait-cycle viewer.
- **Sweep scaffolding:** [`../scripts/overnight/`](../scripts/overnight/)
  — multi-experiment wrappers used by [`RESTART_LOG.md § Batch 3`](RESTART_LOG.md#batch-3--2026-04-29--overnight-19-experiment-sweep--negative-result).
- **Legacy code:** [`../src/legacy/`](../src/legacy/) — frozen tracks;
  see [`LEGACY_TRACKS.md`](LEGACY_TRACKS.md) before extending.
- **Assets:** [`../assets/`](../assets/) — MuJoCo MJCFs and the
  gait-cycle reference array + measured biomech targets.
- **Results:** [`../results/`](../results/) — training output dirs.

## Conventions for AI agents working in these docs

1. **One source of truth per fact.** If a constant, flag, or behavior
   is described in two places, one of them is going to drift.
   Cross-link instead of restating.
2. **Code wins over docs on values.** When a doc cites a code fact, it
   cites the file path (e.g. `src/walker2d/ppo_walker2d_phase.py:103`).
   If a doc and the code disagree on a default flag, weight, or
   threshold, fix the doc.
3. **Writeup wins over docs on narrative.** When a doc cites a
   writeup section ("writeup §6.3"), the writeup is authoritative for
   *scope, methods, and results narrative*. Update writeups when
   project narrative changes; update docstrings when operational
   detail changes.
4. **Each doc starts with `**Purpose:** … **Read this when:** …`.**
   This lets a fresh agent decide in 3 seconds whether to keep
   reading.
5. **Don't repeat the Phase-5 sign-error story across files.** It
   lives canonically in [`PROJECT_TIMELINE.md § Phase 5`](PROJECT_TIMELINE.md#phase-5--the-sign-error-discovery-2026-04-28).
   Other docs link to that anchor in a single sentence.
6. **Don't write ephemeral handoff docs into `docs/`.** Per-run
   working notes belong in the run dir under `results/`, not in the
   permanent doc tree.
