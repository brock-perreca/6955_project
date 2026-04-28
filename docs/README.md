# Documentation index

This is the documentation root. The project is structured for AI-first
navigation: every important fact about the project lives in one of these
files, and `CLAUDE.md` at the repo root points here as the first stop.

## Where to start

| If you want to… | Read |
|---|---|
| Understand what this project *currently* is | [`PROJECT_STATUS.md`](PROJECT_STATUS.md) |
| Understand how it got here (proposal → pivot → now) | [`PROJECT_TIMELINE.md`](PROJECT_TIMELINE.md) |
| Find the file that does X | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Understand the env, reward, and training loop | [`METHODS.md`](METHODS.md) |
| Understand why each reward term exists | [`REWARD_DESIGN.md`](REWARD_DESIGN.md) |
| Look up a past run / failure mode / demo | [`RUN_LOG.md`](RUN_LOG.md) |
| Know what's planned next | [`ROADMAP.md`](ROADMAP.md) |
| Understand a legacy script before touching it | [`LEGACY_TRACKS.md`](LEGACY_TRACKS.md) |
| Know where reference data lives + how it's loaded | [`DATA_SOURCES.md`](DATA_SOURCES.md) |
| Read primary-source papers (DeepMimic, GAIL, AMP×2, AIRL, OpenCap, KinTwin, KINESIS) | [`papers/papers.md`](papers/papers.md) |
| Read the formal writeup | [`reports/`](reports/) |

## Quick navigation

- **Active code:** [`../src/walker2d/`](../src/walker2d/) — the phase-conditioned
  PPO pipeline that produces the current canonical walking policy.
- **Diagnostics:** [`../src/diagnostics/`](../src/diagnostics/) — standalone
  sanity-check scripts for the reference cycle, Walker2d masses, and OSIM mass.
- **Legacy code:** [`../src/legacy/`](../src/legacy/) — earlier Walker2d
  attempts and the original 3D musculoskeletal track. Frozen for reference;
  see [`LEGACY_TRACKS.md`](LEGACY_TRACKS.md) before extending.
- **Assets:** [`../assets/`](../assets/) — MuJoCo MJCF files and the gait-cycle
  reference array.
- **Results:** [`../results/`](../results/) — training output directories
  (model.zip, checkpoints/, reference.npy). The current canonical run is
  documented in [`PROJECT_STATUS.md`](PROJECT_STATUS.md).
- **Writeups:** [`reports/`](reports/) — the original proposal and the
  current authoritative writeup.
- **Papers:** [`papers/`](papers/) — primary-source PDFs (DeepMimic,
  GAIL, AMP-for-animation, AMP-for-robots, AIRL, OpenCap, KinTwin,
  KINESIS, smartphone-mocap validation) plus
  [`papers/papers.md`](papers/papers.md), an index describing what each
  one is and when to read it.

## Cross-reference conventions

When a doc cites a code fact, it cites the file path (e.g.
`src/walker2d/ppo_walker2d_phase.py:103`). Treat that as the source of
truth — if a doc and the code disagree, fix the doc, not the code.

When a doc cites a number that comes from the writeup (return curves,
hypothesis labels, scale-collapse claims), it cites the section
(e.g. "writeup §6.3"). The writeup is authoritative for *scope, methods,
and results*; the code is authoritative for *what is currently running*.
