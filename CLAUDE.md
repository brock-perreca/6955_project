# CLAUDE.md — orientation hub

This file is the **AI-first entry point** for this repo. It's a small
hub that points at the right doc in [`docs/`](docs/) for whatever you're
trying to do. Read this first; then read whichever linked doc actually
matches the task.

The project pivoted significantly from the original proposal. **Don't
assume anything about the file structure or scope from generic Walker2d
imitation work** — read [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md)
and [`docs/PROJECT_TIMELINE.md`](docs/PROJECT_TIMELINE.md) first.

---

## TL;DR for an agent that just opened this repo

- **What this project is:** RL gait imitation on MuJoCo Walker2d-v4
  conditioned on Ulrich treadmill IK (Subject 1, baseline, 1.25 m/s).
  Two methods: phase-conditioned PPO + DeepMimic reward (working,
  primary track) and Adversarial Motion Priors / AIRL (comparison
  track, partial / pending GPU port).
- **Active code:** [`src/walker2d/`](src/walker2d/). Modify these.
- **Frozen code:** [`src/legacy/`](src/legacy/). Don't extend without
  asking the user first.
- **Authoritative writeup:** [`docs/reports/writeup_filled_1.docx`](docs/reports/writeup_filled_1.docx)
  (joint with Brian Keller). The PDF in the same folder is the
  *original proposal* and the project pivoted away from it.
- **Authors:** joint with **Brian Keller**. Brian works on AMP/AIRL
  (not yet committed to this repo); Brock works on phase-conditioned
  PPO + BC warm-start (the code that *is* committed).

---

## Where to read for what

| Task / question | Read |
|---|---|
| What is this project right now? | [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) |
| Why is the codebase shaped this way? Original proposal vs current scope. | [`docs/PROJECT_TIMELINE.md`](docs/PROJECT_TIMELINE.md) |
| Where does file X live? What's the import graph? | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Implementation details: env, reward, RSI, BC, optimizer, termination | [`docs/METHODS.md`](docs/METHODS.md) |
| Why does each reward term exist? What exploit closes which gap? | [`docs/REWARD_DESIGN.md`](docs/REWARD_DESIGN.md) |
| Past runs / failure modes / curated demos with reproduce commands | [`docs/RUN_LOG.md`](docs/RUN_LOG.md) |
| Future work (MJX, multi-step preview, DTW, multi-cycle) | [`docs/ROADMAP.md`](docs/ROADMAP.md) |
| Why this legacy file exists; what to verify before re-running | [`docs/LEGACY_TRACKS.md`](docs/LEGACY_TRACKS.md) |
| Reference data formats (Ulrich, OpenCap, .osim) | [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) |
| User-facing setup + quickstart commands | [`README.md`](README.md) |

If a doc and the code disagree on a *value* (default flag, weight,
threshold), the code wins. If they disagree on a *narrative* (what the
project is doing or why), the writeup wins.

---

## Working rules for an agent in this repo

1. **Modify `src/walker2d/`, not `src/legacy/`.** If you find yourself
   reaching for a legacy file to add a feature, stop and ask: does the
   active pipeline have a place for this? It almost always does.
2. **Run scripts from the project root.** Each entry-point script does
   `Path(__file__).resolve().parents[2]` to find the repo root. Path
   resolution depends on this.
3. **Don't add error handling, fallbacks, or shims for cases that
   can't happen.** Trust the active code's invariants. Validate at
   boundaries (CLI, file IO) only.
4. **Don't comment what the code already says.** Comments only earn
   their place when they record *why* — a non-obvious constraint, a
   tuned constant, a workaround for a specific bug.
5. **Don't write new top-level files at the repo root.** Documentation
   goes under [`docs/`](docs/), code under [`src/`](src/), assets under
   [`assets/`](assets/), data references under
   `<repo>/Ulrich_Treadmill_Data/` (gitignored). The repo root holds
   `CLAUDE.md`, `README.md`, `.gitignore`, and platform-agnostic config
   only.
6. **`writeup_filled_1.docx` is authoritative for scope/methods/results.
   `Advanced_AI_Project_Report.pdf` is historical motivation, not a
   spec.** Both live under [`docs/reports/`](docs/reports/).
7. **Don't delete legacy code or data.** The user has flagged that some
   original-proposal ideas may be revisited. "Move and document" is
   always preferred over "remove."

---

## File-tree at a glance

```
6955_Project/
├── CLAUDE.md                      ← you are here
├── README.md                       ← user quickstart
├── .gitignore
├── docs/                            ← all documentation (start at docs/README.md)
│   ├── PROJECT_TIMELINE.md          ←   how we got here
│   ├── PROJECT_STATUS.md            ←   where we are
│   ├── ARCHITECTURE.md              ←   directory map + import graph
│   ├── METHODS.md                   ←   implementation details
│   ├── REWARD_DESIGN.md             ←   reward + exploit taxonomy
│   ├── ROADMAP.md                   ←   future work
│   ├── LEGACY_TRACKS.md             ←   what each old track was
│   ├── DATA_SOURCES.md              ←   Ulrich + OpenCap formats
│   ├── RUN_LOG.md                   ←   past runs with reproduce/render commands
│   ├── reports/                      ← the writeups
│   └── figures/                      ← diagnostic plots
├── src/
│   ├── walker2d/                     ← ACTIVE phase-conditioned imitation
│   ├── diagnostics/                  ← standalone sanity checks
│   └── legacy/                       ← FROZEN: walker2d_v1/, musculoskeletal/
├── assets/
│   ├── mjcf/                          ← MuJoCo XML (walker2d_subject1.xml is missing)
│   └── reference/                     ← gait_cycle_reference.npy
├── requirements/
│   ├── windows_5090.txt
│   ├── windows_cpu.txt
│   └── macos.txt
└── results/                            ← training outputs
```

For the full directory map with per-file roles, see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
