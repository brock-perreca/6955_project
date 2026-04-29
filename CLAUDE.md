# CLAUDE.md — orientation hub

This file is the **AI-first entry point** for this repo. It's a small
hub that points at the right doc in [`docs/`](docs/) for whatever you're
trying to do. Read this first; then read whichever linked doc actually
matches the task.

The project pivoted significantly from the original proposal. **Don't
assume anything about the file structure or scope from generic Walker2d
imitation work** — read the narrative arc below, then
[`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) and
[`docs/PROJECT_TIMELINE.md`](docs/PROJECT_TIMELINE.md).

---

## The narrative arc — what we wanted, what we had to settle for, what's still the dream

**Where we started**
([`docs/reports/Advanced_AI_Project_Report.pdf`](docs/reports/Advanced_AI_Project_Report.pdf),
the original proposal): *Lab-to-Field Transfer in Musculoskeletal
Reinforcement Learning.* A 6-condition study (R1–R6) training **3D,
80-muscle, 20-DoF MyoLeg** agents on **OpenCap markerless** vs
**lab-grade marker + force-plate + EMG** references with SAC and
SAC+GAIL, evaluating emergent ground reaction forces, EMG, and joint
contact forces against ground truth. The driving question: how much
biomechanical fidelity is lost when markerless mocap replaces a $150k
motion-capture lab.

**Why we pivoted.** That scope was too large for one semester. 3D +
Hill-type muscle actuators + adversarial imitation + multi-condition
comparison needed infrastructure (OpenCap pipeline, OpenSim post-hoc
analysis, muscle-actuator hyperparameter tuning) we could not bring up
in time. On top of that, the first 2D fallback (Phase 1) failed for
unrelated reasons — 50 vs 125 Hz reference speed mismatch, phase-blind
observation, concatenated multi-trial reference — and a symmetry-reward
pretraining detour (Phase 2) hit four characteristic local optima
(two-legged hopping, one-legged hopping, ankle paddling, standing in
place tapping feet). Phase conditioning, not clever reward shaping,
turned out to be the missing ingredient.

**Where we landed**
([`docs/reports/writeup_filled_1.docx`](docs/reports/writeup_filled_1.docx)
is the authoritative current writeup): a pragmatic backup track on
**2D MuJoCo Walker2d-v4** (torque-actuated, 6 joints) conditioned on
**Ulrich treadmill IK** (Subject 1, 1.25 m/s). Two methods:
- **Phase-conditioned PPO + DeepMimic-style multi-term reward + BC
  warm-start** — Brock's track, committed, primary baseline.
- **Adversarial Motion Priors / AIRL** — Brian's track,
  [`src/walker2d/amp_walker2d.py`](src/walker2d/amp_walker2d.py) and
  [`src/walker2d/airl_walker2d.py`](src/walker2d/airl_walker2d.py).
  Both reuse `Walker2dPhaseAware` from the PPO track but replace the
  hand-crafted reward with a learned discriminator (LSGAN for AMP,
  AIRL-shaped BCE for AIRL). Collapses at 8-env CPU scale (writeup
  §6.3) due to discriminator memorisation of the compact expert
  manifold; needs the MJX-parallelised port for stable training.

**Where we are now (Phase 5b, 2026-04-29).** On 2026-04-28 we
discovered the on-disk reference was hip-and-ankle inverted (the
knee-only flip is the correct sign convention; the loaders had been
flipping all six joints). Every pre-restart PPO/AMP/AIRL run had been
trained against a self-contradictory target. The loaders were
corrected and the pipeline is being rebuilt from a DeepMimic-faithful
baseline. Four batches done: the post-restart prior best was
`results/restart_b2_xvel/` (walks, but with stiff hips ~2° vs
reference 45°). The 2026-04-29 overnight 19-experiment sweep
(Batch 3) read as "reward-driven trap"; **Batch 4 superseded that
diagnosis on the same day** — the stiff-hip basin was actually a
**physical reachability** problem in the MJCF. Stock `walker2d.xml`
constrains `thigh_joint` to `[-150°, 0°]`, but the reference asks for
+30° hip flexion. Opening the range to `[-30°, +60°]` in
`assets/mjcf/walker2d_hipopen.xml` raised hip ROM from 1.8° to
**91.5°** in 2M steps (`results/restart_b4_hipopen/`); a 5M follow-up
(`results/restart_b4_hipopen_5M/`, **the current best policy**)
narrowed it to 63° ROM at 1.40 m/s with all eval episodes surviving
1000 steps. The 19 overnight ablations were all chasing a problem
that lived upstream of the reward. See:

- [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) — current state
- [`docs/RESTART_LOG.md § Batch 4`](docs/RESTART_LOG.md#batch-4--2026-04-29--joint-range-hypothesis-open-hip-mjcf--positive)
  — joint-range diagnosis and fix
- [`docs/REWARD_DESIGN.md § The stiff-hip trap`](docs/REWARD_DESIGN.md#the-stiff-hip-trap-2026-04-29-diagnosis)
  — mechanism (with the Batch-4 update at the top)
- [`docs/ROADMAP.md § 0`](docs/ROADMAP.md#0-narrow-the-hipopen-gait-toward-reference-tracking-new-2026-04-29)
  — narrow the over-flexed hipopen gait toward reference tracking
- [`docs/PROJECT_TIMELINE.md § Phase 5`](docs/PROJECT_TIMELINE.md#phase-5--the-sign-error-discovery-2026-04-28)
  — full sign-error story

**What we still want — the focus that has not changed.** The current
DeepMimic reward is heavily *engineered*: every weight, exponential
sharpness, contact threshold, and termination condition was tuned to
close a specific reward-hacking exploit
([`docs/REWARD_DESIGN.md`](docs/REWARD_DESIGN.md)). That gets a 2D
walker walking, but it is the opposite of the original spirit.

**The dream we would love to land before this project is done: a 2D
Walker2d actually walking from mostly or purely imitation data — with
minimal or no hand-crafted gait-shaping reward terms — and producing
biomechanically realistic kinematics and contact patterns.** That is
what AMP/AIRL were meant to deliver, and it is the through-line back
to the original proposal's core question of how faithfully imitation
alone can recover real human movement. The current engineered-reward
track is the **alternate backup** that got something working; it is
not the destination. Treat work that moves the project toward the
imitation-only ideal — MJX-parallelized AMP, multi-step preview
observations, DTW-based shape-fidelity rewards, richer multi-cycle or
multi-subject reference data, or a measured return to the
musculoskeletal track — as on-mission, not scope creep.

---

## TL;DR for an agent that just opened this repo

- **What this project is:** RL gait imitation on MuJoCo Walker2d-v4
  conditioned on Ulrich treadmill IK (Subject 1, baseline, 1.25 m/s).
  Two methods: phase-conditioned PPO + DeepMimic reward (working,
  primary track) and Adversarial Motion Priors / AIRL (comparison
  track, committed in `src/walker2d/{amp,airl}_walker2d.py`, pending
  GPU/MJX port for stable training at scale).
- **Active code:** [`src/walker2d/`](src/walker2d/). Modify these.
- **Frozen code:** [`src/legacy/`](src/legacy/). Don't extend without
  asking the user first.
- **Authoritative writeup for the *current* (backup) scope:**
  [`docs/reports/writeup_filled_1.docx`](docs/reports/writeup_filled_1.docx)
  (joint with Brian Keller). The PDF in the same folder
  ([`Advanced_AI_Project_Report.pdf`](docs/reports/Advanced_AI_Project_Report.pdf))
  is the original musculoskeletal proposal — its big-picture goal
  (imitation-only biomechanically realistic locomotion) is still what
  we'd love to reach; see the narrative arc above.
- **Authors:** joint with **Brian Keller**. Brian's AMP/AIRL code
  lives in [`src/walker2d/amp_walker2d.py`](src/walker2d/amp_walker2d.py)
  and [`src/walker2d/airl_walker2d.py`](src/walker2d/airl_walker2d.py)
  (cherry-picked from upstream `bk-37/6955_Project@3e4c3fa`). Brock
  works on phase-conditioned PPO + BC warm-start
  ([`src/walker2d/ppo_walker2d_phase.py`](src/walker2d/ppo_walker2d_phase.py)).

---

## Where to read for what

| Task / question | Read |
|---|---|
| What is this project right now? | [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) |
| **What's been tried since the 2026-04-28 restart?** Per-batch setup + observations + render commands. | [`docs/RESTART_LOG.md`](docs/RESTART_LOG.md) |
| Why is the codebase shaped this way? Original proposal vs current scope. | [`docs/PROJECT_TIMELINE.md`](docs/PROJECT_TIMELINE.md) |
| Where does file X live? What's the import graph? | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Implementation details: env, reward, RSI, BC, optimizer, termination | [`docs/METHODS.md`](docs/METHODS.md) |
| **How do I validate progress against real biomechanics?** Two-tool flow: `extract_reference_biomech.py` → `eval_biomech.py --targets` → `scripts/biomech_report.py`. Emits `vs_reference` deltas, a 0–4 progress score, and a writeup-ready figure. | [`docs/METHODS.md § Held-out biomechanical evaluation`](docs/METHODS.md#held-out-biomechanical-evaluation-the-two-tool-flow), [`src/diagnostics/README.md`](src/diagnostics/README.md) |
| Why does each reward term exist? What exploit closes which gap? | [`docs/REWARD_DESIGN.md`](docs/REWARD_DESIGN.md) |
| Past runs / failure modes / curated demos with reproduce commands | [`docs/RUN_LOG.md`](docs/RUN_LOG.md) |
| Future work (MJX, multi-step preview, DTW, multi-cycle) | [`docs/ROADMAP.md`](docs/ROADMAP.md) |
| Why this legacy file exists; what to verify before re-running | [`docs/LEGACY_TRACKS.md`](docs/LEGACY_TRACKS.md) |
| Reference data formats (Ulrich, OpenCap, .osim) | [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) |
| Primary-source papers (DeepMimic, GAIL, AMP×2, AIRL, OpenCap, KinTwin, KINESIS) | [`docs/papers/papers.md`](docs/papers/papers.md) |
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
6. **`writeup_filled_1.docx` documents the current (backup) scope and
   recent advancements. `Advanced_AI_Project_Report.pdf` is the
   original musculoskeletal proposal — historical for the *file
   structure*, but its big-picture goal (imitation-only,
   biomechanically realistic locomotion) is still the project's
   north star. See the narrative arc at the top of this file.** Both
   live under [`docs/reports/`](docs/reports/).

---

## File-tree at a glance

```
6955_Project/
├── CLAUDE.md                      ← you are here
├── README.md                       ← user quickstart
├── .gitignore
├── docs/                            ← all documentation (start at docs/README.md)
│   ├── PROJECT_STATUS.md            ←   where we are right now
│   ├── PROJECT_TIMELINE.md          ←   how we got here
│   ├── RESTART_LOG.md               ←   post-2026-04-28 batches
│   ├── ROADMAP.md                   ←   future work
│   ├── ARCHITECTURE.md              ←   directory map + import graph
│   ├── METHODS.md                   ←   implementation details
│   ├── REWARD_DESIGN.md             ←   reward + exploit taxonomy
│   ├── DATA_SOURCES.md              ←   Ulrich + OpenCap formats
│   ├── LEGACY_TRACKS.md             ←   what each old track was
│   ├── RUN_LOG.md                   ←   legacy symmetry-pretrain demos
│   ├── papers/                       ←   primary-source PDFs + index
│   ├── reports/                      ← the writeups
│   └── figures/                      ← diagnostic plots + curated mp4s
├── src/
│   ├── walker2d/                     ← ACTIVE phase-conditioned imitation
│   ├── diagnostics/                  ← standalone sanity checks + biomech eval
│   └── legacy/                       ← FROZEN: walker2d_v1/, musculoskeletal/
├── scripts/
│   ├── biomech_report.py             ← writeup-ready biomech table + figure
│   └── overnight/                    ← multi-experiment sweep scaffolding
├── assets/
│   ├── mjcf/                          ← MuJoCo XML (walker2d_subject1.xml is missing)
│   └── reference/                     ← gait_cycle_reference.npy + biomech_targets.json
├── requirements/
│   ├── windows_5090.txt
│   ├── windows_cpu.txt
│   └── macos.txt
└── results/                            ← training outputs
```

For the full directory map with per-file roles, see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
