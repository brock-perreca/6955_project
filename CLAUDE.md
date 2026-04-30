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
([`report/Advanced_AI_Project_Report.pdf`](report/Advanced_AI_Project_Report.pdf),
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
([`report/writeup_filled_1.docx`](report/writeup_filled_1.docx)
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

**Where we are now (Phase 5b, post-Tier-0, 2026-04-29).** On 2026-04-28
we discovered the on-disk reference was hip-and-ankle inverted (the
knee-only flip is the correct sign convention; the loaders had been
flipping all six joints). Every pre-restart PPO/AMP/AIRL run had been
trained against a self-contradictory target. The loaders were
corrected and the pipeline is being rebuilt from a DeepMimic-faithful
baseline.

The 2026-04-29 overnight 19-experiment sweep (Batch 3) initially
read as a "reward-driven trap"; **two independent same-day Batch 4
diagnostics — one per laptop — superseded that diagnosis** and
landed on the same root cause: the stock `walker2d.xml`
`thigh_joint range="-150 0"` is a **kinematic ceiling**. The
reference asks for hip flexion up to +29.97°; ~68 % of every gait
cycle is therefore *outside* the joint range. xvel-5M's hip was
parked at +0° for 95.3 % of frames. The 19 overnight ablations
were all chasing a problem that lived upstream of the reward.

Both machines ran a single-knob morphology ablation; the two
parallel variants together bracket the answer:

- **`assets/mjcf/walker2d_hipopen.xml`** (Brock-Asus-Laptop, Batch 4
  + Batch 5 sweeps): `thigh_joint range="-30 60"` — permissive both
  directions. Hip ROM jumped 1.8° → **91.5°** in 2M steps
  (`results/restart_b4_hipopen/`); a 5M follow-up
  (`results/restart_b4_hipopen_5M/`) narrowed it to ~63° at 1.40 m/s.
  Batch 5 (`pose_scale20`, `min_joint`) tightened it further but a
  visual A/B found all three indistinguishable to the eye.
- **`assets/mjcf/walker2d_hiprelax.xml`** (Brock-O11, Tier 0 C):
  `thigh_joint range="-150 35"` — minimal +5° headroom over the
  reference peak. Three-seed × 5M sweep
  (`results/restart_b4_hiprelax_s11/.../s13/`); hip ROM 1.8° →
  ~30° (per-stride median, post strike-detector fix); reference
  shape tracked but amplitude ~67 % of the reference's 45°,
  cadence ~1.95× too fast, vGRF too high.

Together the two ablations confirm **morphology was the dominant
cause** of stiff-hip in every pre-Tier-0 run. The hipopen variant
*overshoots* on full-rollout max−min (91.5° on a 45° target, kicks
included) but settles to ~30° per-stride median; the hiprelax
variant lands at the same ~30°.

**Current lead policy: `results/restart_b5_min_joint/`** (named
2026-04-29 after a strike-detector bug fix in `eval_biomech.py`
— pre-fix the same scorecard had ranked `b4_hiprelax_s11` first.
See
[`docs/figures/biomech_realism_dashboard.png`](docs/figures/biomech_realism_dashboard.png)
and [`docs/PROJECT_STATUS.md § Biomechanical-realism finding`](docs/PROJECT_STATUS.md#biomechanical-realism-finding-2026-04-29--end-of-road-on-the-engineered-reward-track)).
min_joint wins on the corrected scorecard: highest progress score
(2.66), lowest peak vGRF among the post-Tier-0 candidates (3.70 BW),
lowest double-support deviation, per-stride hip ROM 30° (~67 % of
ref). It is the most reference-faithful of the four, even though
it still misses double-support and peak-vGRF by ~90 % / ~240 %.

The other three (`b4_hipopen_5M`, `b5_pose_scale20`,
`b4_hiprelax_s11`) are kept as comparison points but **superseded
as the lead**.

**Eval-detector fix (2026-04-29).** Pre-fix the eval reported
stride ~0.36 s and "cadence 3× too fast" — that was a strike-
detector artifact. `_rising_edges` had `min_gap=25` hardcoded,
which at the 125 Hz sim rate is only 0.2 s; the high-impact
contact chatter (4–5 BW slams, 30 ms bouts) in these stiff-legged
gaits registered as 2–3 separate strikes per real stride. The
reference extractor used a 0.5-s debounce on the 50 Hz force-
plate stream (25 samples = 0.5 s — the right *time* but
accidentally the same *frame count*). Fix: scale the eval's
debounce to `int(0.5 * CTRL_HZ) = 62` frames so both detectors
share the same 0.5-s window. Pre-fix artifacts archived at
`results/biomech_candidates_eval.pre-mingap-fix.json`.

**End-of-road finding (2026-04-29, post strike-detector fix):**
every candidate fails the biomech scorecard. Stride 0.57–0.68 s
vs ref 1.12 s (~1.7–2.0× too fast, *not* 3× as previously
claimed), double-support ~1–2 % vs ref 23 % (these are
bouncing/skipping, not walking), peak vGRF 3.7–4.8 BW vs ref 1.10
BW (slamming, not loading), hip ROM 30° (per-stride median) vs
ref 45°. The post-Tier-0 candidates only modestly beat the
pre-Tier-0 `b2_xvel` baseline (score 2.47) on the same scorecard. The conclusion is
that **phase-conditioned PPO + DeepMimic-style engineered reward
on Walker2d does not recover human walking biomechanics**, even
with the kinematic-ceiling fix. Further reward-knob experiments
on this stack are deprioritised; the path to biomechanically
realistic gait runs through the imitation-only dream (AMP/MJX) or
back to the musculoskeletal track. See:

- [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) — current state
- [`docs/TIER0_DIAGNOSTICS.md`](docs/TIER0_DIAGNOSTICS.md) — Tier 0 per-experiment ledger + verdict (Brock-O11)
- [`docs/RESTART_LOG.md`](docs/RESTART_LOG.md) — per-batch progress incl. both Batch 4 ablations and Batch 5 hipopen sweeps
- [`docs/REWARD_DESIGN.md § The stiff-hip trap`](docs/REWARD_DESIGN.md#the-stiff-hip-trap-2026-04-29-diagnosis)
  — pre-Tier-0 reward analysis (superseded as the *dominant* cause; reward is the *secondary* cause per both Batch 4 ablations)
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — top-priority next steps: structural reward reform on top of the relaxed-hip MJCFs, plus narrowing the hipopen gait
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
  [`report/writeup_filled_1.docx`](report/writeup_filled_1.docx)
  (joint with Brian Keller). The PDF in the same folder
  ([`Advanced_AI_Project_Report.pdf`](report/Advanced_AI_Project_Report.pdf))
  is the original musculoskeletal proposal — its big-picture goal
  (imitation-only biomechanically realistic locomotion) is still what
  we'd love to reach; see the narrative arc above. The Overleaf
  template (`report/template.tex`, `project.sty`, `sample.bib`) and
  the assignment rubric (`report/Final_Project_Report.pdf`) also live
  in [`report/`](report/) — see [`report/README.md`](report/README.md).
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
| **Tier 0 morphology-vs-reward diagnostic (2026-04-29).** Why we're now training on `walker2d_hiprelax.xml`, what xvel-5M's stiff hip really was, the experiment-C verdict. | [`docs/TIER0_DIAGNOSTICS.md`](docs/TIER0_DIAGNOSTICS.md) |
| **What's been tried since the 2026-04-28 restart?** Per-batch setup + observations + render commands. | [`docs/RESTART_LOG.md`](docs/RESTART_LOG.md) |
| Why is the codebase shaped this way? Original proposal vs current scope. | [`docs/PROJECT_TIMELINE.md`](docs/PROJECT_TIMELINE.md) |
| Where does file X live? What's the import graph? | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Implementation details: env, reward, RSI, BC, optimizer, termination | [`docs/METHODS.md`](docs/METHODS.md) |
| **How do I validate progress against real biomechanics?** Two-tool flow: `extract_reference_biomech.py` → `eval_biomech.py --targets` → `scripts/biomech_report.py`. Emits `vs_reference` deltas, a 0–4 progress score, and a writeup-ready figure. For multi-run side-by-side L+R kinematics, both-leg vGRF curves, and a ±20% credible-band scorecard, use `scripts/biomech_realism_dashboard.py` on the eval JSON. | [`docs/METHODS.md § Held-out biomechanical evaluation`](docs/METHODS.md#held-out-biomechanical-evaluation-the-two-tool-flow), [`src/diagnostics/README.md`](src/diagnostics/README.md), [`scripts/README.md`](scripts/README.md) |
| Why does each reward term exist? What exploit closes which gap? | [`docs/REWARD_DESIGN.md`](docs/REWARD_DESIGN.md) |
| Past runs / failure modes / curated demos with reproduce commands | [`docs/RUN_LOG.md`](docs/RUN_LOG.md) |
| Future work (MJX, multi-step preview, DTW, multi-cycle) | [`docs/ROADMAP.md`](docs/ROADMAP.md) |
| Which MJCF should I use? hipopen vs hiprelax vs stock | [`assets/mjcf/README.md`](assets/mjcf/README.md) |
| Tooling index — what each `scripts/` and diagnostic does | [`scripts/README.md`](scripts/README.md), [`src/diagnostics/README.md`](src/diagnostics/README.md) |
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
   [`assets/`](assets/), formal write-up materials (Overleaf template,
   .docx writeups, assignment rubric) under [`report/`](report/), data
   references under `<repo>/Ulrich_Treadmill_Data/` (gitignored). The
   repo root holds `CLAUDE.md`, `README.md`, `.gitignore`, and
   platform-agnostic config only.
6. **`writeup_filled_1.docx` documents the current (backup) scope and
   recent advancements. `Advanced_AI_Project_Report.pdf` is the
   original musculoskeletal proposal — historical for the *file
   structure*, but its big-picture goal (imitation-only,
   biomechanically realistic locomotion) is still the project's
   north star. See the narrative arc at the top of this file.** Both
   live under [`report/`](report/), alongside the Overleaf template
   and the final-report assignment rubric.

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
│   ├── RESTART_LOG.md               ←   post-2026-04-28 batches (Batch 4 + 4b + 5)
│   ├── TIER0_DIAGNOSTICS.md         ←   morphology-vs-reward Tier 0 ledger
│   ├── ROADMAP.md                   ←   future work (item 0a/0b)
│   ├── ARCHITECTURE.md              ←   directory map + import graph
│   ├── METHODS.md                   ←   implementation details
│   ├── REWARD_DESIGN.md             ←   reward + exploit taxonomy
│   ├── DATA_SOURCES.md              ←   Ulrich + OpenCap formats
│   ├── LEGACY_TRACKS.md             ←   what each old track was
│   ├── RUN_LOG.md                   ←   legacy symmetry-pretrain demos
│   ├── papers/                       ←   primary-source PDFs + index
│   └── figures/                      ←   diagnostic plots + curated mp4s
├── report/                            ← formal write-up materials (see report/README.md)
│   ├── Final_Project_Report.pdf      ←   assignment rubric (Canvas handout)
│   ├── template.tex / project.sty / sample.bib  ←  Overleaf template
│   ├── writeup_filled_1.docx         ←   current authoritative narrative writeup
│   ├── writeup_extracted.txt         ←   plain-text extraction (grep-friendly)
│   ├── methods_analysis.docx         ←   pre-pivot methods doc
│   └── Advanced_AI_Project_Report.pdf ←  original musculoskeletal proposal
├── src/
│   ├── walker2d/                     ← ACTIVE phase-conditioned imitation
│   ├── diagnostics/                  ←   sanity checks + biomech eval (see README)
│   └── legacy/                       ← FROZEN: walker2d_v1/, musculoskeletal/
├── scripts/                            ← (see scripts/README.md)
│   ├── biomech_report.py             ←   writeup-ready biomech table + figure
│   ├── eval_hip_rom.py               ←   single-source-of-truth hip ROM metric
│   ├── debug_joint_range_hypothesis.py  ← end-to-end joint-range diagnostic
│   ├── make_hipinvert_reference.py   ←   build the hipinvert reference variant
│   ├── smoke_test_warmstart.py       ←   smoke test the MJCF-read warm-start
│   ├── render_all_results.ps1        ←   PowerShell: render every run to mp4
│   ├── tier0/                        ←   Tier 0 morphology-vs-reward harness
│   └── overnight/                    ←   multi-experiment sweep scaffolding
├── assets/
│   ├── mjcf/                          ← MuJoCo XML (see assets/mjcf/README.md)
│   │   ├── walker2d_hipopen.xml      ←   thigh range [-30, 60] (Asus track)
│   │   ├── walker2d_hiprelax.xml     ←   thigh range [-150, 35] (O11 track)
│   │   └── walker2d_subject1.xml     ←   missing on this checkout
│   └── reference/                     ← gait_cycle_reference.npy + biomech_targets.json
├── requirements/
│   ├── windows_5090.txt
│   ├── windows_cpu.txt
│   └── macos.txt
└── results/                            ← training outputs (model.zips kept on disk)
```

For the full directory map with per-file roles, see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
