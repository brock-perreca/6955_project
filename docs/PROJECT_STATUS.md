# Project status — current snapshot

**Purpose:** what's running *right now* — the active code, the current
best policy, what's known broken, and where the next move is queued.
**Read this when:** opening the project for the first time, or coming
back after a few days away.
**Adjacent:** [`PROJECT_TIMELINE.md`](PROJECT_TIMELINE.md) for the
chronological story · [`RESTART_LOG.md`](RESTART_LOG.md) for the most
recent batches with full setup/observation/render commands ·
[`ROADMAP.md`](ROADMAP.md) for what's queued next ·
[`reports/writeup_filled_1.docx`](reports/writeup_filled_1.docx) for
the formal joint writeup with Brian.

*Last updated: 2026-04-29.*

---

## Where we are

**Phase 5b.** The reference data was corrected on 2026-04-28
(`extract_gait_cycle.py` and `ulrich_loader.py` flip the knee only;
hip and ankle now match OpenSim signs — see
[`PROJECT_TIMELINE.md § Phase 5`](PROJECT_TIMELINE.md#phase-5--the-sign-error-discovery-2026-04-28)).
Every PPO/AMP/AIRL run on disk *before* the restart was trained
against a self-contradictory target; those checkpoints and the
pre-restart engineered-reward constants are kept only as a historical
record.

The post-restart pipeline rebuild is in progress. Four batches done
(see [`RESTART_LOG.md`](RESTART_LOG.md) for full details):

- **Batch 2** found the prior best policy (`results/restart_b2_xvel/`)
  via the `--xvel_term 0.3` floor termination. Walks, but with stiff
  hips (~2° ROM vs reference 45°) and 3× cadence as a downstream
  consequence.
- **Batch 3** (2026-04-29 overnight, 19 experiments) tested 8
  reward-aggregator/termination ablations + 4 AMP/AIRL warm-started
  runs + 3 multi-step preview runs + an SAC variant. **All 19 fall
  into the same stiff-hip basin or worse.** Headline read at the time
  was "reward-driven trap"; batch 4 superseded that.
- **Batch 4** (2026-04-29) **diagnosed the stiff-hip basin as a
  joint-range problem in the MJCF**, not a reward problem. Stock
  `walker2d.xml` constrains `thigh_joint` to `[-150°, 0°]`; the
  reference's +30° hip-flexion peaks are unreachable. Opening the
  range to `[-30°, +60°]` (`results/restart_b4_hipopen/`, 2M steps)
  jumped hip ROM from 1.8° to **91.5°**. Every previous batch was
  fighting an unreachable target. Variant B (re-invert hip column,
  keep stock MJCF) only partially escapes — the +13° extension peak
  still exceeds the 0° MJCF limit, producing a brittle deep-extension
  kicking gait. See [`RESTART_LOG.md § Batch 4`](RESTART_LOG.md#batch-4--2026-04-29--joint-range-hypothesis-open-hip-mjcf--positive).
- **Batch 5** (2026-04-29) ran the two queued aggregator-narrowing
  ablations on top of `hipopen`: `--pose_scale 20` (sharper mean
  aggregator) and `--min_joint_pose` (worst-joint floor). Both
  produced **partial improvements** — hip ROM narrowed from 63° → 57°
  on both variants and forward velocity dropped from 1.40 m/s toward
  the 1.25 target (`min_joint` lands at 1.231 m/s — essentially
  spot-on). Neither hits the prompt's full win criterion (hip ROM
  ~43°), but both are valid candidates for the new current best,
  pending visual A/B. See
  [`RESTART_LOG.md § Batch 5`](RESTART_LOG.md#batch-5--2026-04-29--narrow-the-hipopen-over-flex--partial-positive-both-variants).

**Current best policy (pending visual review):**
`results/restart_b4_hipopen_5M/` (5M steps, seed 6). Hip ROM 63.2°
vs reference 43°, mean fwd vel 1.40 m/s vs target 1.25, all 4
deterministic eval episodes survive 1000 steps. Still over-flexes
by ~10° at the swing-forward peak — pose-tracking forgives one
overshooting joint when the others track — but a robust, real
walking gait.

**Batch-5 candidates** (`results/restart_b5_min_joint/`,
`results/restart_b5_pose_scale20/`) both improve on baseline along
hip-ROM and forward-velocity axes; `min_joint` is the leading
candidate to supersede `b4_hipopen_5M` once visual A/B confirms.

**Top-priority next steps** (see [`ROADMAP.md § 0`](ROADMAP.md#0-narrow-the-hipopen-gait-toward-reference-tracking-new-2026-04-29)):

1. Visual A/B: `docs/figures/restart_b4_hipopen_5M.mp4` vs
   `restart_b5_min_joint.mp4` vs `restart_b5_pose_scale20.mp4`.
2. If still over-flexed, stack the two batch-5 knobs
   (`--pose_scale 20 --min_joint_pose`) — neither single-knob run
   tested this combination.
3. If stacking still doesn't reach reference shape, the next
   escalation is the peaked-forward-reward fallback
   (`--fwd_weight 0.15 --xvel_term -1e9`) per
   [`ROADMAP.md § 0`](ROADMAP.md#0-narrow-the-hipopen-gait-toward-reference-tracking-new-2026-04-29).
4. Re-run AMP/AIRL warm-started from the best of the b4/b5 set —
   batch-3 AMP failed partly because the underlying PPO couldn't
   produce reference-like hip flexion; that constraint is now
   removed.

---

## What this project is

A reinforcement learning study of **gait imitation on the MuJoCo
Walker2d-v4 planar biped**, conditioned on inverse-kinematics reference
data from the Ulrich treadmill walking dataset (Subject 1, 1.25 m/s).

Two complementary imitation methods are studied:

1. **Phase-conditioned PPO + DeepMimic 4-term reward** — primary
   working track. Active code in [`../src/walker2d/ppo_walker2d_phase.py`](../src/walker2d/ppo_walker2d_phase.py).
   Off-policy SAC sibling at [`../src/walker2d/sac_walker2d_phase.py`](../src/walker2d/sac_walker2d_phase.py).
2. **Adversarial Motion Priors (AMP) + AIRL** — comparison track
   ([`../src/walker2d/amp_walker2d.py`](../src/walker2d/amp_walker2d.py),
   [`../src/walker2d/airl_walker2d.py`](../src/walker2d/airl_walker2d.py),
   cherry-picked from upstream `bk-37/6955_Project@3e4c3fa` on
   2026-04-28). AMP collapses at 8 CPU envs (writeup §6.3); the
   recommended workflow today is to finetune from a working
   PPO+DeepMimic checkpoint, but the warm-started Batch 3 runs found
   the discriminator gradient pushes the policy out of stiff-hip into
   asymmetric kicks rather than natural gait. The full unblock is the
   GPU/MJX port — see [`ROADMAP.md § 1`](ROADMAP.md#1-mujoco-mjx--gpu-port-for-amp-writeup-71).

Three top-line scientific contributions (from the writeup):

- A working phase-conditioned imitation policy on real human IK data.
- A mechanistic taxonomy of reward-hacking failure modes (ankle
  paddling, one-legged hopping, toe-walking, plus the post-restart
  stiff-hip trap) framed as canonical Goodhart's-Law cases. See
  [`REWARD_DESIGN.md`](REWARD_DESIGN.md).
- A characterization of AMP's discriminator collapse at small env
  counts (writeup §6.3) and the mechanism that explains it.

---

## How to validate progress

`eval_biomech.py` compares every checkpoint against a *measured*
reference (computed from Subject 1's force plates and IK by
`extract_reference_biomech.py`), and `scripts/biomech_report.py`
renders a writeup-ready markdown table + 6-panel figure. After any
training batch:

```
python src/diagnostics/eval_biomech.py --xml walker2d.xml --eps 6 --steps 2500 \
    results/<run>:final:<label> --out results/<run>_eval.json --csv results/biomech_history.csv
python scripts/biomech_report.py results/<run>_eval.json --rerollout
```

The eval JSON's `vs_reference` block carries `delta` and `pct_err` per
metric, plus a single `progress_score` in [0, 4]. See
[`METHODS.md § Held-out biomechanical evaluation`](METHODS.md#held-out-biomechanical-evaluation-the-two-tool-flow).
**Anti-Goodhart caveat from Batch 3:** `progress_score` and
`hip_knee_dtw` flatter stand-and-wiggle policies; pair with
`hip_r_rom_deg` and visual review.

---

## What's currently running

- **Active training script:** `src/walker2d/ppo_walker2d_phase.py` —
  the DeepMimic-faithful 4-term reward (sum of `exp(−k·err²)` on pose,
  velocity, end-effector, root height). All exploit-patch terms
  (swing_pen, contact_r, per-joint sharpness/weights, per-joint
  termination thresholds, BC) are off-by-default kwargs/CLI flags. See
  [`METHODS.md § Reward`](METHODS.md#reward--deepmimic-four-term-sum)
  and [`RESTART_LOG.md`](RESTART_LOG.md).
- **Active reference:** `assets/reference/gait_cycle_reference.npy` —
  one clean stride from Ulrich Subject 1 baseline (56 frames @ 50 Hz,
  resampled to 140 frames @ 125 Hz inside the env). FK-verified after
  the 2026-04-28 sign fix to encode forward walking.
- **Current best policy (pending visual review of batch 5):**
  `results/restart_b4_hipopen_5M/` — 5M steps, seed 6, 8 envs,
  `--xvel_term 0.3`, `--xml walker2d_hipopen.xml` (custom MJCF with
  hip range opened to `[-30°, +60°]`). Hip ROM 63.2° vs reference
  43°; mean fwd vel 1.40 m/s vs target 1.25; all 4 deterministic
  eval episodes survive 1000 steps. Over-flexes ~10° at swing peak.
  Two batch-5 candidates (`results/restart_b5_min_joint/`,
  `results/restart_b5_pose_scale20/`) both narrow ROM to ~57° and
  pull fwd vel toward target; `min_joint` lands at fwd vel 1.231
  m/s and is the leading candidate to supersede this once visual
  A/B confirms.
- **Pre-batch-4 baseline (kept for reference):** `results/restart_b2_xvel/`
  — 5M steps, stock `walker2d.xml`, single-cycle reference, 8 envs.
  Same recipe minus the hipopen MJCF. Stiff-hip basin (hip ROM ~2°
  vs reference 45°, cadence 3× too fast). Useful as the "before"
  policy for showing what opening the joint range did.
- **Pre-restart canonical** (kept for historical comparison only —
  trained on the inverted reference; do not branch new work off
  these): `results/walker2d_phase_cycle_s1scaled_sum_20260423-213031/`
  (100M, scaled MJCF, engineered reward).

## Comparison runs on disk

Sorted by training date; `restart_*` runs are post-restart on the
corrected reference, `walker2d_phase_*` runs are pre-restart on the
inverted reference and require the missing `walker2d_subject1.xml`
MJCF to render.

| Result dir | Steps | Notes |
|---|---|---|
| **`results/restart_b4_hipopen_5M/`** | **5M** | **Current best (pending visual A/B vs batch-5).** DeepMimic 4-term + `--xvel_term 0.3` + `--xml walker2d_hipopen.xml`. seed=6. Hip ROM 63°, fwd vel 1.40 m/s, 1000×4 eval survival. |
| `results/restart_b5_min_joint/`     | 5M     | Batch 5 Variant B: above + `--min_joint_pose`. seed=8. Hip ROM 57°, fwd vel **1.23 m/s** (~target). Leading batch-5 candidate. |
| `results/restart_b5_pose_scale20/`  | 5M     | Batch 5 Variant A: above + `--pose_scale 20`. seed=7. Hip ROM **56.6°**, fwd vel 1.35 m/s. Tightest ROM but fwd vel still 8% over. |
| `results/restart_b4_hipopen/`       | 2M     | Pre-5M `b4_hipopen` checkpoint. seed=4. Hip ROM 91° (under-trained). |
| `results/restart_b4_hipinvert/`     | 2M     | Batch 4 Variant B (re-invert hip ref, stock MJCF). seed=5. Brittle deep-extension kicking gait, episodes die at 47-250 steps. |
| `results/restart_b2_xvel/` | 5M | Pre-batch-4 stiff-hip baseline. DeepMimic 4-term + `--xvel_term 0.3`. Stock walker2d.xml, seed=2. ep_len 2120, all-episode 2500-step survival on eval, but hip ROM ~2° vs ref 45°. |
| `results/restart_b2_k30/` | 5M | DeepMimic + `--pose_scale 30`. Unstable; 4/6 eval episodes fall in <120 steps. |
| `results/restart_b1_dm/` | 2M (killed) | Pure DeepMimic baseline. Stand-and-wiggle exploit. |
| `results/restart_b1_dm_bc/` | 2M (killed) | DeepMimic + 5-epoch BC. Same exploit, marginally varied across seeds. |
| `results/walker2d_phase_cycle_s1scaled_sum_20260423-213031/` | 100M | Pre-restart canonical (inverted reference, scaled MJCF). |
| `results/walker2d_phase_cycle_s1scaled_sum_20260422-175117/` | 60M | Earlier pre-restart canonical, scaled MJCF. |
| `results/walker2d_phase_full_sum_20260410-124935/` | 18M | Stock Walker2d, full-trial ref, uniform-k=8 (pre-restart). |
| `results/walker2d_phase_full_sum_20260410-105306/` | 45M | Earlier pre-restart DeepMimic-reward run. |
| `results/walker2d_phase_cycle_sum_20260409-211537/` | 10.5M | First single-cycle reference run (pre-restart). |
| `results/walker2d_pretrain_symmetry_20260407-172719/` | 5M | Symmetry-pretrain ankle-paddling demo (legacy). |

The 2026-04-29 overnight sweep produced 19 additional runs under
`results/overnight_<TIMESTAMP>/` (not all retained on every checkout —
see `RESTART_LOG.md § Batch 3`).

Render any of them with:

```
python src/walker2d/render_phase.py --xml walker2d.xml --live results/restart_b2_xvel:final
python src/walker2d/render_phase.py --xml walker2d.xml --mp4 docs/figures/foo.mp4 results/restart_b2_xvel:final
```

`render_phase.py` defaults to `--xml walker2d_subject1.xml`; for stock
Walker2d runs (all post-restart and most early pre-restart), override
with `--xml walker2d.xml`.

---

## What still needs to happen

For the **current writeup-driven scope**, see [`ROADMAP.md`](ROADMAP.md).
Top items:

0. **Reward reform** (NEW priority): restore `forward_reward` peaked
   term, drop `xvel_term` floor.
1. **MJX/GPU port** to make AMP function (4,000-env parallelism).
2. **Multi-step future context** in the observation — implemented as
   `--preview_k`; ineffective on the broken reward (Batch 3); worth
   one more pass after item 0.
3. **DTW-based shape-fidelity evaluation** — `hip_knee_dtw` and
   `all_joints_dtw` shipped in `eval_biomech.py`; pair with hip ROM
   per Batch 3 caveat.
4. **Multi-cycle / multi-subject reference** for temporal smoothness
   and robustness.

For **revisiting the original 3D / musculoskeletal scope**, see
[`LEGACY_TRACKS.md`](LEGACY_TRACKS.md). Code is preserved in
`src/legacy/musculoskeletal/`.

---

## Known gaps in this checkout

- **`assets/mjcf/walker2d_subject1.xml` is missing.** The pre-restart
  canonical run was trained against it; required for `--scale_model`
  and is the default `--xml` for `render_phase.py`. Must be
  regenerated or copied from the user's other machine before training
  / rendering with scaled geometry. Stock-Walker2d runs (the entire
  `restart_*` family and most pre-restart full-trial runs) load and
  roll out fine without it — pass `--xml walker2d.xml` to
  `render_phase.py`.
- **`amp_walker2d.py` and `airl_walker2d.py`** were cherry-picked from
  upstream commit `3e4c3fa` on 2026-04-28; relocated from the repo
  root into `src/walker2d/` and rewired to import the active loader.
  `--ref_cycle` works out-of-the-box; `--ref_all` does not receive
  per-trial segment lengths from the loader, so boundary transitions
  are not filtered out of the expert buffer.
- **`Ulrich_Treadmill_Data/`** is gitignored. Users supply their own
  copy at `<repo>/Ulrich_Treadmill_Data/Subject{1..10}/IK/walking_*/output/results_ik.sto`.
  See [`DATA_SOURCES.md`](DATA_SOURCES.md).

### Per-machine setup notes

- **Current laptop (no GPU, Python 3.13):** `Ulrich_Treadmill_Data/` is
  a directory junction to `CoordinationRetrainingData/forSimTK/`. Venv
  at `.venv/` was built with the CPU build of PyTorch 2.7.0 and
  `requirements/windows_cpu.txt`. MyoSuite is **not** in this venv —
  it pins `gymnasium==1.2.3` / `mujoco==3.6.0` (incompatible with the
  active Walker2d stack) and requires Python 3.10–3.12. For the
  legacy `src/legacy/musculoskeletal/` track, a sibling venv lives at
  `.venv-myo/` (Python 3.12, MyoSuite 2.12.1, verified
  `myoLegWalk-v0` reset+step). See `README.md` "MyoAssist (legacy
  musculoskeletal track)" for the recipe.
- **numpy must be 2.x** to load existing checkpoints — their pickle
  blobs reference `numpy._core` (a 2.x-only path). The requirements
  files now pin `numpy>=2.0,<3.0`.
