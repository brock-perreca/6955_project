# Restart log — rebuilding the imitation pipeline on the corrected reference

**Purpose:** per-batch progress on the post-2026-04-28 rebuild — what
was tried, what happened, render commands.
**Read this when:** picking the next batch, or coming back to compare
"the old engineered reward did X" vs "the new DeepMimic baseline does
Y." For the trigger event (the sign-error discovery), see
[`PROJECT_TIMELINE.md § Phase 5`](PROJECT_TIMELINE.md#phase-5--the-sign-error-discovery-2026-04-28).
For the current best policy, see
[`PROJECT_STATUS.md`](PROJECT_STATUS.md).

**Format.** One entry per batch. Each entry: setup, expectation,
observation, render command. Optimised for "user opens 4 mp4s and
forms an opinion in 10 minutes."

---

## Batch 1 — 2026-04-28 — DeepMimic baseline ± BC

### Setup

Stripped `ppo_walker2d_phase.py` back to the DeepMimic core. The
previous reward (engineered against the corrupted reference) had
per-joint sharpness/weights, swing-foot contact penalty, stance-foot
contact alternation, per-joint pose/ankle termination thresholds. All
of those exploit-patch terms are now off-by-default; the reward is
just the four DeepMimic Eq. 6 terms:

```
r = 0.65 · r_p + 0.10 · r_v + 0.15 · r_e + 0.10 · r_c
r_p = exp(−10 · mean_j (q_j − q_ref_j)²)
r_v = exp(−0.1 · mean_j (dq_j − dq_ref_j)²)
r_e = exp(−40 · sum_foot ((Δx)² + (Δz)²))     root-relative
r_c = exp(−10 · (h − h_ref)²)
```

Termination: Walker2d-v4 default height [0.8, 2.0] + |pitch| > 0.3 rad.
No per-joint pose/ankle thresholds, no x-velocity floor.

RSI: uniform initial phase, qpos[3:9] = ref, qvel[3:9] = ref_vel,
qvel[0] = 1.25 m/s.

Stock walker2d.xml (Subject-1-scaled MJCF is missing on this checkout).

PPO: 8 envs, 5M steps, linear LR 3e-4 → 3e-5, ent_coef 0.005,
target_kl 0.015, 256-256 MLP.

| Variant | What changes | Output dir |
|---|---|---|
| `dm`    | Vanilla DeepMimic (no BC, no extras)         | `results/restart_b1_dm/` |
| `dm_bc` | Same + 5-epoch BC warm-start (100k PD steps) | `results/restart_b1_dm_bc/` |

### Expectation

Two open questions:

1. **Does DeepMimic-faithful reward + RSI alone produce walking on the
   corrected reference?** The previous engineered reward was layered
   with exploit-patches (swing_pen, contact_r, per-joint k/weights)
   that may have been compensating for the corrupted reference rather
   than for fundamental algorithmic gaps. If the corrupted reference
   was the dominant problem, the simple reward might just walk.
2. **Does the PD-rollout BC warm-start matter on the corrected
   reference?** BC was justified previously as a hopping-prevention
   tool; that exploit was partly a self-contradictory-reward symptom.
   On a clean reference, BC may be unnecessary overhead — or it may
   still help the policy escape the early-training basin around
   "stand still."

Success looks like: > 1500-step episodes, visible bilateral foot
alternation under the live viewer, no obvious hopping/paddling/dragging.

Failure looks like: (a) episode lengths plateau low (~200), (b) the
live viewer shows a familiar local optimum (one-leg hop, ankle
paddle, foot tap), (c) reward/r_pose flat near floor.

### Observation (cut at 2.5M; killed before 5M — basin clearly settled)

Both variants reach high `ep_rew` and `ep_len` headline numbers, but
the headline numbers are hiding a **stand-and-wiggle exploit**, not
walking. Killed at 2.5M because the basin had clearly settled — going
to 5M was unlikely to escape it. 2M checkpoints saved; final TB
scalars (vanilla / BC):

| metric             | vanilla @ 2.5M | bc @ 2.34M |
|---|---|---|
| `rollout/ep_rew_mean` | 685.1   | 481.6 |
| `rollout/ep_len_mean` | 1466.8  | 970.8 |
| `reward/r_pose`       | 0.544   | 0.578 |
| `reward/r_vel`        | 0.228   | 0.218 |
| `reward/r_ee`         | 0.054   | 0.058 |
| `reward/r_root`       | 0.952   | 0.962 |
| pitch terminations    | 2/rollout | 3/rollout |

Visual diagnostics on the **1M checkpoint** (deterministic rollout):

- Vanilla 1M, `seed=0`: 500 steps survived; avg forward velocity
  **0.35 m/s** (vs. 1.25 m/s target), `hip_r` range **[-12°, +2°]**
  while reference sweeps **[-13°, +30°]**. The hip never flexes
  forward; knee and ankle wiggle to the time-locked target while the
  body drifts on decaying RSI warm-start qvel until momentum dies.
  Foot z barely lifts (`foot_r_z ∈ [-0.05, +0.01]` over 500 steps;
  reference swing peak is `+0.27`).
- BC 1M is more varied across seeds — one seed reaches `hip_r ∈
  [-4.6°, +24.4°]` and avg vel 0.57 m/s (close to walking); other
  seeds collapse to 0.0–0.2 m/s standing patterns.

Why the headline numbers don't reveal this:
`r_pose ≈ 0.55` is hiding *partial* tracking — knees/ankles wiggle
correctly while the hip sits stiff at ≈0°. Mean-of-squares per-step
pose reward is forgiving on a single outlier joint (5 of 6 right ⇒
`mean(diff²)` stays small enough that `exp(-10·mean)` ≈ 0.55).
`r_ee ≈ 0.06` is the *only* signal that should punish standing
strongly (foot positions wildly off when the body doesn't move), but
its 0.15 weight buys only ~0.009 per step vs. ~0.36 from pose —
nowhere near enough to dislodge the basin.

Other tells of the stand-still basin:
- `r_vel ≈ 0.22` — when joints don't sweep through reference
  velocities, dq tracking is poor. Consistent with stiff joints.
- Eval-biomech stride period **0.24 s** vs. reference **1.12 s** —
  the heel-strike detector is firing on small force oscillations of a
  stationary stance, not real foot strikes. Cadence "504 steps/min"
  is an artifact of stand-still.

The two clear candidates to test in batch 2: (1) tighter pose tracking
so partial 5/6-joint tracking earns less reward; (2) a direct
"non-stationary" signal — either a forward-velocity reward term or an
`xvel_term` floor termination. Both go to batch 2.

### Render

```
python src/walker2d/render_phase.py --xml walker2d.xml --live results/restart_b1_dm:2000000:vanilla-2M results/restart_b1_dm_bc:2000000:bc-2M

# Pre-rendered preview mp4s already on disk:
#   docs/figures/restart_b1_preview_1M.mp4         (vanilla 1M only)
#   docs/figures/restart_b1_preview_2M_*.mp4       (vanilla 2M, bc 2M)
```

---

## Batch 2 — 2026-04-28 — escape the stand-still basin

### Setup

Both variants are vanilla batch-1 baseline + a single targeted change
to the stand-still failure. The single-knob design is so we can read
which mechanism is actually doing the work if either succeeds.

| Variant | Change                          | Rationale                                                           | Output dir |
|---|---|---|---|
| `xvel`  | `--xvel_term 0.3`               | Termination floor: episode ends if forward velocity drops below 0.3 m/s. Direct stand-still kill. | `results/restart_b2_xvel/` |
| `k30`   | `--pose_scale 30`               | Pose `exp(-30·mean(diff²))` → 50% reward at ≈ 0.15 rad RMS (was ≈ 0.26 rad). Stiff hip becomes unprofitable. | `results/restart_b2_k30/` |

Everything else identical to batch 1: 8 envs, 5M steps, stock
walker2d.xml, single-cycle reference, RSI + warm-start qvel,
height + |pitch|>0.3 termination only (no per-joint pose/ankle
thresholds), no swing_pen, no contact_r, no BC. Seeds 2 and 3
respectively (vs. 0/1 in batch 1).

### Expectation

- `xvel`: kills stand-still episodes hard, so ep_len should *drop*
  initially (every standing run terminates at xvel) and then
  recover *only* if the policy actually learns forward motion.
  Healthy signature: ep_len curve dips below batch 1, then climbs
  past it; `term/xvel` peaks early and falls; hip excursion
  approaches reference range; visual rollout shows real foot lifts.
- `k30`: same ep_len trajectory as batch 1 (no new termination), but
  `r_pose` should plateau lower (0.3 rather than 0.55) because mean
  squared error of the stand-still basin no longer earns 0.55. The
  policy is forced to find a better basin to recover ep_rew.

If both succeed: `xvel_term` is the simpler, more DeepMimic-faithful
choice (analogous to "fall = die"). If one succeeds and one fails:
diagnostic.

### Observation (full 5M for both)

**`xvel` is the keeper.** This is the best policy the project has produced.
Visual review (Brock): "best run I've seen in the entirety of the project".
Both runs ran the full 5M cleanly in ~30 min each (alone on the box).

| metric                 | xvel @ 5M | k30 @ 5M |
|---|---|---|
| `rollout/ep_rew_mean`  | 1052.6    | 234.6    |
| `rollout/ep_len_mean`  | 2120.9    | 868.9    |
| `reward/r_pose`        | 0.564     | 0.215    |
| `reward/r_vel`         | 0.240     | 0.291    |
| `reward/r_ee`          | 0.072     | 0.133    |
| `reward/r_root`        | 0.973     | 0.955    |
| pitch/height term      | 0 / 0     | 0 / 0    |
| xvel term              | 5         | 0        |

`eval_biomech` over 6 deterministic episodes × 2500 steps:

Reference targets in the right column are now *measured* (added
2026-04-29) — they come from Subject 1's GRF + IK files via
`src/diagnostics/extract_reference_biomech.py`. Per-joint ROM,
stride period, cadence, double-support, and peak vGRF/BW are all
computed from the same Ulrich data the gait cycle was extracted from.
Earlier versions of this table used bibliographic ranges — those are
preserved in the git history of this file. See `METHODS.md` for the
two-tool eval flow.

| metric                  | xvel-5M    | k30-5M     | reference (measured)  |
|---|---|---|---|
| ep_len_steps (median)   | 2500       | 92         | —                 |
| n_strides (median)      | 61         | 1          | —                 |
| stride_period_s         | 0.323      | 0.252      | **1.120**         |
| cadence (steps/min)     | 372        | 476        | **107.1**         |
| double_support_frac     | 0.074      | 0.298      | **0.227**         |
| LR_stride_asymmetry     | 0.099      | 0.330      | < 0.10            |
| swing_drag_frac         | 0.0        | 0.0        | 0.0               |
| hip_knee_dtw            | 0.148      | 0.185      | lower is better   |
| peak_vgrf_bw            | 3.20       | 2.58       | **1.10**          |
| hip_r ROM (deg)         | 1.8        | 1.6        | **45.4**          |
| knee_r ROM (deg)        | 21.2       | 18.0       | **65.7**          |
| ankle_r ROM (deg)       | 20.3       | 11.7       | **40.0**          |
| progress_score (0–4)    | 2.466      | 0.644      | **4.000**         |

**`xvel`** survives every episode (6/6 hit the 2500-step cap), is
symmetric (LR_asymmetry 0.066), and never drags the swing foot. It is
robustly walking. Two real residual problems:

1. **Cadence ~3× too fast** (stride 0.32 s vs reference 1.12 s; 371
   steps/min vs ~107). Body is running-cadence-but-walking-speed —
   ~0.34 m/step. A single 1.12-s reference cycle is being "consumed"
   by ~3.5 physical strides.
2. **Thighs barely move** (Brock, eyeball check). Confirmed by the
   numerical diagnostic from batch 1 (hip_r ∈ [-12°, +2°] vs reference
   [-13°, +30°]) — `r_pose = 0.564` is hiding stiff hips behind
   compliant knee/ankle on a 6-joint mean. The fast cadence is
   downstream of stiff hips: foot can't reach reference x-excursion
   (-0.40 → +0.69 m), so the body strides multiple short steps to
   cover the same x distance per phase cycle.

**`k30`** is unstable — 4/6 episodes fall in <120 steps; the surviving
ones limp asymmetrically (LR_asymmetry 1.47). Tighter pose alone
without forward-velocity pressure didn't escape.

**Verdict on batch-2 hypothesis:** the `xvel_term` floor is the right
mechanism (defensible: "if you stop walking forward, you've fallen off
the back of the treadmill"). Tighter `k_pose` *alone* is destabilising
without a forward-motion constraint.

### Render

```
python src/walker2d/render_phase.py --xml walker2d.xml --live results/restart_b2_xvel:final results/restart_b2_k30:final

python src/diagnostics/eval_biomech.py --xml walker2d.xml --eps 6 --steps 2500 results/restart_b2_xvel:final:xvel-5M results/restart_b2_k30:final:k30-5M --out results/restart_b2_eval.json --csv results/biomech_history.csv

python scripts/biomech_report.py results/restart_b2_eval.json --rerollout
```

Pre-rendered mp4: `docs/figures/restart_b2_xvel-5M.mp4`,
`docs/figures/restart_b2_k30-5M.mp4`. Pre-rendered comparison
figure: `docs/figures/biomech_report.png` (shows the stiff-hip basin
and the single-peak vGRF vs the reference's classic double-hump).

---

## Batch 3 — 2026-04-29 — overnight 19-experiment sweep — **negative result**

> **Headline:** none of the 19 experiments produced a visibly improved
> gait. The stiff-hip basin is **reward-driven**, not optimizer-driven,
> and **not escapable by aggregator/weighting changes alone.** Brock's
> visual review (morning of 2026-04-29) overruled several
> metric-flagged "wins" — the AMP/AIRL runs that scored highest on
> hip-ROM were sporadic kicks, not flexion-during-walking. See
> `results/overnight_20260429-0211/OVERNIGHT_SUMMARY.md` for the
> per-experiment table and post-review writeup.

### Setup

Single overnight branch `overnight/phase_flags`. 19 experiments run in
pairs of 2 in parallel, ~30 min each, total wall ~5 h. All additive
CLI flags wired through `Walker2dPhaseAware`; no destructive code
changes to `xvel-5M`.

The plan: 8 single-knob ablations to escape stiff hip (Phase 1), 4
AMP/AIRL warm-started runs (Phase 2), 3 multi-step preview-obs runs
(Phase 3), 3 curriculum/optimizer runs (Phase 5), plus a Phase 4
code-only DTW eval extension (`all_joints_dtw`).

### Expectation

At least one of:
- Reward-aggregator change escapes stiff hip via stricter pose
  tracking (Phase 1 prod_reward, min_joint, hip2x).
- AMP/AIRL discriminator pushes the policy toward natural gait when
  warm-started from a working policy (Phase 2 — was the strongest
  conceptual bet pre-overnight).
- Preview-obs lookahead lets the policy anticipate hip flexion
  (Phase 3).
- SAC's off-policy exploration finds a non-stiff-hip basin (Phase 5).

### Observation

**0/19 experiments produced a visibly improved gait.** All policies
fall into one of two visual states:

1. **Stiff-hip walking with knee/ankle wiggle** (most B1 + B3 runs).
   Body translates forward at near-treadmill speed; thighs stay
   nearly parallel; reference shows ~45° hip excursion, sim shows
   ≤3°. Same basin as `xvel-5M`.
2. **Sporadic / collapsing motion** (B2 AMP/AIRL, B5 reverse-curriculum).
   Discriminator gradient or curriculum transfer pushes the policy
   *out* of stiff-hip but into asymmetric kicks, ankle paddling, or a
   too-fast unnatural gait. Metrics misread the kicks as "hip ROM."

`hip_r ROM` summary across all 19 runs (reference: 45.4°):

| run | hip_r ROM | survival | visual | composite |
|---|---|---|---|---|
| `b1_hip2x` (top composite) | 2.16° | 100% | stiff hip + walking | 4.08 |
| `b1_hip4x` | 1.58° | 100% | stiff hip + asym | 3.26 |
| `b1_prod_reward` | 1.66° | 100% | stiff hip + walking | 3.66 |
| `b1_min_joint` | 1.64° | 100% | stiff hip + walking | 3.43 |
| `b1_ee30` | 2.04° | 85% | stiff hip + walking | 3.74 |
| `b1_velw5` | 1.87° | 100% | stiff hip + walking | 3.36 |
| `b1_hipterm` (term=0.4) | 3.0° | 4% | dies in 96 steps | 2.21 |
| `b1_energy` | 2.07° | 100% | stiff hip + walking | 3.27 |
| `b2_amp_ft_xvel` | 1.46° | 27% | sporadic | 2.98 |
| `b2_amp_16env` | 4.72° | 47% | ankle paddle (65°) | 2.30 |
| `b2_amp_ft_winner` | 1.60° | 14% | broken survival | 2.23 |
| `b2_airl_ft_winner` | 6.69° | 29% | sporadic kicks (102° ankle paddle) | 3.15 |
| `b3_preview_k4` | 2.24° | 46% | choppier than B1 | 3.44 |
| `b3_preview_k4_winner` | 2.44° | 100% | choppier than B1 | 3.52 |
| `b3_preview_k8` | 1.90° | 100% | similar to B1 | 3.47 |
| `b5_sac` (200k SAC) | 1.53° | 79% | same basin as PPO | 3.67 |
| `b5_reverse_curriculum_a` (v_target=0.6) | 2.71° | 41% | unstable | 2.76 |
| `b5_reverse_curriculum_b` (finetune to 1.25) | 2.63° | 26% | broken at slow→fast transition | 2.55 |

### Why metrics misled the agent

1. **`xvel_term=0.3` survival is satisfied at any forward velocity ≥0.31 m/s,** so survival reward dominates as long as the policy translates *at all*. Standing-with-knee-wiggle earns full survival.
2. **DTW finds the closest cyclic alignment;** stand-and-wiggle scores OK on DTW because *one* valid stride matches the reference cycle even if the body barely moves.
3. **`stride_period_s` from rising-edge heel-strike fires on small force oscillations of a stationary stance** (the same flaw as batch-2's xvel-5M cadence reading; the agent re-imported the metric without correcting).
4. **`hip_r ROM` IS legit** — that one really did say all 8 Phase 1 runs stayed at 1.6–3°. The negative result is solid; the positive results were illusory metric-noise.
5. **MP4 first frame** has hips wide because RSI sets `qpos[3:9] = ref[phase]`. The trained policy's first action collapses the hips back to ~0° within 1-2 frames. The "good" opening frame is the reset state, not behaviour.

### What this means for next steps

- **Skip more aggregator variants and warm-started AMP/AIRL.** They are not where the trap is.
- **Restore the deleted `forward_reward = exp(-3·(v-1.25)²)` term and remove `xvel_term` floor.** A bell-curve forward target replaces a survival-floor — standing-still no longer earns survival reward.
- **Test that change ALONE before stacking other modifications.** If it escapes the basin, *then* revisit AMP/AIRL with the new reward. If it doesn't, the trap is deeper than reward (gait-cycle frame rate? phase obs?).

### Render

```
# Single canonical "stiff hip but walking" video (use as visual baseline):
results/overnight_20260429-0211/b1_hip2x/preview.mp4

# Full ranking + per-run reports:
results/overnight_20260429-0211/RANKING.md
results/overnight_20260429-0211/<exp>/REPORT.md   (one per run, 19 total)
results/overnight_20260429-0211/OVERNIGHT_SUMMARY.md (post-review writeup)
```

---

## Batch 4 — 2026-04-29 — Tier 0 morphology ablation (hip-range relaxation)

### Setup

Before launching the planned reward-structure reform, we paused for
a Tier 0 diagnostic: was the stiff-hip basin **reward-driven** (as
Batch 3's overnight concluded) or **morphology-driven**?

Two static probes settled the framing before any retraining:

- **A.1 (`src/diagnostics/check_reference_jnt_range.py`).** Stock
  `walker2d.xml` `thigh_joint range="-150 0"` — the hip cannot flex
  forward past 0°. Reference peaks at +29.97°. **~68 % of every
  reference cycle is outside the joint range.** Knees and ankles fit.
- **A.2 (xvel-5M deterministic rollout, 5 seeds × 600 steps).** hip_r
  median +1.39°, std 2.22°, **95.3 % of frames within 0.5° of the
  +0° upper limit, 93.45 % above it**. The policy was actively
  pushing into the soft-constraint wall. xvel-5M's "stiff hip" was
  the joint-range ceiling, not policy stagnation below it.

Pre-Batch-3 conclusion was therefore wrong about the dominant cause
of stiff hip — every restart batch *and* all 19 overnight experiments
trained against a target ~half of which was unreachable. See
[`docs/TIER0_DIAGNOSTICS.md`](TIER0_DIAGNOSTICS.md) for the full
ledger.

**The Batch 4 experiment (C in the Tier 0 ledger):** single-knob
morphology ablation. Created `assets/mjcf/walker2d_hiprelax.xml` with
`thigh_joint range="-150 35"` (5° headroom over the reference peak,
no looser to avoid creating an overswing basin). Otherwise verbatim
xvel-5M recipe.

| Variant | Change (vs xvel-5M) | Output dir | Seed |
|---|---|---|---|
| `hiprelax_s11` | `--xml walker2d_hiprelax.xml` | `results/restart_b4_hiprelax_s11/` | 11 |
| `hiprelax_s12` | `--xml walker2d_hiprelax.xml` | `results/restart_b4_hiprelax_s12/` | 12 |
| `hiprelax_s13` | `--xml walker2d_hiprelax.xml` | `results/restart_b4_hiprelax_s13/` | 13 |

Three seeds in parallel for seed-fragility insurance — the overnight
showed seed-dependent behavior is real on this pipeline. 8 envs each,
5M steps each, ~54 min wall-clock per seed under 24-env / 16-core
contention.

Tooling shipped in this batch:
- `--xml` CLI flag added to `ppo_walker2d_phase.py` (was previously
  only `--scale_model`).
- `env_kwargs.json` written by training now records `xml_file`;
  `run_dashboard.py`, `eval_biomech.py`, `render_phase.py` prefer
  the saved value over their CLI default.
- `src/diagnostics/check_reference_jnt_range.py` — reachability
  gate that catches this class of bug in <1 s without training.
- `scripts/tier0/evaluate_C.py` — full validation pipeline
  (dashboards, eval_biomech, mp4s, comparison panel, summary).

### Expectation

- If pure morphology, hip ROM grows to reference and tracks shape.
- If mixed (range + reward), hip swings emerge but stay
  amplitude-truncated.
- If reward-only despite range (unlikely given A.2 numbers), no
  improvement.

### Observation — MIXED, both fixes needed

All three seeds qualitatively identical. Best on every aggregate
metric is **seed 11** (LR asymmetry 0.097 — the only seed under the
< 0.10 threshold; lowest all-joints DTW 0.532; highest progress
score 2.41; lowest peak vGRF/BW 3.97). Brock visual review
(2026-04-29): "all of the videos look great"; xvel-5M by comparison
"looks pretty bad".

| metric (median, 6 eps × 2500 steps) | xvel-5M | s11 (canonical) | s12 | s13 | reference |
|---|---|---|---|---|---|
| **hip_r ROM (deg)** | **1.77** | **19.79** | **19.92** | **16.50** | **45.4** |
| **hip_l ROM (deg)** | **1.94** | **15.27** | **16.56** | **18.52** | **45.4** |
| knee_r ROM (deg)    | 22.18 | 26.57 | 38.56 | 32.70 | 65.7 |
| knee_l ROM (deg)    | 22.50 | 26.30 | 36.80 | 31.80 | 66.1 |
| stride_period_s     | 0.327 | 0.361 | 0.347 | 0.371 | **1.120** |
| cadence (steps/min) | 367.6 | 332.7 | 346.3 | 323.6 | **107.1** |
| LR_stride_asymmetry | 0.138 | **0.097** | 0.143 | 0.122 | < 0.10 |
| peak_vgrf_bw        | 3.28 | **3.97** | 4.02 | 4.18 | 1.10 |
| all_joints_dtw      | n/a   | **0.532** | 0.641 | 0.543 | lower better |
| progress_score (0–4)| 2.31 | **2.41** | 2.19 | 2.11 | 4.00 |

Hip-trace probe (5 seeds × 600 steps, same as A.2): xvel-5M had
95.3 % of frames within 0.5° of the +0° wall. After relaxation, only
15-21 % of frames touch the new +35° limit. The wall is no longer
binding.

**Verdict.** **Mixed (range + reward), both fixes needed.**

The morphology hypothesis is strongly confirmed — hip ROM grew ~10×
across all three seeds, the flat-topped clamping signature
disappeared, the trace tracks reference shape and frequency. But
amplitude plateaued at ~40 % of reference, cadence stayed ~3× too
fast, and peak vGRF/BW *worsened* (4.0 vs 3.3 — running-not-walking
signature persists). With `xvel_term=0.3` rewarding survival floor
at any forward velocity ≥ 0.31 m/s, the policy converges to fast,
low-amplitude strides regardless of morphology.

**Next step (now-Batch-5):** Tier 1 reward reform —
`forward_reward = exp(-3·(v-1.25)²)` + drop `xvel_term` floor —
**on top of `walker2d_hiprelax.xml`**, not stock walker2d.xml. See
[`ROADMAP.md § 0`](ROADMAP.md#0-structural-reward-reform-forward_reward--remove-xvel_term-floor-new-2026-04-29).

### Render / eval

```
# 30-second visual triage: open the comparison panel
docs/figures/tier0/C_hiprelax/C_hip_trace_comparison.png

# Watch the videos in order:
#   docs/figures/tier0/C_hiprelax/00_reference_replay.mp4   (kinematic ceiling)
#   docs/figures/tier0/C_hiprelax/xvel-5M_final.mp4         (pre-Tier-0 best)
#   docs/figures/tier0/C_hiprelax/hiprelax_s11_final.mp4    (canonical)
#   docs/figures/tier0/C_hiprelax/hiprelax_s12_final.mp4
#   docs/figures/tier0/C_hiprelax/hiprelax_s13_final.mp4

# Live policy view:
python src/walker2d/render_phase.py --live results/restart_b4_hiprelax_s11:final

# Full pipeline (dashboards × ckpts, eval_biomech, mp4s, comparison panel, summary):
python scripts/tier0/evaluate_C.py
```

`docs/figures/tier0/C_hiprelax/C_summary.md` carries the cleaned-up
table; per-seed `*_eval.json` files have the full vs_reference
deltas.
