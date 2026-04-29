# Restart log — rebuilding the imitation pipeline on the corrected reference

> **Why this file exists.** On 2026-04-28 we discovered that
> `assets/reference/gait_cycle_reference.npy` had been computed with
> `walker = -opensim` applied to all six joints — a flip that's correct
> only for the knee. Hip and ankle were inverted, so every PPO/AMP/AIRL
> run on disk was trained against a self-contradictory target. See
> [`PROJECT_TIMELINE.md` § Phase 5](PROJECT_TIMELINE.md#phase-5--the-sign-error-discovery-2026-04-28).
>
> The data is fixed (verified by FK probe on 2026-04-28: at peak
> hip_r flexion the right foot is at +0.69 m relative to root, i.e.
> in front of the body — forward walking). This log records the
> ground-up rebuild of the imitation pipeline against the corrected
> reference.
>
> **Format.** One entry per batch. Each entry: setup, expectation,
> observation, render command. Optimised for "user opens 4 mp4s and
> forms an opinion in 10 minutes."

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

### Observation

(in progress)

### Render

```
python src/walker2d/render_phase.py --xml walker2d.xml --live results/restart_b2_xvel:final results/restart_b2_k30:final

python src/diagnostics/eval_biomech.py --xml walker2d.xml results/restart_b2_xvel:final results/restart_b2_k30:final --out results/restart_b2_eval.json
```

### Render

```
# Live MuJoCo viewer (requires display):
python src/walker2d/render_phase.py --xml walker2d.xml results/restart_b1_dm:final results/restart_b1_dm_bc:final --live

# Held-out biomech metrics (deterministic rollouts → JSON):
python src/diagnostics/eval_biomech.py --xml walker2d.xml results/restart_b1_dm:final results/restart_b1_dm_bc:final --out results/restart_b1_eval.json
```
