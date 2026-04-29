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

## Batch 4 — 2026-04-29 — joint-range hypothesis: open hip MJCF — **POSITIVE**

> **Headline:** the post-restart stiff-hip basin was a **physical
> reachability** problem, not a reward / optimizer / discriminator problem.
> Stock `walker2d.xml` constrains `thigh_joint` to `[-150°, 0°]` — the
> reference asks for hip flexion peaks of **+29.7°** on both sides, which
> the simulator cannot reach. Opening the hip MJCF range bidirectionally
> to `[-30°, +60°]` (Variant A `b4_hipopen`) **escaped the basin in 2M
> steps** — hip_r ROM went from `1.8°` (xvel-5M, 5M steps) to `91.5°`.
> The 19 overnight reward/aggregator/discriminator/curriculum experiments
> were chasing a problem that lived upstream of the reward.

### Diagnosis (verified before training)

Independent verification with `scripts/debug_joint_range_hypothesis.py`:

1. **MJCF inspection.** Stock `walker2d.xml` (the gym default the
   `Walker2d-v4` env loads) has:
   ```
   thigh_joint            range="-150  0"  (deg)
   thigh_left_joint       range="-150  0"  (deg)
   leg_joint              range="-150  0"  (deg)
   foot_joint             range="-45  +45" (deg)
   ```
2. **Reference inspection.** `assets/reference/gait_cycle_reference.npy`
   per-joint ranges (post-2026-04-28 hip un-inversion):

   | joint    | min (deg) | max (deg) | XML range  | fits? |
   |---|---|---|---|---|
   | hip_r    | -13.39    | **+29.69**| -150 …  0  | **NO**  |
   | knee_r   | -64.51    |   -0.19   | -150 …  0  | yes |
   | ankle_r  | -23.41    |   +6.16   |  -45 … +45 | yes |
   | hip_l    | -13.44    | **+29.94**| -150 …  0  | **NO**  |
   | knee_l   | -62.34    |   -0.24   | -150 …  0  | yes |
   | ankle_l  | -17.16    |  +10.86   |  -45 … +45 | yes |

3. **Dynamics-respecting probe.** Setting `qpos[3:9] = ref[peak]` then
   stepping the env one Walker2d-v4 frame (4 substeps × 2 ms) with
   `action=0`: at peak hip_r flexion (+29.69°), the joint settles at
   +27.89° within 8 ms — i.e. the constraint solver is actively pulling
   the joint back toward the 0° limit. The same is true at peak hip_l
   (+29.94° → +28.12°) and at every other phase where the reference
   demands hip flexion (phases 0, 14, 28, 42 all show ≥0.8° pullback).
4. **Trained xvel-5M policy probe.** Deterministic rollout, 280 steps:
   - hip_r within 1° of upper limit (0°): **97.5% of steps**
   - hip_r at limit AND ref demanding ≥5° flexion: **62.1% of steps**
   - hip_l mirror: 92.1% / 56.1%

   The xvel-5M policy spent essentially the entire rollout pinned to the
   joint upper limit while the reference was sweeping +30°.

The previous engineered-reward exploit-patch terms
(`swing_pen`, `contact_r`, per-joint `hip_term`) were partly compensating
for this hard limit, not just for reward-hacking around the corrupted
reference. The pre-2026-04-28 reference's hip range (`[-30°, +13°]`
under the inverted sign convention) clipped less catastrophically (only
the +13° tail clipped) than the post-restart `[-13°, +30°]`, and the
patches absorbed what slipped through.

### Setup

Two parallel single-knob ablations against the `xvel-5M` recipe:

| Variant | Change | Output dir |
|---|---|---|
| `hipopen` | Custom MJCF `assets/mjcf/walker2d_hipopen.xml`: `thigh_joint range="-30  60"` (and `thigh_left_joint`); knee + ankle unchanged. Reference unchanged. | `results/restart_b4_hipopen/` |
| `hipinvert` | Reference `assets/reference/gait_cycle_reference_hipinvert.npy`: cols 0 (hip_r) and 3 (hip_l) negated; knee+ankle unchanged. Stock `walker2d.xml`. | `results/restart_b4_hipinvert/` |

Both: 8 envs, **2M steps** (vs 5M batch-2; basin escape readable from hip
ROM by 1–2M), single-cycle reference, RSI + warm-start qvel,
height + |pitch|>0.3 termination, `--xvel_term 0.3`, no swing_pen, no
contact_r, no BC. Seeds 4 and 5 respectively.

CLI exposed via the new `--xml` flag in `ppo_walker2d_phase.py`
(`--xml walker2d_hipopen.xml`); `xml_file` is now persisted in
`env_kwargs.json` so renderer/eval pick up the right model automatically.

The third option Brock raised (a Subject-1-scaled `walker2d_subject1.xml`)
was deprioritised: scaling segment lengths to Subject 1 doesn't fix the
hip-range bottleneck unless the joint range is also opened, so it's
strictly a follow-up to a working hipopen baseline.

> **Side note on Variant B's framing.** Brock's task description claimed
> re-inverting the hip column "puts the reference target back in the
> reachable side of the joint range." This is mostly true but slightly
> off: re-inverting moves hip_r to `[-29.69°, +13.39°]` — the +13.39°
> extension peak still exceeds the 0° MJCF upper bound. So Variant B is
> *less clipped* than the corrected reference but not actually fully
> reachable. We ran it anyway as the requested single-knob ablation;
> it's the more partial of the two fixes.

### Expectation

- **`hipopen`**: positive — basin escape, hip ROM ≥ 20°. May overshoot
  the reference (hip wants only ~43° ROM) until the pose-tracking
  reward's 6-joint mean penalises the overshoot enough; could converge
  cleanly with longer training.
- **`hipinvert`**: weak positive at best. Removes the catastrophic
  +30° clipping (now only the +13° tail is clipped, 13° outside vs 30°
  outside), but the policy will still bottom out on flexion and may
  compensate via deep extension. Brittle gait expected; useful as the
  control showing that "fix the reference" alone is not enough.

### Observation (full 2M for both, plus 5M follow-up of `hipopen`)

Eval: deterministic rollout, 4 episodes × 1000 steps, RSI from seed
42..45 (`scripts/eval_hip_rom.py`).

| metric                  | xvel-5M (b2) | b4_hipinvert  | b4_hipopen (2M) | b4_hipopen_5M | reference  |
|---|---|---|---|---|---|
| training steps          | 5M           | 2M            | 2M              | 5M            | —          |
| ep_len mean (training)  | 2120         | 127           | 660             | 4841          | —          |
| ep_rew mean (training)  | 1052         | 54            | 323             | 3507          | —          |
| eval episodes survived  | 6/6 × 2500   | 47, 188, 250, 148 | 392, 1000×3 | **1000×4**    | —          |
| mean fwd vel (m/s)      | 0.35         | 1.327         | 2.065           | **1.395**     | 1.25       |
| **hip_r ROM (deg)**     | **1.8**      | **77.32**     | **91.54**       | **63.23**     | **43.18**  |
| hip_l ROM (deg)         | ~2           | 45.71         | 86.99           | 59.35         | 43.30      |
| hip_r min/max (deg)     | -12 / +2     | -66.4 / +10.9 | -32.2 / +59.3   | **-23.2 / +40.1** | -13 / +30 |
| hip_l min/max (deg)     | similar      | -33.2 / +12.6 | -31.1 / +55.9   | -17.7 / +41.7 | -13 / +30  |
| % steps within 1° of upper hip limit | 97.5% | 32.1% (lim=0°) | 0.1% (lim=60°) | 0.0% | —      |

**`b4_hipopen` (Variant A) — primary winner.** Hip ROM jumped from 1.8°
to 91.5° on a single MJCF change — direct confirmation of the
hypothesis. Policy spends 0.1% of steps near the new upper limit
(+60°), so the new range is comfortably wider than what the policy
wants. Three of four episodes survive the full 1000-step eval cap.

The gait at 2M is **over-flexed and over-fast**: 91° ROM vs reference
43°, mean velocity 2.07 m/s vs target 1.25, per-episode hip_r maxes
at 55–59°. Consistent with under-training on a freshly opened solution
space — the pose-tracking `r_pose = exp(-10·mean(diff²))` is forgiving
of a single overshooting joint when the other five track. A 5M
follow-up was queued and ran; it narrows substantially (see below).

**`b4_hipopen_5M` (Variant A, 5M follow-up) — current best policy.**
Same recipe, seed 6, ran cleanly. Convergence trajectory in training
log: ep_len 660 (2M) → 1325 (1.84M) → 3052 (2.87M) → 5837 (4.51M);
ep_rew 323 → 4163. Eval shows the gait narrowed:

- Hip ROM 91.5° → 63.2° (still 20° over reference, but no longer
  using the full opened range; both peaks now sit ~10° inside the
  new joint limits).
- Mean fwd vel 2.07 → 1.40 m/s (target 1.25; ~12% over).
- All 4 eval episodes survive the full 1000-step cap.
- Hip max +40° vs reference +30°: the policy still over-flexes ~10°
  on the swing-forward half. Pose-tracking forgiveness still
  asymmetric (one hip vs five other joints).

The 5M run is the new current-best policy — supersedes
`results/restart_b2_xvel/` (which is now stiff-hip-baseline only).
Visual review is the load-bearing check; metrics suggest a real,
stable, slightly over-flexed walk.

**`b4_hipinvert` (Variant B) — partial / brittle.** Hip ROM technically
> 20° (77.3°), but reading the per-episode min values (-37.7°, -66.2°,
-66.2°, -66.4°) shows the policy is bottoming the hip joint at
*deep extension* (well past the reference's -29.7° minimum) on the
extension half of the cycle, then slamming against the +0° upper limit
on the flexion half (32.1% of steps near limit). It's an unstable
kicking gait: episodes die at 47–250 steps. Velocity ~1.33 m/s
matches treadmill speed only because the deep kick produces forward
momentum; this is not a tracking gait.

**Verdict on the hypothesis.** Confirmed. The 19 overnight Phase-1/2/3/5
experiments and the entire `restart_b1`–`restart_b3` arc were
struggling against a hard joint limit at hip flexion. Variant A's
single-line MJCF change broke the basin; Variant B's reference-only
change reduces the severity of clipping but does not eliminate it,
producing a brittle compensation gait.

### Next steps

1. **5M follow-up of `b4_hipopen`** ✓ done (`results/restart_b4_hipopen_5M/`,
   seed 6). Narrowed from 91°→63° hip ROM and 2.07→1.40 m/s; all
   eval episodes survive 1000 steps. Still over-flexes ~10° beyond
   reference's +30° peak.
2. **Tighter pose tracking to narrow the 10° overshoot** (queued).
   Either `--pose_scale 20` (50% reward at 0.18 rad RMS, harder for
   one stiff/overshooting joint to hide) or `--product_reward`
   (geometric-mean per-joint exps; one bad joint hurts the whole
   reward). Single-knob ablation against `b4_hipopen_5M`. Now that
   the basin is escaped, the batch-3 aggregator ablations may finally
   pull their weight.
3. **Re-evaluate AMP/AIRL warm-starts from `b4_hipopen_5M`.** The
   discriminator-collapse / sporadic-kicks failures from batch-3
   Phase-2 were partly a stiff-hip data-distribution mismatch (the
   policy could not produce reference-like hip flexion). With a
   real walking baseline, the discriminator should have a learnable
   signal. Brian's track.
4. **Optional: `walker2d_subject1.xml`.** With segment lengths scaled
   to Subject 1's IK + hip range opened, the foot-x trajectories will
   match the reference more cleanly. Lower priority than (2)–(3).

### Render

```
# render_phase.py now reads xml_file from each run's env_kwargs.json (added
# 2026-04-29 alongside this batch), so each rollout uses the MJCF it was
# trained against; no --xml needed.
python src/walker2d/render_phase.py --live results/restart_b4_hipopen_5M:final results/restart_b4_hipopen:final results/restart_b4_hipinvert:final

# Eval hip ROM (single source of truth — see the "metric traps" warning in Batch 3):
python scripts/eval_hip_rom.py results/restart_b4_hipopen_5M results/restart_b4_hipopen results/restart_b4_hipinvert

# Pre-rendered:
docs/figures/restart_b4_hipopen_5M.mp4  (1000-step deterministic rollout — current best)
docs/figures/restart_b4_hipopen.mp4     (1000-step rollout — 2M under-trained)
docs/figures/restart_b4_hipinvert.mp4   (231-step rollout — episode dies)
```

### What to watch

Three mp4s, watch in this order:

1. **`docs/figures/restart_b4_hipopen_5M.mp4` — THE WINNER.** This is
   the new current-best policy (supersedes `restart_b2_xvel`). Watch
   the thighs: previous batches showed two parallel sticks; this
   should show real, large-amplitude hip flexion-extension on both
   legs at near-treadmill speed (1.4 m/s vs target 1.25). Survives
   the full 1000 frames cleanly. The hips still over-flex by ~10°
   compared to the reference's +30° peak (eval shows max +40°), so
   the gait may look slightly exaggerated. **This is the load-bearing
   visual confirmation that the joint-range hypothesis was right and
   the fix works.**
2. **`docs/figures/restart_b4_hipopen.mp4` — same recipe, 2M
   under-trained.** Watch this if you want to see what an
   under-trained version of the winner looks like — should be a
   visibly faster, more aggressive gait with bigger hip ROM. Useful
   for understanding the trajectory; skip if short on time.
3. **`docs/figures/restart_b4_hipinvert.mp4` — Variant B control /
   failure mode.** Expect a brief, brittle motion before falling at
   ~step 231. Watch only to see the failure mode the alternative
   ("re-invert reference, keep stock MJCF") falls into — legs kick
   deep backward (the -66° extension that the metrics flagged) and
   then it loses balance. Useful evidence that fixing the reference
   sign alone is insufficient.

Skip `docs/figures/restart_b2_xvel-5M.mp4` (the prior stiff-hip
baseline) unless you specifically want to A/B against it; the
batch-3 review already covered it.

---

## Batch 5 — 2026-04-29 — narrow the `hipopen` over-flex — **partial positive (both variants)**

> **Headline:** both single-knob aggregator variants against
> `b4_hipopen_5M` produced **partial improvements** in the same
> direction: narrower hip ROM, lower peak hip flexion, no survival
> regression. Neither hits the "ROM drops to ~43°" target — both
> land at ~57° vs reference 43° (baseline 63°). `min_joint` is the
> more reference-faithful of the two (mean fwd vel 1.231 m/s vs
> target 1.25, vs `pose_scale20` 1.354 m/s and baseline 1.395 m/s);
> `pose_scale20` wins by a hair on hip ROM (56.6° vs 57.1°). Both
> are queued for visual A/B against `b4_hipopen_5M.mp4`.

### Setup

Two parallel single-knob ablations against the proven `b4_hipopen_5M`
recipe (8 envs, `--xvel_term 0.3`, `--xml walker2d_hipopen.xml`,
single-cycle reference). Trained from scratch with RSI rather than
finetuning — `b4_hipopen_5M`'s over-flex is an established local
optimum that a small finetune may not be able to climb out of.

| Variant | Change | Mechanism | Output dir |
|---|---|---|---|
| `pose_scale20` | `--pose_scale 20` (default 10) | Doubles the exponential sharpness of `r_pose = exp(-k·mean(diff²))`. A 10° overshoot now costs ~10% of pose reward instead of ~5%. Preserves the mean aggregator. | `results/restart_b5_pose_scale20/` |
| `min_joint`    | `--min_joint_pose` flag | Worst-joint floor: `r_pose = min_j exp(-k·w_j·diff_j²)`. One bad joint kills the whole pose reward — directly attacks the 5-of-6-joint loophole that lets the policy hide a single overshooting hip behind 5 compliant joints. | `results/restart_b5_min_joint/` |

Both: 8 envs, 5M steps, `--xvel_term 0.3`, `--xml walker2d_hipopen.xml`,
single-cycle reference, RSI + warm-start qvel, height + |pitch|>0.3
termination only, no swing_pen, no contact_r, no BC. Seeds 7 and 8.
Ran sequentially (~18.5 min each) on the 8-core box.

### Expectation

- **`pose_scale20`**: smallest change. Should narrow ROM somewhat
  by raising the cost of single-joint overshoots, but the mean
  aggregator still forgives 5/6 compliant joints. Risk: if the
  exponential is too sharp the reward gradient flattens and training
  slows; with k=20 vs k=10 at 0.18 rad RMS, gradient is still
  well-conditioned.
- **`min_joint`**: most aggressive. Forces all 6 joints to track
  simultaneously. Risk per the prompt: policy can't satisfy all 6
  and collapses to a different basin. Lower training ep_rew expected
  (lower headline reward is a *signal* the loophole closed, not a
  failure).

### Observation (full 5M for both)

Both ran cleanly to 5M with no instability. Final training scalars
(read from the train log):

| metric                        | b4_hipopen_5M | b5_pose_scale20 | b5_min_joint |
|---|---|---|---|
| training ep_len mean (final) | 4841          | **6211** (peak iter 1200) | 3389 |
| training ep_rew mean (final) | 3507          | **4153**                  | 2071 |

`pose_scale20` looks "best" on training-time ep_rew/ep_len because
its sharper-but-still-forgiving aggregator is genuinely easier to
optimise than the baseline at the same gait. `min_joint`'s lower
ep_rew is *expected and correct* — the worst-joint floor makes pose
reward harder to earn, even with the same gait. The training-time
numbers are not the load-bearing comparison; eval is.

`scripts/eval_hip_rom.py` over 4 deterministic episodes × 1000 steps,
RSI seeds 42..45:

| metric                             | b4_hipopen_5M (baseline) | b5_pose_scale20 | b5_min_joint |
|---|---|---|---|
| eval episodes survived             | 1000 × 4                 | 1000 × 4                  | 1000 × 4 |
| mean fwd velocity (m/s, target 1.25) | 1.395                  | 1.354                     | **1.231** |
| **hip_r ROM (deg, ref 43.18)**     | 63.23                    | **56.58**                 | 57.08 |
| **hip_l ROM (deg, ref 43.30)**     | 59.35                    | 58.10                     | **56.49** |
| hip_r min / max (deg, ref -13/+30) | -23.2 / +40.1            | -16.7 / **+39.9**         | -19.2 / **+37.9** |
| hip_l min / max (deg, ref -13/+30) | -17.7 / +41.7            | -21.9 / **+36.2**         | -20.5 / **+36.0** |
| per-ep hip_r max (deg)             | 38.7, 37.4, 40.1, 37.7   | 37.5, 39.3, 38.0, 39.9    | 35.4, 36.3, **37.9, 37.2** |
| % steps within 1° of upper limit   | 0.0%                     | 0.0%                      | 0.0% |

**Both variants narrowed the gait in the right direction.** Hip ROM
fell from 63° → 57°. The peak hip flexion (the load-bearing
over-flex) fell from +40° → +37–40°, with `min_joint` showing the
tightest peak (per-ep maxes 35-38° vs baseline 37-40°). The hip min
also tightened (`pose_scale20` -23° → -17°), suggesting both
variants reduce extension overshoot in addition to flexion.

The clearest single signal is mean forward velocity. Baseline
`b4_hipopen_5M` ran at 1.40 m/s (12% over the 1.25 m/s treadmill
target); `min_joint` settles at **1.231 m/s — essentially exactly the
target**. This is consistent with the prompt's diagnosis: in the
baseline, the policy buys forward momentum from over-flexion at a
small pose-reward cost. With `min_joint` enforcing all-joint
tracking, that trade-off no longer favours over-flexion, and the
forward velocity decays to the value that the reference kinematics
actually encode.

`pose_scale20` lands in between: hip ROM the tightest of the three
(56.58°), but mean fwd vel still 1.354 m/s — the sharper mean
aggregator narrows the *shape* of the gait without fully closing the
forward-velocity loophole.

**Verdict:** both are partial wins. `min_joint` is the more
reference-faithful of the two on the metric that's hardest to fake
(`fwd_vel`); `pose_scale20` has marginally tighter hip ROM. Per
the prompt's win criterion ("hip_r ROM drops to ~43° AND fwd_vel
~1.25 AND survival 1000×4"), neither hits the ROM target. Survival
and fwd_vel are both met by `min_joint`; only survival is met by
`pose_scale20`.

### Render

```
# Eval (single source of truth):
python scripts/eval_hip_rom.py results/restart_b5_pose_scale20 results/restart_b5_min_joint results/restart_b4_hipopen_5M

# Pre-rendered:
docs/figures/restart_b5_pose_scale20.mp4   (1000-step deterministic rollout)
docs/figures/restart_b5_min_joint.mp4      (1000-step deterministic rollout)
docs/figures/restart_b4_hipopen_5M.mp4     (current best — visual baseline for A/B)

# Re-render either:
python src/walker2d/render_phase.py --eps 1 --steps 1000 --mp4 docs/figures/restart_b5_<variant>.mp4 results/restart_b5_<variant>:final:<label>
```

`render_phase.py` auto-loads the trained-against MJCF from
`env_kwargs.json`; no `--xml` needed.

### What to watch

Three mp4s for visual A/B (use the same metric-trap discipline as
[Batch 3](#batch-3--2026-04-29--overnight-19-experiment-sweep--negative-result):
trust visual + hip-ROM-from-deterministic-rollout, ignore DTW /
stride-period readouts on stationary force oscillations):

1. **`docs/figures/restart_b4_hipopen_5M.mp4` — baseline.** Rewatch
   first to refresh the over-flexed-but-walks signature. Hip flexion
   visibly past the reference's neutral leg position; mean fwd vel
   1.40 m/s.
2. **`docs/figures/restart_b5_min_joint.mp4` — leading candidate
   for new current best.** Watch the thigh peak forward flexion vs
   the baseline. Should look slightly less exaggerated. Mean fwd
   vel 1.23 m/s — the body should advance at perceptibly closer to
   "treadmill speed" rather than "running across the plane".
3. **`docs/figures/restart_b5_pose_scale20.mp4` — runner-up.**
   Tightest hip ROM of the three numerically (56.6°), but mean fwd
   vel 1.35 m/s (still 8% over). May look more visibly tracking the
   reference shape than `min_joint`, but at a faster body translation.

If `min_joint` looks visibly closer to a reference-paced walk,
**it becomes the new current best** (supersedes `b4_hipopen_5M`).
If both still look too over-flexed, the next escalation is to stack
the two: `--pose_scale 20 --min_joint_pose` (+ seed 9), which neither
of these single-knob runs tested. Beyond that, the prompt's
peaked-forward-reward fallback (`--fwd_weight 0.15
--xvel_term -1e9 --pose_weight 0.50` to make room) replaces the
survival floor with a target-velocity bell curve; this is the
ROADMAP § 0 step (3) escalation.

