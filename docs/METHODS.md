# Methods — implementation details

**Purpose:** authoritative implementation reference for the active
phase-conditioned imitation pipeline.
**Read this when:** modifying the env, the reward, RSI, BC warm-start,
or the optimizer; or looking up a CLI flag default.
**Adjacent docs:** [`REWARD_DESIGN.md`](REWARD_DESIGN.md) for the *why*
behind each reward term · [`ARCHITECTURE.md`](ARCHITECTURE.md) for the
import graph and entry points · [`reports/writeup_filled_1.docx`](reports/writeup_filled_1.docx)
for the formal-paper version.

The active code is `src/walker2d/ppo_walker2d_phase.py`. The off-policy
sibling is `src/walker2d/sac_walker2d_phase.py`; the AMP/AIRL tracks are
documented at the bottom of this file.

The current reward is the **DeepMimic-faithful baseline** that landed
during the 2026-04-28 restart (see
[`PROJECT_TIMELINE.md § Phase 5`](PROJECT_TIMELINE.md#phase-5--the-sign-error-discovery-2026-04-28)
for the why). Optional exploit-patch terms (per-joint sharpness, swing
penalty, contact-alternation, per-joint termination thresholds) are
preserved as off-by-default kwargs/CLI flags. They get re-enabled only
after a trained policy on the corrected reference shows the specific
failure they were meant to patch.

---

## Frequencies and resampling

Walker2d-v4 has `frame_skip=4`, `dt=0.002s` → control runs at **125 Hz**.
Ulrich IK is at **50 Hz**. The reference is resampled to 125 Hz on load
via `scipy.interpolate.CubicSpline` in `load_ref_cycle`. Linear interp
was the original default; the cubic spline avoids velocity-step
artifacts that hurt `r_vel`.

---

## Joint sign convention

Walker2d's joint axes are all `[0, -1, 0]`. Empirical FK probes (the
snippet in `src/diagnostics/view_reference.py`'s docstring) give:

| Joint | OpenSim convention | Walker2d convention | Flip on load |
|---|---|---|---|
| Hip   | +flexion = leg forward (anterior)  | + = foot in +x = leg forward  | **No** |
| Knee  | +flexion, range [0°, +66°]         | − = flexion, range [-150°, 0°]| **Yes** |
| Ankle | +dorsiflexion (toe up)             | + = dorsiflexion (toe up)     | **No** |

Walker2d's foot geom toes point **+x** at neutral and the gym
`forward_reward = forward_reward_weight * x_velocity` rewards motion in
**+x**. So a forward-walking heel strike has *positive* hip in both
conventions. The active loaders (`extract_gait_cycle.py`,
`ulrich_loader.py`) flip the knee only.

`src/legacy/walker2d_v1/ppo_walker2d.py` still applies the old
all-six-joint flip and is wrong for hip and ankle — preserved as a
historical record (do not extend without updating).

Joint order in the 6-vector is
`[hip_r, knee_r, ankle_r, hip_l, knee_l, ankle_l]`.

---

## Joint limits (slightly relaxed at hip)

`_JNT_LO`/`_JNT_HI` in `src/walker2d/ppo_walker2d_phase.py`:

```python
_JNT_LO = [-2.618, -2.618, -0.785, -2.618, -2.618, -0.785]   # rad
_JNT_HI = [ 0.550,  0.000,  0.785,  0.550,  0.000,  0.785]
```

Hip flexion is allowed up to **+31.5°** to cover the Ulrich reference
range (stock Walker2d caps hip at 0). Knee/ankle untouched.

---

## Phase observation

Default 25-D obs = `[base(17) | q_ref(6) | sin φ | cos φ]`.

With `--preview_k K > 1` the `q_ref` slot becomes a `(N_REF · K)` window
of upcoming reference frames (`q_ref(φ), q_ref(φ+1), …, q_ref(φ+K−1)`)
and the obs grows accordingly.

φ is normalized to a fixed module-level constant
**`GAIT_CYCLE_FRAMES = 140`** (~1.1 s @ 125 Hz), *not* to the full
reference length. With long continuous references (e.g. 7570 frames),
normalizing to `ref_len` would stretch sin/cos φ across 60 s and destroy
the within-stride signal.

The `observation_space` override happens *after* `MujocoEnv.__init__()`
because `MujocoEnv` assigns to `self.observation_space` directly (no
property setter), so a property override would `AttributeError`.

---

## Custom MJCF support

`Walker2dPhaseAware` accepts `xml_file=`. Resolution order:

1. `"walker2d.xml"` → looked up in gymnasium's MuJoCo asset dir (stock).
2. Absolute path → used as-is.
3. Bare filename → looked up in `assets/mjcf/` first, then falls back
   to `PROJECT_ROOT` for backward compatibility with older runs that
   placed the MJCF at the repo root.

The CLI `--scale_model` selects `walker2d_subject1.xml` (the
Subject-1-scaled MJCF; missing on the current checkout — see
[`PROJECT_STATUS.md § Known gaps`](PROJECT_STATUS.md#known-gaps-in-this-checkout)).

---

## Fixed-clock phase tracking

`_advance_phase` advances by exactly 1 frame per env step — no
matching, no adaptation. This replaced an earlier adaptive scheme that
let stiff-legged policies "shop" for extended-knee reference frames; a
fixed clock forces the agent to track the reference at the correct
time regardless of its current state.

---

## Reference FK pre-computation

Once at env init, every reference frame is pushed into
`mujoco.mj_kinematics` (with torso clamped at z=1.28, pitch=0) to cache
per-frame:

- root height (`_ref_root_height`)
- right/left foot x relative to root (`_ref_foot_{r,l}_xrel`)
- right/left foot z relative to root (`_ref_foot_{r,l}_zrel`)

These cached values drive the EE and root reward terms.
`--ref_root_drop d` lowers the pinned FK root height by `d` meters for
the cached root target while preserving the existing root-relative foot
targets. This is an ablation knob for stock-Walker2d contact-clearance
mismatch; default `0.0` preserves the current 1.28 m reference.

---

## Reward — DeepMimic four-term sum

```
r = w_p · r_p + w_v · r_v + w_e · r_e + w_c · r_c        + ctrl_cost
                                                         (+ optional terms)

r_p = exp(−k_p · mean_j (q_j − q_ref_j)²)              k_p = 10
r_v = exp(−k_v · mean_j (dq_j − dq_ref_j)²)            k_v = 0.1
r_e = exp(−k_e · sum_foot ((Δx)² + (Δz)²))             k_e = 40
r_c = exp(−k_c · (h − h_ref)²)                         k_c = 10
```

Defaults: `w_p = 0.65`, `w_v = 0.10`, `w_e = 0.15`, `w_c = 0.10`. Each
sub-reward is in `[0, 1]`. Per-step reward is in roughly the same
range; **no `dt` scaling** (the pre-restart code did scale by dt; the
restart matches DeepMimic Eq. 6 literally).

`ctrl_cost = -1e-3 · sum(ctrl²)` is always on — small, helps the value
baseline.

For *why* each term is shaped this way (DeepMimic adaptation, weight
choices, what fails without it), see
[`REWARD_DESIGN.md`](REWARD_DESIGN.md).

### Optional aggregator alternatives (off by default)

The pose term can be swapped from arithmetic-mean to two stricter
forms via CLI flags:

| Flag | What changes |
|---|---|
| `--product_reward` | `r_p = (∏_j exp(−k_p · w_j · diff_j²))^(1/6)` (geometric mean across joints) |
| `--min_joint_pose` | `r_p = min_j exp(−k_p · w_j · diff_j²)` (worst-joint floor) |
| `--pose_joint_weights w₁ … w₆` | Per-joint weighting `w_j` inside any of the three aggregators |

These were added during the 2026-04-29 sweep ([`RESTART_LOG.md § Batch 3`](RESTART_LOG.md))
to attack the stiff-hip-with-compliant-knee/ankle exploit; none escaped
the basin under visual review.

### Optional exploit-patch terms (off by default)

| Flag | Default | Adds to reward |
|---|---|---|
| `--swing_pen_weight w` | 0.0 | `−w · tanh(F_swing/50)` (anti toe-drag) |
| `--contact_weight w`   | 0.0 | `+w · max(0, tanh(F_stance/50) − tanh(F_swing/50))` (stance-side dominance) |
| `--energy_weight w`    | 0.0 | `−w · sum(action²)` (anti-jerk) |

Stance side per frame is precomputed at reset from reference hip angles:
`stance_right[t] = ref[t,0] >= ref[t,3]` (right hip more extended →
right is in stance). Used only when `contact_weight > 0`.

---

## Forward-velocity warm start (RSI)

`reset()` samples a phase uniformly, places `qpos[3:9]` at the
reference, sets joint velocities from the reference derivative, and
sets `qvel[0] = v_target` (default 1.25 m/s). The forward-velocity
warm-start matters because the reference kinematics are mid-stride at
1.25 m/s; without it the body lags the joints for ~50 frames every
episode.

`_ref_vel` (used for the velocity reward and the warm-start joint
velocities) is computed by padding the reference with
`[ref[-1], …, ref[0]]` before central differencing, so frame 0's
velocity is consistent with the looping cycle.

---

## Termination

Up to six checks; whichever fires first sets `info["term_cause"]` ∈
`{"height", "pitch", "ankle", "hip", "pose", "xvel", "other"}` and the
TB callback histograms it as `term/<cause>`.

| Cause | Condition | Default |
|---|---|---|
| `height` | Walker2d-v4 default `[0.8, 2.0]` root height | always on |
| `pitch`  | `\|θ\| > pitch_term_thresh` | 0.3 rad — controlled-fall guard |
| `pose`   | `max(\|hip_r\|, \|knee_r\|, \|hip_l\|, \|knee_l\|) > pose_term_thresh` | 9999 (off) |
| `ankle`  | `max(\|ankle_r\|, \|ankle_l\|) > ankle_term_thresh` | 9999 (off) |
| `hip`    | `max(\|hip_r\|, \|hip_l\|) > hip_term_thresh` | 9999 (off) |
| `xvel`   | `x_vel < xvel_term_thresh` | −∞ (off) |

The pitch guard is the only added termination that's on by default.
Without it the agent learns a controlled forward fall — the height
bound only fires *after* the lean is irrecoverable. Each of the
optional terminations targets a specific exploit (see
[`REWARD_DESIGN.md`](REWARD_DESIGN.md)).

---

## Optimizer schedule

From-scratch runs: linear LR decay **3e-4 → 3e-5** over training,
`ent_coef=0.005`, `target_kl=0.015`. PPO with `n_steps=512`,
`batch_size=4096`, `n_epochs=10`, `gamma=0.99`, `gae_lambda=0.95`,
`clip_range=0.2`, `vf_coef=0.5`, `max_grad_norm=0.5`,
`policy_kwargs={"net_arch": [256, 256]}`.

**Finetune mode (`--finetune <model.zip>`):** drops `learning_rate=1e-5`,
`ent_coef=0`, `target_kl=0.005`. Conservative — pre-restart canonical
runs were sensitive to larger updates. BC warm-start is skipped when
finetuning.

The SAC sibling (`sac_walker2d_phase.py`) uses `learning_rate=3e-4`,
`batch_size=256`, `buffer_size=300_000`, 1 env, 1M total steps.

---

## Behavioral cloning warm-start (`--bc_epochs N`)

Optional pretraining stage. `compute_bc_dataset` rolls out a PD tracking
controller (Kp=200, Kd=20 in torque space, normalized by gear) inside the
actual MuJoCo simulation, collecting (obs, action) pairs that are
*physically consistent with ground contact* — unlike `mj_inverse`, which
ignores contact forces and produces wrong torques during stance.

`pretrain_bc` then does supervised MSE on `π_mean(obs) → action` for N
epochs (LR drops to lr/10 in the second half).

| Flag | Default | Description |
|---|---|---|
| `--bc_epochs` | 0 | If >0, run BC warm-start before PPO |
| `--bc_steps`  | 200_000 | PD-rollout samples to collect |
| `--bc_kp` / `--bc_kd` | 200 / 20 | PD gains (torque space, divided by gear) |
| `--bc_only`   | off | Stop after BC, save BC-only model, skip PPO |

Mutually exclusive with `--finetune` (BC is skipped if finetuning).

---

## TensorBoard logging

`PPO` is constructed with `tensorboard_log=str(log_dir / "tb")` (disable
with `--no_tb`). Beyond SB3's built-in `train/*` and `rollout/*`
scalars, `LogCallback` records per-rollout:

- `reward/{r_pose, r_vel, r_ee, r_root, contact_r, swing_pen, ctrl_cost, energy_pen}`
  — mean per-component value across all env steps in the rollout.
- `term/{height, pitch, ankle, hip, pose, xvel, other}` — count of
  episodes in the rollout that ended for each termination cause.

This is the lowest-cost diagnostic for catching saturated terms and
failure-mode shifts during training. Open with
`tensorboard --logdir results/<run-dir>/tb`.

---

## Checkpoint cadence

`CheckpointCallback` saves every `1_000_000 // num_envs` per-env steps
→ a checkpoint roughly every **1M total env steps**. At the default
`--num_envs 16` that's every ~62.5k per-env steps.

---

## Reference cycle details

Active reference: Subject 1, baseline trial, 1.25 m/s, extracted as a
single clean stride: **56 frames @ 50 Hz → 140 frames @ 125 Hz** (cubic
spline, ~1.12 s).

The on-disk artifact in `assets/reference/gait_cycle_reference.npy` is
the 56-frame, 50 Hz version (so it's diagnosable with low-frequency
tools). Each training run resamples it to 140 frames @ 125 Hz at
env-init time and saves *that* array as `<run-dir>/reference.npy`, so
`render_phase.py` reproduces the exact array the training env saw.

`GAIT_CYCLE_FRAMES = 140` is the constant used to normalize sin/cos φ
in the observation. It happens to coincide with the post-resample
length of the current single-cycle reference, so for this reference
the encoding wraps exactly once per cycle. With the long full-trial
reference (`--ref_all`, 7570 frames @ 125 Hz ≈ 60 s) the encoding
wraps every ~1.1 s instead of every 60 s — which is what makes
sin/cos φ a useful within-stride signal regardless of reference
length.

---

## Full CLI reference (`ppo_walker2d_phase.py`)

The authoritative source is `python src/walker2d/ppo_walker2d_phase.py --help`.
The table below is a snapshot.

### Reference + run setup

| Flag | Default | Description |
|------|---------|-------------|
| `--ref_cycle` | — | Path to gait cycle `.npy` (mutex with `--ref_all`) |
| `--ref_all`   | — | Use full concatenated Ulrich reference (mutex with `--ref_cycle`) |
| `--subjects`  | None | Subjects to include in `--ref_all` |
| `--trial_filter` | None | Substring filter for trial names in `--ref_all` |
| `--num_envs`  | 16 | Parallel environments |
| `--total_steps` | 5e6 | Total env steps |
| `--device`    | cpu | PPO device (cpu recommended for MLP) |
| `--seed`      | 0 | RNG seed for SB3 + env |
| `--scale_model` | off | Use `walker2d_subject1.xml` (Subject-1-scaled MJCF) |
| `--finetune`  | None | Pretrained `.zip` to finetune from (lr→1e-5, entropy→0, target_kl→0.005) |
| `--no_tb`     | off | Disable TensorBoard logging |
| `--out_dir`   | None | Override output directory (default: `results/<auto-stamped>`) |

### BC warm-start (skipped if `--finetune`)

| Flag | Default | Description |
|------|---------|-------------|
| `--bc_epochs` | 0 | BC warm-start before PPO |
| `--bc_steps`  | 200_000 | PD-rollout samples |
| `--bc_kp` / `--bc_kd` | 200 / 20 | PD gains |
| `--bc_only`   | off | Stop after BC, save BC-only model |

### Reward weights and exp scales

| Flag | Default | Term |
|------|---------|------|
| `--pose_weight` | 0.65 | `r_p` weight |
| `--vel_weight`  | 0.10 | `r_v` weight |
| `--ee_weight`   | 0.15 | `r_e` weight |
| `--root_weight` | 0.10 | `r_c` weight |
| `--pose_scale`  | 10.0 | `k_p` |
| `--vel_scale`   | 0.1  | `k_v` |
| `--ee_scale`    | 40.0 | `k_e` |
| `--root_scale`  | 10.0 | `k_c` |
| `--ref_root_drop` | 0.0 | Lower cached reference root target by this many meters |

### Aggregator + exploit-patch flags (off by default)

| Flag | Default | Description |
|------|---------|-------------|
| `--pose_joint_weights w₁ … w₆` | `1 1 1 1 1 1` | Per-joint pose weights inside `mean(w · diff²)` |
| `--product_reward` | off | Geometric mean of per-joint exps for `r_p` |
| `--min_joint_pose` | off | Worst-joint floor for `r_p` |
| `--swing_pen_weight` | 0.0 | Direct swing-foot contact penalty |
| `--contact_weight`   | 0.0 | Stance-side foot dominance reward |
| `--energy_weight`    | 0.0 | `sum(action²)` penalty |
| `--preview_k`        | 1 | Frames of upcoming `q_ref` to expose in obs |

### Termination thresholds

| Flag | Default | Cause |
|------|---------|-------|
| `--pitch_term` | 0.3 | `pitch` |
| `--pose_term`  | 9999 | `pose` (off) |
| `--ankle_term` | 9999 | `ankle` (off) |
| `--hip_term`   | 9999 | `hip` (off) |
| `--xvel_term`  | -∞ | `xvel` (off) |
| `--v_target`   | 1.25 | Treadmill speed for warm-start qvel |

The `--xvel_term 0.3` recipe is what produced the current best policy
(`results/restart_b2_xvel/`); see
[`RESTART_LOG.md § Batch 2`](RESTART_LOG.md#batch-2--2026-04-28--escape-the-stand-still-basin).

---

## Adversarial-imitation tracks: AMP and AIRL (Brian's track)

Both `src/walker2d/amp_walker2d.py` and `src/walker2d/airl_walker2d.py`
share the env, BC helpers, and reference loader from
`ppo_walker2d_phase.py`. They differ in *how the reward signal is
produced*: a discriminator scores transitions `(s, s′)` against an
expert buffer built from consecutive IK reference frames. The
hand-crafted DeepMimic reward is set to 0 (AIRL) or partly retained
(AMP — see below).

### Shared design

- **Expert buffer** (`make_expert_buffer` in `airl_walker2d.py`):
  consecutive IK frames are paired into `(s_t, s_{t+1})` rows. For
  `--ref_cycle` the wraparound pair `(s_{T-1}, s_0)` is included
  because the cycle loops cleanly. For `--ref_all`, trial-boundary
  transitions are excluded *only when the loader reports per-trial
  segment lengths* — the active loader does not, so `--ref_all`
  currently includes a small number of inter-trial boundary rows.
- **State features** (`extract_airl_state`):
  `[q_joint(6), dq_joint(6)] = 12-dim` by default.
  `--no_joint_vel` ablates to 6-dim positions-only. Phase is
  *deliberately omitted* — the (s, s′) transition structure encodes
  gait sequencing implicitly, and adding redundant phase features
  collapsed the expert manifold to identical zeros in earlier runs.
- **Per-rollout loop** (in the SB3 callback, `_on_rollout_end`):
  1. Pull policy `(s, s′)` rows from the rollout buffer, masking out
     transitions where `episode_starts[t+1]=True` (terminals).
  2. Update the discriminator on a balanced expert / policy batch.
  3. Rewrite `rollout_buffer.rewards` with the discriminator signal.
     PPO's downstream advantage / value updates then optimise the
     policy against this learned reward.

### AIRL specifics (`airl_walker2d.py`)

- **Discriminator architecture:**
  `g(s, s′) = f(s, s′) + γ · h(s′) − h(s)` where `f` and `h` are
  separate Tanh MLPs (default hidden 256). The shaping potential `h`
  cancels environment dynamics so the recovered reward is dynamics-
  invariant.
- **Loss:** binary cross-entropy with label smoothing 0.1
  (expert→0.9, policy→0.1); WGAN-GP gradient penalty (default 10) on
  interpolated expert↔policy samples; Gaussian noise (`--expert_noise
  0.05`) added to expert (s, s′) before each disc step to blur the
  140-frame manifold.
- **Reward:** `r = g(s, s′)` directly, with running-window
  normalisation (50k samples) so PPO's advantage scale stays stable.
- **Adaptive freeze** (`--min_frac_expert 0.05`): if the fraction of
  policy transitions the disc scores expert-like drops below the
  floor, the disc is frozen for that rollout — the policy gets a
  chance to catch up before the disc trains further. Directly
  targets the runaway-discriminator collapse.
- **Cold-start failure:** without `--finetune`, the discriminator
  reaches near-perfect separation before the policy learns to walk
  and gradients vanish. The recommended setup is `--finetune <ppo
  walker>.zip`. A small `--loco_bonus` (default 0.05 ·
  max(0, x_vel)) is layered on the env reward as an additional
  cold-start cushion.
- **Optional BC pretrain** (`--bc_epochs`): same PD-rollout dataset
  used by `ppo_walker2d_phase.py`, applied to the policy before AIRL
  begins.

### AMP specifics (`amp_walker2d.py`)

- **Discriminator architecture:** ELU MLP `[1024, 512, 1]` matching
  Escontrela et al. — purely `D(s, s′)` with no shaping potential.
- **Loss (LSGAN, paper Eq. 3):**
  `L = E_E[(D − 1)²] + E_π[(D + 1)²] + (w_gp/2) · E_E[‖∇_φ D‖²]`.
  The gradient penalty is **zero-centered on expert samples only**
  (not interpolated), penalising non-zero gradients on the data
  manifold — different from AIRL's WGAN-GP.
- **Style reward (paper Eq. 4):**
  `r_s = max(0, 1 − 0.25 · (D(s, s′) − 1)²) ∈ [0, 1]`. Bounded by
  construction, so no running normalisation is needed.
- **Combined reward:**
  `r = 0.35 · r_task + 0.65 · r_style` (paper weights `w_g`, `w_s`).
  The task term is `r_task = exp(-5 · (v_x − v_target)²)` (default
  `v_target = 1.25 m/s`). Critically, the task reward is **scaled,
  not zeroed** — this keeps a usable gradient alive from step 1, so
  AMP from-scratch is feasible without `--finetune` (whereas AIRL is
  not).
- **Disc batch sizing:** `--disc_batch_size 4096` is enforced as a
  cap on each gradient step. Building the gradient-penalty graph
  with `create_graph=True` over the full rollout (~16k samples on
  CPU) crashed at the C level; sub-batching avoids this.
- **Expert noise:** same 0.05-std augmentation as AIRL, motivated by
  the same 140-frame memorisation risk.

### Why both tracks collapse at 8-env CPU scale

In short: the expert manifold is tiny (140 cycle transitions) and the
policy's per-rollout transition count at 8 envs × 512 steps ≈ 4k
already gives the discriminator more than enough capacity to memorise
the expert set within the first few updates. Once `frac_expert`
crashes to 0 the reward signal collapses and the policy stops
improving. Mitigations layered into the current code (label smoothing,
expert noise, gradient penalty, AIRL's adaptive freeze, AMP's bounded
reward + retained task term) help but do not eliminate the failure;
the writeup §6.3 conclusion is that the right unblock is MJX/GPU
parallelism (4,000+ envs → ~2M policy transitions per rollout, which
swamps the 140-frame expert buffer and forces the discriminator to
generalise instead of memorise). The 2026-04-29 overnight Batch 3
also tested AMP/AIRL warm-started from a working PPO policy — that
prevents immediate cold-start collapse but still pushes the policy
into asymmetric kicks rather than natural gait. See
[`ROADMAP.md § 1`](ROADMAP.md#1-mujoco-mjx--gpu-port-for-amp-writeup-71).

---

## Diagnostic scripts (`src/diagnostics/`)

Standalone, not imported by the training pipeline. Run from the project root.

| Script | Purpose |
|---|---|
| `diag_cycle.py` | Plot 3 looped gait cycles + print per-joint discontinuity at the seam. Writes `docs/figures/cycle_continuity.png`. |
| `diag_ref.py`   | Print per-joint reference ranges and run open-loop FK at fixed pitch to confirm the reference stays upright. |
| `diag_walker_mass.py` | Dump Walker2d per-body masses (total ≈ 23.68 kg) and the scale factor for comparing to a 75 kg subject. |
| `extract_osim_mass.py` | Parse `*.osim` XML to pull per-subject total body mass. Used for BW-normalized GRF comparison. |
| `view_reference.py` | MuJoCo-viewer playback of the on-disk gait cycle on a Walker2d skeleton (`qpos[3:9] = ref[t]`, body translated at 1.25 m/s). The original sign-error discovery tool. |
| `compare_tb.py` | Side-by-side TensorBoard scalar comparison across runs. |
| `extract_reference_biomech.py` | Compute *measured* biomech targets from a subject's GRF + IK + scaled OpenSim model. Writes `assets/reference/biomech_targets.json` (stride period, cadence, double-support, peak vGRF/BW, per-joint ROM) + `.vgrf_curves.npz` (normalised stance-phase vGRF curves). Run once per subject/trial. Defaults: Subject 1, `walking_baseline1`. |
| `eval_biomech.py` | Held-out biomech metrics for a checkpoint: stride period, cadence, double-support fraction, peak vGRF (BW-normalised, per foot), swing-drag fraction, L-R stride asymmetry, hip-knee phase-plane DTW vs the reference cycle, all-six-joint DTW, and per-joint ROM. With `biomech_targets.json` present, also emits a `vs_reference` block (`delta` and `pct_err` per metric) plus a `progress_score` in [0, 4]. `--csv` appends one row per eval to a history file. Use this — not training reward — to grade reward variants. |
| `scripts/biomech_report.py` | Convert one or more `eval_biomech` JSONs into (a) a markdown comparison table and (b) a 6-panel figure overlaying sim hip/knee/ankle traces and stance-vGRF curve on the Ulrich reference. Defaults write to `docs/figures/biomech_report.{md,png}`. `--rerollout` adds policy traces to the figure (slower); without it the figure shows reference + per-run bars only. |

### Held-out biomechanical evaluation: the two-tool flow

The validation loop the agent should run after every batch:

```
python src/diagnostics/extract_reference_biomech.py            # once per subject/trial
python src/diagnostics/eval_biomech.py --xml walker2d.xml --eps 6 --steps 2500 \
    results/<run>:final:<label> --out results/<run>_eval.json --csv results/biomech_history.csv
python scripts/biomech_report.py results/<run>_eval.json --rerollout
```

The `vs_reference` block in `<run>_eval.json` is the answer to "are we
making progress" — every metric has a measured Subject 1 target and a
`pct_err`. The `progress_score` is a single 0–4 number so an agent can
compare runs without interpreting nine metrics. The targets are
*measured*, not bibliographic — they come from the same Ulrich force
plates and IK files our reference cycle was extracted from.

> **Anti-Goodhart caveat:** the 2026-04-29 sweep
> ([`RESTART_LOG.md § Batch 3`](RESTART_LOG.md#batch-3--2026-04-29--overnight-19-experiment-sweep--negative-result))
> showed `progress_score` and `hip_knee_dtw` can both flatter a
> stand-and-wiggle policy where one valid stride matches the
> reference but the body barely translates. Pair these metrics with
> `hip_r_rom_deg`, `cadence`, and visual review.

---

## Overnight scaffolding (`scripts/overnight/`)

Used by the 2026-04-29 sweep ([`RESTART_LOG.md § Batch 3`](RESTART_LOG.md#batch-3--2026-04-29--overnight-19-experiment-sweep--negative-result)).
Not on the training path; useful when running multi-experiment sweeps
with consistent artefact contracts.

| Script | Purpose |
|---|---|
| `scripts/overnight/run_experiment.py` | Wrapper: train → eval_biomech → preview.mp4 → meta JSON. The standard way to launch sweep experiments. |
| `scripts/overnight/rank_runs.py`      | Composite-score ranking across all runs in a sweep dir. |
| `scripts/overnight/write_report.py`   | Fill `REPORT.md` from a run's eval JSON (template at `REPORT_TEMPLATE.md`). |
| `scripts/overnight/STATUS_TEMPLATE.md` / `REPORT_TEMPLATE.md` | Templates for sweep status and per-run reports. |
