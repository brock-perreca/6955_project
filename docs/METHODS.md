# Methods — implementation details

Implementation-level reference for the active phase-conditioned
imitation pipeline (`src/walker2d/ppo_walker2d_phase.py`). For the formal
methods description (problem statement, reward formula, hypothesis
labels), see [`reports/writeup_filled_1.docx`](reports/writeup_filled_1.docx).
For the *reasoning* behind each reward term and the failure modes that
motivate it, see [`REWARD_DESIGN.md`](REWARD_DESIGN.md).

> **2026-04-28 restart — read this before trusting the reward,
> termination, BC, or per-joint-scaling sections below.** Those
> sections describe the engineered reward that was tuned against the
> corrupted reference (see
> [`PROJECT_TIMELINE.md` § Phase 5](PROJECT_TIMELINE.md#phase-5--the-sign-error-discovery-2026-04-28)).
> The active code in `src/walker2d/ppo_walker2d_phase.py` was simplified
> back to a DeepMimic-faithful baseline on 2026-04-28 — sum of four
> `exp(−k·err²)` terms, no per-joint sharpness/weights, no
> swing_pen/contact_r by default, no per-joint pose/ankle termination.
> Reward components live as code constants and CLI flags; see the
> module docstring of `ppo_walker2d_phase.py` for the current spec and
> [`RESTART_LOG.md`](RESTART_LOG.md) for what's been tried since the
> restart. The headings below are kept for historical context — they
> document the pre-restart engineered reward, which the active code no
> longer uses by default.

---

## Frequencies and resampling

Walker2d-v4 has `frame_skip=4`, `dt=0.002s` → control runs at **125 Hz**.
Ulrich IK is at **50 Hz**. The reference is resampled to 125 Hz on load
via `scipy.interpolate.CubicSpline` in `load_ref_cycle`
(`src/walker2d/ppo_walker2d_phase.py`). Linear interp was the original
default; the cubic spline avoids the velocity-step artifacts that hurt
`vel_r`. The Phase 1 runs *forgot to resample*, which played the gait
2.5× too fast — one of three root causes of the original failure (see
[`PROJECT_TIMELINE.md`](PROJECT_TIMELINE.md)).

---

## Joint sign convention

> **2026-04-28 — this section was wrong, and it took down every
> training run that trusted it.** See
> [`PROJECT_TIMELINE.md § Phase 5`](PROJECT_TIMELINE.md#phase-5--the-sign-error-discovery-2026-04-28)
> for the discovery story. The corrected facts are below; the source
> files (`extract_gait_cycle.py`, `ulrich_loader.py`,
> `src/legacy/walker2d_v1/ppo_walker2d.py:91-109`) still apply
> `walker = -opensim` to all six joints and are now known to be wrong
> for hip and ankle.

Walker2d's joint axes are all `[0, -1, 0]`. Empirical FK probes (run
the snippet in `src/diagnostics/view_reference.py`'s docstring or
`mj_kinematics` directly) give:

| Joint | OpenSim convention | Walker2d convention | Flip needed? |
|---|---|---|---|
| Hip   | +flexion = leg forward (anterior)  | + = foot in +x = leg forward  | **No** |
| Knee  | +flexion, range [0°, +66°]         | − = flexion, range [-150°, 0°]| **Yes** |
| Ankle | +dorsiflexion (toe up)             | + = dorsiflexion (toe up)     | **No** |

Walker2d's foot geom toes point **+x** at neutral and the gym
`forward_reward = forward_reward_weight * x_velocity` rewards motion in
**+x**. So a forward-walking heel strike has *positive* hip in both
conventions. The `walker = -opensim` flip is correct only for the
knee. Applying it to hip and ankle inverts the gait (heel-strike pose
becomes toe-off pose; dorsiflexion becomes plantarflexion).

Joint order in the 6-vector is
`[hip_r, knee_r, ankle_r, hip_l, knee_l, ankle_l]`.

---

## Joint limits (slightly relaxed at hip)

`_JNT_LO`/`_JNT_HI` in `src/walker2d/ppo_walker2d_phase.py`:

```python
_JNT_LO = [-2.618, -2.618, -0.785, -2.618, -2.618, -0.785]   # rad
_JNT_HI = [ 0.349,  0.000,  0.785,  0.349,  0.000,  0.785]
```

Hip extension is allowed up to **+20°** to cover the Ulrich range
(stock Walker2d caps hip at 0). Knee/ankle untouched.

---

## Phase observation

25-D obs = `[base(17) | q_ref(6) | sin φ | cos φ]`.

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

The env now takes an `xml_file=` kwarg (CLI `--scale_model` selects
`walker2d_subject1.xml`, otherwise stock `walker2d.xml`). Because
`Walker2dEnv.__init__` hardcodes `"walker2d.xml"`, the env replicates
Walker2d's attribute setup (`_forward_reward_weight`, `_healthy_z_range`,
etc.) and calls `MujocoEnv.__init__` directly. See
[`ARCHITECTURE.md § XML resolution`](ARCHITECTURE.md#xml-resolution-walker2dphaseawareinit)
for resolution order.

---

## Fixed-clock phase tracking (`_advance_phase`)

Phase advances by exactly 1 frame per env step — no matching, no
adaptation. This replaced the earlier adaptive scheme that searched
forward up to `max_phase_advance` frames for the best-matching reference
pose. The adaptive version let stiff-legged policies "shop" for
extended-knee reference frames and never learn swing flexion; a fixed
clock forces the agent to track the reference at the correct time
regardless of its current state.

The `--max_phase_advance` CLI flag is still accepted but no longer
consumed by the reward loop.

---

## Reference FK pre-computation (`_precompute_reference_kinematics`)

Once at env init, every reference frame is pushed into
`mujoco.mj_kinematics` (with torso clamped at z=1.28, pitch=0) to cache
per-frame:

- root height (`_ref_root_height`)
- right/left foot x relative to root (`_ref_foot_{r,l}_xrel`)
- right/left foot world-z (`_ref_foot_{r,l}_z`)

These cached values drive the EE and root reward terms. **Note:** an
earlier draft used foot world-z directly; the in-file comment correction
explains that root-relative z is what the EE reward needs because
world-z is ~0 at stance and useful as a swing-clearance signal — but
the *current* code stores root-relative z and the EE reward consumes it
as such.

---

## Stance side detection

Pre-computed at reset from reference hip angles:
`stance_right[t] = ref[t,0] >= ref[t,3]` (right hip more extended than
left → right is in stance). Used by the contact alternation reward.

---

## Reward (weighted sum, per-joint scaled)

Each sub-reward is in `[0, 1]` and combined as a weighted sum, scaled by
`dt` so returns are time-invariant. Per-joint scales/weights live as
module-level constants in `src/walker2d/ppo_walker2d_phase.py`:

```python
_JSCALE   = [10, 20, 40, 10, 20, 40]      # k_hip=10, k_knee=20, k_ankle=40
_JWEIGHTS = [0.4, 1.0, 2.5, 0.4, 1.0, 2.5]  # ankle weighted highest
_KVSCALE  = [0.05, 0.1, 0.2, 0.05, 0.1, 0.2]  # tighter vel scale on ankles
```

Sharper k on the ankle reflects that heel-strike timing matters far more
than hip posture — k=40 → 50% reward at ~7°, k=10 → 50% reward at ~15°.

For the per-component formulas, defaults, and the rationale for each
term, see [`REWARD_DESIGN.md`](REWARD_DESIGN.md).

`--product_reward` switches `imit_r` from arithmetic to geometric mean
of per-joint exps.

---

## Periodic velocity computation

`_ref_vel` is computed by padding the reference with
`[ref[-1], …, ref[0]]` before central differencing, so frame 0's velocity
is consistent with the looping cycle. Without the wrap, `np.gradient`'s
one-sided edge difference produced a velocity discontinuity at the seam.

---

## Forward-velocity warm start (RSI fix)

`reset()` sets `qvel[0] = self._v_target` when warm-starting from a
reference frame. Previously `qvel[0]` defaulted to ~0 even though joints
were placed mid-stride at 1.25 m/s reference kinematics. That mismatch
caused both feet to land simultaneously (standing pattern) for the first
~50 frames of every episode and was a major contributor to poor early
tracking.

---

## Termination

Four checks, any of which ends the episode. The termination cause for
the step that ended an episode is exposed via `info["term_cause"]` ∈
`{"height", "pitch", "ankle", "pose", "xvel"}` and is histogrammed by
the TB callback as `term/<cause>`.

1. **Root height out of `[0.8, 2.0]`** (inherited from Walker2d-v4's
   default `super().step`). Cause: `height`.
2. **Pitch magnitude > 0.3 rad (~17°).** Added because without it the
   agent learns a controlled forward fall — height only drops below 0.8
   *after* the lean becomes irrecoverable. Cause: `pitch`.
3. **Hip/knee `pose_term`** (0.9 rad, cause: `pose`) and **ankle**
   `ankle_term` (0.40 rad, cause: `ankle`). The asymmetry — ankle is the
   *tighter* threshold — exists because the agent will exploit large
   plantarflexion for hopping if you let the ankle drift as far as
   hip/knee.
4. **`x_vel < -0.1`** (moving backwards). Cause: `xvel`.

---

## Optimizer schedule

From-scratch runs use a linear LR decay **3e-4 → 3e-5** over training
(`lr_schedule(progress_remaining)`), `ent_coef=0.005`, `target_kl=0.015`.
The decay prevents large destabilizing updates once the policy finds a
good gait. Replaced the earlier flat `1e-4` learning rate.

**Finetune mode (`--finetune`):** `learning_rate=1e-5`, `ent_coef=0`,
`target_kl=0.005`. Much more conservative than the original `3e-5`
finetune setting — the canonical s1scaled run is sensitive to larger
updates.

---

## Behavioral cloning warm-start (`--bc_epochs N`)

Optional pretraining stage. `compute_bc_dataset` rolls out a PD tracking
controller (Kp=200, Kd=20 in torque space, normalized by gear) inside the
actual MuJoCo simulation, collecting (obs, action) pairs that are
*physically consistent with ground contact* — unlike `mj_inverse`, which
ignores contact forces and produces wrong torques during stance.

`pretrain_bc` then does supervised MSE on `π_mean(obs) → action` for N
epochs (LR drops to lr/10 in the second half).

Flags:

| Flag | Default | Description |
|---|---|---|
| `--bc_epochs` | 0 | If >0, run BC warm-start before PPO |
| `--bc_steps` | 200_000 | PD-rollout samples to collect |
| `--bc_kp` / `--bc_kd` | 200 / 20 | PD gains (torque space, divided by gear) |
| `--bc_only` | off | Stop after BC, save BC-only model, skip PPO |

Mutually exclusive with `--finetune` (BC is skipped if finetuning).

---

## TensorBoard logging

`PPO` is constructed with `tensorboard_log=str(log_dir / "tb")` (disable
with `--no_tb`). Beyond SB3's built-in `train/*` and `rollout/*`
scalars, `LogCallback` records per-rollout:

- `reward/{imit_r, vel_r, ee_r, root_r, contact_r, swing_pen, ctrl_cost}`
  — mean per-component value across all env steps in the rollout.
- `term/{height, pitch, ankle, pose, xvel, other}` — count of episodes
  in the rollout that ended for each termination cause.

This is the lowest-cost diagnostic for catching saturated terms and
failure-mode shifts during training. Open with
`tensorboard --logdir results/<run-dir>/tb`.

---

## Checkpoint cadence

`CheckpointCallback` saves every ~5M env steps (was 500k earlier).
At 32 envs this means a checkpoint every ~156k per-env steps.

---

## Reference cycle details (per writeup §5.2)

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

| Flag | Default | Description |
|------|---------|-------------|
| `--ref_cycle` | — | Path to gait cycle `.npy` (required unless `--ref_all`) |
| `--ref_all` | — | Use full concatenated Ulrich reference (mutex with `--ref_cycle`) |
| `--subjects` | None | Subjects to include in `--ref_all` |
| `--trial_filter` | None | Substring filter for trial names in `--ref_all` |
| `--num_envs` | 32 | Parallel environments |
| `--total_steps` | 5e6 | Total env steps |
| `--device` | cpu | PPO device (cpu recommended for MLP) |
| `--scale_model` | off | Use `walker2d_subject1.xml` (Subject-1-scaled MJCF) |
| `--finetune` | None | Pretrained `.zip` to finetune from (lr→1e-5, entropy→0, target_kl→0.005) |
| `--bc_epochs` | 0 | BC warm-start before PPO (skipped when finetuning) |
| `--bc_steps` | 200000 | PD-rollout samples for BC dataset |
| `--bc_kp` / `--bc_kd` | 200 / 20 | PD gains for BC data collection |
| `--bc_only` | off | Stop after BC, save BC-only model, skip PPO |
| `--imit_weight` | 4.0 | Per-joint pose tracking weight |
| `--vel_weight` | 1.0 | Per-joint velocity tracking weight |
| `--ee_weight` | 4.0 | End-effector (foot x + z) tracking weight |
| `--root_weight` | 2.0 | Root height tracking weight (height-only after 2026-04-28) |
| `--contact_weight` | 1.0 | Stance-side foot contact alternation weight |
| `--swing_pen_weight` | 2.0 | Penalty on swing-foot ground contact (anti toe-drag) |
| `--v_target` | 1.25 | Treadmill speed (m/s) used by warm-start qvel |
| `--product_reward` | off | Pose term as geometric mean (default arithmetic) |
| `--max_phase_advance` | 4 | (Inert) accepted but unused after fixed-clock switch |
| `--pose_term` | 0.9 rad | Hip/knee deviation termination threshold |
| `--ankle_term` | 0.40 rad | Ankle deviation termination threshold |
| `--no_pose_term` | off | Disable pose termination (sets pose_term=9999; ankle_term still applies) |
| `--no_tb` | off | Disable TensorBoard logging (default: write to `<log_dir>/tb`) |
| `--out_dir` | None | Override output directory (default: `results/<auto-stamped>`) |

> **Removed flags (2026-04-28).** `--peak_bonus_weight`, `--fwd_weight`,
> `--action_rate_weight` were default-0 and never enabled in any
> canonical run. Removed during a reward-cleanup pass; restore from git
> history if you need them for an ablation.

---

## Adversarial-imitation tracks: AMP and AIRL (Brian's track)

Both `src/walker2d/amp_walker2d.py` and `src/walker2d/airl_walker2d.py`
share the env, BC helpers, and reference loader from
`ppo_walker2d_phase.py`. They differ from the engineered-reward track
in *how the reward signal is produced*: a discriminator scores
transitions `(s, s′)` against an expert buffer built from consecutive
IK reference frames. The hand-crafted DeepMimic reward is set to 0.

### Shared design (both tracks)

- **Expert buffer** (`make_expert_buffer` in `airl_walker2d.py`):
  consecutive IK frames are paired into `(s_t, s_{t+1})` rows. For
  `--ref_cycle` the wraparound pair `(s_{T-1}, s_0)` is included
  because the cycle loops cleanly. For `--ref_all`, trial-boundary
  transitions are excluded *only when the loader reports per-trial
  segment lengths* — the active loader does not, so `--ref_all`
  currently includes a small number of inter-trial boundary rows.
- **State features** (`extract_airl_state`):
  `[q_joint(6), dq_joint(6)] = 12-dim` by default.
  `--no_joint_vel` ablates to the 6-dim positions-only form. Phase is
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
  walker>.zip`, which already produces transitions that overlap
  with the expert manifold. A small `--loco_bonus` (default 0.05 ·
  max(0, x_vel)) is layered on the env reward as an additional
  cold-start cushion.
- **Optional BC pretrain** (`--bc_epochs`): same PD-rollout dataset
  used by `ppo_walker2d_phase.py`, applied to the policy before AIRL
  begins.

### AMP specifics (`amp_walker2d.py`)

- **Discriminator architecture:** ELU MLP `[1024, 512, 1]` matching
  Escontrela et al. — purely `D(s, s′)` with no shaping potential.
  AMP isn't trying to recover a transferable reward; it just needs a
  useful style signal.
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
policy's per-rollout transition count at 8 envs × 512 steps ≈ 4 k
already gives the discriminator more than enough capacity to memorise
the expert set within the first few updates. Once `frac_expert`
crashes to 0 the reward signal collapses and the policy stops
improving. Mitigations layered into the current code (label smoothing,
expert noise, gradient penalty, AIRL's adaptive freeze, AMP's bounded
reward + retained task term) help but do not eliminate the failure;
the writeup §6.3 conclusion is that the right unblock is MJX/GPU
parallelism (4,000+ envs → ~2M policy transitions per rollout, which
swamps the 140-frame expert buffer and forces the discriminator to
generalise instead of memorise).

---

## Diagnostic scripts (`src/diagnostics/`)

Standalone, not imported by the training pipeline. Run from the project root.

| Script | Purpose |
|---|---|
| `diag_cycle.py` | Plot 3 looped gait cycles + print per-joint discontinuity at the seam. Writes `docs/figures/cycle_continuity.png`. |
| `diag_ref.py` | Print per-joint reference ranges and run open-loop FK at fixed pitch to confirm the reference stays upright. |
| `diag_walker_mass.py` | Dump Walker2d per-body masses (total ≈ 23.68 kg) and the scale factor for comparing to a 75 kg subject. |
| `extract_osim_mass.py` | Parse `*.osim` XML to pull per-subject total body mass. Used for BW-normalized GRF comparison. |
| `eval_biomech.py` | Held-out biomech metrics for a checkpoint: stride period, cadence, double-support fraction, peak vGRF (BW-normalised, per foot), swing-drag fraction, L-R stride asymmetry, and a hip-knee phase-plane DTW distance vs the reference cycle. Run on a deterministic rollout; usage matches `render_phase.py` (`run_dir:ckpt[:label]`). Writes JSON. Use this — not training reward — to grade reward variants. |
