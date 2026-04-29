# Reward design + exploit taxonomy

**Purpose:** the *why* behind every term in the imitation reward, and a
catalogue of the failure modes each term closes (or fails to close).
**Read this when:** adding/removing a reward term, debugging a policy
that scores well numerically but looks wrong on video, or writing the
methods/results section.
**Adjacent:** [`METHODS.md`](METHODS.md) for the implementation-level
formula and CLI flags · [`RESTART_LOG.md`](RESTART_LOG.md) for the
recent batches that probed the current reward · DeepMimic
([`papers/Peng_2018_DeepMimic.pdf`](papers/Peng_2018_DeepMimic.pdf))
for the original tracking-reward design.

The current reward is a Walker2d-on-IK adaptation of DeepMimic. After
the 2026-04-28 restart it was simplified back to the literal DeepMimic
4-term sum; exploit-patches that the older engineered reward used to
ship with are preserved as off-by-default knobs. This file is organised
around (1) the current default reward, (2) the optional patches with
the exploit each was built to close, and (3) the **structural
stiff-hip trap** the 2026-04-29 sweep diagnosed in this reward family.

---

## Current default reward (DeepMimic 4-term sum)

Per-step reward, in roughly `[0, 1]` (no `dt` scaling):

```
r = 0.65 · r_p + 0.10 · r_v + 0.15 · r_e + 0.10 · r_c   − 1e-3 · ‖ctrl‖²

r_p = exp(−10 · mean_j (q_j − q_ref_j)²)         pose tracking, 6 joints
r_v = exp(−0.1 · mean_j (dq_j − dq_ref_j)²)      velocity tracking
r_e = exp(−40 · sum_foot ((Δx)² + (Δz)²))        end-effector (root-relative)
r_c = exp(−10 · (h − h_ref)²)                    root height tracking
```

Each term targets a biomechanical signal that, when missing, the
optimizer exploits in a characteristic way (see "Exploit taxonomy"
below). Defaults match DeepMimic Eq. 6 weights/scales as closely as
the Walker2d obs/joint structure allows.

### Why each term

- **`r_p` (pose).** Joint tracking is the load-bearing imitation signal.
  Without it the policy just collects forward velocity and falls.
- **`r_v` (velocity).** Without velocity tracking the policy can hit
  the right joint *positions* via a slow, high-impedance trajectory
  with the wrong derivative profile (push-off and loading dynamics
  vanish).
- **`r_e` (end-effector).** Foot positions in root-relative frame
  constrain the *spatial* gait shape independently of joint angles.
  Without it, ankle posture exploits open up (toe-walking, paddling).
- **`r_c` (root height).** A simple height tracking term keeps the
  body upright. The pre-restart code added a pitch² piece inside the
  exponent; it was removed because the pitch *termination* (next
  section) already closes the controlled-fall exploit and a residual
  pitch reward did no scientific work.
- **`ctrl_cost` (DeepMimic-style).** Tiny. Keeps the value baseline
  well-behaved.

---

## Termination as part of reward design

Termination is not a reward term but it shapes optimisation pressure
just as strongly. The active default is **height + pitch** (see
[`METHODS.md § Termination`](METHODS.md#termination)):

- `height ∉ [0.8, 2.0]` → `term_cause="height"`. Walker2d-v4 default.
- `|pitch| > 0.3 rad` → `term_cause="pitch"`. **The controlled-fall
  guard.** Without it the agent learns to lean forward indefinitely
  because the height bound only fires *after* the lean is
  irrecoverable.

All other terminations (`pose`, `ankle`, `hip`, `xvel`) are
off-by-default sentinels (`9999`, `−∞`). They re-enter the picture
when a specific exploit appears in visual review.

---

## Optional exploit-patch reward terms

These were standard in the pre-restart engineered reward; they're now
kept as gated patches because each one solves a specific failure mode
that's predictable from the missing biomechanical signal.

### `swing_pen` — direct swing-foot contact penalty

```
swing_pen = tanh(F_swing/50)         (added with weight ≥ 0)
```

`r = … − w_swing_pen · swing_pen`. Swing detection uses
`ref_foot_zrel > -1.15` to flag whichever foot is supposed to be aerial
at this phase. **Closes ankle-paddling and toe-drag exploits** —
without it the agent rapidly oscillates the ankle through the full
gait phase angle while keeping both feet planted, satisfying ankle
tracking with no aerial phase.

### `contact_r` — stance-side foot dominance reward

```
contact_r = max(0, tanh(F_stance/50) − tanh(F_swing/50))
```

`r += w_contact · contact_r`. Stance side per frame is precomputed at
reset from reference hip angles. **Closes one-legged-hopping** — the
agent learns to plant one foot and use the other as a balance pole if
contact alternation isn't rewarded. Stable failure mode, hard to
escape once converged.

### Per-joint pose weighting (`pose_joint_weights`, `product_reward`, `min_joint_pose`)

Three alternative aggregators for `r_p`, all gated. The structural
issue they target is described under "The stiff-hip trap" below.

| Aggregator | Formula | When useful |
|---|---|---|
| Default arithmetic mean | `r_p = exp(−k · mean_j(w_j · diff_j²))` | DeepMimic-style; forgiving on a single outlier joint |
| Geometric mean | `r_p = (∏_j exp(−k · w_j · diff_j²))^(1/6)` | Closer to DeepMimic's multiplicative form; one bad joint costs more |
| Worst-joint floor | `r_p = min_j exp(−k · w_j · diff_j²)` | Hardest fix for "5 joints carry 1 stiff joint"; risks instability |

`pose_joint_weights` `[w_hip_r, w_knee_r, w_ankle_r, w_hip_l, …]` works
inside any of the three. Pre-restart engineered runs used per-joint
sharpness `(10, 20, 40, 10, 20, 40)` weighted `(0.4, 1, 2.5, …)` —
sharper on ankle because heel-strike timing matters far more than hip
posture. The current default (1's, k=10 globally) reverts to the
DeepMimic-faithful baseline.

### `energy_weight` — torque-squared penalty

`r −= w · sum(action²)`. Off by default. Useful if torque traces look
noisy or the policy is dissipating power inefficiently.

### Per-joint termination thresholds

`pose_term_thresh`, `ankle_term_thresh`, `hip_term_thresh` (and
`xvel_term_thresh` for the floor termination from
[`RESTART_LOG.md § Batch 2`](RESTART_LOG.md#batch-2--2026-04-28--escape-the-stand-still-basin)).
The asymmetry — ankle tighter than hip/knee — exists in pre-restart
configs because the agent will exploit large plantarflexion for
hopping if you let the ankle drift as far as hip/knee.

---

## The stiff-hip trap (2026-04-29 diagnosis)

> **Update 2026-04-29 (later, post-Batch-4):** the stiff-hip trap is
> primarily a **physical reachability** problem in `walker2d.xml`, not
> a reward-shaping problem. Stock gym `walker2d.xml` constrains
> `thigh_joint` to `[-150°, 0°]` while the reference asks for hip
> flexion peaks of +30° on both sides — the simulator literally cannot
> reach the reference target. The `restart_b2_xvel` policy spent 97.5%
> of its rollout pinned within 1° of the upper joint limit. Opening
> the MJCF range to `[-30°, +60°]` (`results/restart_b4_hipopen/`,
> 2M steps) raised hip ROM from 1.8° to 91.5° in a single change; see
> [`RESTART_LOG.md § Batch 4`](RESTART_LOG.md#batch-4--2026-04-29--joint-range-hypothesis-open-hip-mjcf--positive).
>
> The reward-side mechanism described below (5-of-6-joint loophole +
> survival floor) is real and contributes — it explains why the policy
> *settles* at the joint limit rather than fighting it. But the joint
> limit is what made the reference unreachable in the first place;
> reward changes alone cannot fix that.

The 19-experiment overnight sweep
([`RESTART_LOG.md § Batch 3`](RESTART_LOG.md#batch-3--2026-04-29--overnight-19-experiment-sweep--negative-result))
demonstrated that the current reward family — DeepMimic 4-term + the
`xvel_term=0.3` survival floor that produced the prior best policy
(`results/restart_b2_xvel/`) — is a **strong attractor for stiff-hip
walking**: hips pinned at ~0° vs reference ~45°, knees and ankles
wiggling around the reference, body translating forward at v_target.
Eight reward-aggregator/weighting/termination knobs and an SAC
optimizer swap all failed to escape — because the escape required a
joint range the MJCF didn't expose, not a different aggregator.

### The mechanism

`xvel_term=0.3` is a **floor**, not a target. Any forward velocity
≥ 0.31 m/s satisfies survival. A standing-with-knee-wiggle policy
that drifts at 0.4 m/s collects:

- ~1.0 healthy_reward per step (from staying alive)
- ~0.55 r_pose (mean-of-squares over 6 joints lets 5 wiggling joints
  hide one stiff hip — `mean_squared_error` stays small)
- ~0.95 r_root (height tracks fine because the body doesn't fall)

The per-step pose loss from a stiff hip is ~0.07 reward (`r_pose=0.55`
vs `r_pose=0.62`), which is *less than the survival increment from
staying alive*. The policy's optimum is "stay alive, keep collecting
healthy_reward, ignore hip flexion."

### Why metrics missed it

| Metric | What it said | What was actually true |
|---|---|---|
| `ep_rew_mean` | High ✓ | Survival reward dominates; pose/EE near-saturated |
| `r_pose ≈ 0.55` | Mediocre but plausible | Hides a single stiff joint behind 5 compliant ones |
| `progress_score 2.5/4` | "Closer to walking" | Body barely translates |
| `hip_knee_dtw` | "OK" | Finds the closest cyclic alignment; stand-and-wiggle scores well if any one stride matches |
| **`hip_r_rom_deg`** | **2°** | **Hip is stuck. The one metric that didn't lie.** |

Visual review (Brock, morning of 2026-04-29) is what cleanly
distinguished the trap from real progress. The headline numbers
flattered every Phase 1 variant.

### Code residue from the limit-compensation era (audit 2026-04-29)

After Batch 4 we audited what in the active code had been compensating
for the hidden hip limit. The full list:

- **Hardcoded RSI warm-start clip constants `_JNT_LO/_JNT_HI`**
  (`ppo_walker2d_phase.py`, removed). The upper hip bound was
  `+0.550 rad ≈ +31.5°`, but the loaded MJCF enforced `0°`. The
  warm-start qpos thus *advertised* hip flexion the simulator
  immediately overruled. Replaced with per-instance `_jnt_lo / _jnt_hi`
  read from `self.model.joint(...).range` so the clip always matches
  the actual MJCF — no more silent "the joint started here, then the
  physics moved it" disagreement at frame 1 of every episode.
- **Optional exploit-patch reward terms** (`swing_pen_weight`,
  `contact_weight`, `pose_term_thresh`, `ankle_term_thresh`,
  `hip_term_thresh`). All off-by-default CLI flags; not used by the
  current `b4_hipopen_5M` recipe. They are correctly quarantined as
  ablation knobs, but worth being aware that some of what they used
  to "fix" was the hip limit, not just exploits — re-enabling them
  on the opened MJCF may turn out to do nothing, or to over-constrain
  a now-functional reward.
- **`xvel_term=0.3` survival floor** — kept. This wasn't compensating
  for the joint limit; it kills "stand still and collect healthy
  reward," a separate exploit. It's also part of the proven
  `b4_hipopen_5M` recipe.

### What actually fixed it (Batch 4, 2026-04-29)

**Open the hip joint range in the MJCF.** A custom
`assets/mjcf/walker2d_hipopen.xml` with `thigh_joint range="-30  60"`
(`thigh_left_joint` matching) makes the reference's +30° flexion peaks
reachable. Trained from scratch with the proven `xvel-5M` recipe (8
envs, `--xvel_term 0.3`, no other changes), 2M steps was sufficient to
escape the basin entirely: hip ROM 91.5°, fwd vel 2.07 m/s. The
resulting gait is over-flexed and over-fast — the pose-tracking reward
is too forgiving of a single overshooting joint and the over-flexion
buys forward momentum — so a 5M follow-up + possibly `--pose_scale 20`
or `--product_reward` are queued to narrow the gait toward reference
tracking.

Note that the reward-side fix originally proposed here (replace
`--xvel_term` with a peaked `fwd_r = exp(−3·(v_x−1.25)²)`) was never
tested standalone, because the MJCF fix made it moot for basin escape.
A peaked forward reward may still be useful for *narrowing* the
hipopen gait's 2.07 m/s toward 1.25; that's a tunable for batch 5.

---

## Exploit taxonomy (writeup §6.2, Goodhart's-Law cases)

The history of this reward is an iterative visual-inspection loop:
each unconstrained DoF in a partial reward produced a characteristic
degenerate strategy that was locally optimal for the partial reward
but biomechanically implausible. The canonical cases:

### Ankle paddling (closed by `swing_pen`)

The agent exploits ankle joint tracking by oscillating the ankle
rapidly with no net ground loading. The reference ankle oscillates
through the full gait cycle regardless of foot contact, so ankle
motion alone satisfies the tracking reward while generating no forward
progress. Demo: `results/walker2d_pretrain_symmetry_20260407-172719/`
(symmetry-pretrain detour, no reference; same pattern resurfaces under
phase imitation if `swing_pen=0` and the agent is bumped off the
walking basin).

### One-legged hopping (closed by `contact_r` + `swing_pen`)

Without an explicit contact-alternation reward, the agent plants one
foot and uses the other as a balance pole. Stable over long episodes
and difficult to dislodge once converged.

### Toe-walking (closed by `r_e` + tighter ankle weighting)

Without explicit end-effector tracking, the agent satisfies the ankle
joint reward by maintaining permanent plantarflexion — walking on its
toes. The EE term (foot z-relative-to-root) closes this DoF.

### Forward fall / controlled lean (closed by pitch termination)

Without a hard pitch *termination*, the agent learns a controlled
forward fall — the lean is irrecoverable but the height-out-of-bounds
termination only triggers after the fall has already started. A
`|pitch| > 0.3 rad` termination is the load-bearing fix.

### Standing-and-tapping (`pretrain_walker2d.py` failure mode)

When forward reward is too low and bilateral contact is rewarded, the
agent stands still and alternates foot contacts without translating.
A symmetry-pretrain artefact; the active pipeline closes it with
phase conditioning + reference imitation (the reference moves through
1.25 m/s of forward kinematics every cycle). Three keeper checkpoints
demonstrate this failure mode under the symmetry-pretrain reward.

### Stiff-hip drift (the structural trap, 2026-04-29)

After the 2026-04-28 reference-sign correction, the policy converged
on a *new* exploit specific to the post-correction reward: walking
forward at v_target with thighs pinned at ~0° (reference ~45°) and
knees+ankles wiggling. **Not fixed by per-joint weighting, geometric
mean, worst-joint floor, hip-only termination, energy penalty, AMP
warm-start, or SAC optimizer** (see [Batch 3](RESTART_LOG.md#batch-3--2026-04-29--overnight-19-experiment-sweep--negative-result)).
Mechanism in "The stiff-hip trap" above; planned fix in
[`ROADMAP.md § 0`](ROADMAP.md#0-structural-reward-reform-forward_reward--remove-xvel_term-floor-new-2026-04-29).

---

## Why a weighted sum and not a product

DeepMimic's original paper used a multiplicative reward
(`exp(−Σ wⱼ kⱼ Δqⱼ²)`), which requires *all* tracking terms to be
satisfied simultaneously to earn any reward. The default here is a
weighted sum (arithmetic mean within each term, then summed across
terms): more forgiving and gives partial credit for partial success.

The trade-off is exactly the stiff-hip trap above. `--product_reward`
switches to the geometric mean inside `r_p`, recovering DeepMimic's
multiplicative form on demand for ablations.
