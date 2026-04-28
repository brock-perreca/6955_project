# Reward design + exploit taxonomy

This is the *why* of the reward. For the implementation-level details,
see [`METHODS.md`](METHODS.md). For the failure runs that motivated each
term, see [`RUN_LOG.md`](RUN_LOG.md).

---

## High-level shape

The reward is a **weighted sum** of several `[0, 1]` tracking terms, each
scaled by `dt` so the return is time-invariant:

```
r = w_imit · imit_r
  + w_vel  · vel_r
  + w_ee   · ee_r
  + w_root · root_r
  + w_cont · contact_r
  − w_swing_pen · swing_pen
  + w_peak · peak_bonus      (optional, off by default)
  + w_fwd  · fwd_r            (optional, off by default)
  − w_act  · action_rate_pen  (optional, off by default)
```

Each component except the optional bonuses corresponds to a specific
biomechanical signal that, if missing, the optimizer exploits in a
characteristic way. The reward design therefore reads as a list of
"unconstrained DoFs and the patches that close them."

---

## Per-joint scaling

Pose and velocity terms use **per-joint sharpness `k_j`** and **per-joint
weights `w_j`**:

```python
_JSCALE   = [10, 20, 40, 10, 20, 40]      # hip / knee / ankle, bilateral
_JWEIGHTS = [0.4, 1.0, 2.5, 0.4, 1.0, 2.5]  # ankle weighted 2.5×
_KVSCALE  = [0.05, 0.1, 0.2, 0.05, 0.1, 0.2]  # vel: tighter on ankle
```

- `k=10` → 50% reward at ≈ 0.26 rad (15°) — slow postural joint (hip).
- `k=20` → 50% reward at ≈ 0.19 rad (11°) — moderate (knee).
- `k=40` → 50% reward at ≈ 0.13 rad (7°) — tight (ankle, heel-strike timing).

Heel-strike timing is the most temporally sensitive event in human
walking — small ankle errors translate to large differences in foot
contact dynamics. Hip posture is comparatively forgiving.

---

## Components

### `imit_r` — per-joint pose tracking (default weight 4)

```
imit_r = mean_j ( w_j · exp(−k_j · (q_j − q_ref_j)²) )
```

Replaces the earlier uniform `exp(−8 · Δq²)` formulation, which weighted
hip/knee/ankle equally and let the agent earn most of the imitation
reward by parking hip and knee while ignoring ankle.

### `vel_r` — per-joint velocity tracking (default weight 1)

```
vel_r = mean_j ( exp(−kv_j · (q̇_j − q̇_ref_j)²) )
```

Velocity tracking prevents the policy from matching pose statically while
missing push-off and loading dynamics. Without it, the agent can hit the
right joint *positions* via a slow, high-impedance trajectory that has
the wrong derivative profile.

### `ee_r` — end-effector foot tracking (default weight 4)

Forward kinematics is precomputed at init to obtain reference foot
positions (x relative to root, z relative to root) for each reference
frame. During rollout, foot world positions are queried from MuJoCo
geometry and tracked with `k=40`.

The **earlier swing-phase k=200 sharpening was removed** in favor of a
separate explicit swing-foot contact penalty (`swing_pen` below).
`SWING_CLEARANCE` was lowered from −1.05 to **−1.15** (root-relative z)
so toe-off triggers the penalty earlier.

### `root_r` — torso height + pitch (default weight 2)

```
root_r = exp(−10 · (Δh² + 1·θ²))
```

The pitch coefficient was lowered from 3.0 → **1.0** when the
termination check on pitch (`|pitch| > 0.3 rad`) was added — the
termination handles the forward-lean exploit (controlled fall), so the
reward no longer needs to over-penalize pitch.

### `contact_r` — stance-side foot force dominance (default weight 1)

Contact forces on each foot are extracted from MuJoCo's `cfrc_ext`. The
reward is `tanh(F_stance/50) − tanh(F_swing/50)`, gated by a precomputed
`stance_right[t]` array. The tanh normalization prevents reward from
growing unboundedly with contact force magnitude.

### `swing_pen` — direct swing-foot contact penalty (default weight 2)

```
swing_pen = tanh(F_swing/50)
```

Catches **toe-drag** forces that the alternation reward misses. The
alternation reward is a *difference* — when both feet have small forces
the difference is small but the swing foot is still in unwanted contact.
The direct penalty closes that gap.

### `peak_bonus` — high-excursion phase bonus (default weight 0)

```
peak_bonus = mean_j ( excursion_j[φ] · exp(−k_j · Δq_j²) )
```

where `excursion_j ∈ [0, 1]` is the per-joint normalized distance from
the midpoint of the reference range. Bonuses match at peak knee flex,
peak ankle push-off, etc. — the kinematically dramatic moments. Off by
default; useful as a finetune tool when peak excursions are
under-tracked.

### `fwd_r` — forward velocity reward (default weight 0)

```
fwd_r = exp(−3 · (x_vel − v_target)²),    v_target = 1.25 m/s
```

Off by default because `contact_r` + `ee_r` already constrain forward
speed implicitly. Useful as a gentle nudge during finetuning if drift
appears.

### `action_rate_pen` — anti-jerk penalty (default weight 0)

```
action_rate_pen = Σ_j (a_j(t) − a_j(t−1))²
```

Off by default; turn on if torque traces look noisy.

---

## Exploit taxonomy (writeup §6.2, Goodhart's-Law cases)

The multi-term reward design is the result of an iterative
visual-inspection loop: each unconstrained DoF in the partial reward
produced a characteristic degenerate strategy that was locally optimal
for the partial reward but biomechanically implausible. The three
canonical cases:

### Ankle paddling (closed by `swing_pen`)

Before the swing-foot contact penalty was added, the agent exploited the
ankle joint tracking term by producing rapid ankle oscillation with no
net ground loading. The reference ankle oscillates through the full gait
cycle regardless of foot contact, so ankle motion alone satisfies the
tracking reward while generating no forward progress.

> **Demo:** `results/walker2d_pretrain_symmetry_20260407-172719/` (from
> the symmetry-pretrain detour, but the same exploit pattern applied to
> phase imitation before `swing_pen` was added).

### One-legged hopping (closed by `contact_r` + `swing_pen`)

Without an explicit contact-alternation reward, the agent discovered that
planting one foot and using the other as a balance pole satisfied the
joint tracking reward for the stance leg while requiring no bilateral
coordination. This strategy is stable over long episodes and difficult
to dislodge once converged.

### Toe-walking (closed by `ee_r` + tighter `_JSCALE` on ankle)

Without explicit end-effector tracking, the agent satisfied the ankle
joint reward by maintaining a permanently plantarflexed posture,
effectively walking on its toes. Foot placement constraints (the EE
term, with the foot z-relative-to-root signal) close this DoF.

### Forward fall / controlled lean (closed by pitch termination)

Without a pitch *termination* (just a pitch reward penalty), the agent
learns a controlled forward fall — the lean is irrecoverable but the
height-out-of-bounds termination only triggers after the fall already
started, so the agent collects pitch-penalized but otherwise positive
reward during the lean. Adding a hard termination at `|pitch| > 0.3 rad`
closes this.

### Standing-and-tapping (from the symmetry-pretrain detour)

When forward reward is too low and bilateral contact is rewarded, the
agent stands still and alternates foot contacts without translating.
This was a **`pretrain_walker2d.py`** failure mode (no reference) — see
[`RUN_LOG.md`](RUN_LOG.md) "Run 4". The fix in the active pipeline is
phase conditioning + reference imitation, which makes standing
incompatible with the reward (the reference moves through 1.25 m/s of
forward kinematics every cycle).

---

## Why a weighted sum and not a product

DeepMimic's original paper used a multiplicative reward
(`exp(−Σ wⱼ kⱼ Δqⱼ²)`), which requires *all* tracking terms to be
satisfied simultaneously to earn any reward. Our weighted-sum form
(arithmetic mean within each term, then summed across terms) is more
forgiving: the agent gets partial credit for partial success.

The trade-off: arithmetic mean lets the agent ignore individual joints
if the others compensate. We close this in two ways:

1. **Per-joint weights** (`_JWEIGHTS`) — under-tracking the ankle costs
   2.5× as much reward as under-tracking the hip.
2. **`--product_reward` flag** — switches `imit_r` to the geometric mean
   of per-joint exps, giving a DeepMimic-style multiplicative form on
   demand.

The default canonical run uses the arithmetic-mean form. The flag is
functional and exists for ablations.
