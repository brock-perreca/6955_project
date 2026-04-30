# Roadmap

**Purpose:** prioritized list of what's planned next.
**Read this when:** picking the next experiment, scoping a writeup
§7, or deciding whether a proposed change is on-mission.
**Adjacent:** [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for "right now"
· [`RESTART_LOG.md`](RESTART_LOG.md) for what just shipped.

**Item 0 is DEPRIORITISED as of 2026-04-29.** The held-out
biomechanical-realism scorecard
([`PROJECT_STATUS.md § Biomechanical-realism finding`](PROJECT_STATUS.md#biomechanical-realism-finding-2026-04-29--end-of-road-on-the-engineered-reward-track))
ran on all four post-Tier-0 candidates plus the pre-Tier-0
`b2_xvel` baseline and found that **none of them produce a
biomechanically realistic gait**: double-support ~1% (vs ref 23%),
peak vGRF 3.7–4.8 BW (vs ref 1.10), cadence 178–210 spm (vs ref
107). The post-Tier-0 candidates barely beat the pre-Tier-0
baseline on the 0–4 progress score. **Lead policy: `b5_min_joint`**
(highest score 2.66, lowest peak vGRF 3.70 BW, lowest double-support
deviation; named on the post-strike-detector-fix scorecard).

The new top priority is **item 1 (MJX/AMP)**, with item 0 retained
below for historical context. Items 2–4 follow.

---

## 0. Close the residual reward gap on relaxed-hip MJCFs — **DEPRIORITISED 2026-04-29**

**Status:** retained as historical context only. The biomech-realism
scorecard
([`PROJECT_STATUS.md`](PROJECT_STATUS.md#biomechanical-realism-finding-2026-04-29--end-of-road-on-the-engineered-reward-track))
showed that closing the residual reward gap on the engineered-reward
stack would require hand-engineering double-support, vGRF, and
cadence into the reward — which defeats the imitation-only goal.
Reward-knob experiments on this stack are paused until a more
fundamental change (AMP/MJX or muscle actuators) lands.

**Why (original framing):** opening the hip joint range fixed the
dominant cause of stiff-hip walking, but each variant leaves a
different residual gap that points back at the reward:

- **hipopen (`[-30, 60]`)**: at 5M (`results/restart_b4_hipopen_5M/`)
  hip ROM is **63°** vs reference 43°, mean fwd vel 1.40 m/s vs
  target 1.25. **Over-flexed and slightly over-fast.** Pose-tracking
  `exp(-10·mean(diff²))` is too forgiving of one overshooting joint
  when the other five track.
- **hiprelax (`[-150, 35]`)**: at 5M (`results/restart_b4_hiprelax_s11/`)
  hip ROM is **17–20°** vs reference 43° (per-stride medians sliced
  by the pre-fix detector; post strike-detector-fix re-eval reads
  ~30° per stride, ~67 % of ref), cadence **~1.95× too fast** (not
  3× as previously framed — see PROJECT_STATUS.md), peak vGRF/BW
  worsens (4.0 vs 3.3). **Under-flexed and over-fast.** `xvel_term=0.3`
  is a *floor* — any forward velocity ≥ 0.31 m/s satisfies survival,
  so the policy's optimum is "drift fast and short" even with the
  wall relaxed.

The two variants together bracket the reward question: hipopen
overshoots ROM, hiprelax undershoots ROM, and **both** are over-fast.
The shared diagnosis is that `xvel_term=0.3` (or the `mean()`
aggregator that hides single-joint overshoots) makes "fast,
low-amplitude" the local optimum regardless of MJCF.

**Plan, in two parallel threads** — both are on-mission, run them
together so the results are interpretable side-by-side:

### 0a. Narrow the `hipopen` gait (Asus-laptop track)

In order of escalation:

1. ✓ **5M follow-up of `b4_hipopen`** — `results/restart_b4_hipopen_5M/`,
   seed 6. Narrowed 91°→63° hip ROM, 2.07→1.40 m/s.
2. ✓ **Sharpen pose tracking** (Batch 5). `--pose_scale 20` and
   `--min_joint_pose` each narrow hip ROM 63°→57° and pull fwd vel
   toward target; `min_joint` lands at 1.231 m/s (essentially target).
   Neither hits ROM ~43°. See
   [`RESTART_LOG.md § Batch 5`](RESTART_LOG.md#batch-5--2026-04-29--narrow-the-hipopen-over-flex--partial-positive-both-variants).
3. **Stack the two batch-5 knobs** — `--pose_scale 20 --min_joint_pose`
   together (untested combination, both moved the gait in the same
   direction). One 5M run before escalating to a peaked-forward
   reward.
4. **If still over-fast, add the peaked forward reward** —
   `fwd_r = exp(-3·(v-1.25)²)` with `--fwd_weight 0.15`, drop
   `--xvel_term`. Replaces the survival floor with a target-velocity
   bell curve. Currently not a CLI flag — needs a small code change.
5. **Once a clean tracking gait exists, retry AMP/AIRL warm-start**
   from `b4_hipopen_5M`. Batch 3's AMP runs failed partly because
   the underlying PPO couldn't produce reference-like hip flexion;
   with a tracking baseline the discriminator should have a learnable
   signal.

### 0b. Structural reward reform on `walker2d_hiprelax.xml` (O11 track)

Restore the `forward_reward = exp(-3·(v-1.25)²)` term that was
deleted as default-off on 2026-04-28
([`REWARD_DESIGN.md § Removed terms`](REWARD_DESIGN.md#fwd_r--forward-velocity-reward)).
Drop `xvel_term`. The bell-curve forward target replaces a survival
floor with a peaked reward — drifting at v=0.4 no longer maxes out;
only matching v_target does. **Train on `walker2d_hiprelax.xml`** —
the +5° headroom is just enough for the policy to express the
reference's +30° peak without exploring far-off-distribution
overswing the way hipopen does.

Restoring `fwd_r` as a CLI flag `--fwd_weight` (default ~0.10–0.20)
is the minimal-risk path; this is the same code change as 0a step 4,
so the two tracks share that work.

**Test plan for 0b:** ONE training run from scratch with
`--xml walker2d_hiprelax.xml --fwd_weight 0.15`, no `--xvel_term`,
otherwise the xvel-5M recipe. Baseline: `results/restart_b4_hiprelax_s11/`
(Tier 0 canonical pick — hip ROM 17-20°, cadence 333). If hip ROM >
30° AND cadence < 200 in eval, the residual reward trap is broken on
hiprelax; queue stacked variants with `--product_reward` and a
warm-started AMP run on the new policy. If hip amplitude still
plateaus at ~20°, the trap is deeper than reward+range — frame rate,
phase obs rate, or body-mass scaling vs the 75 kg subject — and
Tier 2 begins.

### Joint readout

Running 0a step 4 and 0b together (both use the new `--fwd_weight`
flag, just on different MJCFs) gives the cleanest experimental
design: hipopen with the new reward should *narrow* toward 45°;
hiprelax with the new reward should *grow* toward 45°. If both
converge near 45° at cadence ~110, reward was the dominant remaining
cause and we move to writeup §7.1 (MJX/AMP). If both stay where
they are, Tier 2 (frame rate / phase observation) opens.

**Owner:** Brock. 0a step (3) is two existing CLI flags stacked.
0a step (4) and 0b both need the same new code (peaked-forward
reward term). 0a step (5) is the comparison track.

**Skip:** more aggregator variants on the *stock* `walker2d.xml`,
hip_term, energy penalty, reverse curriculum, preview_k > 1. Tier 0
+ Batch 3 say these are not where the trap is on stock; on the
relaxed MJCFs some may help narrow the gait but they're lower
priority than the peaked-forward reward.

---

## 1. MuJoCo MJX / GPU port for AMP (writeup §7.1)

**Why:** AMP collapses at 8 CPU envs (writeup §6.3). The discriminator
achieves near-perfect separation before the policy has produced any
walking, driving style reward to ~0.03 and eliminating the gradient.
Mechanism: 24-D discriminator input space, only 349 expert transitions,
8-env policy diversity insufficient to force discriminator generalization.

**2026-04-29 update:** the overnight Batch 3 tested AMP/AIRL warm-started
from a working PPO policy as a workaround. The warm-start *does*
prevent immediate cold-start collapse (style_r stayed in [0.2, 0.4]
for ~3M steps), but the discriminator gradient pushes the policy into
a *different* bad basin (asymmetric kicks, ankle paddling at 100°+)
rather than toward natural gait. Visual review (Brock) called all 4
B2 runs "pretty terrible." So the warm-start workaround is **not** a
substitute for the GPU port; the discriminator still needs more
policy diversity than 8 envs can provide. Revisit AMP/AIRL after item 0.

**Plan:** Port `Walker2dPhaseAware` to MJX (JAX GPU backend) targeting
**2,000–4,000 parallel envs on a single RTX 5090**. At that scale, policy
diversity itself prevents discriminator memorization. Reproduces the
condition under which Escontrela et al. ran AMP successfully (4096 envs
on a GPU cluster) — see
[`papers/Escontrela_2022_AMP_legged_robots.pdf`](papers/Escontrela_2022_AMP_legged_robots.pdf)
(robotics) and the original
[`papers/Peng_2021_AMP_animation.pdf`](papers/Peng_2021_AMP_animation.pdf)
(character-animation framing); for AIRL-style transferable rewards
see [`papers/Fu_2018_AIRL.pdf`](papers/Fu_2018_AIRL.pdf). Index entries:
[`papers/papers.md § Adversarial imitation track`](papers/papers.md#2-adversarial-imitation-track).

**Owner:** likely Brian (AMP track), Brock (env port).

**Risk:** MJX is more constrained than vanilla MuJoCo (XLA tracing,
limited dynamic shapes). The contact model and `cfrc_ext` extraction
used in `contact_r` may need rewriting. Reference FK precomputation can
stay CPU-only.

---

## 2. Multi-step future context in the observation (writeup §7.2)

**Why:** The single-cycle reference has a small velocity discontinuity
at the wrap-around boundary (frame 349 → frame 0). The current reward
has a brief dip there that the policy cannot smooth away because it
doesn't know it's coming.

**Plan:** Extend the obs from `[base | q_ref(φ) | sin φ | cos φ]` to
`[base | q_ref(φ) | q_ref(φ+1) | … | q_ref(φ+K−1) | sin φ | cos φ]` —
i.e. give the policy a small preview window of upcoming reference
frames. Analogous to MPC with reference preview. Probably K = 4–8.

**2026-04-29 update:** implemented as `--preview_k` CLI flag in the
overnight Batch 3 sweep and tested at K=4, K=8. Visual review
(Brock): preview_k runs are "just a little choppier than B1," no
real improvement. **Hypothesis discounted but not eliminated** —
this was tested *with* the broken reward structure (item 0). After
item 0, preview_k is worth one more pass to see whether the
anticipation signal becomes useful when survival pressure no longer
dominates.

**Risk:** Larger obs space → policy has to learn to use the preview
window without overfitting to it. If the preview length doesn't match
the gait cycle period, the encoding has aliasing.

---

## 3. DTW-based reference selection and evaluation (writeup §7.3)

**Why:** Two complementary uses of Dynamic Time Warping distance:

1. **Held-out evaluation metric.** A policy can collect high return
   while exhibiting any of the [`REWARD_DESIGN.md`](REWARD_DESIGN.md)
   exploits if reward weights are slightly miscalibrated. DTW between
   the policy's full-episode joint trajectory and the reference cycle
   is a *shape-fidelity* signal that's robust to phase drift and
   independent of reward calibration.
2. **Reference selection.** DTW distance between cycles within the
   Ulrich dataset can cluster similar gait cycles, enabling
   multi-cycle reference training without trial-boundary discontinuities.

**Plan:** Implement DTW alongside an existing evaluation script, run it
on the canonical run + a couple of failure-mode runs to validate that
DTW separates "real walking" from "reward hacked" trajectories.

**2026-04-29 update:** `hip_knee_dtw` (2 joints) and `all_joints_dtw`
(6 joints) are both now in `eval_biomech.py`. **Caveat from
overnight Batch 3:** DTW finds the closest cyclic alignment, so a
stand-and-wiggle gait with one valid stride scores OK on DTW even
when the body barely translates. DTW is *necessary but not
sufficient* for diagnosing real walking; pair with `hip_r_rom_deg`
and visual review.

---

## 4. Multi-cycle / multi-subject reference (writeup §7.4)

**Why:** Single-cycle reference has limited variance — the policy
overfits to the exact stride pattern from Subject 1. Multi-cycle and
multi-subject references would improve both temporal smoothness (by
closing the seam discontinuity with neighboring cycles) and robustness
(generalize across between-subject variability).

**Plan:** Use the DTW clustering from item 3 to choose 4–8 high-quality
cycles. Concatenate them carefully (or sample uniformly per episode).
Re-test the failure-mode taxonomy on the multi-cycle reference.

---

## 5. Possible return-to-scope — musculoskeletal track

The original proposal scope (3D 80-muscle MyoLeg, OpenCap markerless
data, GRF/EMG/joint-contact-force comparison) was preserved as legacy
code in `src/legacy/musculoskeletal/`. If this is revisited:

- Start with `src/legacy/musculoskeletal/ppo_walk.py` (MyoSuite
  myoLegWalk-v0 baseline) to verify the muscle-actuated env still
  works on the current MuJoCo / MyoSuite versions.
- `train.py`, `bc_policy.py`, `gail.py`, and `data_utils.py` are the
  BC + GAIL pipeline driver. They use OpenCap data layout
  (`subject{N}/OpenSimData/...`) — see [`DATA_SOURCES.md`](DATA_SOURCES.md).
- See [`LEGACY_TRACKS.md`](LEGACY_TRACKS.md) for the things to verify
  before re-running.

This is **out of scope for the current writeup** but the user has
indicated they may go back to some of the original ideas. The code is
preserved precisely to make that path easier.

---

## 6. Speculative — Walker2d → 3D Humanoid

Walker2d is a 2D sagittal-plane benchmark. A 3D bipedal humanoid (e.g.
Gymnasium's `Humanoid-v4`, or MyoSuite's myoLeg) would re-enable
out-of-plane balance and torsional dynamics. The phase-conditioned reward
generalizes naturally; the bigger questions are reference data (do we
have 3D IK?) and AMP/AIRL discriminator scaling. Probably blocked on
items 1 (MJX) and 5 (musculoskeletal track).
