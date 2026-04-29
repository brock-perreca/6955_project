# Roadmap

**Purpose:** prioritized list of what's planned next.
**Read this when:** picking the next experiment, scoping a writeup
§7, or deciding whether a proposed change is on-mission.
**Adjacent:** [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for "right now"
· [`RESTART_LOG.md`](RESTART_LOG.md) for what just shipped.

**Item 0 is the new top priority** after the 2026-04-29 Batch 4
([`RESTART_LOG.md § Batch 4`](RESTART_LOG.md#batch-4--2026-04-29--joint-range-hypothesis-open-hip-mjcf--positive))
diagnosed the stiff-hip basin as a joint-range problem in `walker2d.xml`,
fixed by `assets/mjcf/walker2d_hipopen.xml`. Items 1-4 from the writeup §7
follow once the hipopen baseline is solid.

---

## 0. Narrow the `hipopen` gait toward reference tracking (NEW, 2026-04-29)

**Why:** Batch 4 escaped the stiff-hip basin by opening
`walker2d.xml`'s hip range from `[-150°, 0°]` to `[-30°, +60°]`
(`assets/mjcf/walker2d_hipopen.xml`). At 2M steps the policy's hip ROM
is **91.5°** vs reference 43°, mean fwd vel 2.07 m/s vs target 1.25.
The basin is gone; the gait is over-flexed and over-fast. The
pose-tracking reward `exp(-10·mean(diff²))` is too forgiving of one
overshooting joint when the other five track.

**Plan, in order of escalation:**

1. **5M follow-up of `b4_hipopen` with the same recipe.** Already
   queued (`results/restart_b4_hipopen_5M/`, seed 6). Hypothesis:
   additional rollouts shrink the 91°→43° overshoot.
2. **If (1) plateaus over-flexed, sharpen pose tracking.** Try
   `--pose_scale 20` (50% reward at 0.18 rad RMS rather than 0.26)
   *or* `--product_reward` (geometric-mean per-joint exps; one bad
   joint hurts the whole reward). Single-knob ablation against (1).
3. **If still over-fast, add the peaked forward reward** that was
   originally Batch 4's plan: `fwd_r = exp(-3·(v-1.25)²)` with
   `--fwd_weight 0.15`, drop `--xvel_term`. Now the survival floor
   no longer rewards drift at any forward speed.
4. **Once a clean tracking gait exists, retry AMP/AIRL warm-start.**
   Batch 3's AMP runs failed partly because the underlying PPO policy
   could not produce reference-like hip flexion (data-distribution
   mismatch with the expert manifold). With a hipopen baseline that
   actually tracks, the discriminator should have a learnable signal.

**Owner:** Brock. Code change for (1) is zero (already running). (2)
and (3) are existing CLI flags. (4) is the comparison track.

**Skip:** more aggregator variants on the *stock* MJCF, hip_term,
energy penalty, reverse curriculum, preview_k > 1. Batch 3 already
ruled all of those out as fixes for the basin; on the open MJCF some
may help narrow the gait but they're lower-priority than (1)–(3).

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
