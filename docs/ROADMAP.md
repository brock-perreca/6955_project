# Roadmap

**Purpose:** prioritized list of what's planned next.
**Read this when:** picking the next experiment, scoping a writeup
§7, or deciding whether a proposed change is on-mission.
**Adjacent:** [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for "right now"
· [`RESTART_LOG.md`](RESTART_LOG.md) for what just shipped.

**Item 0 is the new top priority** after the 2026-04-29 overnight
([`RESTART_LOG.md § Batch 3`](RESTART_LOG.md#batch-3--2026-04-29--overnight-19-experiment-sweep--negative-result));
items 1-4 from the writeup §7 follow once the reward trap is fixed.

---

## 0. Structural reward reform: forward_reward + remove xvel_term floor (NEW, 2026-04-29)

**Why:** the 19-experiment overnight sweep
([`RESTART_LOG.md § Batch 3`](RESTART_LOG.md)) demonstrated that the
current reward structure is a strong attractor for stiff-hip walking.
Eight reward-aggregator/weighting/termination knobs and an SAC
optimizer swap all failed to escape. The diagnosis: `xvel_term=0.3` is
a *floor* — once the policy moves at v ≥ 0.31 m/s it earns full
survival reward, regardless of joint kinematics. The per-step pose
gradient toward hip flexion is smaller than the survival reward, so
the policy's optimum is "drift forward stiffly, collect survival."

**Plan:** Restore the `forward_reward = exp(-3·(v-1.25)²)` term that
was deleted as default-off on 2026-04-28
([`REWARD_DESIGN.md § Removed terms`](REWARD_DESIGN.md#fwd_r--forward-velocity-reward)).
Drop `xvel_term`. The bell-curve forward target replaces a survival
floor with a peaked reward — drifting at v=0.4 no longer maxes out;
only matching v_target does. Restoring as a CLI flag `--fwd_weight`
(default ~0.10–0.20) is the minimal-risk path.

**Test plan:** ONE training run from scratch with `--fwd_weight 0.15`,
no `--xvel_term`, otherwise the xvel-5M recipe. If hip ROM > 15° in
visual review, the trap is broken; queue stacked variants with
`--product_reward` and warm-started AMP. If still stiff-hip, the trap
is deeper than reward (frame rate? phase obs?) and we move to a
diagnostic experiment.

**Owner:** Brock. Code change is small (~10 LOC; the `fwd_r` term and
its CLI flag previously existed and are git-recoverable).

**Skip:** more aggregator variants, hip_term, energy penalty, reverse
curriculum, preview_k > 1. The data says these are not where the trap
is.

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
