# Roadmap

Planned future directions, prioritized roughly. The first four items
come directly from the writeup §7
([`reports/writeup_filled_1.docx`](reports/writeup_filled_1.docx)). The
last items are speculative — possible returns to original-proposal scope.

For what is *currently* working, see [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

---

## 1. MuJoCo MJX / GPU port for AMP (writeup §7.1)

**Why:** AMP collapses at 8 CPU envs (writeup §6.3). The discriminator
achieves near-perfect separation before the policy has produced any
walking, driving style reward to ~0.03 and eliminating the gradient.
Mechanism: 24-D discriminator input space, only 349 expert transitions,
8-env policy diversity insufficient to force discriminator generalization.

**Plan:** Port `Walker2dPhaseAware` to MJX (JAX GPU backend) targeting
**2,000–4,000 parallel envs on a single RTX 5090**. At that scale, policy
diversity itself prevents discriminator memorization. Reproduces the
condition under which Escontrela et al. ran AMP successfully (4096 envs
on a GPU cluster).

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
