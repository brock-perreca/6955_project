# Project timeline

**Purpose:** the authoritative chronological record of how this
project's scope evolved, what was tried, what worked, what didn't, and
why we're where we are.
**Read this when:** you find a legacy file or stale assumption and
want to know "why is this still here / why was this ever here." For
*right now*, see [`PROJECT_STATUS.md`](PROJECT_STATUS.md). For per-run
reproduce/render commands on the legacy symmetry-pretrain runs, see
[`RUN_LOG.md`](RUN_LOG.md). For per-batch progress on the post-restart
rebuild, see [`RESTART_LOG.md`](RESTART_LOG.md).

---

## Phase 0 — Original proposal (proposal stage, see `reports/Advanced_AI_Project_Report.pdf`)

**Title:** *Lab-to-Field Transfer in Musculoskeletal Reinforcement Learning.*

**Vision:** A 6-condition study (R1–R6) training **3D, 80-muscle, 20-DoF
MyoLeg** agents on **OpenCap markerless** vs **lab-grade marker + force
plate + EMG** references, comparing emergent ground reaction forces, EMG,
and joint contact forces under SAC and SAC+GAIL. Goal: measure how much
biomechanical fidelity is lost when markerless mocap replaces a $150k lab.

**What was scoped:**
- 3D MyoLeg model, ~80 Hill-type muscles, ~20 DoF
- OpenCap markerless data + traditional lab-grade reference
- SAC and SAC+GAIL imitation
- Evaluation: GRF, EMG, joint contact forces, vs ground truth

**Outcome:** Found to be too ambitious for a one-semester course project.
The 3D + muscle + adversarial + multi-condition combination would have
required infrastructure (OpenCap pipeline, OpenSim post-hoc analysis,
muscle-actuator hyperparameter tuning) that was not realistic given the
time budget. Pivot.

**Code that survives from this era:** see
[`LEGACY_TRACKS.md § Musculoskeletal track`](LEGACY_TRACKS.md#musculoskeletal-track)
— `src/legacy/musculoskeletal/` (`ppo_myoassist.py`, `train.py`,
`bc_policy.py`, `gail.py`, etc.). The `OpenCap_data/` dataset is
preserved on disk where users have it locally.

---

## Phase 1 — First Walker2d attempt: phase-blind imitation (early April 2026)

**Decision:** Drop the 3D / muscle / OpenCap stack. Switch to **2D
Walker2d-v4** (torque-actuated, 6 joints) so we can iterate quickly on a
clean RL benchmark while still using real human IK data.

**Approach:** `ppo_walker2d.py` — feed the Ulrich treadmill IK reference
directly as a per-joint target into a DeepMimic-style imitation reward.

**Failed.** Three compounding bugs (documented in [`RUN_LOG.md`](RUN_LOG.md)
— "Background: why the first approach failed"):

1. **2.5× speed mismatch.** Ulrich IK is at 50 Hz; Walker2d runs at 125 Hz
   (frame_skip=4). The reference was not resampled, so the gait played out
   2.5× too fast. The policy was chasing a target that changed faster than
   it could respond.
2. **Phase blindness.** The observation contained no information about
   where in the gait cycle the agent currently was. The policy could only
   learn an average response, which collapsed to a partial-extension of
   the stance leg.
3. **Concatenated reference.** All trials from all subjects were
   concatenated into a single 413k-frame reference, creating
   discontinuous joint jumps at trial boundaries that corrupted the
   reward signal.

The result: agent kicks one leg back (partial reward for hip extension)
and falls over.

**Representative failure run:** `walker2d_ulrich_all_20260406-221644`
(deleted to save disk; render command in [`RUN_LOG.md`](RUN_LOG.md)).

**Code that survives:** `src/legacy/walker2d_v1/ppo_walker2d.py` —
preserved for historical reference. `load_sto`, `load_ulrich_reference`,
and `ULRICH_ROOT` were extracted into
[`src/walker2d/ulrich_loader.py`](../src/walker2d/ulrich_loader.py).

---

## Phase 2 — Symmetry-reward pretraining detour (April 7–8, 2026)

**Decision:** Try to bootstrap walking *without* a reference, using
biomechanically-motivated reward shaping (gait symmetry + foot
alternation). The hypothesis was that reference imitation could be a
finetune step on top of a pretrained "knows how to alternate feet" policy.

**Approach:** `pretrain_walker2d.py` — a custom `Walker2dContactWalk`
wrapper that adds gait-symmetry rewards (left-right hip anti-phase, foot
contact alternation, ROM range), with no reference data.

**Hit four distinct local optima** (see [`RUN_LOG.md`](RUN_LOG.md) for
each):

1. **Two-legged hopping with some swing** — symmetry collected, but
   bilateral hopping wins on forward reward.
2. **One-legged hopping** — asymmetric strategy more reward-efficient
   when symmetry weight is too low to overcome the hop local optimum.
3. **Ankle paddling** — increased gravity + ankle torque cap suppressed
   hopping; agent shuffles forward via pure ankle plantarflexion against
   the ground. No aerial phase.
4. **Stands still and taps feet** — when forward reward is low enough,
   the agent maximizes height + bilateral contact reward by standing.

**Conclusion: dead end.** Reward shaping without phase information
cannot produce real walking on Walker2d. The local optima are too
attractive. The actual fix turned out to be phase conditioning, not
reward design.

**Code that survives:** `src/legacy/walker2d_v1/pretrain_walker2d.py` and
`src/legacy/walker2d_v1/render_walker.py`. Three keeper checkpoints from
this phase (`results/walker2d_pretrain_symmetry_*/`) are still on disk as
demos of canonical reward-hacking failures.

---

## Phase 3 — First walking policy: phase-conditioned DeepMimic (mid April 2026)

**Decision:** Add the phase information that Phase 1 was missing. New
script: `ppo_walker2d_phase.py`.

**What changed vs Phase 1:**

| Phase 1 (failed) | Phase 3 (works) |
|---|---|
| 50 Hz reference, no resampling | Cubic-spline resample 50 → 125 Hz |
| Phase-blind 17-D obs | 25-D obs = `[base(17) | q_ref(6) | sin φ | cos φ]` |
| Concatenated 413k frames | Single clean stride extracted via `extract_gait_cycle.py` |
| Uniform `exp(-8 · Δq²)` reward | DeepMimic multi-term reward |
| No reference state init | RSI: warm-start from a uniformly-sampled reference frame |

**First walking policy:** `walker2d_phase_cycle_20260408-115434` (now
superseded). Heel-strike events visible, bilateral foot alternation,
sustained 2000-step episodes.

**Subsequent iterations within this phase** (each fixing a specific
exploit found by visual inspection):
- **Adaptive → fixed-clock phase.** Adaptive matching let stiff-legged
  policies "shop" for extended-knee reference frames. Replaced with
  `_phase = (_phase + 1) % T`.
- **Per-joint reward weights.** Uniform `k=8` replaced with
  `k_hip=10, k_knee=20, k_ankle=40` and ankle weighted 2.5×; heel-strike
  timing matters more than hip posture.
- **Swing-foot contact penalty.** Added explicit `tanh(F_swing/50)`
  penalty to catch toe-drag exploits the contact alternation reward
  missed.
- **Pitch-magnitude termination.** Without it, the agent learned a
  controlled forward fall — height only drops below 0.8 once the lean
  is irrecoverable.
- **Forward-velocity warm start.** RSI fix: set `qvel[0] = v_target`
  to match the mid-stride 1.25 m/s reference kinematics.
- **Subject-1-scaled MJCF.** `walker2d_subject1.xml` (now in
  `assets/mjcf/`) replaces the stock Walker2d body proportions with
  ones that reflect adult human limb segments.
- **BC warm-start with PD rollouts.** Inverse dynamics ignores ground
  contact (wrong torques during stance). Roll out a PD controller
  inside the actual MuJoCo sim instead — contact-aware (s, a) pairs
  for supervised pretraining.

**Current canonical policy:** `walker2d_phase_cycle_s1scaled_sum_20260423-213031/`
(100M steps, Subject-1-scaled MJCF, single-cycle reference, per-joint
weighted-sum reward; replaces the prior 60M
`walker2d_phase_cycle_s1scaled_sum_20260422-175117/`, which is still on
disk). See [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

---

## Phase 4 — Adversarial methods (Brian's track)

**Decision:** Compare the engineered DeepMimic reward against learned
adversarial alternatives. Two new methods:

- **AMP (Adversarial Motion Priors)** — LSGAN discriminator on (s, s')
  joint-space pairs, zero-centered gradient penalty, bounded style
  reward. Combined with a forward-velocity task reward.
- **AIRL (Adversarial IRL)** — disentangled reward with shaping
  potential `f(s,s') + γh(s') − h(s)`.

**Status:** Both methods are described in the writeup
(`docs/reports/writeup_filled_1.docx` §4.4 and §6.3). Code is committed
at [`../src/walker2d/amp_walker2d.py`](../src/walker2d/amp_walker2d.py)
and [`../src/walker2d/airl_walker2d.py`](../src/walker2d/airl_walker2d.py)
(cherry-picked from upstream `bk-37/6955_Project@3e4c3fa` on 2026-04-28).
Both reuse `Walker2dPhaseAware` from the PPO track and the BC helpers
from `ppo_walker2d_phase.py`; only the reward signal differs.

**Key finding (writeup §6.3):** AMP collapses at 8 CPU envs. The
discriminator achieves near-perfect separation before the policy has
learned anything, driving style reward to ~0.03 and eliminating the
gradient. Mechanism: 24-D discriminator input space, only 349 expert
transitions, 8-env policy diversity insufficient to force discriminator
generalization. The Escontrela et al. paper used 4096 envs on a GPU
cluster — at that scale, policy diversity itself prevents memorization.

**Path forward:** Port the env to MuJoCo MJX (JAX, GPU) targeting
2,000–4,000 parallel envs on a single RTX 5090. See
[`ROADMAP.md`](ROADMAP.md).

---

## Phase 5 — The sign-error discovery (2026-04-28)

**What happened.** Brock asked the assistant to add a kinematic
playback tool so we could watch the reference data on a Walker2d
skeleton and confirm what motion the imitation pipeline was actually
trying to enforce. The new `src/diagnostics/view_reference.py` plays
the on-disk `assets/reference/gait_cycle_reference.npy` directly into
`qpos[3:9]` with the body drifted forward at 1.25 m/s.

The legs visibly walked **backward** — feet sweeping back-to-front
during stance, body translating in the wrong direction. Empirical FK
probes on Walker2d-v4 (set `qpos[3] = ±0.5`, read `body('foot').xpos`)
showed:

- positive hip joint → foot in **+x** direction (same as OpenSim's
  +hip_flexion = leg forward)
- positive ankle joint → toe rotates upward = dorsiflexion (same as
  OpenSim's +ankle_angle)
- foot geom toes point **+x** at neutral
- gym Walker2d-v4 reward is `forward_reward_weight * x_velocity`,
  positive for **+x** motion

OpenSim and Walker2d-v4 agree on sign for hip and ankle on this model.
The `walker = -opensim` negation in `extract_gait_cycle.py:38-43` and
`ulrich_loader.py` is correct only for the knee (OpenSim
`knee_angle ∈ [0°, +66°]` maps to Walker2d `leg_joint ∈ [-150°, 0°]`,
opposite signs). For hip and ankle it inverts the gait — heel-strike
pose becomes toe-off pose, dorsiflexion becomes plantarflexion.

The `METHODS.md § Joint sign convention` section and the legacy
comment block at `src/legacy/walker2d_v1/ppo_walker2d.py:91-103` both
asserted the all-six-joint flip as fact. They were wrong.

**Aftermath.**

- **Every PPO and AMP/AIRL run on disk was trained on a corrupted
  reference** — hip and ankle imitation targets are gait-inverted, knee
  is correct. The DeepMimic pose-tracking reward and the
  `forward_reward = +x_velocity` term were pulling the policy in
  *opposite* directions for two of three joints.
- The Phase 2 local-optima taxonomy (two-legged hopping, one-legged
  hopping, ankle paddling, foot-tapping in place) and the AMP/AIRL
  collapse described in writeup §6.3 are partly explained by this:
  policies were finding minimum-conflict equilibria between two
  contradictory reward signals on a self-inconsistent kinematic target.
- The engineered DeepMimic reward in [`REWARD_DESIGN.md`](REWARD_DESIGN.md)
  was tuned to patch reward-hacking exploits that *are themselves
  partly symptoms* of the corrupted reference. Each weight, sharpness,
  and termination threshold was set on a self-contradictory target.
  Some terms may not be needed at all on a clean reference.
- Trained checkpoints, BC warm-start data, and reward-tuning constants
  are all suspect. Code (`Walker2dPhaseAware`, phase observation, RSI,
  optimizer setup, AMP/AIRL discriminators, render and diagnostic
  scripts) is unaffected.

**Decision (2026-04-28).** Restart the imitation pipeline from the
ground up. Fix the data, then rebuild the simplest plausible
DeepMimic-faithful method first and only add complexity as we run into
specific failures. Treat the new run as a clean ablation: every
reward term, termination condition, and BC choice has to be
re-justified on the corrected reference.

**Aftermath (still 2026-04-28).** The loaders were corrected:
`extract_gait_cycle.py` and `ulrich_loader.py` now flip only the
knee. The on-disk `assets/reference/gait_cycle_reference.npy` was
regenerated and FK-verified to encode forward walking. Per-batch
progress on the rebuild is in [`RESTART_LOG.md`](RESTART_LOG.md). The
`src/legacy/walker2d_v1/ppo_walker2d.py` file still contains the old
all-six-joint flip — it's preserved as historical evidence and is not
on the active import path.

---

## Phase 5b — The kinematic-ceiling discovery (2026-04-29)

**What happened.** Post-restart Batches 1–3 reproduced the same
failure mode regardless of reward / aggregator / optimizer: a
"stiff-hip" gait with hip ROM ~2° while the corrected reference
asks for ~45°. The 19-experiment overnight sweep (Batch 3) tested
8 reward-aggregator/termination ablations, 4 AMP/AIRL warm-starts,
3 preview-obs runs, 1 SAC variant, and 3 curriculum runs; **all 19
landed in the same basin**. Read-at-the-time diagnosis: a reward
trap.

**The actual root cause** turned out to be one line in stock
`walker2d.xml`: `thigh_joint range="-150 0"`. The reference's hip
flexion peaks at +29.97°. ~68% of every reference cycle was outside
the joint range. The `restart_b2_xvel` policy spent 95.3% of frames
within 0.5° of the +0° wall — pre-Tier-0 reward sweeps had been
incapable of producing reference-like hip flexion regardless of
what they tweaked, because the simulator's constraint solver was
actively pulling the joint back from the commanded value.

**Two independent same-day diagnostics** caught this on two
machines:

- **Brock-Asus-Laptop, Batch 4 (commit `7724ff9`).** Built
  `assets/mjcf/walker2d_hipopen.xml` (`thigh_joint range="-30 60"`,
  permissive both sides). 2M training steps on the same xvel-5M
  recipe raised hip ROM from 1.8° to 91.5°. A 5M follow-up
  (`results/restart_b4_hipopen_5M/`) settled at ROM 63°, fwd vel
  1.40 m/s. Subsequent Batch 5 narrowing variants
  (`pose_scale20`, `min_joint`) tightened metrics but visual A/B
  found all three indistinguishable.
- **Brock-O11, Tier 0 ledger ([`TIER0_DIAGNOSTICS.md`](TIER0_DIAGNOSTICS.md)).**
  Built `assets/mjcf/walker2d_hiprelax.xml` (`thigh_joint range="-150 35"`,
  +5° headroom only — minimal-change variant). Tier 0 experiment C
  (3 seeds × 5M) recovered hip ROM 17–20°. Tracks reference shape
  and frequency but *under-amplitudes* the reference's 45° peak.

The two ablations bracket the residual reward gap: hipopen
*overshoots*, hiprelax *undershoots*. Both confirm morphology was
the dominant cause of pre-Tier-0 stiff-hip; the residual gap points
at `xvel_term=0.3` (a survival floor) plus the pose-tracking
mean-aggregator as the secondary cause.

**The code merge (commit `cf54014`).** Both machines independently
added the same `--xml` flag and `xml_file`-in-env_kwargs.json
persistence; the Asus laptop additionally removed the hardcoded
`_JNT_LO/_JNT_HI` constants in `Walker2dPhaseAware` (the previous
`+0.55 rad` hip-flexion advertisement was a lie the loaded MJCF's
`+0 rad` limit overruled — exactly the masking that hid this
hypothesis from earlier batches). Both XML variants ship in
`assets/mjcf/`; both result trees ship under `results/`. See
[`assets/mjcf/README.md`](../assets/mjcf/README.md) for picking
between them.

**Where Phase 5b lands.** Four candidate "current best" policies
on disk (the user has not picked a single favorite):

- hipopen: `restart_b4_hipopen_5M`, `restart_b5_pose_scale20`, `restart_b5_min_joint`
- hiprelax: `restart_b4_hiprelax_s11`

Tier 1 — restore `forward_reward = exp(-3·(v-1.25)²)`, drop
`xvel_term`, run on **both** MJCFs to bracket the residual reward
gap from below and above — is the next move
([`ROADMAP.md`](ROADMAP.md) item 0).

---

## What changes when the user "goes back to old ideas"

The user has noted that some of the original 3D / musculoskeletal scope
may be revisited. If that happens, the legacy code in
`src/legacy/musculoskeletal/` is the starting point — see
[`LEGACY_TRACKS.md`](LEGACY_TRACKS.md) for what each file does and what
to verify before re-running.

Specifically, **OpenCap data** is the lab-vs-field comparison angle from
the original proposal; if that experiment is revisited, the data is
preserved (gitignored on disk, not in the repo).

---

## Authors

**Brock Pereca** — phase-conditioned PPO, BC warm-start, reward design,
infrastructure.

**Brian Keller** — AMP and AIRL implementations, discriminator-collapse
analysis, MJX migration plan.

This timeline emphasizes Brock's track because the engineered-reward
PPO line drives the active narrative; Brian's AMP/AIRL code now lives
under `src/walker2d/` (cherry-picked from upstream on 2026-04-28).
