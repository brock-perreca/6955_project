# Reference papers

PDFs of the papers this project leans on. They live here so an agent
working in the repo can read the actual primary sources rather than
relying on summaries. Each entry below says **what the paper is** and
**when an agent in this repo should reach for it**.

The papers are grouped by which track of the project they support:

1. **Active pipeline** — phase-conditioned PPO + DeepMimic-style reward
   (Brock's track, the code that's committed).
2. **Adversarial imitation track** — GAIL → AMP / AIRL (Brian's track,
   committed in
   [`../../src/walker2d/amp_walker2d.py`](../../src/walker2d/amp_walker2d.py)
   and
   [`../../src/walker2d/airl_walker2d.py`](../../src/walker2d/airl_walker2d.py);
   this is also "the dream" of imitation-driven gait per
   [`../../CLAUDE.md`](../../CLAUDE.md)).
3. **Musculoskeletal imitation — the original-proposal direction** —
   muscle-driven walkers trained with imitation. The 3D / 80-muscle plan
   that was descoped for this semester but is the through-line back to
   the original question.
4. **Markerless motion capture** — the data side of the original
   lab-vs-field comparison.

For the project narrative around these, see
[`../../CLAUDE.md`](../../CLAUDE.md) ("the narrative arc") and
[`../PROJECT_TIMELINE.md`](../PROJECT_TIMELINE.md).

---

## 1. Active pipeline

### `Peng_2018_DeepMimic.pdf` — DeepMimic
> Peng, Abbeel, Levine, van de Panne. *DeepMimic: Example-Guided Deep
> Reinforcement Learning of Physics-Based Character Skills.*
> ACM Transactions on Graphics (SIGGRAPH), 2018.
> [arXiv:1804.02717](https://arxiv.org/abs/1804.02717) ·
> [project page + code](https://github.com/xbpeng/DeepMimic)

The paper our active reward is a 2D Walker2d-on-IK adaptation of. The
multi-term tracking reward (per-joint pose, end-effector, root, contact),
the phase-indexed reference clock, and Reference State Initialisation
(RSI) all come from here.

**Read this when:**
- modifying anything in [`../REWARD_DESIGN.md`](../REWARD_DESIGN.md) or
  the reward components in `src/walker2d/ppo_walker2d_phase.py` —
  DeepMimic is the canonical statement of what each tracking term is
  *supposed* to do before our exploit-driven tweaks.
- explaining why phase conditioning is the load-bearing fix for the
  Phase 1 / Phase 2 failures (see
  [`../PROJECT_TIMELINE.md`](../PROJECT_TIMELINE.md)).
- evaluating any new reward term — DeepMimic's exponential
  `exp(-k · err²)` form and per-joint scaling are the baseline to
  justify departures from.

---

## 2. Adversarial imitation track

### `Ho_2016_GAIL.pdf` — GAIL
> Ho, Ermon. *Generative Adversarial Imitation Learning.*
> NeurIPS, 2016. [arXiv:1606.03476](https://arxiv.org/abs/1606.03476)

The foundation paper for adversarial imitation. A discriminator learns
to tell expert state-action pairs from policy ones; the policy is
trained with the discriminator's score as reward. Every later
adversarial imitation method (AIRL, AMP, ASE) is a variant of this
recipe.

**Read this when:**
- thinking about the AMP track and why the discriminator collapses at
  small env counts (see writeup §6.3 and
  [`../ROADMAP.md § 1`](../ROADMAP.md)) — the issue is fundamentally
  GAIL's: a too-good discriminator gives no usable gradient. The
  vocabulary for diagnosing it (mode collapse, discriminator
  saturation, gradient vanishing) is established here.
- reviewing `src/legacy/walker2d_v1/gail_walker2d.py`. That file is the
  direct GAIL-on-Walker2d attempt that AMP/AIRL eventually superseded.
- comparing AIRL vs GAIL framings of the reward.

### `Peng_2021_AMP_animation.pdf` — AMP (original, character animation)
> Peng, Ma, Abbeel, Levine, Kanazawa. *AMP: Adversarial Motion Priors
> for Stylized Physics-Based Character Control.*
> ACM Transactions on Graphics (SIGGRAPH), 2021.
> [arXiv:2104.02180](https://arxiv.org/abs/2104.02180) ·
> [project page](https://xbpeng.github.io/projects/AMP/)

The paper that introduced AMP. Replaces DeepMimic's per-frame phase
tracking with a *style* discriminator over short state-transition
windows: the policy is rewarded for producing transitions that look
in-distribution to the mocap dataset, with task reward layered on top.
This is the framework Brian's track is implementing, and it's the
direct conceptual predecessor of [`Escontrela_2022_AMP_legged_robots.pdf`](#escontrela_2022_amp_legged_robotspdf--amp-for-legged-robots).

**Read this when:**
- defining or debugging the AMP discriminator in Brian's track —
  observation features, replay buffer, gradient penalty, and the
  logistic discriminator loss are all specified here.
- explaining why AMP is a structurally different alternative to the
  engineered DeepMimic reward, not just a swap of one tracking term
  for another.
- comparing AMP-style imitation against the active phase-conditioned
  pipeline in the writeup.

### `Escontrela_2022_AMP_legged_robots.pdf` — AMP for legged robots
> Escontrela, Peng, Yu, Zhang, Iscen, Goldberg, Abbeel.
> *Adversarial Motion Priors Make Good Substitutes for Complex Reward
> Functions.* IROS, 2022.
> [arXiv:2203.15103](https://arxiv.org/abs/2203.15103) ·
> [project page](https://sites.google.com/berkeley.edu/amp-in-real)

The robotics translation of AMP. Core claim: a learned style reward
from mocap can *replace* hand-engineered task rewards while still
producing naturalistic, sim-to-real-transferable gaits. Trains 4096
parallel environments on GPU — the regime we don't yet have access to.

**Read this when:**
- planning the MJX port in [`../ROADMAP.md § 1`](../ROADMAP.md). The
  4096-env scale here is the empirical anchor for "AMP works at scale,
  collapses at 8 envs" — Brian's writeup §6.3 result is consistent
  with this paper, not in tension with it.
- justifying the project pitch that AMP is the principled path away
  from the engineered DeepMimic reward (the through-line in
  [`../../CLAUDE.md`](../../CLAUDE.md) "the dream").
- comparing what an "expert manifold" looks like in their setting
  (~thousands of mocap clips) vs ours (one ~349-frame Ulrich cycle) —
  this is the clearest statement of why our discriminator memorizes.

### `Fu_2018_AIRL.pdf` — AIRL
> Fu, Luo, Levine. *Learning Robust Rewards with Adversarial Inverse
> Reinforcement Learning.* ICLR, 2018.
> [arXiv:1710.11248](https://arxiv.org/abs/1710.11248)

The "AIRL" half of Brian's "AMP / AIRL" track. Frames the
discriminator so that, at optimum, it factors out the dynamics and
recovers a *reward function* (specifically the advantage-shaped form
`f(s,a,s') = r(s) + γV(s') − V(s)`) instead of just a discriminator
score. That makes the recovered reward transferable across changes in
dynamics — the property GAIL doesn't have.

**Read this when:**
- writing or interpreting Brian's AIRL implementation. Pay particular
  attention to the state-only reward parameterisation and the way
  AIRL's discriminator absorbs the policy log-prob — these are the
  bits that distinguish it from GAIL/AMP at the loss level.
- discussing why imitation-derived rewards might be reusable across
  envs (e.g. across torque-Walker2d → muscle-Walker2d → 3D
  musculoskeletal). AIRL's transfer experiments are the canonical
  citation for that claim.
- evaluating whether scale-collapse pathologies in AMP also apply to
  AIRL — the discriminator architectures are similar enough that
  GAIL-family failure modes typically transfer.

---

## 3. Musculoskeletal imitation — original-proposal direction

These two papers are the closest current art to the original 3D /
80-muscle / OpenCap-vs-lab proposal scope (see
[`../PROJECT_TIMELINE.md § Phase 0`](../PROJECT_TIMELINE.md) and
[`../LEGACY_TRACKS.md`](../LEGACY_TRACKS.md)). Both came out *after*
the proposal was written and they are the strongest case for why the
original direction was a real research question.

### `Simos_2025_KINESIS.pdf` — KINESIS
> Simos, Chiappa, Mathis. *KINESIS: Motion Imitation for Human
> Musculoskeletal Locomotion* (also titled
> *Reinforcement learning-based motion imitation for physiologically
> plausible musculoskeletal motor control* in earlier versions).
> arXiv, 2025. [arXiv:2503.14637](https://arxiv.org/abs/2503.14637)

A model-free imitation framework for muscle-driven humanoids — up to
**290 muscles**, trained on ~1.8 hours of locomotion data, learns
muscle activations that correlate with real human EMG.

**Read this when:**
- considering a return to the musculoskeletal track. KINESIS
  demonstrates the core feasibility (muscle-driven imitation at scale)
  that the original proposal hypothesised. The MyoLeg / MyoSuite stack
  in `src/legacy/musculoskeletal/` is the same family of models.
- writing or defending the writeup section that explains why we
  descoped from muscle-driven to torque-driven Walker2d. KINESIS is the
  state-of-the-art benchmark for what the abandoned scope would have
  required.

### `Cotton_2025_KinTwin.pdf` — KinTwin
> Cotton. *KinTwin: Imitation Learning with Torque and Muscle Driven
> Biomechanical Models Enables Precise Replication of Able-Bodied and
> Impaired Movement from Markerless Motion Capture.*
> arXiv, 2025. [arXiv:2505.13436](https://arxiv.org/abs/2505.13436)

KINESIS' clinical sibling. Imitation policies on **LocoMujoco**
biomechanical models reproduce kinematics from markerless motion
capture across able-bodied and impaired participants, including
assistive-device gait. Demonstrates that imitation learning can solve
the inverse-dynamics problem (joint torques, muscle activations) from
markerless input alone.

**Read this when:**
- thinking about clinical / impaired-gait extensions, or any framing
  that combines the musculoskeletal track with the markerless-mocap
  track. This is exactly the lab-to-field transfer question the
  original proposal was about, executed with 2025 tools.
- evaluating whether [LocoMujoco](https://github.com/robfiras/loco-mujoco)
  could replace the MyoSuite-based setup in
  `src/legacy/musculoskeletal/`. It's a more recent, better-maintained
  imitation-friendly muscle env stack.

---

## 4. Markerless motion capture (data side)

### `Uhlrich_2023_OpenCap.pdf` — OpenCap
> Uhlrich, Falisse, Kidziński, Muccini, Ko, Chaudhari, Hicks, Delp.
> *OpenCap: Human movement dynamics from smartphone videos.*
> PLOS Computational Biology 19(10): e1011462, 2023.
> [DOI:10.1371/journal.pcbi.1011462](https://doi.org/10.1371/journal.pcbi.1011462) ·
> [project + code](https://www.opencap.ai/) ·
> [GitHub](https://github.com/stanfordnmbl/opencap-core)

OpenCap is the smartphone-based markerless mocap pipeline that the
original 6-condition study (R1–R6 in the proposal) was going to use as
the *field* arm of the lab-vs-field comparison. The resulting `.mot`
files are exactly the format
`src/legacy/musculoskeletal/data_utils.py` was written to consume — see
[`../DATA_SOURCES.md § OpenCap markerless mocap data`](../DATA_SOURCES.md).

**Read this when:**
- working with anything under `src/legacy/musculoskeletal/`, the
  legacy `OpenCap_data/` layout, or revisiting the original-proposal
  scope. The four IK sources (Mocap + HRNet + OpenPose-default +
  OpenPose-highAccuracy) come straight from this pipeline.
- contextualising why a "smartphone vs $150k lab" comparison was the
  core proposal pitch — OpenCap is *the* artifact that made the field
  arm of that comparison feasible in the first place.

### `Horsak_2023_smartphone_markerless_validity.pdf` — Smartphone-mocap concurrent validity
> Horsak, Eichmann, Lauer, Prock, Krondorfer, Siragy, Dumphart.
> *Concurrent validity of smartphone-based markerless motion capturing
> to quantify lower-limb joint kinematics in healthy and pathological
> gait.* Journal of Biomechanics 159, 111801, 2023.
> [DOI:10.1016/j.jbiomech.2023.111801](https://doi.org/10.1016/j.jbiomech.2023.111801)

Independent validation of smartphone markerless mocap (OpenCap-class
methods) against marker-based gold standard for lower-limb kinematics
in *healthy and pathological* gait — i.e. it's the empirical answer to
"how much fidelity do you actually lose when you swap a $150k lab for a
phone."

**Read this when:**
- justifying claims about how much error the lab→field swap introduces.
  This paper has the per-joint error budget; cite it rather than
  hand-waving.
- planning any future evaluation that involves comparing markerless and
  marker-based references.

---

## Other things worth linking (not in this folder)

These are not currently downloaded. Add them here if/when an agent
needs to read primary sources rather than abstracts.

### Adjacent papers
- **ASE** — Peng, Guo, Halper, Levine, Fidler. *ASE: Large-Scale
  Reusable Adversarial Skill Embeddings for Physically Simulated
  Characters.* SIGGRAPH 2022.
  [arXiv:2205.01906](https://arxiv.org/abs/2205.01906). AMP's successor
  for skill-conditioned imitation.
- **AddBiomechanics** — Werling et al. *AddBiomechanics Dataset:
  Capturing the Physics of Human Motion at Scale.* ECCV 2024.
  [paper](https://addbiomechanics.org/assets/AddBiomechanics_Dataset_Paper.pdf).
  Largest open biomechanics dataset; alternative to or supplement for
  the Ulrich data.

### Code repos worth knowing
- [`xbpeng/DeepMimic`](https://github.com/xbpeng/DeepMimic) — original
  DeepMimic.
- [`nv-tlabs/ASE`](https://github.com/nv-tlabs/ASE) — Nvidia's reference
  implementation of AMP and ASE in IsaacGym.
- [`robfiras/loco-mujoco`](https://github.com/robfiras/loco-mujoco) —
  MuJoCo locomotion env stack used by KinTwin; richer torque + muscle
  models than Walker2d.
- [`MyoHub/myosuite`](https://github.com/MyoHub/myosuite) — MyoSuite,
  the muscle-actuated env stack used by `src/legacy/musculoskeletal/`.
- [`stanfordnmbl/opencap-core`](https://github.com/stanfordnmbl/opencap-core)
  — OpenCap pipeline source.
- [`google-deepmind/mujoco_mjx`](https://github.com/google-deepmind/mujoco_mjx)
  — MuJoCo's JAX/XLA backend, the target of
  [`../ROADMAP.md § 1`](../ROADMAP.md) (AMP at 4k envs).
- [SimTK: Muscle Coordination Retraining to Reduce Knee Loading](https://simtk.org/projects/coordretraining)
  — the SimTK project that the local
  `CoordinationRetrainingData/forSimTK/` data comes from. This is the
  *origin* of the "Ulrich" treadmill data (note: actually Uhlrich,
  Stanford NMBL); the local junction `Ulrich_Treadmill_Data/` points
  here. See [`../DATA_SOURCES.md`](../DATA_SOURCES.md).

### Other directions, if/when they become relevant
- **DTW for motion evaluation.** No single canonical paper to grab;
  Sakoe & Chiba 1978 is the original DTW algorithm. The interesting
  question for [`../ROADMAP.md § 3`](../ROADMAP.md) is which DTW
  *variant* — soft-DTW (Cuturi & Blondel 2017,
  [arXiv:1703.01541](https://arxiv.org/abs/1703.01541)) is the
  differentiable choice if DTW ever needs to be in the loss.
- **Phase-Functioned Neural Networks** — Holden, Komura, Saito,
  *Phase-Functioned Neural Networks for Character Control*
  (SIGGRAPH 2017). Pre-DeepMimic phase-conditioning paper, useful as
  background if the phase representation needs revisiting.

---

## Conventions

- Filenames use `Firstauthor_Year_ShortDesc.pdf`. Keep this when adding
  new papers so the directory listing reads as a citation list.
- Cite papers in the rest of the docs by **filename + section anchor**
  (e.g. "see [`papers/Peng_2018_DeepMimic.pdf`](papers/Peng_2018_DeepMimic.pdf)
  + [`papers.md § Active pipeline`](papers/papers.md#1-active-pipeline)").
- If a paper here becomes outdated or wrong, prefer **annotating** this
  index over deleting the PDF — superseded references are still useful
  for understanding why earlier project decisions were made.
