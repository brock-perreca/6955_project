# Legacy tracks

**Purpose:** describe what's frozen in [`../src/legacy/`](../src/legacy/),
why it's frozen, and what to verify before re-running.
**Read this when:** you find yourself reaching into a legacy file to
add or extend a feature, or you're considering revisiting the original
3D / musculoskeletal scope.

**Do not extend without confirming with the user first.** If you want
to add features here, the right move is almost always to ask whether
the new feature should go into the active pipeline under
[`../src/walker2d/`](../src/walker2d/) instead.

For the chronological story of why each track exists, see
[`PROJECT_TIMELINE.md`](PROJECT_TIMELINE.md).

---

## `src/legacy/walker2d_v1/` — earlier 2D Walker2d attempts

These are the Walker2d scripts from before phase conditioning was
adopted. They share heritage with the active pipeline (`Walker2d-v4`
torque actuation, Ulrich IK reference) but use **different envs and
different reward designs** that were superseded.

| File | Status | Why it's legacy | Safe to delete? |
|---|---|---|---|
| `ppo_walker2d.py` | Frozen | Phase-blind imitation. Three compounding bugs (no resampling, phase-blind obs, concatenated 413k-frame ref) all closed in `ppo_walker2d_phase.py`. Loaders extracted to `src/walker2d/ulrich_loader.py`. **Still applies the old all-six-joint sign flip** that the active loader corrected on 2026-04-28; do not re-import these helpers. | No — preserved for historical reference + blame |
| `pretrain_walker2d.py` | Dead end | Symmetry-reward pretraining without a reference. Hit four canonical local optima (hopping, paddling, standing). The fix turned out to be phase conditioning, not reward shaping. | No — produces the demo runs in `RUN_LOG.md` |
| `gail_walker2d.py` | Dead end | GAIL approach for Walker2d. Not part of the writeup. Superseded in the active code by `src/walker2d/amp_walker2d.py` (LSGAN-style AMP) and `src/walker2d/airl_walker2d.py` (AIRL with shaping potential). | No — kept as a comparison baseline against the current AMP / AIRL implementations |
| `render_walker.py` | Frozen | Renderer for the legacy `Walker2dImitation` env and vanilla Walker2d-v4 (`--vanilla`). Still useful for rendering legacy checkpoints (e.g. `results/walker2d_pretrain_symmetry_*/`). | No |

**Reproduce/render commands** for each demo in this group are in
[`RUN_LOG.md`](RUN_LOG.md).

### What to do if a legacy walker2d_v1 file matters again

- If you want to re-run `pretrain_walker2d.py`: it stands alone (no
  imports from other project files). Should still work.
- If you want to re-run `ppo_walker2d.py`: same. The loaders inside it
  are preserved (duplicated with `src/walker2d/ulrich_loader.py`); they
  can read Ulrich data from the standard `Ulrich_Treadmill_Data/` layout
  off the project root.
- If you want to extend (e.g. add a new reward variant to legacy):
  **ask first.** The active pipeline almost certainly has a more
  principled place for the new feature.

---

## `src/legacy/musculoskeletal/` — original 3D 80-muscle plan

This is the original proposal scope: **3D, 80-muscle, 20-DoF MyoLeg**
agents trained on **OpenCap markerless** vs **lab-grade marker + force
plate + EMG** references, with SAC and SAC+GAIL. See
[`PROJECT_TIMELINE.md § Phase 0`](PROJECT_TIMELINE.md#phase-0--original-proposal-proposal-stage-see-reportsadvanced_ai_project_reportpdf)
and `docs/reports/Advanced_AI_Project_Report.pdf`.

It was dropped because the 3D + muscle + adversarial + multi-condition
combination was too ambitious for a one-semester course project. The
user has flagged that they **may revisit** these ideas — the code is
preserved precisely to make that easier.

If you're considering picking this track up again, read the two recent
musculoskeletal-imitation papers in [`papers/`](papers/) first:
[`Simos_2025_KINESIS.pdf`](papers/Simos_2025_KINESIS.pdf) (290-muscle
imitation, ~1.8h locomotion data) and
[`Cotton_2025_KinTwin.pdf`](papers/Cotton_2025_KinTwin.pdf)
(LocoMujoco-based muscle imitation from markerless mocap, including
impaired gait). Both demonstrate that the original-proposal direction
is now empirically tractable. See
[`papers/papers.md § Musculoskeletal imitation`](papers/papers.md#3-musculoskeletal-imitation--original-proposal-direction).

| File | Role |
|---|---|
| `ppo_myoassist.py` | PPO training on the MyoAssist env. Muscle-actuated. |
| `ppo_walk.py` | MyoSuite `myoLegWalk-v0` baseline run loop. |
| `render_myoassist.py` | Visualize a trained MyoAssist policy. |
| `train.py` | Driver for two-stage training: BC pretraining → GAIL. |
| `bc_policy.py` | Behavioural-cloning policy network + supervised trainer. |
| `gail.py` | GAIL discriminator + PPO inner loop. |
| `data_utils.py` | OpenCap / SimTK data loading + preprocessing. |
| `evaluate.py` | Quantitative policy evaluation (per-step kinematics). |

### What to verify before re-running

MyoSuite's pinned deps conflict with the active Walker2d venv, so the
musculoskeletal track lives in its own environment. Before running
anything in this directory, confirm:

- **Separate venv.** MyoSuite **must not** share the active `.venv` —
  it pins `gymnasium==1.2.3` and `mujoco==3.6.0`, which would break
  the Walker2d stack (`gymnasium==0.29.x`,
  `stable-baselines3==2.8.x`, `mujoco>=3.1`). Use a sibling venv
  named `.venv-myo` (gitignored via `.venv-*/`).
- **Python 3.10–3.12.** MyoSuite declares
  `Requires-Python: >=3.10,<=3.12`. 3.13 is unsupported upstream
  (and pip will silently fall back to MyoSuite ≤ 2.11 plus the
  unbuildable `labmaze`).
- **MyoSuite version.** Verified working: **2.12.1** on Python 3.12.
  This release dropped the `dm-control` dependency, so the old
  CMake / Bazelisk requirement (for `dm-tree` and `labmaze`) is gone.
  If you pin to ≤ 2.11 you will need those build tools back.
- **Setup recipe** (uses [`uv`](https://docs.astral.sh/uv/)):
  ```bash
  uv python install 3.12
  uv venv --python 3.12 .venv-myo
  VIRTUAL_ENV=.venv-myo uv pip install myosuite
  ```
  Run scripts directly: `.venv-myo/Scripts/python src/legacy/musculoskeletal/...`.
- **Data:** `OpenCap_data/` directory is gitignored. Layout described
  in [`DATA_SOURCES.md`](DATA_SOURCES.md).

If extending this track, the best approach is probably to **fork to a
new directory** (e.g. `src/musculoskeletal/`) rather than modifying the
preserved legacy files. That way the old code stays working as a
reference point.

---

## What's *not* in `src/legacy/`

A few things that look like they could be legacy but aren't:

- **`src/walker2d/ulrich_loader.py`** — extracted from the original
  `ppo_walker2d.py`, but it's the *active* loader. Imports from this
  file are encouraged.
- **`src/walker2d/extract_gait_cycle.py`** — uses Ulrich data, but it's
  the active script for building `assets/reference/gait_cycle_reference.npy`.
  Re-run it whenever the canonical reference cycle needs to change.
- **`src/diagnostics/`** — not legacy, just standalone. Not on the
  training path but actively useful for sanity checks.
- **`assets/mjcf/walker2d_custom.xml`** — predates the decision to stick
  with stock Walker2d-v4 geometry. Could be deleted but the user
  prefers preservation. Don't reference from active code.

---

## Asking before extending

If you (an AI assistant or human collaborator) find yourself reaching
into `src/legacy/` to add something, please pause and check whether the
right move is one of:

1. The active pipeline already has a place for this (most common).
2. A new file in `src/walker2d/` or a new sibling directory.
3. A genuine extension to the legacy code, after confirming with the user
   that the legacy track is being revisited.

Option 3 is rare. The default is option 1 or 2.
