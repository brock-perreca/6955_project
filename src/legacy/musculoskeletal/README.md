# `src/legacy/musculoskeletal/` — 3D 80-muscle MyoLeg track (preserved for return)

The original proposal scope: **3D, 80-muscle, 20-DoF MyoLeg** agents
trained on **OpenCap markerless** vs **lab-grade marker + force plate +
EMG** references, with SAC and SAC+GAIL. See
[`../../../report/Advanced_AI_Project_Report.pdf`](../../../report/Advanced_AI_Project_Report.pdf)
for the original proposal.

**Out of scope for the current writeup** but preserved here in case the
user returns to any of these ideas. Code is frozen at the pre-pivot state.

| File | Role |
|---|---|
| `ppo_myoassist.py` | PPO training on the MyoAssist env. Muscle-actuated. |
| `ppo_walk.py` | MyoSuite `myoLegWalk-v0` baseline run loop. |
| `render_myoassist.py` | Visualize a trained MyoAssist policy. |
| `train.py` | Driver for two-stage training: BC pretraining → GAIL. |
| `bc_policy.py` | Behavioural-cloning policy network + supervised trainer. |
| `gail.py` | GAIL discriminator + PPO inner loop. |
| `data_utils.py` | OpenCap / SimTK data loading + preprocessing. |
| `evaluate.py` | Quantitative policy evaluation. |

## Before re-running

MyoSuite must run in a **separate venv** from the active Walker2d
pipeline — it pins `gymnasium==1.2.3` and `mujoco==3.6.0`, which
conflict with the active stack (`gymnasium==0.29.x`,
`stable-baselines3==2.8.x`). Before running anything here:

- **Separate venv** at `.venv-myo` (gitignored via `.venv-*/`).
  Don't `pip install myosuite` into the main `.venv`.
- **Python 3.10–3.12.** MyoSuite declares
  `Requires-Python: >=3.10,<=3.12`. 3.13 is unsupported upstream
  (and on 3.13 pip silently falls back to MyoSuite ≤ 2.11, which
  pulls in `labmaze` with no Python 3.13 wheel).
- **MyoSuite ≥ 2.12** dropped the `dm-control` dependency, so the
  old CMake / Bazelisk requirement (for `dm-tree` and `labmaze`) is
  gone. Verified working: **2.12.1** on Python 3.12.
- **Setup recipe** (uses [`uv`](https://docs.astral.sh/uv/)):
  ```bash
  uv python install 3.12
  uv venv --python 3.12 .venv-myo
  VIRTUAL_ENV=.venv-myo uv pip install myosuite
  .venv-myo/Scripts/python src/legacy/musculoskeletal/<script>.py
  ```
- **OpenCap data** at `<repo>/OpenCap_data/subject{N}/...`.

See [`../../../docs/LEGACY_TRACKS.md`](../../../docs/LEGACY_TRACKS.md)
and [`../../../docs/DATA_SOURCES.md`](../../../docs/DATA_SOURCES.md)
for full layouts and data conventions.

If extending this track, the suggested approach is to **fork to a new
top-level directory** (e.g. `src/musculoskeletal/`) rather than editing
the preserved legacy files. That way the old code stays working as a
reference point.
