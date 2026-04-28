# `src/legacy/musculoskeletal/` — 3D 80-muscle MyoLeg track (preserved for return)

The original proposal scope: **3D, 80-muscle, 20-DoF MyoLeg** agents
trained on **OpenCap markerless** vs **lab-grade marker + force plate +
EMG** references, with SAC and SAC+GAIL. See
[`../../../docs/reports/Advanced_AI_Project_Report.pdf`](../../../docs/reports/Advanced_AI_Project_Report.pdf)
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

The musculoskeletal stack depends on heavy system libraries that have
moved over time. Before running anything in this directory:

- **Python 3.11** required (MyoSuite's C++ deps don't build on 3.12+).
- **CMake** on PATH (for `dm-tree`).
- **Bazelisk** aliased to `bazel` (for `labmaze`).
- **MyoSuite** version compatible with current MuJoCo / Gymnasium —
  last verified early April 2026.
- **OpenCap data** at `<repo>/OpenCap_data/subject{N}/...`.

See [`../../../docs/LEGACY_TRACKS.md`](../../../docs/LEGACY_TRACKS.md)
and [`../../../docs/DATA_SOURCES.md`](../../../docs/DATA_SOURCES.md)
for full layouts and data conventions.

If extending this track, the suggested approach is to **fork to a new
top-level directory** (e.g. `src/musculoskeletal/`) rather than editing
the preserved legacy files. That way the old code stays working as a
reference point.
