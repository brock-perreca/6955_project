# `src/walker2d/` — phase-conditioned Walker2d imitation (active)

The active training pipeline. This is the code that produces the current
canonical walking policy (see
[`../../docs/PROJECT_STATUS.md`](../../docs/PROJECT_STATUS.md)).

## Files

| File | Role |
|---|---|
| `ppo_walker2d_phase.py` | **Main training script.** `Walker2dPhaseAware` env (25-D obs, fixed-clock phase, per-joint weighted-sum DeepMimic reward), optional BC warm-start via PD-rollout dataset, finetune support. CLI entry point. |
| `render_phase.py` | Render one or more trained phase-aware policies side-by-side. CLI: positional `result_dir:checkpoint[:label]` specs. |
| `extract_gait_cycle.py` | One-shot script: build `assets/reference/gait_cycle_reference.npy` from Subject 1 baseline IK. Detects two consecutive right heel strikes and saves the inter-strike segment. |
| `ulrich_loader.py` | `load_sto`, `load_ulrich_reference`, `ULRICH_ROOT`, `PROJECT_ROOT`. Imported by both `ppo_walker2d_phase.py` and `extract_gait_cycle.py`. Was originally in the legacy `ppo_walker2d.py` — extracted to free the active pipeline from the legacy import. |

## Quickstart

Run all commands from the project root.

```bash
# 1. Build the reference cycle (one-time, requires Ulrich data on disk)
python src/walker2d/extract_gait_cycle.py

# 2. Train from scratch (Subject-1-scaled MJCF, recommended)
python src/walker2d/ppo_walker2d_phase.py \
    --ref_cycle assets/reference/gait_cycle_reference.npy \
    --scale_model --num_envs 32 --total_steps 5e6

# 3. Render the result
python src/walker2d/render_phase.py results/<latest-run-dir>:final
```

For the full quickstart (BC warm-start, finetuning, comparing runs,
stock Walker2d geometry), see [`../../README.md`](../../README.md).

For the full method spec — env construction, reward components,
termination logic, BC warm-start mechanics, optimizer schedule — see
[`../../docs/METHODS.md`](../../docs/METHODS.md).

For why each reward term exists, see
[`../../docs/REWARD_DESIGN.md`](../../docs/REWARD_DESIGN.md).
