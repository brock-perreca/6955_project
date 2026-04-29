# `src/walker2d/` — Walker2d imitation tracks (active)

The active training code. Two tracks share the same env
(`Walker2dPhaseAware`), reference loader, and BC warm-start helpers,
but use different reward formulations:

- **Engineered DeepMimic reward** (`ppo_walker2d_phase.py`) — Brock's
  primary working baseline. Hand-crafted per-joint imitation reward;
  the canonical walking policy in
  [`../../docs/PROJECT_STATUS.md`](../../docs/PROJECT_STATUS.md) comes
  from this track.
- **Learned-reward AMP / AIRL** (`amp_walker2d.py`, `airl_walker2d.py`)
  — Brian's comparison track. Replaces the engineered reward with a
  discriminator over reference (s, s′) transitions. Collapses at 8-env
  CPU scale (writeup §6.3); the GPU/MJX-parallelised port is the
  unblock.

## Files

| File | Role |
|---|---|
| `ppo_walker2d_phase.py` | **PPO + DeepMimic reward (Brock's track).** `Walker2dPhaseAware` env (25-D obs, fixed-clock phase, per-joint weighted-sum DeepMimic reward), optional BC warm-start via PD-rollout dataset, finetune support. CLI entry point. |
| `amp_walker2d.py` | **AMP (Brian's track).** LSGAN discriminator over `(s, s′) = [q, dq]` transitions with zero-centered gradient penalty on expert samples. Combined reward `r = 0.35·r_task + 0.65·r_style` (paper weights) keeps a task gradient alive from step 1, so from-scratch is feasible. Subclasses `Walker2dPhaseAware` as `Walker2dAMP` (replaces imitation reward with `exp(-5·(v_x - v_target)²)`). |
| `airl_walker2d.py` | **AIRL (Brian's track).** Discriminator with shaping potential `g(s, s′) = f(s, s′) + γ·h(s′) - h(s)` so the recovered reward is dynamics-invariant. BCE loss + WGAN-GP, label smoothing, expert-noise augmentation, and an adaptive freeze when `frac_expert < 0.05` to stop the disc from running away. Cold-start collapses without `--finetune`; warm-starting from a working PPO walker is the recommended setup. |
| `render_phase.py` | Render one or more trained phase-aware policies side-by-side. CLI: positional `result_dir:checkpoint[:label]` specs. Works for any policy trained on `Walker2dPhaseAware` or its subclasses (AMP / AIRL). |
| `extract_gait_cycle.py` | One-shot script: build `assets/reference/gait_cycle_reference.npy` from Subject 1 baseline IK. Detects two consecutive right heel strikes and saves the inter-strike segment. |
| `ulrich_loader.py` | `load_sto`, `load_ulrich_reference`, `ULRICH_ROOT`, `PROJECT_ROOT`. Imported by `ppo_walker2d_phase.py`, `extract_gait_cycle.py`, and the AMP/AIRL scripts (for `--ref_all`). Was originally in the legacy `ppo_walker2d.py` — extracted to free the active pipeline from the legacy import. |

## Quickstart

Run all commands from the project root.

```bash
# 1. Build the reference cycle (one-time, requires Ulrich data on disk)
python src/walker2d/extract_gait_cycle.py

# 2a. PPO + DeepMimic — train from scratch (Subject-1-scaled MJCF, recommended)
python src/walker2d/ppo_walker2d_phase.py \
    --ref_cycle assets/reference/gait_cycle_reference.npy \
    --scale_model --num_envs 32 --total_steps 5e6

# 2b. AMP — paper weights, finetune off a working walker
python src/walker2d/amp_walker2d.py \
    --ref_cycle assets/reference/gait_cycle_reference.npy \
    --finetune results/<phase-run>/model.zip \
    --num_envs 32 --total_steps 5e6

# 2c. AIRL — same finetune pattern (cold-start tends to collapse)
python src/walker2d/airl_walker2d.py \
    --ref_cycle assets/reference/gait_cycle_reference.npy \
    --finetune results/<phase-run>/model.zip \
    --num_envs 32 --total_steps 5e6

# 3. Render any of the above
python src/walker2d/render_phase.py results/<run-dir>:final
```

For the full quickstart (BC warm-start, finetuning, comparing runs,
stock Walker2d geometry), see [`../../README.md`](../../README.md).

For the full method spec — env construction, reward components,
termination logic, BC warm-start mechanics, optimizer schedule — see
[`../../docs/METHODS.md`](../../docs/METHODS.md).

For why each reward term exists, see
[`../../docs/REWARD_DESIGN.md`](../../docs/REWARD_DESIGN.md).
