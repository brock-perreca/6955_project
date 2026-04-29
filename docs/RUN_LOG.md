# Run log — pre-restart demo runs and failure modes

**Purpose:** preserve the four canonical exploit demos from the Phase
2 symmetry-pretrain detour, with reproduce/render commands.
**Read this when:** writing about reward-hacking failure modes, or
you want to see the local-optima taxonomy in motion (literally — each
run has a 500-frame keepsake video).
**Adjacent:** [`REWARD_DESIGN.md § Exploit taxonomy`](REWARD_DESIGN.md#exploit-taxonomy-writeup-62-goodharts-law-cases)
for the analysis of these failure modes ·
[`PROJECT_TIMELINE.md § Phase 2`](PROJECT_TIMELINE.md#phase-2--symmetry-reward-pretraining-detour-april-78-2026)
for the chronological story · [`RESTART_LOG.md`](RESTART_LOG.md) for
post-restart batches.

> Every run on this page (and every PPO/AMP/AIRL run under
> `results/walker2d_phase_*/` not listed here) was trained against
> the **inverted reference** (hip + ankle channels gait-flipped).
> See [`PROJECT_TIMELINE.md § Phase 5`](PROJECT_TIMELINE.md#phase-5--the-sign-error-discovery-2026-04-28).
> Don't branch new work off these checkpoints. They are kept as
> behavioral demos.

---

## Setup

Run from the project root with the active venv. The render entry
point is the legacy renderer (these runs were trained on the legacy
`Walker2dContactWalk` env, not the active phase-aware env):

```
python src/legacy/walker2d_v1/render_walker.py \
    --model results/walker2d_pretrain_symmetry_<TIMESTAMP>/model.zip \
    --vanilla --steps 500
```

All four runs were trained with `pretrain_walker2d.py --mode symmetry`
on PPO, 32 envs, 5M steps. Differences are in `--weight`, gravity
multiplier, and ankle torque cap (set in `make_env()` of
`pretrain_walker2d.py`). The single training command is:

```
python src/legacy/walker2d_v1/pretrain_walker2d.py --mode symmetry \
    --num_envs 32 --total_steps 5e6 [--weight W]
```

---

## The four demos

| Run dir | Failure mode | `--weight` | Gravity | Ankle cap | Reward at 5M |
|---|---|---|---|---|---|
| `walker2d_pretrain_symmetry_20260407-111838` | **Two-legged hopping** with some swing | 3.0 | 2× | ±0.3 | 1611 |
| `walker2d_pretrain_symmetry_20260407-114136` | **One-legged hopping**, trailing leg stabilising | 3.0 | 2× | ±0.3 | 3271 |
| `walker2d_pretrain_symmetry_20260407-172719` | **Ankle paddling** (no aerial phase) | 8.0 | 2.5× | ±0.3 | ~1280 |
| `walker2d_pretrain_symmetry_20260408-110759` | **Stands still and taps feet** | 8.0 | 2.5× | ±0.3 | ~1140 |

The two `weight=3.0` runs have identical commands; the seed differs
(reward landscape has both basins). The two `weight=8.0` runs differ
only in `forward_reward_weight` (lowered interactively during the
standing-and-tapping run; exact value not logged — try 0.1 in
`make_env()` to reproduce).

Each demo is referenced by [`REWARD_DESIGN.md`](REWARD_DESIGN.md) as
the visual evidence for the corresponding Goodhart's-Law case.

---

## Pre-restart phase-aware canonical runs

Listed in [`PROJECT_STATUS.md § Comparison runs on disk`](PROJECT_STATUS.md#comparison-runs-on-disk).
All trained against the inverted reference; rendering them works (with
`--xml walker2d.xml` for stock-geometry runs, or with the missing
`walker2d_subject1.xml` for scaled runs) but the kinematics are
hip-and-ankle-inverted relative to OpenSim.
