# Walker2d Gait Imitation — RL pipeline

Reinforcement learning pipeline for learning human-like walking gait
from Ulrich treadmill IK reference data, using MuJoCo's Walker2d-v4
torque-actuated model.

Joint project with **Brian Keller**. The current authoritative writeup
is [`docs/reports/writeup_filled_1.docx`](docs/reports/writeup_filled_1.docx).
The project compares two imitation methods on Subject 1's baseline
treadmill walking trial (1.25 m/s):

1. **Phase-conditioned PPO** with a multi-term DeepMimic-style reward,
   reference-state initialization, and an optional behavioral-cloning
   warm-start using PD rollouts inside MuJoCo (contact-aware, unlike
   `mj_inverse`). This is the active code in
   [`src/walker2d/`](src/walker2d/) and produces the canonical walking
   policy.
2. **Adversarial Motion Priors (AMP)** + **AIRL** as comparison
   methods (Brian's track), in
   [`src/walker2d/amp_walker2d.py`](src/walker2d/amp_walker2d.py) and
   [`src/walker2d/airl_walker2d.py`](src/walker2d/airl_walker2d.py).
   Both reuse `Walker2dPhaseAware` from the PPO track and replace the
   hand-crafted reward with a learned discriminator. AMP collapses at
   8 CPU envs (writeup §6.3); the recommended workflow today is to
   finetune from a working PPO+DeepMimic checkpoint. The full unblock
   is a MuJoCo MJX / GPU-parallel port — see
   [`docs/ROADMAP.md`](docs/ROADMAP.md). For implementation specifics
   (LSGAN vs AIRL discriminator, reward shaping, gradient penalty,
   adaptive freeze) see
   [`docs/METHODS.md`](docs/METHODS.md#adversarial-imitation-tracks-amp-and-airl-brians-track).

A secondary muscle-actuated track using MyoAssist (3D, 80-muscle) is
preserved as legacy code under
[`src/legacy/musculoskeletal/`](src/legacy/musculoskeletal/) — the
original proposal scope, kept for possible return.

---

## Documentation

This repo is documented for AI-first navigation. Start at
[`CLAUDE.md`](CLAUDE.md) (orientation hub) or jump straight to
[`docs/`](docs/):

- [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) — current snapshot
- [`docs/PROJECT_TIMELINE.md`](docs/PROJECT_TIMELINE.md) — chronological story
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — directory map + import graph
- [`docs/METHODS.md`](docs/METHODS.md) — env, reward, training, BC details
- [`docs/REWARD_DESIGN.md`](docs/REWARD_DESIGN.md) — reward + exploit taxonomy
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — planned future work
- [`docs/RUN_LOG.md`](docs/RUN_LOG.md) — past runs and failure modes
- [`docs/LEGACY_TRACKS.md`](docs/LEGACY_TRACKS.md) — frozen tracks
- [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) — Ulrich + OpenCap layouts

---

## Environment setup

**Active Walker2d pipeline:** Python 3.11 or 3.13 both work. Use a
plain venv (`python -m venv .venv`) or conda — either is fine.

**Legacy musculoskeletal track:** requires a **separate** Python 3.12
venv (e.g. `.venv-myo`). MyoSuite cannot share the active venv —
it pins `gymnasium==1.2.3` and `mujoco==3.6.0`, which would break
the Walker2d stack (gymnasium 0.29 + stable-baselines3 2.8). MyoSuite
also requires Python 3.10–3.12; 3.13 is unsupported upstream. See
the "MyoAssist (legacy musculoskeletal track)" section below.

```bash
# venv (works on Python 3.11 or 3.13)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
```

### Option A — NVIDIA RTX 5090 (CUDA 12.8)

Requires the [CUDA 12.8 toolkit](https://developer.nvidia.com/cuda-12-8-0-download-archive).

```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements/windows_5090.txt
```

### Option B — CPU only / other Windows machines

```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements/windows_cpu.txt
```

### Option C — macOS

```bash
pip install -r requirements/macos.txt
```

### Linking the Ulrich dataset

The loader expects `<repo>/Ulrich_Treadmill_Data/Subject{N}/IK/...`.
If the dataset arrives as the SimTK distribution
(`CoordinationRetrainingData/forSimTK/`), create a directory junction
or symlink. On Windows (no admin needed):

```cmd
mklink /J "Ulrich_Treadmill_Data" "CoordinationRetrainingData\forSimTK"
```

On macOS/Linux:

```bash
ln -s CoordinationRetrainingData/forSimTK Ulrich_Treadmill_Data
```

### MyoAssist (legacy musculoskeletal track)

MyoSuite is **not** in `requirements/<platform>.txt` and **must not**
be installed into the active Walker2d venv: it pins
`gymnasium==1.2.3` and `mujoco==3.6.0`, which would break the
Walker2d stack (`gymnasium==0.29.x` + `stable-baselines3==2.8.x`).
MyoSuite also requires Python 3.10–3.12 — 3.13 is unsupported
upstream (`Requires-Python: >=3.10,<=3.12`).

If you need the legacy musculoskeletal track, create a **separate
venv** on Python 3.12. The recipe used on this dev box (uses
[`uv`](https://docs.astral.sh/uv/) to install Python 3.12 if needed):

```bash
# 1. Install Python 3.12 (one-time, if you don't already have it)
uv python install 3.12

# 2. Create a sibling venv at .venv-myo (gitignored via .venv-*/)
uv venv --python 3.12 .venv-myo

# 3. Install myosuite into it
VIRTUAL_ENV=.venv-myo uv pip install myosuite          # bash
# $env:VIRTUAL_ENV=".venv-myo"; uv pip install myosuite  # PowerShell

# 4. Run legacy scripts via that venv directly (don't activate it
#    in a shell that has the main .venv active):
.venv-myo/Scripts/python src/legacy/musculoskeletal/ppo_myoassist.py ...
```

MyoSuite ≥ 2.12 dropped the `dm-control` dependency, so the old
`labmaze` / `dm-tree` build-tool dance (CMake + Bazelisk) is no
longer required. If you pin to ≤ 2.11 for any reason, you'll need
those tools back.

---

## Quickstart — active Walker2d pipeline

Run all commands **from the project root**.

### 1. Build the gait-cycle reference (one-time)

Requires Ulrich data on disk at `<repo>/Ulrich_Treadmill_Data/Subject{N}/IK/...`.
See [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) for the expected layout.

```bash
python src/walker2d/extract_gait_cycle.py
# → writes assets/reference/gait_cycle_reference.npy
# → writes docs/figures/gait_cycle_check.png
```

### 2. Train phase-aware imitation (from scratch)

> **Throughput tip (CPU box, 16 logical cores):** the script defaults
> to `--num_envs 32` (the value used to train the validated checkpoints
> in `results/`). On a 16-CPU desktop, sweeping showed `--num_envs 48`
> is the throughput peak (~7,600 vs ~7,200 env-steps/sec, +5%). Note
> that `num_envs × n_steps (=512)` sets the PPO rollout buffer, so
> changing `num_envs` shifts learning dynamics — bumping to 48 is fine
> for new experiments but for like-for-like comparisons against
> existing runs, stay at 32. Setting `OMP_NUM_THREADS=1` /
> `MKL_NUM_THREADS=1` was tested and **hurt** throughput by ~15% on
> this stack (PPO update on the main process benefits from BLAS
> threading on the 4096-batch update); leave them at default.

Stock Walker2d geometry:

```bash
python src/walker2d/ppo_walker2d_phase.py \
    --ref_cycle assets/reference/gait_cycle_reference.npy \
    --num_envs 32 --total_steps 5e6
```

Subject-1-scaled geometry (matches the current canonical run; requires
`assets/mjcf/walker2d_subject1.xml` to exist):

```bash
python src/walker2d/ppo_walker2d_phase.py \
    --ref_cycle assets/reference/gait_cycle_reference.npy \
    --scale_model --num_envs 32 --total_steps 5e6
```

With BC warm-start (collects ~200k PD-rollout samples, supervised MSE
for 10 epochs, then PPO):

```bash
python src/walker2d/ppo_walker2d_phase.py \
    --ref_cycle assets/reference/gait_cycle_reference.npy \
    --scale_model --bc_epochs 10 --bc_steps 200000 \
    --num_envs 32 --total_steps 5e6
```

### 3. Finetune from the current canonical checkpoint

```bash
python src/walker2d/ppo_walker2d_phase.py \
    --ref_cycle assets/reference/gait_cycle_reference.npy --scale_model \
    --finetune results/walker2d_phase_cycle_s1scaled_sum_20260422-175117/model.zip \
    --num_envs 32 --total_steps 5e6
```

Finetune mode lowers `learning_rate` to 1e-5, `target_kl` to 0.005,
and zeroes `ent_coef`.

### 4. Render a trained policy

```bash
# Single run, model.zip
python src/walker2d/render_phase.py \
    results/walker2d_phase_cycle_s1scaled_sum_20260422-175117:final

# Specific checkpoint with a custom label
python src/walker2d/render_phase.py \
    results/walker2d_phase_cycle_s1scaled_sum_20260422-175117:60000000:"60M steps"

# Compare multiple runs / checkpoints back-to-back
python src/walker2d/render_phase.py \
    results/walker2d_phase_cycle_s1scaled_sum_20260422-175117:50000000:"50M" \
    results/walker2d_phase_cycle_s1scaled_sum_20260422-175117:60000000:"60M"

# Render an older stock-Walker2d run — must override --xml
python src/walker2d/render_phase.py --xml walker2d.xml \
    results/walker2d_phase_full_sum_20260410-124935:18000000

# Pretrain / vanilla Walker2d (legacy renderer)
python src/legacy/walker2d_v1/render_walker.py \
    --model results/walker2d_pretrain_symmetry_<timestamp>/model.zip \
    --vanilla --steps 500
```

`render_phase.py` defaults: `--xml walker2d_subject1.xml`, `--eps 3`,
`--steps 2000`. Spec format is `result_dir:checkpoint[:label]` where
`checkpoint` is either an integer step count (looks under
`<result_dir>/checkpoints/model_<N>_steps.zip`) or the literal
`final` (loads `<result_dir>/model.zip`).

### 5. Diagnostics

```bash
python src/diagnostics/diag_cycle.py        # 3-cycle plot + seam check
python src/diagnostics/diag_ref.py          # joint ranges + FK upright
python src/diagnostics/diag_walker_mass.py  # Walker2d body masses
```

---

## Key flags — `ppo_walker2d_phase.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--ref_cycle` | — | Path to gait cycle `.npy` (required unless `--ref_all`) |
| `--num_envs` | 32 | Parallel environments |
| `--total_steps` | 5e6 | Total env steps |
| `--scale_model` | off | Use `walker2d_subject1.xml` instead of stock |
| `--finetune` | None | Pretrained `.zip` to finetune from |
| `--bc_epochs` | 0 | If >0, BC warm-start before PPO |
| `--bc_steps` | 200000 | PD-rollout samples for BC |
| `--bc_only` | off | Stop after BC, save BC-only model |
| `--imit_weight` | 4.0 | Per-joint pose tracking weight |
| `--vel_weight` | 1.0 | Per-joint velocity tracking weight |
| `--ee_weight` | 4.0 | End-effector (foot x + z) tracking weight |
| `--root_weight` | 2.0 | Root height + pitch tracking weight |
| `--contact_weight` | 1.0 | Stance-side contact alternation weight |
| `--swing_pen_weight` | 2.0 | Penalty on swing-foot ground contact |
| `--peak_bonus_weight` | 0.0 | Bonus at high-excursion phases |
| `--fwd_weight` | 0.0 | Forward velocity reward (off by default) |
| `--v_target` | 1.25 | Target forward speed (m/s) |
| `--action_rate_weight` | 0.0 | Anti-jerk penalty |
| `--product_reward` | off | Geometric mean of per-joint exps for `imit_r` |
| `--pose_term` | 0.9 rad | Hip/knee deviation termination |
| `--ankle_term` | 0.40 rad | Ankle deviation termination |
| `--no_pose_term` | off | Disable pose termination |

For the full flag list and the schedule details (LR decay, finetune
overrides), see [`docs/METHODS.md`](docs/METHODS.md) or
`python src/walker2d/ppo_walker2d_phase.py --help`.

---

## Quickstart — legacy musculoskeletal track (out of current scope)

Preserved for possible return. See
[`src/legacy/musculoskeletal/README.md`](src/legacy/musculoskeletal/README.md)
and [`docs/LEGACY_TRACKS.md`](docs/LEGACY_TRACKS.md) before running.

```bash
# Train MyoAssist
python src/legacy/musculoskeletal/ppo_myoassist.py --num_envs 16 --total_steps 1e7

# Two-stage BC + GAIL
python src/legacy/musculoskeletal/train.py --mode bc \
    --subject subject10 --trial walking1 --bc_epochs 200
python src/legacy/musculoskeletal/train.py --mode gail \
    --subject subject10 --trial walking1 \
    --bc_ckpt checkpoints/bc_policy_best.pt --gail_steps 500000
```
