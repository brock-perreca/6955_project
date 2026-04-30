# Walker2d Gait Imitation

## Quick tour

- **Want to see a trained walker?** Render any policy under `results/`
  with `python src/walker2d/render_phase.py --live <run_dir>:final` —
  see the [render section](#4-render-a-trained-policy) below.
- **Want the story of how we got here?** Start at
  [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) (current snapshot)
  or [`docs/PROJECT_TIMELINE.md`](docs/PROJECT_TIMELINE.md) (the
  chronological version, including the pivot from a 3D
  musculoskeletal proposal to this 2D backup track).
- **Want to train your own?** Skip to [Environment setup](#environment-setup)
  and the [Quickstart](#quickstart--active-walker2d-pipeline).

## What's actually in here

Two imitation methods on the same phase-conditioned environment:

1. **Phase-conditioned PPO** with a multi-term DeepMimic-style reward,
   reference-state initialization, and an optional behavioral-cloning
   warm-start using PD rollouts inside MuJoCo (contact-aware, unlike
   `mj_inverse`). This is the active code in
   [`src/walker2d/`](src/walker2d/) and produces the canonical walking
   policy.
2. **Adversarial Motion Priors (AMP)** and **AIRL** as comparison
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
- [`docs/RESTART_LOG.md`](docs/RESTART_LOG.md) — recent batches (Batch 4 / 4b / 5)
- [`docs/TIER0_DIAGNOSTICS.md`](docs/TIER0_DIAGNOSTICS.md) — morphology-vs-reward Tier 0 verdict
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — directory map + import graph
- [`docs/METHODS.md`](docs/METHODS.md) — env, reward, training, BC details
- [`docs/REWARD_DESIGN.md`](docs/REWARD_DESIGN.md) — reward + exploit taxonomy
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — planned future work
- [`docs/RUN_LOG.md`](docs/RUN_LOG.md) — past runs and failure modes
- [`docs/LEGACY_TRACKS.md`](docs/LEGACY_TRACKS.md) — frozen tracks
- [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) — Ulrich + OpenCap layouts
- [`assets/mjcf/README.md`](assets/mjcf/README.md) — MJCF picker (hipopen / hiprelax / stock)
- [`scripts/README.md`](scripts/README.md) — tooling index (eval_hip_rom, debug_joint_range, tier0/, etc.)
- [`src/diagnostics/README.md`](src/diagnostics/README.md) — diagnostic-script index

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

After the 2026-04-29 Tier 0 / Batch 4 diagnosis there are **four**
candidate "current best" policies kept on disk for visual A/B (Brock
has not picked one yet); they split into two MJCF tracks. The recipes
below mirror the runs that produced them:

```powershell
# hipopen track — wide bracket [-30, +60] deg (Asus laptop, b4_hipopen_5M)
python src/walker2d/ppo_walker2d_phase.py `
    --ref_cycle assets/reference/gait_cycle_reference.npy `
    --xml walker2d_hipopen.xml `
    --xvel_term 0.3 --num_envs 8 --total_steps 5e6

# hiprelax track — minimal +5 deg headroom [-150, +35] (O11 box, b4_hiprelax_s11)
python src/walker2d/ppo_walker2d_phase.py `
    --ref_cycle assets/reference/gait_cycle_reference.npy `
    --xml walker2d_hiprelax.xml `
    --xvel_term 0.3 --num_envs 8 --total_steps 5e6
```

Stock `walker2d.xml` is the default if `--xml` is omitted, but it
caps `thigh_joint` at `[-150, 0] deg` which makes ~68% of the
reference unreachable — don't use it for a fresh run unless you
specifically want the pre-Tier-0 stiff-hip baseline. See
[`assets/mjcf/README.md`](assets/mjcf/README.md) for the MJCF picker.

To use the Subject-1-scaled body proportions instead of `--xml`, add
`--scale_model` (requires `assets/mjcf/walker2d_subject1.xml`, which
is missing on this checkout — see
[`docs/PROJECT_STATUS.md § Known gaps`](docs/PROJECT_STATUS.md#known-gaps-in-this-checkout)).
`--scale_model` and `--xml` are mutually exclusive.

> **Throughput tip (CPU box, 16 logical cores):** the script default is
> `--num_envs 16`. On a 16-CPU desktop, sweeping showed `--num_envs 48`
> is the throughput peak (~7,600 vs ~7,200 env-steps/sec, +5%) at the
> cost of larger PPO rollout buffer (`num_envs × n_steps=512`); for
> like-for-like comparisons against existing runs, stay at 8 (Batch 2
> recipe) or 16 (script default). `OMP_NUM_THREADS=1` /
> `MKL_NUM_THREADS=1` were tested and **hurt** throughput by ~15% on
> this stack — leave them at default.

With BC warm-start (collects ~200k PD-rollout samples, supervised MSE
for 10 epochs, then PPO):

```powershell
python src/walker2d/ppo_walker2d_phase.py `
    --ref_cycle assets/reference/gait_cycle_reference.npy `
    --xvel_term 0.3 --bc_epochs 10 --bc_steps 200000 `
    --num_envs 8 --total_steps 5e6
```

### 3. Finetune from one of the current-best checkpoints

```powershell
# Pick whichever candidate you want to push further:
python src/walker2d/ppo_walker2d_phase.py `
    --ref_cycle assets/reference/gait_cycle_reference.npy `
    --xml walker2d_hipopen.xml `
    --finetune results/restart_b4_hipopen_5M/model.zip `
    --num_envs 8 --total_steps 5e6

# (or restart_b5_min_joint, restart_b5_pose_scale20, restart_b4_hiprelax_s11.
#  Match --xml to the MJCF the source policy was trained against;
#  hiprelax_s11 was trained on walker2d_hiprelax.xml.)
```

Finetune mode lowers `learning_rate` to 1e-5, `target_kl` to 0.005,
and zeroes `ent_coef`.

### 4. Render a trained policy

Since the 2026-04-29 merge, `render_phase.py` auto-loads the MJCF
each policy was trained against from its `env_kwargs.json`, so the
`--xml` flag is no longer required for any run that includes
`xml_file` in its env_kwargs (every post-2026-04-29 run does).

```powershell
# Single run — auto-loads trained MJCF from env_kwargs.json
python src/walker2d/render_phase.py --live results/restart_b4_hipopen_5M:final

# Specific checkpoint with a custom label
python src/walker2d/render_phase.py `
    results/restart_b4_hipopen_5M:1000000:"1M steps"

# Compare all four current-best candidates back-to-back
python src/walker2d/render_phase.py --live `
    results/restart_b4_hipopen_5M:final `
    results/restart_b5_pose_scale20:final `
    results/restart_b5_min_joint:final `
    results/restart_b4_hiprelax_s11:final

# Pre-2026-04-29 runs need an explicit --xml (their env_kwargs.json
# lacks xml_file). Stock-walker2d runs:
python src/walker2d/render_phase.py --xml walker2d.xml `
    results/restart_b2_xvel:final

# Pre-restart scaled-MJCF run (requires the missing walker2d_subject1.xml):
python src/walker2d/render_phase.py --xml walker2d_subject1.xml `
    results/walker2d_phase_cycle_s1scaled_sum_20260423-213031:final

# Pretrain / vanilla Walker2d (legacy renderer)
python src/legacy/walker2d_v1/render_walker.py `
    --model results/walker2d_pretrain_symmetry_20260407-172719/model.zip `
    --vanilla --steps 500

# Bulk re-render every run dir to mp4 (PowerShell driver)
.\scripts\render_all_results.ps1
```

`render_phase.py` defaults: `--xml walker2d_subject1.xml` (only used
as fallback when env_kwargs.json doesn't carry `xml_file`),
`--eps 3`, `--steps 2000`. Spec format is `result_dir:checkpoint[:label]`
where `checkpoint` is either an integer step count (looks under
`<result_dir>/checkpoints/model_<N>_steps.zip`) or the literal
`final` (loads `<result_dir>/model.zip`).

### 5. Diagnostics + tooling

Per-script docs live in
[`src/diagnostics/README.md`](src/diagnostics/README.md) and
[`scripts/README.md`](scripts/README.md). Most-reached-for tools:

```bash
# Reachability gate: does the reference fit a given MJCF? (PNG + JSON)
python src/diagnostics/check_reference_jnt_range.py --xml walker2d_hipopen.xml

# Single-source-of-truth hip ROM metric (4-ep deterministic rollout)
python scripts/eval_hip_rom.py results/restart_b4_hipopen_5M

# End-to-end joint-range hypothesis verification (MJCF + ref + dynamics + policy)
python scripts/debug_joint_range_hypothesis.py

# Per-run dashboard PNG (sim/ref overlay, reward decomp, action hist, foot xz)
python src/diagnostics/run_dashboard.py results/restart_b4_hipopen_5M:final --steps 600

# Held-out biomech eval vs measured Subject-1 targets
python src/diagnostics/eval_biomech.py results/restart_b4_hipopen_5M:final `
    --out results/restart_b4_hipopen_5M_eval.json

# Writeup-ready biomech table + 6-panel figure
python scripts/biomech_report.py results/restart_b4_hipopen_5M_eval.json --rerollout

# Tier 0 experiment-C panel (3-seed dashboards + eval + mp4s + comparison plot)
python scripts/tier0/evaluate_C.py

# Reference / model sanity checks
python src/diagnostics/diag_cycle.py        # 3-cycle plot + seam check
python src/diagnostics/diag_ref.py          # joint ranges + FK upright
python src/diagnostics/diag_walker_mass.py  # Walker2d body masses
```

---

## Key flags — `ppo_walker2d_phase.py`

The current default reward is the DeepMimic 4-term sum (Eq. 6); see
[`docs/METHODS.md § Reward`](docs/METHODS.md#reward--deepmimic-four-term-sum).
Most exploit-patch terms are off by default and gated behind explicit
CLI flags — [`docs/METHODS.md § Full CLI reference`](docs/METHODS.md#full-cli-reference-ppo_walker2d_phasepy)
has the complete table. The most common flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--ref_cycle` | — | Path to gait cycle `.npy` (mutex with `--ref_all`) |
| `--num_envs` | 16 | Parallel environments |
| `--total_steps` | 5e6 | Total env steps |
| `--seed` | 0 | RNG seed |
| `--xml` | None | Custom MJCF filename under `assets/mjcf/` (e.g. `walker2d_hipopen.xml`, `walker2d_hiprelax.xml`); use `walker2d.xml` for the gym default. Mutex with `--scale_model`. |
| `--scale_model` | off | Use `walker2d_subject1.xml` instead of stock. Mutex with `--xml`. |
| `--finetune` | None | Pretrained `.zip` to finetune from |
| `--bc_epochs` | 0 | If >0, BC warm-start before PPO |
| `--pose_weight` | 0.65 | `r_p` weight |
| `--vel_weight`  | 0.10 | `r_v` weight |
| `--ee_weight`   | 0.15 | `r_e` weight |
| `--root_weight` | 0.10 | `r_c` weight |
| `--pose_scale`  | 10.0 | `k_p` |
| `--xvel_term`   | -∞ | Forward-velocity floor termination (the `0.3` recipe produced every current-best candidate) |
| `--pitch_term`  | 0.3 | Pitch-magnitude termination (rad) |
| `--product_reward` / `--min_joint_pose` | off | Aggregator alternatives for `r_p` |
| `--preview_k`   | 1 | Frames of upcoming `q_ref` exposed in obs |
| `--ref_root_drop` | 0.0 | Lower the FK-derived reference root-height target by this many meters. Stock-geometry contact-clearance ablation. |
| `--out_dir`     | None | Override output directory |

For everything else (BC parameters, exploit-patch weights, joint-term
thresholds, AMP/AIRL flags), use `--help` or
[`docs/METHODS.md`](docs/METHODS.md).

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
