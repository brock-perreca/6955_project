# Project status — current snapshot

*Last updated: 2026-04-29.*

> **STATUS: RESTARTING.** The reference data has been corrected
> (`extract_gait_cycle.py` and `ulrich_loader.py` now flip only the
> knee — `walker = -opensim` was wrong for hip and ankle on this
> Walker2d model). The on-disk
> `assets/reference/gait_cycle_reference.npy` was regenerated and
> verified by FK probe to encode forward walking. The imitation
> pipeline is being rebuilt from a DeepMimic-faithful baseline; per-
> batch progress lives in [`RESTART_LOG.md`](RESTART_LOG.md).
> Pre-restart runs (the `walker2d_phase_*` directories below) and the
> exploit-patch reward terms (`swing_pen`, `contact_r`, per-joint
> sharpness) were tuned against the corrupted reference and are kept
> only as a historical record — see
> [`REWARD_DESIGN.md`](REWARD_DESIGN.md) for the warning at the top.
> [`METHODS.md § Joint sign convention`](METHODS.md#joint-sign-convention)
> documents the corrected facts.

For a chronological story of how we got here, see
[`PROJECT_TIMELINE.md`](PROJECT_TIMELINE.md). For the formal writeup, see
[`reports/writeup_filled_1.docx`](reports/writeup_filled_1.docx) (joint
with Brian Keller).

---

## What this project is

A reinforcement learning study of **gait imitation on the MuJoCo
Walker2d-v4 planar biped**, conditioned on inverse-kinematics reference
data from the Ulrich treadmill walking dataset (Subject 1, 1.25 m/s).

Two complementary imitation methods are studied:

1. **Phase-conditioned PPO + multi-term DeepMimic reward** — the primary
   track. Active code in [`../src/walker2d/`](../src/walker2d/).
   *Working*: produces a policy with heel-strike events, bilateral foot
   alternation, and 2000-step sustained walking.
2. **Adversarial Motion Priors (AMP) + AIRL** — comparison track.
   Brian's code, committed at
   [`../src/walker2d/amp_walker2d.py`](../src/walker2d/amp_walker2d.py)
   and
   [`../src/walker2d/airl_walker2d.py`](../src/walker2d/airl_walker2d.py)
   (cherry-picked from upstream `bk-37/6955_Project@3e4c3fa` on
   2026-04-28). *Pending GPU/MJX port*: AMP collapses at 8 CPU envs;
   the recommended workflow today is to finetune from a working
   PPO+DeepMimic checkpoint via `--finetune`.

Three top-line scientific contributions (from the writeup):

- A working phase-conditioned imitation policy on real human IK data.
- A mechanistic taxonomy of reward-hacking failure modes (ankle paddling,
  one-legged hopping, toe-walking) framed as canonical Goodhart's-Law
  cases. See [`REWARD_DESIGN.md`](REWARD_DESIGN.md).
- A characterization of AMP's discriminator collapse at small env counts
  (writeup §6.3) and the mechanism that explains it.

---

## How to validate progress (the agent-facing eval loop)

Added 2026-04-29: `eval_biomech.py` now compares every checkpoint
against a *measured* reference (computed from Subject 1's force plates
and IK by `extract_reference_biomech.py`), and `scripts/biomech_report.py`
renders a writeup-ready markdown table + 6-panel figure. After any
training batch:

```
python src/diagnostics/eval_biomech.py --xml walker2d.xml --eps 6 --steps 2500 \
    results/<run>:final:<label> --out results/<run>_eval.json --csv results/biomech_history.csv
python scripts/biomech_report.py results/<run>_eval.json --rerollout
```

The eval JSON's `vs_reference` block carries `delta` and `pct_err` per
metric, plus a single `progress_score` in [0, 4]. See
[`METHODS.md § Diagnostic scripts`](METHODS.md#diagnostic-scripts-srcdiagnostics)
for details.

## What's currently running

- **Active training script:** `src/walker2d/ppo_walker2d_phase.py` —
  rewritten 2026-04-28 as a DeepMimic-faithful baseline (sum of four
  `exp(−k·err²)` terms: pose / vel / EE / root). All exploit-patch
  terms (swing_pen, contact_r, per-joint sharpness/weights, per-joint
  pose/ankle thresholds, BC) are off-by-default kwargs/CLI flags. See
  the module docstring and [`RESTART_LOG.md`](RESTART_LOG.md).
- **Active reference:** `assets/reference/gait_cycle_reference.npy` —
  one clean stride from Ulrich Subject 1 baseline (56 frames @ 50 Hz,
  resampled to 140 frames @ 125 Hz inside the env). FK-verified after
  the 2026-04-28 sign fix to encode forward walking.
- **Current best policy:**
  `results/restart_b2_xvel/model.zip` (5M steps, stock walker2d.xml,
  single-cycle reference, 8 envs). Diff from default DeepMimic
  baseline: one CLI flag, `--xvel_term 0.3` (forward-velocity floor
  termination). Visual review: best policy in the project's history
  (Brock, 2026-04-28). Quantitative residuals:
  - Cadence ~3× too fast (stride 0.32 s vs reference 1.12 s).
  - Hip excursion stiff (`hip_r ∈ [-12°, +2°]` in batch-1 diagnostic;
    visual confirms thighs barely move). Cause and primary
    target for the next batch — see batch 3 in
    [`RESTART_LOG.md`](RESTART_LOG.md).
- **Pre-restart canonical** (kept for historical comparison):
  `results/walker2d_phase_cycle_s1scaled_sum_20260423-213031/model.zip`
  (100M, scaled MJCF, engineered reward, **trained on the corrupted
  reference** — kinematics are gait-inverted on hip and ankle).

## Comparison runs on disk

| Result dir | Steps | Notes |
|---|---|---|
| **`results/restart_b2_xvel/`** | **5M** | **Current best.** DeepMimic 4-term reward + `--xvel_term 0.3`. Stock walker2d.xml, seed=2. ep_len 2120, all-episode 2500-step survival on eval. |
| `results/restart_b2_k30/` | 5M | DeepMimic + `--pose_scale 30`. Unstable; 4/6 eval episodes fall in <120 steps. Tighter pose alone without an xvel floor doesn't escape. |
| `results/restart_b1_dm/` | 2M (killed at 2.5M) | Pure DeepMimic baseline. Stand-and-wiggle exploit — long episodes hide stiff hips + zero forward motion. |
| `results/restart_b1_dm_bc/` | 2M (killed at 2.34M) | DeepMimic + 5-epoch BC. Same exploit, marginally varied across seeds. |
| `results/walker2d_phase_cycle_s1scaled_sum_20260423-213031/` | 100M | Pre-restart canonical, scaled MJCF, single-cycle ref. **Trained on inverted reference.** From upstream commit `3e4c3fa`. |
| `results/walker2d_phase_cycle_s1scaled_sum_20260422-175117/` | 60M | Earlier pre-restart canonical, scaled MJCF. |
| `results/walker2d_phase_full_sum_20260410-124935/` | 18M | Stock Walker2d, full-trial ref, uniform-k=8 (pre-restart). |
| `results/walker2d_phase_full_sum_20260410-105306/` | 45M | Earlier pre-restart DeepMimic-reward run. |
| `results/walker2d_phase_cycle_sum_20260409-211537/` | 10.5M | First single-cycle reference run (pre-restart). |
| `results/walker2d_pretrain_symmetry_20260407-172719/` | 5M | Symmetry-pretrain ankle-paddling demo (legacy). |

Render any of them with (PowerShell):

```
python src/walker2d/render_phase.py --xml walker2d.xml --live results/restart_b2_xvel:final
python src/walker2d/render_phase.py --xml walker2d.xml --mp4 docs/figures/foo.mp4 results/restart_b2_xvel:final
```

The default `--xml` is `walker2d_subject1.xml` (missing on this checkout).
For all post-restart runs and stock-geometry pre-restart runs, override
with `--xml walker2d.xml`.

---

## What still needs to happen

For the **current writeup-driven scope**, see [`ROADMAP.md`](ROADMAP.md).
Highlights:

1. **MJX/GPU port** to make AMP function (4,000-env parallelism).
2. **Multi-step future context** in the observation
   (`[q_ref_φ, q_ref_{φ+1}, …, q_ref_{φ+K−1}]`) to smooth wrap-around
   jerkiness at the gait-cycle seam.
3. **DTW-based evaluation and reference selection** — held-out
   shape-fidelity metric independent of phase drift; clustering signal
   for picking reference cycles from the Ulrich dataset.
4. **Multi-cycle / multi-subject reference** for temporal smoothness
   and robustness.

For the **possibility of revisiting the original 3D / musculoskeletal
scope**, see [`LEGACY_TRACKS.md`](LEGACY_TRACKS.md). Code is preserved
in `src/legacy/musculoskeletal/`.

---

## Known gaps in this checkout

- `assets/mjcf/walker2d_subject1.xml` — **missing**. The current canonical
  run was trained against it. Required for `--scale_model` and is the
  default `--xml` for `render_phase.py`. Must be regenerated or copied
  from the user's other machine before training/rendering the canonical
  policy. Stock-Walker2d runs (`walker2d_phase_full_sum_*` and
  `walker2d_phase_cycle_sum_*`) load and roll out fine without it —
  pass `--xml walker2d.xml` to `render_phase.py`.
- `amp_walker2d.py` and `airl_walker2d.py` — checked in (cherry-picked
  from upstream commit `3e4c3fa` on 2026-04-28). Both relocated from
  the repo root into `src/walker2d/` and rewired to import the active
  loader; `--ref_cycle` works out-of-the-box, `--ref_all` no longer
  receives per-trial segment lengths (boundary transitions are not
  filtered out of the expert buffer).
- `Ulrich_Treadmill_Data/` — gitignored. Users supply their own copy at
  `<repo>/Ulrich_Treadmill_Data/Subject{1..10}/IK/walking_*/output/results_ik.sto`.
  See [`DATA_SOURCES.md`](DATA_SOURCES.md).

### Per-machine setup notes

- **Current laptop (no GPU, Python 3.13):** `Ulrich_Treadmill_Data/`
  is a directory junction to `CoordinationRetrainingData/forSimTK/`.
  Venv at `.venv/` was built with the CPU build of PyTorch 2.7.0 and
  `requirements/windows_cpu.txt`. MyoSuite is **not** in this venv —
  it pins `gymnasium==1.2.3` / `mujoco==3.6.0` (incompatible with
  the active Walker2d stack) and requires Python 3.10–3.12. For the
  legacy `src/legacy/musculoskeletal/` track, a sibling venv lives
  at `.venv-myo/` (Python 3.12, MyoSuite 2.12.1, verified
  `myoLegWalk-v0` reset+step). See `README.md` "MyoAssist (legacy
  musculoskeletal track)" for the recipe.
- **numpy must be 2.x** to load existing checkpoints — their pickle
  blobs reference `numpy._core` (a 2.x-only path). The requirements
  files now pin `numpy>=2.0,<3.0`.
