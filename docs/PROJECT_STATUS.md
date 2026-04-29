# Project status — current snapshot

*Last updated: 2026-04-28.*

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

## What's currently running

- **Active training script:** `src/walker2d/ppo_walker2d_phase.py`.
- **Active reference:** `assets/reference/gait_cycle_reference.npy` —
  one clean stride from Ulrich Subject 1 baseline (56 frames @ 50 Hz,
  resampled to 140 frames @ 125 Hz inside the env).
- **Current canonical policy:**
  `results/walker2d_phase_cycle_s1scaled_sum_20260423-213031/model.zip`
  (cherry-picked from upstream `3e4c3fa`).
  - 100M env steps (`checkpoints/model_100000000_steps.zip` snapshot
    is also on disk)
  - Subject-1-scaled MJCF (`assets/mjcf/walker2d_subject1.xml`,
    *currently missing on this checkout* — must be regenerated/copied
    before training/rendering)
  - Per-joint weighted-sum reward (no product reward)
  - Single-cycle reference (no `--ref_all`)
  - Previous 60M canonical run
    (`walker2d_phase_cycle_s1scaled_sum_20260422-175117/`) is still on
    disk for comparison.
- **Most recent training computer:** the user has alternate machines;
  the canonical run was trained on the other one.
- **Reward cleanup (2026-04-28):** Removed three default-off terms
  (`peak_bonus`, `fwd_r`, `action_rate_pen`) and the pitch piece inside
  `root_r`. The reward is now 6 weighted terms + a small `‖ctrl‖²` cost.
  Per-component reward means (`reward/*`) and per-rollout termination-
  cause counts (`term/*`) now log to TensorBoard at
  `results/<run-dir>/tb`.
- **Held-out biomech eval:** `src/diagnostics/eval_biomech.py` produces
  stride period, cadence, double-support fraction, peak vGRF/BW,
  swing-drag fraction, L-R stride asymmetry, and a hip-knee phase-plane
  DTW vs the reference. Use this — not training reward — to compare
  reward variants and seeds. Sample baseline runs land in
  `results/<run-dir>/eval_biomech.json`.

## Comparison runs on disk

| Result dir | Steps | Notes |
|---|---|---|
| `results/walker2d_phase_cycle_s1scaled_sum_20260423-213031/` | 100M | **Latest extended run**, scaled MJCF, single-cycle ref. Saved as `model.zip` + `checkpoints/model_100000000_steps.zip`. From upstream commit `3e4c3fa`. |
| `results/walker2d_phase_cycle_s1scaled_sum_20260422-175117/` | 60M | Earlier canonical run, scaled MJCF, single-cycle ref |
| `results/walker2d_phase_full_sum_20260410-124935/` | 18M | Stock Walker2d, full-trial ref, uniform-k=8 — useful as a `--finetune` base for stock-geometry runs |
| `results/walker2d_phase_full_sum_20260410-105306/` | 45M | Earlier DeepMimic-reward run |
| `results/walker2d_phase_cycle_sum_20260409-211537/` | 10.5M | First single-cycle reference run |
| `results/walker2d_pretrain_symmetry_20260407-172719/` | 5M | Symmetry-pretrain ankle-paddling demo (legacy) |

Render any of them with:

```bash
python src/walker2d/render_phase.py results/<run-dir>:final
# or, for an integer checkpoint step under checkpoints/:
python src/walker2d/render_phase.py results/<run-dir>:18000000:"18M"
```

The default `--xml` is `walker2d_subject1.xml`. For runs trained on stock
Walker2d, override with `--xml walker2d.xml`.

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
