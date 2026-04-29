# Code architecture

**Purpose:** directory map, import graph, path constants, and entry
points.
**Read this when:** you want to know "where does X live" or "what
imports what."
**Adjacent:** [`METHODS.md`](METHODS.md) for env/reward internals ·
[`REWARD_DESIGN.md`](REWARD_DESIGN.md) for the reward formula and
exploit taxonomy.

The project is split by *status* (active / diagnostic / legacy), not
by topic. This makes the safe-to-modify surface obvious — anything
under `src/walker2d/` is the live pipeline; anything under
`src/legacy/` is frozen.

---

## Directory map

```
6955_Project/
│
├── CLAUDE.md                           # AI orientation hub (read first)
├── README.md                            # User-facing setup + quickstart
├── .gitignore
│
├── docs/                                # All project documentation
│   ├── README.md                        #   ← index
│   ├── PROJECT_STATUS.md                #   ← right now
│   ├── PROJECT_TIMELINE.md              #   ← chronological history
│   ├── RESTART_LOG.md                   #   ← post-2026-04-28 batches
│   ├── ROADMAP.md                       #   ← what's queued next
│   ├── ARCHITECTURE.md                  #   ← you are here
│   ├── METHODS.md                       #   ← implementation reference
│   ├── REWARD_DESIGN.md                 #   ← reward + exploit taxonomy
│   ├── DATA_SOURCES.md                  #   ← Ulrich + OpenCap layouts
│   ├── LEGACY_TRACKS.md                 #   ← what's frozen in src/legacy/
│   ├── RUN_LOG.md                       #   ← legacy symmetry-pretrain demos
│   ├── reports/                          # Original proposal + current writeup
│   ├── papers/                           # Primary-source PDFs + index
│   └── figures/                          # Diagnostic plot outputs
│
├── src/
│   ├── walker2d/                         # ── ACTIVE: phase-conditioned imitation ──
│   │   ├── ppo_walker2d_phase.py        #   PPO + DeepMimic 4-term reward (Brock's track)
│   │   ├── sac_walker2d_phase.py        #   Off-policy SAC sibling (same env + reward)
│   │   ├── amp_walker2d.py              #   AMP — LSGAN disc, paper combined reward (Brian's track)
│   │   ├── airl_walker2d.py             #   AIRL — disc with shaping potential, BCE+GP (Brian's track)
│   │   ├── render_phase.py              #   Render one or more trained policies
│   │   ├── extract_gait_cycle.py        #   Build gait_cycle_reference.npy from Ulrich IK
│   │   └── ulrich_loader.py             #   load_sto / load_ulrich_reference / ULRICH_ROOT
│   │
│   ├── diagnostics/                      # Standalone sanity-check scripts (not on training path)
│   │   ├── diag_cycle.py                 #   3-cycle plot + seam discontinuity
│   │   ├── diag_ref.py                   #   Per-joint ranges + open-loop FK upright check
│   │   ├── diag_walker_mass.py           #   Walker2d body-mass dump
│   │   ├── extract_osim_mass.py          #   Per-subject body mass from .osim XML
│   │   ├── view_reference.py             #   MuJoCo-viewer playback of the on-disk gait cycle
│   │   ├── compare_tb.py                 #   Side-by-side TensorBoard scalar comparison
│   │   ├── extract_reference_biomech.py  #   Measured biomech targets from GRF + IK + .osim
│   │   └── eval_biomech.py               #   Held-out biomech metrics + vs_reference + progress_score
│   │
│   └── legacy/                           # ── FROZEN: do NOT extend without confirmation ──
│       ├── walker2d_v1/                   # Earlier Walker2d attempts (Phase 1 + 2)
│       │   ├── ppo_walker2d.py           #   Phase-blind imitation (failed; still has all-six-joint flip)
│       │   ├── pretrain_walker2d.py      #   Symmetry-reward shaping (dead end)
│       │   ├── gail_walker2d.py          #   GAIL approach (superseded by AMP/AIRL)
│       │   └── render_walker.py          #   Renderer for legacy env
│       └── musculoskeletal/              # Original 3D 80-muscle plan (preserved for return)
│           ├── ppo_myoassist.py          #   PPO on MyoAssist env
│           ├── ppo_walk.py               #   MyoSuite myoLegWalk-v0 baseline
│           ├── render_myoassist.py
│           ├── train.py                   #   BC + GAIL pipeline driver
│           ├── bc_policy.py
│           ├── gail.py
│           ├── data_utils.py             #   OpenCap / SimTK loading
│           └── evaluate.py
│
├── scripts/                              # Wrappers + reporting
│   ├── biomech_report.py                 #   Render writeup-ready biomech table + 6-panel figure
│   └── overnight/                        #   Multi-experiment sweep scaffolding
│       ├── run_experiment.py             #     train + eval_biomech + preview.mp4 + meta
│       ├── rank_runs.py                  #     composite-score ranking
│       ├── write_report.py               #     fill REPORT.md from eval JSON
│       ├── REPORT_TEMPLATE.md
│       └── STATUS_TEMPLATE.md
│
├── assets/                              # Static project assets
│   ├── mjcf/                              # MuJoCo MJCF files
│   │   ├── walker2d_subject1.xml         #   (missing on this checkout — see PROJECT_STATUS.md)
│   │   └── walker2d_custom.xml           #   Legacy custom MJCF
│   └── reference/
│       ├── gait_cycle_reference.npy      #   Single Ulrich stride @ 50Hz, (56, 6)
│       ├── biomech_targets.json          #   Measured Subject 1 stride/cadence/ROM/vGRF targets
│       └── biomech_targets.vgrf_curves.npz #  Normalised stance-phase vGRF curves
│
├── requirements/                          # Pip requirement files by platform
│   ├── windows_5090.txt
│   ├── windows_cpu.txt
│   └── macos.txt
│
└── results/                              # Training outputs (mostly gitignored)
    └── <run_dir>/
        ├── model.zip                       #   Final SB3 PPO/SAC policy
        ├── reference.npy                  #   Reference array used at training time
        ├── env_kwargs.json                #   Env construction kwargs (so renderer can reconstruct)
        ├── tb/                             #   TensorBoard event files
        └── checkpoints/
            └── model_<N>_steps.zip
```

---

## How the active pipeline is wired

### Imports (active code only)

```
ppo_walker2d_phase.py
  └─ from ulrich_loader import load_ulrich_reference

sac_walker2d_phase.py
  └─ from ppo_walker2d_phase import Walker2dPhaseAware, load_ref_cycle, CTRL_HZ, MJCF_ROOT

airl_walker2d.py
  ├─ from ppo_walker2d_phase import Walker2dPhaseAware, load_ref_cycle, CTRL_HZ,
  │                                  GAIT_CYCLE_FRAMES, compute_bc_dataset, pretrain_bc
  └─ from ulrich_loader        import load_ulrich_reference

amp_walker2d.py
  ├─ from ppo_walker2d_phase import Walker2dPhaseAware, load_ref_cycle, CTRL_HZ,
  │                                  compute_bc_dataset, pretrain_bc, LogCallback
  ├─ from airl_walker2d        import extract_airl_state, make_expert_buffer
  └─ from ulrich_loader        import load_ulrich_reference

render_phase.py
  └─ from ppo_walker2d_phase import Walker2dPhaseAware, _JNT_LO, _JNT_HI

extract_gait_cycle.py
  └─ from ulrich_loader import load_sto, ULRICH_ROOT, PROJECT_ROOT
```

`ppo_walker2d_phase.py` is the spine — every other active script
imports from it. AMP reuses `extract_airl_state` and
`make_expert_buffer` from AIRL because the (s, s′) feature extraction
and expert-buffer construction are identical between the two methods.

The active pipeline does not import from any file in `src/legacy/`.
Path resolution: each active script computes
`PROJECT_ROOT = Path(__file__).resolve().parents[2]` so it can find
`assets/mjcf/`, `assets/reference/`, and `results/` regardless of cwd.

### Path constants

| Constant | Value | Used for |
|---|---|---|
| `PROJECT_ROOT` | `<repo>` | Resolving asset paths and writing outputs |
| `MJCF_ROOT` | `<repo>/assets/mjcf` | Looking up `walker2d_subject1.xml` etc. |
| `REF_ROOT` | `<repo>/assets/reference` | Looking up `gait_cycle_reference.npy` |
| `ULRICH_ROOT` | `<repo>/Ulrich_Treadmill_Data` | Default Ulrich IK root (data is gitignored) |
| `CTRL_HZ` | `125.0` | Walker2d-v4 control rate (`frame_skip=4`, `dt=0.002s`) |
| `REF_HZ` | `50.0` | Ulrich IK source rate |
| `GAIT_CYCLE_FRAMES` | `140` | Period for sin/cos φ encoding (~1.1s @ 125Hz) |

### XML resolution (`Walker2dPhaseAware.__init__`)

The env is configurable via `xml_file=`:

1. `"walker2d.xml"` → looked up in gymnasium's MuJoCo asset dir (stock).
2. Absolute path → used as-is.
3. Bare filename (e.g. `"walker2d_subject1.xml"`) → looked up in
   `assets/mjcf/` first, then falls back to `PROJECT_ROOT` for
   backward compatibility with older runs that placed the MJCF at the
   repo root.

### Training-output layout

Every run writes a self-contained directory under `results/`:

- `results/<run_name>/reference.npy` — the reference used at training
  time. `render_phase.py` and `eval_biomech.py` re-load this so the
  evaluation env matches the training env exactly.
- `results/<run_name>/env_kwargs.json` — the kwargs used to construct
  `Walker2dPhaseAware` (including `preview_k`, `pose_joint_weights`,
  `xvel_term_thresh`, etc.). Renderer/eval read this so they don't
  need to be told the env config separately.
- `results/<run_name>/model.zip` — final policy. Loaded via
  `<run-dir>:final`.
- `results/<run_name>/checkpoints/model_<N>_steps.zip` — periodic
  checkpoints (~every 1M env steps).
- `results/<run_name>/tb/` — TensorBoard event files.

---

## Entry points

Run all of these from the project root.

| Action | Command |
|---|---|
| Build the gait-cycle reference (one-time) | `python src/walker2d/extract_gait_cycle.py` |
| Train PPO + DeepMimic from scratch (stock Walker2d, current best recipe) | `python src/walker2d/ppo_walker2d_phase.py --ref_cycle assets/reference/gait_cycle_reference.npy --xvel_term 0.3 --num_envs 8 --total_steps 5e6` |
| Train PPO + DeepMimic from scratch (Subject-1-scaled MJCF) | …add `--scale_model` |
| Train PPO + DeepMimic with BC warm-start | …add `--bc_epochs 10 --bc_steps 200000` |
| Finetune PPO + DeepMimic from a checkpoint | …add `--finetune results/<run-dir>/model.zip` |
| Train SAC + DeepMimic (off-policy sibling) | `python src/walker2d/sac_walker2d_phase.py --ref_cycle assets/reference/gait_cycle_reference.npy --xvel_term 0.3 --total_steps 1e6` |
| Train AMP (paper weights, finetuned from a working walker) | `python src/walker2d/amp_walker2d.py --ref_cycle assets/reference/gait_cycle_reference.npy --finetune results/<phase-run>/model.zip --num_envs 32 --total_steps 5e6` |
| Train AIRL (same finetune pattern; cold-start collapses) | `python src/walker2d/airl_walker2d.py --ref_cycle assets/reference/gait_cycle_reference.npy --finetune results/<phase-run>/model.zip --num_envs 32 --total_steps 5e6` |
| Render a single trained run (any track — they share the env) | `python src/walker2d/render_phase.py --xml walker2d.xml results/<run-dir>:final` |
| Compare multiple runs back-to-back | `python src/walker2d/render_phase.py --xml walker2d.xml results/<run-A>:final results/<run-B>:1000000:"1M"` |
| Sanity-check the reference | `python src/diagnostics/diag_cycle.py` &nbsp;&nbsp;and&nbsp;&nbsp;`python src/diagnostics/diag_ref.py` |
| View the on-disk reference cycle on a Walker2d skeleton | `python src/diagnostics/view_reference.py` |
| Compute measured biomech targets (one-time per subject) | `python src/diagnostics/extract_reference_biomech.py` |
| Evaluate a checkpoint vs measured targets | `python src/diagnostics/eval_biomech.py --xml walker2d.xml results/<run>:final --out results/<run>_eval.json` |
| Writeup-ready table + figure | `python scripts/biomech_report.py results/<run>_eval.json --rerollout` |

For the full set of `ppo_walker2d_phase.py` flags and their defaults,
see [`METHODS.md § Full CLI reference`](METHODS.md#full-cli-reference-ppo_walker2d_phasepy)
or `python src/walker2d/ppo_walker2d_phase.py --help`. The AMP/AIRL
CLIs are documented in their respective module docstrings and via
`--help`.
