# Code architecture

The project is split by *status* (active / diagnostic / legacy), not by
topic. This makes the safe-to-modify surface obvious — anything under
`src/walker2d/` is the live pipeline; anything under `src/legacy/` is
frozen.

For implementation details (frequencies, joint conventions, reward
internals, RSI, BC warm-start mechanics), see [`METHODS.md`](METHODS.md).
For the reward formula and exploit taxonomy, see
[`REWARD_DESIGN.md`](REWARD_DESIGN.md).

---

## Directory map

```
6955_Project/
│
├── CLAUDE.md                           # AI orientation hub (read first)
├── README.md                            # User-facing setup + quickstart
├── .gitignore                            # data/, *.pyc, .venv, etc.
│
├── docs/                                # All project documentation
│   ├── README.md                        #   ← index
│   ├── PROJECT_TIMELINE.md
│   ├── PROJECT_STATUS.md
│   ├── ARCHITECTURE.md                  #   ← you are here
│   ├── METHODS.md
│   ├── REWARD_DESIGN.md
│   ├── ROADMAP.md
│   ├── LEGACY_TRACKS.md
│   ├── DATA_SOURCES.md
│   ├── RUN_LOG.md
│   ├── reports/                          # Original proposal + current writeup
│   └── figures/                          # Diagnostic plot outputs (cycle, reference, etc.)
│
├── src/                                  # All Python code
│   ├── walker2d/                         # ── ACTIVE: phase-conditioned imitation ──
│   │   ├── ppo_walker2d_phase.py        #   PPO + DeepMimic engineered reward (Brock's track)
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
│   │   └── extract_osim_mass.py          #   Per-subject body mass from .osim XML
│   │
│   └── legacy/                           # ── FROZEN: do NOT extend without confirmation ──
│       ├── walker2d_v1/                   # Earlier Walker2d attempts (Phase 1 + 2)
│       │   ├── ppo_walker2d.py           #   Phase-blind imitation (failed)
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
├── assets/                              # Static project assets
│   ├── mjcf/                              # MuJoCo MJCF files
│   │   ├── walker2d_subject1.xml         #   (missing on this checkout — see PROJECT_STATUS.md)
│   │   └── walker2d_custom.xml           #   Legacy custom MJCF (predates stick-with-stock decision)
│   └── reference/
│       └── gait_cycle_reference.npy       # Single Ulrich stride @ 50Hz, (56, 6)
│
├── requirements/                          # Pip requirement files by platform
│   ├── windows_5090.txt
│   ├── windows_cpu.txt
│   └── macos.txt
│
└── results/                              # Training outputs (gitignored)
    └── <run_dir>/
        ├── model.zip                       #   Final SB3 PPO policy
        ├── reference.npy                  #   Reference array used at training time
        └── checkpoints/                    #   Periodic snapshots, ~5M steps apart
            └── model_<N>_steps.zip
```

---

## How the active pipeline is wired

### Imports (active code only)

```
ppo_walker2d_phase.py
  └─ from ulrich_loader import load_ulrich_reference

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

`amp_walker2d.py` reuses `extract_airl_state` and `make_expert_buffer`
from `airl_walker2d.py` because the (s, s′) feature extraction and
expert-buffer construction are identical between the two methods —
only the discriminator architecture, loss, and reward formulation
differ.

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
   `assets/mjcf/` first, then falls back to `PROJECT_ROOT` for backward
   compatibility with older runs that placed the MJCF at the repo root.

### Training-output layout

Every run writes a self-contained directory under `results/`:

- `results/<run_name>/reference.npy` — the reference used at training time.
  `render_phase.py` re-loads this so the renderer's env matches the
  training env exactly.
- `results/<run_name>/model.zip` — final policy. Loaded via
  `<run-dir>:final`.
- `results/<run_name>/checkpoints/model_<N>_steps.zip` — periodic
  checkpoints (~every 5M env steps). Loaded via
  `<run-dir>:<N>:"label"`.

---

## Entry points

Run all of these from the project root.

| Action | Command |
|---|---|
| Build the gait-cycle reference (one-time) | `python src/walker2d/extract_gait_cycle.py` |
| Train PPO + DeepMimic from scratch (stock Walker2d) | `python src/walker2d/ppo_walker2d_phase.py --ref_cycle assets/reference/gait_cycle_reference.npy --num_envs 32 --total_steps 5e6` |
| Train PPO + DeepMimic from scratch (Subject-1-scaled) | …add `--scale_model` |
| Train PPO + DeepMimic with BC warm-start | …add `--bc_epochs 10 --bc_steps 200000` |
| Finetune PPO + DeepMimic from a checkpoint | …add `--finetune results/<run-dir>/model.zip` |
| Train AMP (paper weights, finetuned from a working walker) | `python src/walker2d/amp_walker2d.py --ref_cycle assets/reference/gait_cycle_reference.npy --finetune results/<phase-run>/model.zip --num_envs 32 --total_steps 5e6` |
| Train AIRL (same finetune pattern; cold-start collapses) | `python src/walker2d/airl_walker2d.py --ref_cycle assets/reference/gait_cycle_reference.npy --finetune results/<phase-run>/model.zip --num_envs 32 --total_steps 5e6` |
| Render a single trained run (any track — they share the env) | `python src/walker2d/render_phase.py results/<run-dir>:final` |
| Compare multiple runs back-to-back | `python src/walker2d/render_phase.py results/<run-A>:final results/<run-B>:60000000:"60M"` |
| Sanity-check the reference | `python src/diagnostics/diag_cycle.py` &nbsp;&nbsp;and&nbsp;&nbsp;`python src/diagnostics/diag_ref.py` |

For the full set of `ppo_walker2d_phase.py` flags and their defaults,
see [`METHODS.md`](METHODS.md) or `python src/walker2d/ppo_walker2d_phase.py --help`.
The AMP/AIRL CLIs are documented in their respective module docstrings
and via `--help`; the AMP/AIRL discriminator + reward design is
covered in [`METHODS.md`](METHODS.md).
