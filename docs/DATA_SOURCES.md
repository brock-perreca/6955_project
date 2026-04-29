# Data sources

**Purpose:** where the reference data lives, what format it's in, and
how the active pipeline consumes it.
**Read this when:** the loader fails to find files, you need to know
the joint-column order, or you're considering a new dataset.

The actual data directories are **gitignored** — users supply their own
copies on disk. The paths and layouts below are conventions the loaders
expect.

---

## Ulrich Treadmill Walking Dataset (active)

The current scientific question uses **Subject 1, baseline trial,
1.25 m/s** from the Ulrich dataset, extracted as a single clean stride
of **56 frames @ 50 Hz** (~1.12 s), resampled to **140 frames @ 125 Hz**
inside the env (see [`METHODS.md § Frequencies`](METHODS.md#frequencies-and-resampling)).

> **Naming note.** Despite the local folder name "Ulrich", the dataset
> originates from Scott D. **Uhlrich** et al. at Stanford NMBL — the
> SimTK project [Muscle Coordination Retraining to Reduce Knee
> Loading](https://simtk.org/projects/coordretraining), accompanying
> Uhlrich et al., *Muscle coordination retraining inspired by
> musculoskeletal simulations reduces knee contact force* (Sci Rep 12,
> 9842, 2022). The `walking_baseline_*` and `walking_feedback_ON_*`
> trials are the pre/post conditions of the gastrocnemius-vs-soleus
> biofeedback experiment from that paper. The local junction name is
> kept for backwards compatibility.

### Expected on-disk layout

The loader (`src/walker2d/ulrich_loader.py`) looks under
`<repo>/Ulrich_Treadmill_Data/`:

```
Ulrich_Treadmill_Data/
  Subject1/
    IK/
      walking_baseline_01/
        output/
          results_ik.sto      ← OpenSim IK output, 50 Hz, degrees
      walking_baseline_02/
        output/
          results_ik.sto
      walking_feedback_ON_01/
        output/
          results_ik.sto      ← present in the dataset, not used by active pipeline
      ...
  Subject2/
    IK/
      ...
  ...
  Subject10/
```

If your local data lives somewhere else (e.g.
`CoordinationRetrainingData/forSimTK/`), the simplest fix is a symlink
or rename to `Ulrich_Treadmill_Data/`. Alternatively, override
`ULRICH_ROOT` programmatically before importing the loader.

**On Windows**, a directory junction works without admin and is what
this repo uses on machines where the data lives under
`CoordinationRetrainingData/forSimTK/`:

```cmd
mklink /J "Ulrich_Treadmill_Data" "CoordinationRetrainingData\forSimTK"
```

Note that the trial directories in the SimTK dump are named
`walking_baseline1`, `walking_FBcolor1_finalFB1`, etc. — the loader's
`walking_*` glob handles this; the `extract_gait_cycle.py`
`walking_*baseline*` filter matches `walking_baseline1` correctly.

### What gets loaded

`load_ulrich_reference(subjects=None, trial_filter=None, control_hz=50.0)`:

- Iterates over `Subject{N}/IK/walking_*/output/results_ik.sto`.
- Filters by `subjects` (default 1..10) and `trial_filter` (substring,
  e.g. `"baseline"`).
- Each trial's IK is parsed via `load_sto`, resampled to `control_hz`,
  converted to radians, and sign-converted via the **knee-only flip**
  that aligns OpenSim and Walker2d sign conventions (see
  [`METHODS.md § Joint sign convention`](METHODS.md#joint-sign-convention)).
  The pre-2026-04-28 loader applied an all-six-joint flip; that was
  wrong for hip and ankle and was corrected during the Phase 5
  restart. Pre-restart checkpoints under `results/walker2d_phase_*/`
  were trained against the corrupted sign and are kept as a historical
  record only — see
  [`PROJECT_TIMELINE.md § Phase 5`](PROJECT_TIMELINE.md#phase-5--the-sign-error-discovery-2026-04-28).
- All matching trials are concatenated end-to-end into a single
  `(T, 6)` array with column order
  `[hip_r, knee_r, ankle_r, hip_l, knee_l, ankle_l]`.

For the active single-cycle pipeline, the concatenation is bypassed:
`extract_gait_cycle.py` picks the first matching trial, detects two
consecutive right heel strikes via `scipy.signal.find_peaks` on
`-hip_r`, and saves the inter-strike segment to
`assets/reference/gait_cycle_reference.npy`.

### The `feedback_ON` trials

The dataset includes both `walking_baseline_*` and `walking_feedback_ON_*`
trials per subject. An earlier project direction proposed comparing
**baseline vs feedback_ON kinematics** as a 2-condition study; the
current writeup focuses on imitation methods and uses **only the
baseline trial**. The `feedback_ON` data is on disk but unused. If the
project ever revisits the baseline-vs-feedback question, the loader
already supports it via `trial_filter="feedback_ON"`.

### `.sto` parsing

`load_sto(path)` parses OpenSim Storage / Motion files (`.sto` / `.mot`)
into a column dict:

- Skips the file header up to the `endheader` line.
- Reads the column-name row.
- Reads the numeric data rows.
- Returns `{column_name: np.ndarray, ...}`.

This is sufficient for the IK files used here. Not a full OpenSim
parser — handles only space-delimited tabular `.sto`.

---

## OpenCap markerless mocap data (legacy)

Used by the **musculoskeletal** track (`src/legacy/musculoskeletal/`)
from the original 3D project scope (writeup-replaced; see
[`PROJECT_TIMELINE.md § Phase 0`](PROJECT_TIMELINE.md#phase-0--original-proposal-proposal-stage-see-reportsadvanced_ai_project_reportpdf)).
**Not consumed by the active Walker2d pipeline.**

OpenCap pipeline paper:
[`papers/Uhlrich_2023_OpenCap.pdf`](papers/Uhlrich_2023_OpenCap.pdf).
Independent validation of smartphone-based markerless mocap against
marker-based gold standard:
[`papers/Horsak_2023_smartphone_markerless_validity.pdf`](papers/Horsak_2023_smartphone_markerless_validity.pdf).

### Expected layout

```
OpenCap_data/
  subject{N}/
    OpenSimData/
      Mocap/IK/{trial}.mot                           ← markered gold standard (lab-grade)
      Video/HRNet/IK/{trial}.mot                     ← markerless (HRNet pose estimator)
      Video/OpenPose_default/IK/{trial}.mot          ← markerless (OpenPose default)
      Video/OpenPose_highAccuracy/IK/{trial}.mot     ← markerless (OpenPose high-accuracy)
    EMGData/
      {trial}_EMG.sto                                ← per-muscle activations
    ForceData/
      {trial}_forces.mot                             ← ground reaction forces (force plate)
```

Each `subject{N}/` corresponds to one human participant. The four IK
sources (Mocap + 3 video pipelines) enable the **lab-vs-field** comparison
the original proposal targeted. EMG and force-plate data are ground
truth for the muscle-activation and GRF emergent-quantity comparisons
that motivated the original 6-condition study.

Loaded by `src/legacy/musculoskeletal/data_utils.py`. Path resolution
is handled inside that file; consult it directly if revisiting this
track.

---

## OpenSim model files (`.osim`)

The original project pipeline used scaled OpenSim models per subject
(for biomechanical comparison: BW-normalized GRF requires per-subject
total mass). `src/diagnostics/extract_osim_mass.py` parses `<Body><mass>`
elements out of any `.osim` XML it finds under `OSIM_ROOT` (default `.`)
and prints a per-subject mass table.

Walker2d itself has total mass ≈ **23.68 kg** (per
`src/diagnostics/diag_walker_mass.py`). For BW-normalized GRF
comparison the conversion is:

```
sim_GRF_BW   = sim_contact_force / (23.68 · 9.81)
subj_GRF_BW  = forceplate_force  / (subject_mass · 9.81)
```

Both are dimensionless, so the mass-mismatch cancels.

---

## Active reference artifact

`assets/reference/gait_cycle_reference.npy`

- Shape: `(N, 6)`, dtype `float32`, units **radians**, Walker2d sign
  convention.
- N is whatever `extract_gait_cycle.py` detected as one stride —
  for Subject 1 baseline this is **56 frames @ 50 Hz** (~1.12 s,
  resampled to 140 frames @ 125 Hz inside the env).
- Joint order: `[hip_r, knee_r, ankle_r, hip_l, knee_l, ankle_l]`.
- Resampled to 125 Hz at env-init time, *not* on disk. The on-disk
  artifact stays at 50 Hz so it's diagnosable with low-frequency tools
  (cf. `src/diagnostics/diag_cycle.py`).

Regenerate with:

```bash
python src/walker2d/extract_gait_cycle.py
```

The script writes the new `.npy` to `assets/reference/` and a sanity
plot (`docs/figures/gait_cycle_check.png`).
