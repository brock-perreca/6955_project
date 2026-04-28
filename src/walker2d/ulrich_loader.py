"""
ulrich_loader.py
────────────────
Reference-data loaders for the Ulrich treadmill walking IK dataset.

Originally lived in `ppo_walker2d.py`. Extracted into its own module so the
active phase-conditioned pipeline (`ppo_walker2d_phase.py`,
`extract_gait_cycle.py`) can import it without pulling in the legacy
training body and `Walker2dImitation` env.

Public API
──────────
- `PROJECT_ROOT` : repo root (the directory containing CLAUDE.md / README.md)
- `ULRICH_ROOT`  : default Ulrich data root (`PROJECT_ROOT / "Ulrich_Treadmill_Data"`)
- `load_sto(path)`               : parse OpenSim .sto / .mot file → column dict
- `load_ulrich_reference(...)`   : load + concatenate Ulrich IK trials → (T, 6) array

See `docs/DATA_SOURCES.md` for the full Ulrich data layout and the
sign-convention conversion from OpenSim → Walker2d joints.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
# This file lives at <repo>/src/walker2d/ulrich_loader.py — three parents up
# from the file gets us to the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ULRICH_ROOT  = PROJECT_ROOT / "Ulrich_Treadmill_Data"


# ── parsers ───────────────────────────────────────────────────────────────────
def load_sto(path: Path) -> dict:
    """Parse an OpenSim .sto / .mot file into a column dict."""
    with open(path) as f:
        lines = f.readlines()
    for i, l in enumerate(lines):
        if l.strip() == "endheader":
            header_end = i
            break
    cols = lines[header_end + 1].split()
    data = np.array(
        [[float(x) for x in l.split()] for l in lines[header_end + 2:] if l.strip()]
    )
    return {c: data[:, i] for i, c in enumerate(cols)}


def load_ulrich_reference(subjects: list[int] | None = None,
                          trial_filter: str | None = None,
                          control_hz: float = 50.0) -> np.ndarray:
    """
    Load all matching Ulrich IK walking trials and concatenate into a single
    reference array of shape (T, 6) with column order:

        [hip_r, knee_r, ankle_r, hip_l, knee_l, ankle_l]

    Values are in RADIANS, in the Walker2d sign convention. Walker2d's joint
    axes are all `[0, -1, 0]`, so a positive rotation in OpenSim around +Y
    maps to a negative angle in Walker2d:

        hip:   walker = -opensim   (flexion positive in OpenSim, negative in Walker2d)
        knee:  walker = -opensim   (same sign flip)
        ankle: walker = -opensim   (plantarflex negative in OpenSim, positive in Walker2d
                                    — the negation lines up with Walker2d's foot_joint sign)

    See `docs/DATA_SOURCES.md` for OpenSim joint range references.
    """
    if subjects is None:
        subjects = list(range(1, 11))  # Subject1..Subject10

    segments = []
    total_files = 0

    for subj_id in subjects:
        subj_dir = ULRICH_ROOT / f"Subject{subj_id}" / "IK"
        if not subj_dir.exists():
            print(f"  [warn] missing: {subj_dir}")
            continue

        for trial_dir in sorted(subj_dir.glob("walking_*")):
            if trial_filter and trial_filter not in trial_dir.name:
                continue
            ik_path = trial_dir / "output" / "results_ik.sto"
            if not ik_path.exists():
                continue

            d = load_sto(ik_path)
            # Ulrich IK is at 50 Hz in degrees.
            orig_hz  = 1.0 / (d["time"][1] - d["time"][0])
            orig_len = len(d["time"])
            new_len  = int(orig_len * control_hz / orig_hz)
            orig_x   = np.arange(orig_len)
            new_x    = np.linspace(0, orig_len - 1, new_len)

            from scipy.interpolate import CubicSpline
            def resamp(key):
                return CubicSpline(orig_x, d[key])(new_x)

            seg = np.stack([
                -np.deg2rad(resamp("hip_flexion_r")),
                -np.deg2rad(resamp("knee_angle_r")),
                -np.deg2rad(resamp("ankle_angle_r")),
                -np.deg2rad(resamp("hip_flexion_l")),
                -np.deg2rad(resamp("knee_angle_l")),
                -np.deg2rad(resamp("ankle_angle_l")),
            ], axis=1)
            segments.append(seg)
            total_files += 1

    if not segments:
        raise FileNotFoundError(
            f"No Ulrich IK trials found under {ULRICH_ROOT}. "
            "See docs/DATA_SOURCES.md for the expected layout."
        )

    ref = np.concatenate(segments, axis=0).astype(np.float32)
    duration = len(ref) / control_hz
    print(f"  Loaded {total_files} trials → {len(ref):,} frames @ {control_hz}Hz "
          f"({duration:.0f}s = {duration/60:.1f} min)")
    return ref
