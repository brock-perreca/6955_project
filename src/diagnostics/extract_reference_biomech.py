"""
extract_reference_biomech.py
────────────────────────────
Compute biomechanical *reference targets* from the Ulrich treadmill walking
dataset for a single subject + trial, and write them to
`assets/reference/biomech_targets.json`. This file is the ground-truth
right-hand column of the `eval_biomech.py` summary table — replacing the
literature/textbook ranges that previously lived in `docs/RESTART_LOG.md`
with subject-specific, measured values.

What gets extracted (Subject1 / walking_baseline1 by default)
────────────────────────────────────────────────────────────
From `ExpmtlData/GRF/<trial>_forces.mot` (force plates):
    body_weight_n           — from scaled .osim total mass × 9.81
    stride_period_s         — same-foot strike interval (median over trial)
    cadence_steps_per_min   — 2 steps / stride
    double_support_frac     — frames with both feet loaded / any-foot frames
    peak_vgrf_bw_{r,l}      — mean peak vGRF / BW over all stance bouts
    peak_vgrf_bw            — average of left / right
    contact_thresh_n        — the 5% BW threshold actually used
    fp1_label, fp2_label    — which force plate we treated as which leg

From `IK/<trial>/output/results_ik.sto` (joint angles):
    hip_r_rom_deg, knee_r_rom_deg, ankle_r_rom_deg  (and L counterparts)
    hip_r_min/max_deg, ...  per-joint min/max in degrees, full trial

Plus a normalised stance-phase vGRF curve (100 samples) per foot, saved as
a separate `.npy` next to the JSON, so downstream code (eval_biomech.py)
can DTW-compare GRF *shape*, not just peak magnitude.

Usage
─────
    python src/diagnostics/extract_reference_biomech.py
        [--subject 1] [--trial walking_baseline1]
        [--out assets/reference/biomech_targets.json]

Why this script exists
──────────────────────
Until 2026-04-28, the "reference target" column in eval_biomech tables
was bibliographic — 1.12 s stride from the reference frame count, ~107
cadence as the trivial inverse, ~1.2 BW peak vGRF and 0.20–0.30 double
support from biomech textbooks. None of those were measured from the
Ulrich subject we actually train against. This script fixes that: every
number in `biomech_targets.json` is computed from this subject's own
force-plate and IK files. See `docs/METHODS.md § Held-out biomechanical
evaluation` for how it plugs in.
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "walker2d"))
from ulrich_loader import load_sto, ULRICH_ROOT  # noqa: E402

GRAVITY = 9.81
STANCE_RESAMPLE_N = 100  # normalised stance-phase samples for shape DTW


# ── helpers ────────────────────────────────────────────────────────────────────

def osim_total_mass(osim_path: Path) -> float:
    """Sum every <mass> entry in the OpenSim XML (one per Body)."""
    tree = ET.parse(osim_path)
    return sum(
        float(el.text.strip())
        for el in tree.getroot().iter("mass")
        if el.text and el.text.strip()
    )


def find_contact_runs(in_contact: np.ndarray) -> list[tuple[int, int]]:
    """Return [(start, end), ...] inclusive indices of contiguous in-contact runs."""
    if not in_contact.any():
        return []
    idx = np.where(in_contact)[0]
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]] if len(breaks) else np.array([idx[0]])
    ends = np.r_[idx[breaks], idx[-1]] if len(breaks) else np.array([idx[-1]])
    return list(zip(starts.tolist(), ends.tolist()))


def stance_strike_indices(force: np.ndarray, thresh: float,
                          min_gap: int) -> np.ndarray:
    """First frame of each stance bout (rising edge of in-contact)."""
    in_contact = force > thresh
    runs = find_contact_runs(in_contact)
    if not runs:
        return np.array([], dtype=int)
    starts = np.array([s for s, _ in runs], dtype=int)
    if len(starts) == 0:
        return starts
    keep = [int(starts[0])]
    for s in starts[1:]:
        if s - keep[-1] >= min_gap:
            keep.append(int(s))
    return np.asarray(keep, dtype=int)


def normalised_stance_curve(force: np.ndarray, thresh: float,
                            min_len: int = 10) -> np.ndarray:
    """
    Mean stance-phase vGRF curve, time-normalised to STANCE_RESAMPLE_N samples.

    Returns shape (STANCE_RESAMPLE_N,) or empty array if no usable stances.
    """
    in_contact = force > thresh
    runs = find_contact_runs(in_contact)
    curves = []
    for s, e in runs:
        if e - s + 1 < min_len:
            continue
        bout = force[s:e + 1]
        new_x = np.linspace(0, len(bout) - 1, STANCE_RESAMPLE_N)
        curves.append(np.interp(new_x, np.arange(len(bout)), bout))
    if not curves:
        return np.array([])
    return np.mean(np.stack(curves, axis=0), axis=0)


def per_stride_rom(angle: np.ndarray, strike_idx: np.ndarray) -> dict:
    """Per-stride min/max/range; return median across strides."""
    if len(strike_idx) < 2:
        return {}
    mins, maxs, ranges = [], [], []
    for i in range(len(strike_idx) - 1):
        s, e = int(strike_idx[i]), int(strike_idx[i + 1])
        if e - s < 5:
            continue
        seg = angle[s:e]
        mins.append(float(seg.min()))
        maxs.append(float(seg.max()))
        ranges.append(float(seg.max() - seg.min()))
    if not ranges:
        return {}
    return {
        "min_deg":   float(np.rad2deg(np.median(mins))),
        "max_deg":   float(np.rad2deg(np.median(maxs))),
        "range_deg": float(np.rad2deg(np.median(ranges))),
        "n_strides": len(ranges),
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--trial",   default="walking_baseline1",
                   help="Trial folder name (under SubjectN/IK/)")
    p.add_argument("--out", default=None,
                   help="Output JSON path (default: "
                        "assets/reference/biomech_targets.json)")
    p.add_argument("--bw_thresh_frac", type=float, default=0.05,
                   help="Foot-contact threshold as fraction of body weight (default 0.05)")
    args = p.parse_args()

    subj_dir = ULRICH_ROOT / f"Subject{args.subject}"
    if not subj_dir.exists():
        raise FileNotFoundError(f"Subject dir not found: {subj_dir}")

    # ── body mass from scaled OpenSim model ───────────────────────────────────
    osim_path = subj_dir / "Models" / "RajagopalModified_scaled.osim"
    if not osim_path.exists():
        raise FileNotFoundError(f"Scaled osim not found: {osim_path}")
    mass_kg = osim_total_mass(osim_path)
    bw_n = mass_kg * GRAVITY
    contact_thresh = args.bw_thresh_frac * bw_n
    print(f"[mass] {osim_path.name}: {mass_kg:.2f} kg  ->  BW = {bw_n:.1f} N  "
          f"(contact thresh = {contact_thresh:.1f} N at {args.bw_thresh_frac*100:.0f}% BW)")

    # ── GRF .mot ──────────────────────────────────────────────────────────────
    grf_path = subj_dir / "ExpmtlData" / "GRF" / f"{args.trial}_forces.mot"
    if not grf_path.exists():
        raise FileNotFoundError(f"GRF .mot not found: {grf_path}")
    grf = load_sto(grf_path)
    t_grf = grf["time"]
    grf_dt = float(t_grf[1] - t_grf[0])
    grf_hz = 1.0 / grf_dt
    print(f"[grf] {grf_path.name}: {len(t_grf):,} samples @ {grf_hz:.1f} Hz "
          f"({t_grf[-1] - t_grf[0]:.1f} s)")

    # Vertical force = column ending in `_vy`. fp1 = `ground_force_vy`,
    # fp2 = `1_ground_force_vy`. Treadmill split-belt convention varies;
    # we don't assume R/L mapping — we just call them fp1/fp2 and report
    # the same metrics for both. Symmetry gets reported as the spread
    # between them.
    vgrf_fp1 = np.abs(grf["ground_force_vy"])
    vgrf_fp2 = np.abs(grf["1_ground_force_vy"])

    # Min-gap of half the slowest plausible stride period (≈0.5 s) keeps us
    # from double-counting tiny force-plate noise blips.
    min_gap = max(1, int(0.5 * grf_hz))
    fp1_strikes = stance_strike_indices(vgrf_fp1, contact_thresh, min_gap)
    fp2_strikes = stance_strike_indices(vgrf_fp2, contact_thresh, min_gap)
    print(f"[grf] fp1 strikes: {len(fp1_strikes)}   fp2 strikes: {len(fp2_strikes)}")

    # Stride period: same-foot strike interval. Use whichever plate has more
    # detected strikes (more robust median); they should be ≈ equal.
    strikes_for_stride = fp1_strikes if len(fp1_strikes) >= len(fp2_strikes) else fp2_strikes
    if len(strikes_for_stride) >= 2:
        stride_period_s = float(np.median(np.diff(strikes_for_stride)) * grf_dt)
    else:
        stride_period_s = float("nan")
    cadence_spm = 2.0 * 60.0 / stride_period_s if np.isfinite(stride_period_s) else float("nan")

    # Double-support fraction
    fp1_in = vgrf_fp1 > contact_thresh
    fp2_in = vgrf_fp2 > contact_thresh
    any_in = fp1_in | fp2_in
    ds_frac = float((fp1_in & fp2_in).sum() / any_in.sum()) if any_in.sum() else float("nan")

    # Peak vGRF/BW per plate (mean of per-bout peaks)
    def _peak_per_bout(in_contact: np.ndarray, force: np.ndarray) -> float:
        runs = find_contact_runs(in_contact)
        peaks = [float(force[s:e + 1].max()) for s, e in runs if e > s]
        return float(np.mean(peaks)) if peaks else 0.0

    peak_fp1_bw = _peak_per_bout(fp1_in, vgrf_fp1) / bw_n
    peak_fp2_bw = _peak_per_bout(fp2_in, vgrf_fp2) / bw_n
    peak_bw     = 0.5 * (peak_fp1_bw + peak_fp2_bw)
    print(f"[grf] peak vGRF/BW  fp1: {peak_fp1_bw:.2f}  fp2: {peak_fp2_bw:.2f}  "
          f"mean: {peak_bw:.2f}")

    # Stance-shape curves (saved separately)
    stance_curve_fp1 = normalised_stance_curve(vgrf_fp1, contact_thresh) / bw_n
    stance_curve_fp2 = normalised_stance_curve(vgrf_fp2, contact_thresh) / bw_n

    # ── IK .sto ───────────────────────────────────────────────────────────────
    ik_path = subj_dir / "IK" / args.trial / "output" / "results_ik.sto"
    if not ik_path.exists():
        raise FileNotFoundError(f"IK .sto not found: {ik_path}")
    ik = load_sto(ik_path)
    t_ik = ik["time"]
    ik_dt = float(t_ik[1] - t_ik[0])
    ik_hz = 1.0 / ik_dt
    print(f"[ik]  {ik_path.parent.parent.name}/results_ik.sto: "
          f"{len(t_ik):,} samples @ {ik_hz:.1f} Hz")

    # Sign convention matches ulrich_loader.py / extract_gait_cycle.py:
    # hip & ankle agree with OpenSim; knee flips. ROM is sign-invariant,
    # so we can keep OpenSim signs here and just report deg.
    joints_rad = {
        "hip_r":   np.deg2rad(ik["hip_flexion_r"]),
        "knee_r": -np.deg2rad(ik["knee_angle_r"]),  # match Walker2d sign
        "ankle_r": np.deg2rad(ik["ankle_angle_r"]),
        "hip_l":   np.deg2rad(ik["hip_flexion_l"]),
        "knee_l": -np.deg2rad(ik["knee_angle_l"]),
        "ankle_l": np.deg2rad(ik["ankle_angle_l"]),
    }

    # Per-stride ROM uses heel strikes from the GRF, but on the IK timebase.
    # Map fp1/fp2 strikes (in GRF samples) into IK frame indices.
    fp1_strikes_ik = (fp1_strikes * (ik_hz / grf_hz)).astype(int)
    fp2_strikes_ik = (fp2_strikes * (ik_hz / grf_hz)).astype(int)

    rom: dict = {}
    for name, ang in joints_rad.items():
        # Use whichever plate's strikes correspond to this leg. We don't
        # know the mapping; report ROM under both, plus full-trial range.
        rom[name] = {
            "full_trial_min_deg":   float(np.rad2deg(ang.min())),
            "full_trial_max_deg":   float(np.rad2deg(ang.max())),
            "full_trial_range_deg": float(np.rad2deg(ang.max() - ang.min())),
            "per_stride_fp1": per_stride_rom(ang, fp1_strikes_ik),
            "per_stride_fp2": per_stride_rom(ang, fp2_strikes_ik),
        }

    # ── assemble + write ──────────────────────────────────────────────────────
    out_path = (Path(args.out) if args.out else
                PROJECT_ROOT / "assets" / "reference" / "biomech_targets.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    curve_path = out_path.with_suffix(".vgrf_curves.npz")

    payload = {
        "schema_version": 1,
        "subject":   args.subject,
        "trial":     args.trial,
        "source": {
            "grf_mot": str(grf_path.relative_to(PROJECT_ROOT)),
            "ik_sto":  str(ik_path.relative_to(PROJECT_ROOT)),
            "osim":    str(osim_path.relative_to(PROJECT_ROOT)),
        },
        "body_mass_kg":     mass_kg,
        "body_weight_n":    bw_n,
        "contact_thresh_n": contact_thresh,
        "grf_sample_hz":    grf_hz,
        "ik_sample_hz":     ik_hz,
        "spatiotemporal": {
            "stride_period_s":       stride_period_s,
            "cadence_steps_per_min": cadence_spm,
            "double_support_frac":   ds_frac,
            "n_fp1_strikes":         int(len(fp1_strikes)),
            "n_fp2_strikes":         int(len(fp2_strikes)),
        },
        "vgrf": {
            "peak_bw_fp1": peak_fp1_bw,
            "peak_bw_fp2": peak_fp2_bw,
            "peak_bw":     peak_bw,
            "stance_curve_path": str(curve_path.relative_to(PROJECT_ROOT)),
            "stance_resample_n": STANCE_RESAMPLE_N,
        },
        "kinematics_rom": rom,
    }

    out_path.write_text(json.dumps(payload, indent=2))
    np.savez(curve_path, fp1=stance_curve_fp1, fp2=stance_curve_fp2)
    print(f"\nWrote {out_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {curve_path.relative_to(PROJECT_ROOT)}")
    print(f"\nKey targets:")
    print(f"  stride_period_s:       {stride_period_s:.3f}")
    print(f"  cadence_steps_per_min: {cadence_spm:.1f}")
    print(f"  double_support_frac:   {ds_frac:.3f}")
    print(f"  peak_vgrf_bw:          {peak_bw:.2f}")
    for name in ["hip_r", "knee_r", "ankle_r"]:
        r = rom[name]
        print(f"  {name} ROM (full trial): "
              f"[{r['full_trial_min_deg']:+.1f}°, {r['full_trial_max_deg']:+.1f}°] "
              f"range={r['full_trial_range_deg']:.1f}°")


if __name__ == "__main__":
    main()
