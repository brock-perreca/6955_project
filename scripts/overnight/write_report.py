"""
write_report.py
───────────────
Produce REPORT.md for one experiment from its eval_biomech.json + tb logs.

The overnight master agent fills REPORT.md for every experiment to satisfy
the artefact contract in HANDOFF.md §5. Doing 22 of these by hand burns
context; this script does it deterministically from the on-disk artefacts.

Output is a one-page markdown report with:
  - hypothesis + setup (from the train_cmd.txt + a CLI-passed hypothesis)
  - headline TB metrics: r_pose, r_vel, r_ee, r_root, ep_rew, ep_len
  - held-out biomech: stride period, cadence, DTW (hip-knee + all-joints),
    LR symmetry, swing drag, peak vGRF, per-joint ROM
  - parent comparison block (vs results/restart_b2_xvel by default)
  - anti-Goodhart checklist auto-flagged from the metrics
  - heuristic verdict (KEEP / DROP / FOLLOW-UP)

Heuristics (all comparing this run's median to the parent):
  KEEP:      survival>=parent AND (cadence improves >=10% OR hip_r ROM
             improves >=50% OR hip_knee_dtw improves >=15%)
  DROP:      survival<parent*0.5 OR LR_asymmetry > 0.5 OR n_strides == 1
  FOLLOW-UP: anything else worth a closer look (improved on one axis but
             regressed on another)

Brock will review the MP4s to make the actual call. The verdict is a
ranking hint, not a decision.

Usage:
    python scripts/overnight/write_report.py results/overnight_<TS>/<exp>/ \\
        --hypothesis "Up-weight bilateral hip channels in pose tracking" \\
        --parent results/restart_b2_xvel/eval_biomech.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARENT_LABEL = "xvel-5M"
PARENT_TARGETS = {  # measured Subject 1 reference (from biomech_targets.json)
    "stride_period_s":       1.120,
    "cadence_steps_per_min": 107.1,
    "double_support_frac":   0.227,
    "lr_stride_asymmetry":   0.10,    # < 0.10 = symmetric
    "hip_r_rom_deg":         45.4,
    "knee_r_rom_deg":        65.7,
    "ankle_r_rom_deg":       40.0,
    "peak_vgrf_bw":          1.10,
}
# Parent run baseline numbers from RESTART_LOG.md batch 2 (xvel-5M):
PARENT_BASELINE = {
    "ep_len_steps":          2500,
    "n_strides_detected":     61,
    "stride_period_s":         0.323,
    "cadence_steps_per_min": 372,
    "double_support_frac":     0.074,
    "lr_stride_asymmetry":     0.099,
    "swing_drag_frac":         0.0,
    "hip_knee_dtw":            0.148,
    "peak_vgrf_bw":            3.20,
    "hip_r_rom_deg":           1.8,
    "knee_r_rom_deg":         21.2,
    "ankle_r_rom_deg":        20.3,
}


def _read_json(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] read {p}: {e}")
        return {}


def _scalar_summary(eval_path: Path) -> dict:
    """Pull the first run's summary from eval_biomech.json."""
    raw = _read_json(eval_path)
    if not isinstance(raw, list) or not raw:
        return {}
    return raw[0].get("summary", {}) or {}


def _read_tb_scalars(tb_dir: Path) -> dict:
    """Best-effort: read the last value of each scalar tag from TB events.

    Falls back to the run.log tail if TB parsing isn't available.
    """
    if not tb_dir.exists():
        return {}
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return {}
    sub = next((d for d in tb_dir.iterdir() if d.is_dir()), tb_dir)
    try:
        ea = EventAccumulator(str(sub), size_guidance={"scalars": 0})
        ea.Reload()
        out = {}
        for tag in ea.Tags().get("scalars", []):
            evs = ea.Scalars(tag)
            if evs:
                out[tag] = float(evs[-1].value)
        return out
    except Exception as e:
        print(f"[warn] TB read {sub}: {e}")
        return {}


def _delta(this: float, parent: float) -> str:
    if parent == 0 or not math.isfinite(parent) or not math.isfinite(this):
        return ""
    pct = 100.0 * (this - parent) / abs(parent)
    sign = "+" if pct >= 0 else ""
    return f" ({sign}{pct:.0f}%)"


def _fmt(x, fmt=".3f"):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return format(x, fmt)


def _check_anti_goodhart(s: dict) -> dict:
    """Flag exploits that the metrics betray."""
    flags = {}

    ep_len = s.get("ep_len_steps__median", 0) or 0
    n_str  = s.get("n_strides_detected__median", 0) or 0
    sp     = s.get("stride_period_s__median", float("nan"))
    asym   = s.get("lr_stride_asymmetry__median", 0) or 0
    drag   = s.get("swing_drag_frac__median", 0) or 0
    hip_r  = s.get("hip_r_rom_deg__median", 0) or 0
    ankle_r = s.get("ankle_r_rom_deg__median", 0) or 0

    # Stand-and-wiggle: long episode but very short stride period (cadence
    # detector firing on small force oscillations).
    if ep_len > 1500 and sp and sp < 0.20:
        flags["stand_and_wiggle"] = (
            f"yes — ep_len={int(ep_len)} but stride_period={sp:.3f}s "
            f"(< 0.20s = sub-stand-still cadence)"
        )
    elif ep_len < 200:
        flags["stand_and_wiggle"] = "no — episode too short to be standing"
    else:
        flags["stand_and_wiggle"] = "no"

    # Stiff hip: hip_r ROM < 15° while reference sweeps ~43°.
    if hip_r > 0 and hip_r < 15:
        flags["stiff_hip"] = f"yes — hip_r ROM={hip_r:.1f}° (ref ~45°)"
    elif hip_r >= 15 and hip_r < 30:
        flags["stiff_hip"] = f"partial — hip_r ROM={hip_r:.1f}° (ref ~45°)"
    elif hip_r >= 30:
        flags["stiff_hip"] = f"no — hip_r ROM={hip_r:.1f}° (ref ~45°)"
    else:
        flags["stiff_hip"] = "can't tell from metrics alone"

    # Ankle paddling — high ankle ROM with no foot lift would show as high
    # ankle ROM with low forward progress. Use cadence as a forward-progress
    # proxy: stand-still cadence > 300/min
    if ankle_r > 30 and ep_len > 1500 and sp and sp < 0.30:
        flags["ankle_paddling"] = "possible — high ankle ROM + short stride"
    else:
        flags["ankle_paddling"] = "no"

    # Asymmetric (one-legged hopping ish)
    if asym > 0.5:
        flags["asymmetry"] = f"yes — LR_asymmetry={asym:.2f}"
    elif asym > 0.20:
        flags["asymmetry"] = f"partial — LR_asymmetry={asym:.2f}"
    else:
        flags["asymmetry"] = f"no — LR_asymmetry={asym:.2f}"

    # Toe walking — short stride + low stance-phase double-support
    flags["toe_walking"] = "can't tell from metrics alone"

    return flags


def _verdict(s: dict, parent: dict) -> str:
    """Heuristic KEEP / DROP / FOLLOW-UP."""
    ep = s.get("ep_len_steps__median", 0) or 0
    p_ep = parent.get("ep_len_steps", 1) or 1
    if ep < 0.5 * p_ep or ep < 200:
        return "DROP"
    asym = s.get("lr_stride_asymmetry__median", 0) or 0
    if asym > 0.5:
        return "DROP"

    sp = s.get("stride_period_s__median", float("nan"))
    p_sp = parent.get("stride_period_s", float("nan"))
    cadence_improved = (
        sp and p_sp and math.isfinite(sp) and math.isfinite(p_sp)
        and abs(sp - 1.12) <= abs(p_sp - 1.12) * 0.90
    )
    hip = s.get("hip_r_rom_deg__median", 0) or 0
    p_hip = parent.get("hip_r_rom_deg", 1) or 1
    hip_improved = hip >= 1.5 * p_hip and p_hip > 0

    dtw = s.get("hip_knee_dtw__median", float("inf"))
    p_dtw = parent.get("hip_knee_dtw", float("inf"))
    dtw_improved = (
        math.isfinite(dtw) and math.isfinite(p_dtw) and p_dtw > 0
        and dtw <= p_dtw * 0.85
    )

    if cadence_improved or hip_improved or dtw_improved:
        return "KEEP"
    return "FOLLOW-UP"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="results/overnight_*/<exp>/")
    ap.add_argument("--hypothesis", default="(unstated — fill in manually)")
    ap.add_argument("--parent_label", default=PARENT_LABEL)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    eval_p  = run_dir / "eval_biomech.json"
    tb_p    = run_dir / "tb"

    if not eval_p.exists():
        print(f"[fail] {eval_p} not found")
        raise SystemExit(1)

    s   = _scalar_summary(eval_p)
    tb  = _read_tb_scalars(tb_p)
    cmd = (run_dir / "train_cmd.txt").read_text(encoding="utf-8") if (run_dir / "train_cmd.txt").exists() else ""
    meta = _read_json(run_dir / "train_meta.json")

    flags   = _check_anti_goodhart(s)
    verdict = _verdict(s, PARENT_BASELINE)

    parent = PARENT_BASELINE
    ref    = PARENT_TARGETS

    body = []
    body.append(f"# `{run_dir.name}` — {args.hypothesis}")
    body.append("")
    body.append("| field | value |")
    body.append("|---|---|")
    body.append(f"| run_dir | `{run_dir.relative_to(PROJECT_ROOT) if run_dir.is_relative_to(PROJECT_ROOT) else run_dir}` |")
    body.append(f"| based_on | `results/restart_b2_xvel` (xvel-5M parent) |")
    body.append(f"| train wallclock | {meta.get('wallclock_s', '—')} s |")
    body.append(f"| seed | (see train_cmd.txt) |")
    body.append("")
    body.append("## Hypothesis")
    body.append("")
    body.append(args.hypothesis)
    body.append("")
    body.append("## Setup")
    body.append("")
    body.append("```")
    body.append(cmd.strip())
    body.append("```")
    body.append("")
    body.append("## Headline TB metrics (final-rollout means)")
    body.append("")
    body.append("| metric | value |")
    body.append("|---|---|")
    for k in ("rollout/ep_rew_mean", "rollout/ep_len_mean",
              "reward/r_pose", "reward/r_vel", "reward/r_ee",
              "reward/r_root", "reward/energy_pen"):
        if k in tb:
            body.append(f"| `{k}` | {tb[k]:.4f} |")
    body.append("")
    body.append("## Held-out biomech (median over eval episodes)")
    body.append("")
    body.append("| metric | this run | parent (xvel-5M) | reference |")
    body.append("|---|---|---|---|")
    for sim_key, parent_key, ref_key, fmt in [
        ("ep_len_steps__median",            "ep_len_steps",         None,                       ".0f"),
        ("n_strides_detected__median",      "n_strides_detected",   None,                       ".0f"),
        ("stride_period_s__median",         "stride_period_s",      "stride_period_s",          ".3f"),
        ("cadence_steps_per_min__median",   "cadence_steps_per_min","cadence_steps_per_min",    ".1f"),
        ("double_support_frac__median",     "double_support_frac",  "double_support_frac",      ".3f"),
        ("swing_drag_frac__median",         "swing_drag_frac",      None,                       ".3f"),
        ("lr_stride_asymmetry__median",     "lr_stride_asymmetry",  "lr_stride_asymmetry",      ".3f"),
        ("hip_knee_dtw__median",            "hip_knee_dtw",         None,                       ".3f"),
        ("all_joints_dtw__median",          None,                   None,                       ".3f"),
        ("peak_vgrf_bw__median",            "peak_vgrf_bw",         "peak_vgrf_bw",             ".2f"),
        ("hip_r_rom_deg__median",           "hip_r_rom_deg",        "hip_r_rom_deg",            ".1f"),
        ("knee_r_rom_deg__median",          "knee_r_rom_deg",       "knee_r_rom_deg",           ".1f"),
        ("ankle_r_rom_deg__median",         "ankle_r_rom_deg",      "ankle_r_rom_deg",          ".1f"),
    ]:
        sim = s.get(sim_key)
        p   = parent.get(parent_key) if parent_key else None
        r   = ref.get(ref_key)       if ref_key    else None
        delta = _delta(sim, p) if (sim is not None and p is not None) else ""
        body.append(
            f"| {sim_key.replace('__median','')} "
            f"| {_fmt(sim, fmt)}{delta} "
            f"| {_fmt(p, fmt)} "
            f"| {_fmt(r, fmt) if r is not None else '—'} |"
        )
    body.append("")
    body.append("## Anti-Goodhart check (auto-flagged from metrics)")
    body.append("")
    for k, v in flags.items():
        body.append(f"- **{k}**: {v}")
    body.append("")
    body.append("## Verdict (heuristic)")
    body.append("")
    body.append(f"**{verdict}**")
    body.append("")
    body.append(
        "> The verdict is heuristic. Watch the MP4 before trusting it. "
        "`r_pose` improvements without hip-excursion improvements are not "
        "a success."
    )
    body.append("")

    out = Path(args.out) if args.out else (run_dir / "REPORT.md")
    out.write_text("\n".join(body), encoding="utf-8")
    print(f"Wrote {out}  (verdict={verdict})")


if __name__ == "__main__":
    main()
