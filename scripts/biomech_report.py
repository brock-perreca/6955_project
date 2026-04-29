"""
biomech_report.py
─────────────────
Render a writeup-ready biomech comparison from one or more eval_biomech JSON
files. Produces:

  1. A markdown table: per-run sim values + reference target column + delta/%err.
     Drop straight into `docs/RESTART_LOG.md` or the writeup.
  2. A 6-panel matplotlib figure overlaying sim curves on the Ulrich reference:
       (a) hip angle (R), (b) knee angle (R), (c) ankle angle (R),
       (d) vGRF stance-phase curve (R), (e) hip-knee phase plane (R),
       (f) stride-period bar chart across runs.

Both rely on `assets/reference/biomech_targets.json` (and the sibling
`.vgrf_curves.npz`) produced by `extract_reference_biomech.py`. If you
haven't run that, this script will print one warning and emit a
sim-only figure.

Usage
─────
    python scripts/biomech_report.py
        results/restart_b2_eval.json
        [results/restart_b1_eval_1M.json ...]
        [--out-md docs/figures/biomech_report.md]
        [--out-fig docs/figures/biomech_report.png]
        [--targets assets/reference/biomech_targets.json]
        [--rerollout]   # re-run rollouts to get angle traces (else uses summary only)

The --rerollout path imports Walker2dPhaseAware and the saved policy from
each run dir to produce the angle traces overlaid in the figure. Without
it, the figure only shows the reference traces + per-run stride/cadence
bars (the markdown table is independent and always works).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "walker2d"))

# Reference column order in our reference cycle / sim qpos slice:
# 0=hip_r, 1=knee_r, 2=ankle_r, 3=hip_l, 4=knee_l, 5=ankle_l.
JOINT_NAMES = ["hip_r", "knee_r", "ankle_r", "hip_l", "knee_l", "ankle_l"]


# ── markdown table ─────────────────────────────────────────────────────────────

# Each row: (display_name, summary_key (without __median), is_pct_err)
TABLE_ROWS = [
    ("ep_len_steps (median)",      "ep_len_steps",          False),
    ("n_strides (median)",         "n_strides_detected",    False),
    ("stride_period_s",            "stride_period_s",       True),
    ("cadence (steps/min)",        "cadence_steps_per_min", True),
    ("double_support_frac",        "double_support_frac",   True),
    ("LR_stride_asymmetry",        "lr_stride_asymmetry",   False),
    ("swing_drag_frac",            "swing_drag_frac",       False),
    ("hip_knee_dtw",               "hip_knee_dtw",          False),
    ("peak_vgrf_bw",               "peak_vgrf_bw",          True),
    ("hip_r ROM (deg)",            "hip_r_rom_deg",         True),
    ("knee_r ROM (deg)",           "knee_r_rom_deg",        True),
    ("ankle_r ROM (deg)",          "ankle_r_rom_deg",       True),
    ("hip_l ROM (deg)",            "hip_l_rom_deg",         True),
    ("knee_l ROM (deg)",           "knee_l_rom_deg",        True),
    ("ankle_l ROM (deg)",          "ankle_l_rom_deg",       True),
]


def _ref_value(targets: dict | None, summary_key: str) -> float | None:
    if targets is None:
        return None
    spat = targets.get("spatiotemporal", {})
    vgrf = targets.get("vgrf", {})
    rom  = targets.get("kinematics_rom", {})
    if summary_key == "stride_period_s":       return spat.get("stride_period_s")
    if summary_key == "cadence_steps_per_min": return spat.get("cadence_steps_per_min")
    if summary_key == "double_support_frac":   return spat.get("double_support_frac")
    if summary_key == "peak_vgrf_bw":          return vgrf.get("peak_bw")
    if summary_key.endswith("_rom_deg"):
        joint = summary_key[:-len("_rom_deg")]
        return rom.get(joint, {}).get("full_trial_range_deg")
    return None


def render_table(runs: list[dict], targets: dict | None) -> str:
    """Build a markdown comparison table across runs."""
    headers = ["metric"] + [r["label"] for r in runs] + ["ref"]
    sep = ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(sep) + " |"]
    for disp, key, _ in TABLE_ROWS:
        row = [disp]
        for r in runs:
            val = r["summary"].get(f"{key}__median")
            row.append("—" if val is None else f"{val:.3f}")
        ref = _ref_value(targets, key)
        row.append("—" if ref is None else f"**{ref:.3f}**")
        lines.append("| " + " | ".join(row) + " |")

    # Append a progress-score row if present
    score_row = ["progress_score (0–4)"]
    for r in runs:
        sc = r.get("progress_score")
        score_row.append("—" if sc is None else f"{sc:.3f}")
    score_row.append("**4.000**")
    lines.append("| " + " | ".join(score_row) + " |")
    return "\n".join(lines)


# ── figure ─────────────────────────────────────────────────────────────────────

def _ref_cycle_traces(reference: np.ndarray, ctrl_hz: float = 125.0):
    """Return (t, hip, knee, ankle) in degrees for one reference stride (R leg)."""
    t = np.arange(len(reference)) / ctrl_hz
    return t, np.rad2deg(reference[:, 0]), np.rad2deg(reference[:, 1]), \
              np.rad2deg(reference[:, 2])


def _rollout_traces(run_dir: Path, model_path: Path, xml: str,
                    n_strides: int = 3, max_steps: int = 800):
    """Re-run a deterministic rollout and return per-stride angle/vGRF traces.

    Returns dict with keys: `t` (s), `hip_r`, `knee_r`, `ankle_r` (degrees),
    `vgrf_r_bw`, `r_strikes` (frame idx into the trace). Returns None on
    rollout failure.
    """
    try:
        from stable_baselines3 import PPO
        from ppo_walker2d_phase import Walker2dPhaseAware, load_ref_cycle, CTRL_HZ
    except Exception as e:
        print(f"[rollout] import failed: {e}")
        return None

    ref_path = run_dir / "reference.npy"
    if ref_path.exists():
        reference = np.load(ref_path).astype(np.float32)
    else:
        reference = load_ref_cycle(PROJECT_ROOT / "assets" / "reference"
                                   / "gait_cycle_reference.npy")
    env = Walker2dPhaseAware(reference=reference, xml_file=xml)
    body_weight_n = float(np.sum(env.model.body_mass)) * abs(
        float(env.model.opt.gravity[2]))
    model = PPO.load(str(model_path), device="cpu")

    qpos_log, vgrf_r = [], []
    obs, _ = env.reset()
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        qpos_log.append(env.data.qpos[3:9].copy())
        vgrf_r.append(abs(float(env.data.cfrc_ext[4, 5])))
        if term or trunc:
            break
    env.close()

    qpos = np.asarray(qpos_log)
    vgrf = np.asarray(vgrf_r) / body_weight_n
    if len(qpos) < 30:
        return None

    # Detect right strikes in the rollout trace for the phase plane
    above = vgrf > 0.05
    edges = np.where(above[1:] & ~above[:-1])[0] + 1
    strikes = []
    for e in edges:
        if not strikes or e - strikes[-1] >= 25:
            strikes.append(int(e))
    return {
        "t":        np.arange(len(qpos)) / CTRL_HZ,
        "hip_r":    np.rad2deg(qpos[:, 0]),
        "knee_r":   np.rad2deg(qpos[:, 1]),
        "ankle_r":  np.rad2deg(qpos[:, 2]),
        "vgrf_r_bw": vgrf,
        "r_strikes": strikes,
    }


def render_figure(runs: list[dict], targets: dict | None,
                  out_path: Path, do_rollout: bool, xml: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    (ax_hip, ax_knee, ax_ankle), (ax_grf, ax_phase, ax_bar) = axes

    # Reference traces (right leg, one stride) ─────────────────────────────────
    ref_cycle = None
    ref_path = PROJECT_ROOT / "assets" / "reference" / "gait_cycle_reference.npy"
    if ref_path.exists():
        ref_cycle = np.load(ref_path).astype(np.float32)
        t, hip, knee, ankle = _ref_cycle_traces(ref_cycle)
        for ax, y, name in [(ax_hip, hip, "hip_r"), (ax_knee, knee, "knee_r"),
                            (ax_ankle, ankle, "ankle_r")]:
            ax.plot(t, y, "k-", lw=2, alpha=0.85, label="reference")
            ax.set_title(name); ax.set_xlabel("time (s)"); ax.set_ylabel("deg")
        ax_phase.plot(hip, knee, "k-", lw=2, alpha=0.85, label="reference")
        ax_phase.set_xlabel("hip_r (deg)"); ax_phase.set_ylabel("knee_r (deg)")
        ax_phase.set_title("hip-knee phase (R)")

    # Reference vGRF stance curve ──────────────────────────────────────────────
    if targets is not None:
        curve_rel = targets.get("vgrf", {}).get("stance_curve_path")
        if curve_rel and (PROJECT_ROOT / curve_rel).exists():
            d = np.load(PROJECT_ROOT / curve_rel)
            stance_pct = np.linspace(0, 100, len(d["fp1"]))
            ax_grf.plot(stance_pct, d["fp1"], "k-", lw=2, alpha=0.85,
                        label="reference (fp1)")
    ax_grf.set_xlabel("stance phase (%)"); ax_grf.set_ylabel("vGRF / BW")
    ax_grf.set_title("stance-phase vGRF (R)")

    # Per-run rollouts ─────────────────────────────────────────────────────────
    colours = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for i, run in enumerate(runs):
        c = colours[i % len(colours)]
        label = run["label"]

        if do_rollout:
            run_dir    = Path(run["run_dir"])
            model_path = Path(run["model_path"])
            traces = _rollout_traces(run_dir, model_path, xml)
            if traces is not None and len(traces["r_strikes"]) >= 2:
                s, e = traces["r_strikes"][0], traces["r_strikes"][1]
                tt = traces["t"][s:e] - traces["t"][s]
                ax_hip.plot(tt,   traces["hip_r"][s:e],   color=c, alpha=0.8, label=label)
                ax_knee.plot(tt,  traces["knee_r"][s:e],  color=c, alpha=0.8, label=label)
                ax_ankle.plot(tt, traces["ankle_r"][s:e], color=c, alpha=0.8, label=label)
                ax_phase.plot(traces["hip_r"][s:e], traces["knee_r"][s:e],
                              color=c, alpha=0.8, label=label)

        # Per-run sim vGRF curve from JSON
        sim_curve = run.get("vgrf_curves", {}).get("vgrf_curve_r")
        if sim_curve:
            stance_pct = np.linspace(0, 100, len(sim_curve))
            ax_grf.plot(stance_pct, sim_curve, color=c, alpha=0.8, label=label)

    # Stride-period bar chart ──────────────────────────────────────────────────
    labels = [r["label"] for r in runs]
    sims   = [r["summary"].get("stride_period_s__median", np.nan) for r in runs]
    x = np.arange(len(labels))
    ax_bar.bar(x, sims, color=colours[:len(labels)], alpha=0.8, label="sim")
    if targets is not None:
        ref_stride = targets.get("spatiotemporal", {}).get("stride_period_s")
        if ref_stride is not None:
            ax_bar.axhline(ref_stride, color="k", lw=2, ls="--",
                           label=f"reference ({ref_stride:.2f} s)")
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(labels, rotation=20, ha="right")
    ax_bar.set_ylabel("stride period (s)")
    ax_bar.set_title("stride period: sim vs reference")
    ax_bar.legend(fontsize=8)

    for ax in (ax_hip, ax_knee, ax_ankle, ax_phase, ax_grf):
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

    fig.suptitle("Walker2d biomech: simulated runs vs Ulrich Subject 1 baseline")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path.relative_to(PROJECT_ROOT)}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _flatten_eval_json(blob) -> list[dict]:
    """eval_biomech.py writes a list of run dicts; tolerate single-dict too."""
    if isinstance(blob, list):
        return blob
    if isinstance(blob, dict) and "summary" in blob:
        return [blob]
    raise ValueError("Unexpected eval JSON shape")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("eval_jsons", nargs="+",
                   help="One or more eval_biomech JSON outputs")
    p.add_argument("--targets",
                   default=str(PROJECT_ROOT / "assets" / "reference"
                               / "biomech_targets.json"))
    p.add_argument("--out-md",
                   default=str(PROJECT_ROOT / "docs" / "figures"
                               / "biomech_report.md"))
    p.add_argument("--out-fig",
                   default=str(PROJECT_ROOT / "docs" / "figures"
                               / "biomech_report.png"))
    p.add_argument("--rerollout", action="store_true",
                   help="Re-roll the policies to overlay angle traces "
                        "(slower; otherwise the figure shows reference + bars only)")
    p.add_argument("--xml", default="walker2d.xml")
    args = p.parse_args()

    targets_path = Path(args.targets)
    targets = json.loads(targets_path.read_text()) if targets_path.exists() else None
    if targets is None:
        print(f"[warn] {targets_path} not found; sim-only figure, no ref column.")

    runs: list[dict] = []
    for path in args.eval_jsons:
        blob = json.loads(Path(path).read_text())
        for run in _flatten_eval_json(blob):
            runs.append(run)

    md = render_table(runs, targets)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(md + "\n")
    print(f"Wrote {Path(args.out_md).relative_to(PROJECT_ROOT)}")
    print()
    print(md)
    print()

    render_figure(runs, targets, Path(args.out_fig),
                  do_rollout=args.rerollout, xml=args.xml)


if __name__ == "__main__":
    main()
