"""
biomech_realism_dashboard.py
────────────────────────────
Render a comprehensive biomechanical-realism dashboard comparing one or more
trained policies against the measured Ulrich Subject 1 baseline. This is the
"how realistic is the gait?" report — sister to `biomech_report.py` which
focused on the right leg only and a 6-panel layout.

Inputs: one eval_biomech.py JSON (with one or more runs inside it).
Outputs: a single PNG with three blocks:

  Block A — kinematics: 6 panels (hip/knee/ankle × R/L), one stride window,
            sim trace per run overlaid on the reference.
  Block B — kinetics:   2 panels (R, L), normalised stance-phase vGRF curve
            (BW units), sim per run vs reference.
  Block C — scorecard:  per-metric % error bars, one row per metric, runs
            grouped by colour. ±20% "biomechanically credible" band shaded.

Per-run rollouts are needed for Block A (the eval JSON only stores summary
medians, not per-step traces). Block B uses the per-leg stance curves the
eval already aggregated.

Usage
─────
    python scripts/biomech_realism_dashboard.py results/biomech_candidates_eval.json
        [--out docs/figures/biomech_realism_dashboard.png]
        [--targets assets/reference/biomech_targets.json]
        [--max_steps 1500]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "walker2d"))


# ── per-run rollouts (needed for Block A) ──────────────────────────────────────

def _load_run_extras(run_dir: Path) -> tuple[str, dict]:
    """Return (xml_file, kwargs) restored from `<run_dir>/env_kwargs.json`."""
    p = run_dir / "env_kwargs.json"
    if not p.exists():
        return "walker2d.xml", {}
    meta = json.loads(p.read_text(encoding="utf-8"))
    extras: dict = {}
    for k in ("preview_k", "pose_joint_weights", "product_reward",
              "min_joint_pose", "v_target", "ref_root_drop"):
        if k in meta:
            extras[k] = meta[k]
    if "pose_joint_weights" in extras:
        extras["pose_joint_weights"] = tuple(extras["pose_joint_weights"])
    return str(meta.get("xml_file", "walker2d.xml")), extras


def rollout_for_traces(run_dir: Path, model_path: Path, max_steps: int) -> dict | None:
    """Single deterministic rollout. Returns per-leg angle traces + strikes."""
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

    xml_file, extras = _load_run_extras(run_dir)
    env = Walker2dPhaseAware(reference=reference, xml_file=xml_file, **extras)
    bw_n = float(np.sum(env.model.body_mass)) * abs(
        float(env.model.opt.gravity[2]))
    model = PPO.load(str(model_path), device="cpu")

    qpos_log, vgrf_r, vgrf_l = [], [], []
    obs, _ = env.reset()
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        qpos_log.append(env.data.qpos[3:9].copy())
        vgrf_r.append(abs(float(env.data.cfrc_ext[4, 5])))
        vgrf_l.append(abs(float(env.data.cfrc_ext[7, 5])))
        if term or trunc:
            break
    env.close()

    qpos = np.asarray(qpos_log)
    if len(qpos) < 30:
        return None
    vgrf_r_bw = np.asarray(vgrf_r) / bw_n
    vgrf_l_bw = np.asarray(vgrf_l) / bw_n

    min_gap = int(round(0.5 * CTRL_HZ))  # 0.5-s debounce, matches eval_biomech
    def _strikes(v: np.ndarray) -> list[int]:
        above = v > 0.05
        edges = np.where(above[1:] & ~above[:-1])[0] + 1
        kept: list[int] = []
        for e in edges:
            if not kept or e - kept[-1] >= min_gap:
                kept.append(int(e))
        return kept

    return {
        "t":         np.arange(len(qpos)) / CTRL_HZ,
        "hip_r":     np.rad2deg(qpos[:, 0]),
        "knee_r":    np.rad2deg(qpos[:, 1]),
        "ankle_r":   np.rad2deg(qpos[:, 2]),
        "hip_l":     np.rad2deg(qpos[:, 3]),
        "knee_l":    np.rad2deg(qpos[:, 4]),
        "ankle_l":   np.rad2deg(qpos[:, 5]),
        "vgrf_r_bw": vgrf_r_bw,
        "vgrf_l_bw": vgrf_l_bw,
        "r_strikes": _strikes(vgrf_r_bw),
        "l_strikes": _strikes(vgrf_l_bw),
        "xml":       xml_file,
    }


# ── reference helpers ──────────────────────────────────────────────────────────

def _ref_traces():
    """Return (t, dict[joint -> deg]) for one reference stride from the on-disk
    reference cycle, resampled to 125 Hz the way the env does."""
    from ppo_walker2d_phase import load_ref_cycle, CTRL_HZ
    ref = load_ref_cycle(PROJECT_ROOT / "assets" / "reference"
                         / "gait_cycle_reference.npy")
    t = np.arange(len(ref)) / CTRL_HZ
    cols = ["hip_r", "knee_r", "ankle_r", "hip_l", "knee_l", "ankle_l"]
    return t, {c: np.rad2deg(ref[:, i]) for i, c in enumerate(cols)}


def _ref_vgrf_curves(targets: dict) -> dict[str, np.ndarray]:
    rel = targets.get("vgrf", {}).get("stance_curve_path")
    if not rel:
        return {}
    p = PROJECT_ROOT / rel
    if not p.exists():
        return {}
    d = np.load(p)
    out: dict = {}
    if "fp1" in d.files:
        out["r"] = np.asarray(d["fp1"])
    if "fp2" in d.files:
        out["l"] = np.asarray(d["fp2"])
    return out


# ── scorecard rows ─────────────────────────────────────────────────────────────

# (display_name, summary_key, target_path_in_targets_json,
#  larger_is_worse_for_pct_err)
# All metrics here have a measured Ulrich target.
SCORECARD_ROWS = [
    ("stride period (s)",      "stride_period_s",       ("spatiotemporal", "stride_period_s")),
    ("cadence (steps/min)",    "cadence_steps_per_min", ("spatiotemporal", "cadence_steps_per_min")),
    ("double support",         "double_support_frac",   ("spatiotemporal", "double_support_frac")),
    ("peak vGRF / BW",         "peak_vgrf_bw",          ("vgrf",           "peak_bw")),
    ("hip_r ROM (deg)",        "hip_r_rom_deg",         ("kinematics_rom", "hip_r")),
    ("hip_l ROM (deg)",        "hip_l_rom_deg",         ("kinematics_rom", "hip_l")),
    ("knee_r ROM (deg)",       "knee_r_rom_deg",        ("kinematics_rom", "knee_r")),
    ("knee_l ROM (deg)",       "knee_l_rom_deg",        ("kinematics_rom", "knee_l")),
    ("ankle_r ROM (deg)",      "ankle_r_rom_deg",       ("kinematics_rom", "ankle_r")),
    ("ankle_l ROM (deg)",      "ankle_l_rom_deg",       ("kinematics_rom", "ankle_l")),
]


def _ref_value(targets: dict, path: tuple) -> float | None:
    a, b = path
    blk = targets.get(a, {}).get(b)
    if isinstance(blk, dict):
        # ROM block — full-trial range
        return blk.get("full_trial_range_deg")
    return blk


# ── figure ─────────────────────────────────────────────────────────────────────

def render_dashboard(runs: list[dict], targets: dict, traces: list[dict | None],
                     out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n_runs = len(runs)
    colours = plt.get_cmap("tab10").colors

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 6, figure=fig, hspace=0.75, wspace=0.55,
                  height_ratios=[1.0, 1.0, 1.0, 1.4])

    # Block A: kinematics, 2 rows × 3 cols (top 2 rows)
    joint_axes: dict[str, "plt.Axes"] = {}
    layout = [
        ("hip_r",   0, 0), ("knee_r",   0, 1), ("ankle_r", 0, 2),
        ("hip_l",   1, 0), ("knee_l",   1, 1), ("ankle_l", 1, 2),
    ]
    t_ref, ref_traces = _ref_traces()
    for jn, r, c in layout:
        ax = fig.add_subplot(gs[r, c])
        ax.plot(t_ref, ref_traces[jn], "k-", lw=2.4, alpha=0.9, label="ref")
        joint_axes[jn] = ax
        ax.set_title(jn, fontsize=10, fontweight="bold")
        ax.set_xlabel("time in stride (s)", fontsize=8)
        ax.set_ylabel("angle (deg)", fontsize=8)
        ax.grid(alpha=0.3)

    # vGRF stance curves panel (top right span: rows 0-1, col 3-5 split into r/l)
    ax_grf_r = fig.add_subplot(gs[0, 3:])
    ax_grf_l = fig.add_subplot(gs[1, 3:])
    ax_grf_r.set_title("right vGRF stance-phase curve (BW)",
                       fontsize=10, fontweight="bold")
    ax_grf_l.set_title("left vGRF stance-phase curve (BW)",
                       fontsize=10, fontweight="bold")
    ref_curves = _ref_vgrf_curves(targets)
    for ax, leg in [(ax_grf_r, "r"), (ax_grf_l, "l")]:
        if leg in ref_curves:
            stance_pct = np.linspace(0, 100, len(ref_curves[leg]))
            ax.plot(stance_pct, ref_curves[leg], "k-", lw=2.4, alpha=0.9, label="ref")
        ax.set_xlabel("stance phase (%)", fontsize=8)
        ax.set_ylabel("vGRF / BW", fontsize=8)
        ax.grid(alpha=0.3)
        ax.axhline(1.0, color="gray", lw=1, ls=":", alpha=0.4)

    # Block A overlays from each run's rollout
    for ri, (run, tr) in enumerate(zip(runs, traces)):
        c = colours[ri % len(colours)]
        label = run["label"]

        if tr is not None:
            # take the slice between the first two right strikes
            r_strikes = tr["r_strikes"]
            if len(r_strikes) >= 2:
                s, e = r_strikes[0], r_strikes[1]
                tt = tr["t"][s:e] - tr["t"][s]
                for jn in ["hip_r", "knee_r", "ankle_r",
                           "hip_l", "knee_l", "ankle_l"]:
                    joint_axes[jn].plot(tt, tr[jn][s:e],
                                        color=c, alpha=0.85, lw=1.5, label=label)

        # vGRF curves come from the eval JSON (already aggregated across eps)
        sim_curve_r = run.get("vgrf_curves", {}).get("vgrf_curve_r")
        sim_curve_l = run.get("vgrf_curves", {}).get("vgrf_curve_l")
        if sim_curve_r:
            stance_pct = np.linspace(0, 100, len(sim_curve_r))
            ax_grf_r.plot(stance_pct, sim_curve_r, color=c, alpha=0.85,
                          lw=1.5, label=label)
        if sim_curve_l:
            stance_pct = np.linspace(0, 100, len(sim_curve_l))
            ax_grf_l.plot(stance_pct, sim_curve_l, color=c, alpha=0.85,
                          lw=1.5, label=label)

    # Block B: hip-knee phase plane, R and L (row 2, cols 0-1 R, cols 2-3 L)
    ax_phase_r = fig.add_subplot(gs[2, 0:2])
    ax_phase_l = fig.add_subplot(gs[2, 2:4])
    ax_phase_r.set_title("hip-knee phase plane (R)", fontsize=10, fontweight="bold")
    ax_phase_l.set_title("hip-knee phase plane (L)", fontsize=10, fontweight="bold")
    for ax, hn, kn in [(ax_phase_r, "hip_r", "knee_r"),
                       (ax_phase_l, "hip_l", "knee_l")]:
        ax.plot(ref_traces[hn], ref_traces[kn], "k-", lw=2.4, alpha=0.9, label="ref")
        ax.set_xlabel(f"{hn} (deg)", fontsize=8)
        ax.set_ylabel(f"{kn} (deg)", fontsize=8)
        ax.grid(alpha=0.3)
    for ri, (run, tr) in enumerate(zip(runs, traces)):
        if tr is None:
            continue
        c = colours[ri % len(colours)]
        for ax, hn, kn, strikes in [(ax_phase_r, "hip_r", "knee_r", tr["r_strikes"]),
                                    (ax_phase_l, "hip_l", "knee_l", tr["l_strikes"])]:
            if len(strikes) >= 2:
                s, e = strikes[0], strikes[1]
                ax.plot(tr[hn][s:e], tr[kn][s:e],
                        color=c, alpha=0.85, lw=1.5, label=run["label"])

    # Block C: scorecard — three sub-panels (rows 2-3, cols 4-5 + row 3 spans)
    # Layout: row 2 cols 4-5 = progress score bar; row 3 spans 0-5 = pct_err strip.
    ax_score = fig.add_subplot(gs[2, 4:])
    ax_score.set_title("progress score (0–4, higher better)",
                       fontsize=10, fontweight="bold")
    labels = [r["label"] for r in runs]
    scores = [r.get("progress_score", float("nan")) for r in runs]
    bar_colors = [colours[i % len(colours)] for i in range(n_runs)]
    ax_score.barh(np.arange(n_runs), scores, color=bar_colors, alpha=0.85)
    ax_score.set_yticks(np.arange(n_runs))
    ax_score.set_yticklabels(labels, fontsize=8)
    ax_score.invert_yaxis()
    ax_score.axvline(4.0, color="k", lw=2, ls="--", alpha=0.6, label="ideal=4")
    ax_score.set_xlim(0, 4.05)
    ax_score.set_xlabel("score", fontsize=8)
    ax_score.legend(fontsize=7, loc="lower right")
    ax_score.grid(alpha=0.3, axis="x")

    # Pct-err strip across all metrics (Block C bottom)
    ax_pct = fig.add_subplot(gs[3, :])
    metrics = []
    pct_matrix: list[list[float]] = []  # rows = runs, cols = metrics
    for ri, run in enumerate(runs):
        row = []
        for disp, key, _ in SCORECARD_ROWS:
            blk = run.get("vs_reference", {}).get(key)
            row.append(float(blk["pct_err"]) if blk and "pct_err" in blk else float("nan"))
        pct_matrix.append(row)
    metrics = [d for d, _, _ in SCORECARD_ROWS]
    arr = np.asarray(pct_matrix)

    nM = len(metrics)
    width = 0.8 / max(1, n_runs)
    x = np.arange(nM)
    for ri in range(n_runs):
        ax_pct.bar(x + (ri - (n_runs - 1) / 2) * width, arr[ri],
                   width=width, color=bar_colors[ri], alpha=0.85,
                   label=labels[ri])
    ax_pct.axhspan(-20, 20, color="gray", alpha=0.15,
                   label="±20% (biomech-credible band)")
    ax_pct.axhline(0, color="k", lw=1.5)
    ax_pct.set_xticks(x)
    ax_pct.set_xticklabels(metrics, rotation=20, ha="right", fontsize=8)
    ax_pct.set_ylabel("% error vs Ulrich Subject 1 reference", fontsize=9)
    ax_pct.set_title("biomechanical-realism scorecard "
                     "(closer to 0 = more realistic; outside ±20% gray band = visibly off)",
                     fontsize=10, fontweight="bold")
    # Cap y-axis at +400%/-100% for legibility (some are off the chart)
    ax_pct.set_ylim(-110, 410)
    # Mark out-of-range bars with arrows
    for ri in range(n_runs):
        for mi in range(nM):
            v = arr[ri, mi]
            if not np.isfinite(v):
                continue
            if v > 410:
                ax_pct.annotate(f"{v:+.0f}%",
                                xy=(mi + (ri - (n_runs - 1) / 2) * width, 405),
                                ha="center", va="bottom", fontsize=6, color=bar_colors[ri])
    ax_pct.legend(fontsize=7, ncol=min(n_runs + 1, 6), loc="upper right")
    ax_pct.grid(alpha=0.3, axis="y")

    # Joint-axes legend (single shared legend below the kinematics block so
    # it doesn't cover any of the joint-angle traces).
    handles, labels_ = joint_axes["hip_r"].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center",
               bbox_to_anchor=(0.31, 0.965), ncol=min(len(labels_), 5),
               fontsize=8, frameon=True)
    ax_grf_r.legend(fontsize=7, loc="upper right")

    fig.suptitle("Walker2d biomechanical realism — sim policies vs Ulrich Subject 1 baseline",
                 fontsize=13, fontweight="bold", y=0.995)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"Wrote {out_path.relative_to(PROJECT_ROOT)}")


def render_markdown(runs: list[dict], targets: dict, out_md: Path) -> str:
    """Markdown table mirroring the scorecard, with absolute sim values."""
    headers = ["metric", "ref"] + [r["label"] for r in runs]
    sep = ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(sep) + " |"]
    for disp, key, path in SCORECARD_ROWS:
        ref = _ref_value(targets, path)
        row = [disp, "—" if ref is None else f"**{ref:.3f}**"]
        for r in runs:
            v = r["summary"].get(f"{key}__median")
            if v is None:
                row.append("—")
            else:
                blk = r.get("vs_reference", {}).get(key)
                pct = blk.get("pct_err") if blk else None
                cell = f"{v:.3f}" + (f" ({pct:+.0f}%)" if pct is not None else "")
                row.append(cell)
        lines.append("| " + " | ".join(row) + " |")
    score_row = ["progress_score (0–4)", "**4.000**"]
    for r in runs:
        sc = r.get("progress_score")
        score_row.append("—" if sc is None else f"{sc:.3f}")
    lines.append("| " + " | ".join(score_row) + " |")
    md = "\n".join(lines)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md + "\n")
    print(f"Wrote {out_md.relative_to(PROJECT_ROOT)}")
    return md


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("eval_json")
    p.add_argument("--out", default=str(PROJECT_ROOT / "docs" / "figures"
                                       / "biomech_realism_dashboard.png"))
    p.add_argument("--out_md", default=str(PROJECT_ROOT / "docs" / "figures"
                                          / "biomech_realism_dashboard.md"))
    p.add_argument("--targets",
                   default=str(PROJECT_ROOT / "assets" / "reference"
                               / "biomech_targets.json"))
    p.add_argument("--max_steps", type=int, default=1500,
                   help="Per-run rollout length for Block A traces.")
    args = p.parse_args()

    blob = json.loads(Path(args.eval_json).read_text())
    if isinstance(blob, dict):
        blob = [blob]
    runs: list[dict] = blob
    targets = json.loads(Path(args.targets).read_text())

    print(f"[dashboard] {len(runs)} runs:")
    for r in runs:
        print(f"  - {r['label']}  ({r['run_dir']})")

    traces: list[dict | None] = []
    for r in runs:
        print(f"[rollout] {r['label']}")
        tr = rollout_for_traces(Path(r["run_dir"]),
                                Path(r["model_path"]),
                                max_steps=args.max_steps)
        traces.append(tr)

    md = render_markdown(runs, targets, Path(args.out_md))
    print()
    print(md)
    print()
    render_dashboard(runs, targets, traces, Path(args.out))


if __name__ == "__main__":
    main()
