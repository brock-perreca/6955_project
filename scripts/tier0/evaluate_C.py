"""
evaluate_C.py — produce the Tier 0 experiment-C comparison artifacts.

Inputs:
  - results/restart_b4_hiprelax_s11/  (and s12, s13)  — the trained runs
  - results/restart_b2_xvel/                          — xvel-5M baseline

For each run:
  - Dashboard PNG for `final` and `2000000` checkpoints
  - eval_biomech JSON (6 deterministic episodes × 2500 steps)
  - 600-step MP4 rendered at the same camera as `reference_replay.mp4`
  - A 5-seed × 600-step hip-trace plot (sim hip overlaid on ref + jnt_range)

Outputs collected under `docs/figures/tier0/C_hiprelax/`.

Usage:
    python scripts/tier0/evaluate_C.py
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "walker2d"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "diagnostics"))
from ppo_walker2d_phase import Walker2dPhaseAware, CTRL_HZ  # noqa: E402
from eval_biomech import _load_policy, _load_env_kwargs  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "docs" / "figures" / "tier0" / "C_hiprelax"

# (label, run_dir, ckpts_to_dashboard)
RUNS_DEFAULT = [
    ("xvel-5M",     "results/restart_b2_xvel",       ["final"]),
    ("hiprelax_s11","results/restart_b4_hiprelax_s11",["final", "2000000"]),
    ("hiprelax_s12","results/restart_b4_hiprelax_s12",["final", "2000000"]),
    ("hiprelax_s13","results/restart_b4_hiprelax_s13",["final", "2000000"]),
]


def run_dashboard(label, run_dir, ckpt):
    out_png = OUT_ROOT / f"{label}_{ckpt}_dashboard.png"
    spec = f"{run_dir}:{ckpt}:{label}-{ckpt}"
    cmd = [
        sys.executable, "src/diagnostics/run_dashboard.py", spec,
        "--steps", "600",
        "--out", str(out_png),
    ]
    print(f"\n=== dashboard: {label} {ckpt} ===")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return out_png


def run_eval_biomech(label, run_dir, ckpt="final"):
    out_json = OUT_ROOT / f"{label}_eval.json"
    spec = f"{run_dir}:{ckpt}:{label}"
    cmd = [
        sys.executable, "src/diagnostics/eval_biomech.py",
        "--eps", "6", "--steps", "2500",
        spec, "--out", str(out_json),
    ]
    print(f"\n=== eval_biomech: {label} {ckpt} ===")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return out_json


def render_mp4(label, run_dir, ckpt="final", steps=600, seed=0):
    """Render a 600-step MP4 with the trained camera (matches reference_replay).

    Uses the same kinematic-replay-style frame writer (rgb_array). Reference
    overlay is omitted; comparison happens by viewing the mp4 next to
    `00_reference_replay.mp4` in the same folder.
    """
    out_mp4 = OUT_ROOT / f"{label}_{ckpt}.mp4"
    extras = _load_env_kwargs(run_dir)
    xml_file = extras.pop("xml_file", "walker2d.xml")
    ref_path = Path(run_dir) / "reference.npy"
    ref = (np.load(ref_path).astype(np.float32) if ref_path.exists() else None)
    if ref is None:
        from ppo_walker2d_phase import load_ref_cycle
        ref = load_ref_cycle(PROJECT_ROOT / "assets" / "reference"
                             / "gait_cycle_reference.npy")

    env = Walker2dPhaseAware(
        reference=ref, xml_file=xml_file, render_mode="rgb_array",
        pose_term_thresh=9999.0, ankle_term_thresh=9999.0, **extras,
    )
    model_path = (str(Path(run_dir) / "model") if ckpt == "final"
                  else str(Path(run_dir) / "checkpoints" / f"model_{ckpt}_steps"))
    model = _load_policy(model_path)

    obs, _ = env.reset(seed=seed)
    frames = []
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        frames.append(env.render())
        if term or trunc:
            break
    env.close()

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_mp4, frames, fps=int(CTRL_HZ), macro_block_size=1)
    print(f"  wrote {out_mp4}  ({len(frames)} frames)")
    return out_mp4


def hip_trace_panel(runs):
    """5-seed × 600-step hip_r and hip_l vs ref+jnt_range, one column per run."""
    fig, axes = plt.subplots(2, len(runs), figsize=(4.5 * len(runs), 6),
                             constrained_layout=True, sharex=True)
    if len(runs) == 1:
        axes = axes.reshape(2, 1)
    for col, (label, run_dir, _) in enumerate(runs):
        extras   = _load_env_kwargs(run_dir)
        xml_file = extras.pop("xml_file", "walker2d.xml")
        ref      = np.load(Path(run_dir) / "reference.npy").astype(np.float32)
        env      = Walker2dPhaseAware(reference=ref, xml_file=xml_file, **extras)
        model    = _load_policy(str(Path(run_dir) / "model"))
        upper_r = float(env.model.jnt_range[3, 1])
        upper_l = float(env.model.jnt_range[6, 1])

        # Aggregate 5 seeds × 600 steps
        all_hr, all_hl, all_ph = [], [], []
        for seed in range(5):
            obs, _ = env.reset(seed=seed)
            hr, hl, ph = [], [], []
            for _ in range(600):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                hr.append(env.data.qpos[3])
                hl.append(env.data.qpos[6])
                ph.append(int(info.get("phase", 0)))
                if term or trunc:
                    break
            all_hr.append(np.array(hr))
            all_hl.append(np.array(hl))
            all_ph.append(np.array(ph))

        # Plot seed-0 trace (other seeds visually identical when policy is healthy)
        for axis, sim_q, ref_idx, upper, label_q in (
            (axes[0, col], all_hr[0], 0, upper_r, "hip_r"),
            (axes[1, col], all_hl[0], 3, upper_l, "hip_l"),
        ):
            T = len(sim_q)
            axis.plot(np.arange(T), np.degrees(sim_q),
                      color="C0", lw=1, label=f"sim {label_q}")
            axis.plot(np.arange(T), np.degrees(ref[all_ph[0], ref_idx]),
                      color="black", lw=1, label="ref")
            axis.axhline(np.degrees(upper), color="C2", lw=1.2, ls="--",
                         label=f"upper limit ({np.degrees(upper):+.0f}°)")
            axis.set_ylabel(f"{label_q} (deg)")
            axis.grid(alpha=0.3)
            if col == 0:
                axis.legend(loc="lower right", fontsize=7)

        # Aggregate stats annotation
        flat = np.concatenate(all_hr)
        med = float(np.degrees(np.median(flat)))
        std = float(np.degrees(flat.std()))
        near = 100 * float(np.mean(flat > (upper_r - np.radians(0.5))))
        axes[0, col].set_title(
            f"{label}\nmedian {med:+.1f}°  std {std:.1f}°  "
            f"near-limit {near:.0f}%",
            fontsize=10)
        env.close()
    axes[1, 0].set_xlabel("step")
    axes[1, -1].set_xlabel("step")
    out = OUT_ROOT / "C_hip_trace_comparison.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nWrote {out}")
    return out


def write_summary(runs, eval_jsons, hip_trace_png):
    """Markdown table comparing runs on key biomech metrics.

    eval_biomech.py writes a top-level list of run blobs; each blob has a
    `summary` dict with `<metric>__median`/`__iqr`/`__n` triples. Pull the
    medians for the writeup table.
    """
    def med(summary, key):
        return summary.get(f"{key}__median")

    rows = []
    for (label, run_dir, _), j in zip(runs, eval_jsons):
        with open(j, encoding="utf-8") as f:
            data = json.load(f)
        blob = data[0] if isinstance(data, list) else data
        s = blob["summary"]
        rows.append({
            "label":      label,
            "ep_len":     med(s, "ep_len_steps"),
            "n_strides":  med(s, "n_strides_detected"),
            "stride_s":   med(s, "stride_period_s"),
            "cadence":    med(s, "cadence_steps_per_min"),
            "ds_frac":    med(s, "double_support_frac"),
            "hip_r_rom":  med(s, "hip_r_rom_deg"),
            "hip_l_rom":  med(s, "hip_l_rom_deg"),
            "knee_r_rom": med(s, "knee_r_rom_deg"),
            "lr_asym":    med(s, "lr_stride_asymmetry"),
            "peak_vgrf":  med(s, "peak_vgrf_bw"),
            "score":      blob.get("progress_score"),
        })
    out_md = OUT_ROOT / "C_summary.md"
    def fmt(v, spec):
        return format(v, spec) if isinstance(v, (int, float)) else str(v)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Tier 0 — experiment C summary\n\n")
        f.write(f"Hip-trace comparison: `{hip_trace_png.relative_to(PROJECT_ROOT)}`\n\n")
        f.write("All values are medians over 6 deterministic eval episodes "
                "× 2500 max steps (eval_biomech.py).\n\n")
        f.write("| label | ep_len | strides | stride s | cadence | DS frac | "
                "hip_r ROM | hip_l ROM | knee_r ROM | LR asym | vGRF/BW | "
                "progress |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['label']} "
                    f"| {fmt(r['ep_len'], '.0f')} "
                    f"| {fmt(r['n_strides'], '.0f')} "
                    f"| {fmt(r['stride_s'], '.3f')} "
                    f"| {fmt(r['cadence'], '.1f')} "
                    f"| {fmt(r['ds_frac'], '.3f')} "
                    f"| {fmt(r['hip_r_rom'], '.2f')} "
                    f"| {fmt(r['hip_l_rom'], '.2f')} "
                    f"| {fmt(r['knee_r_rom'], '.2f')} "
                    f"| {fmt(r['lr_asym'], '.3f')} "
                    f"| {fmt(r['peak_vgrf'], '.2f')} "
                    f"| {fmt(r['score'], '.2f')} |\n")
        f.write("\n**Reference (measured Subject 1, baseline 1.25 m/s):** "
                "stride 1.120 s, cadence 107.1, DS 0.227, "
                "hip_r ROM 45.4°, hip_l ROM 45.4°, knee_r ROM 65.7°, "
                "LR asym < 0.10, peak vGRF/BW 1.10. Progress score is in [0, 4].\n")
    print(f"\nWrote {out_md}")
    return out_md


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-mp4",   action="store_true", help="Skip MP4 rendering")
    p.add_argument("--skip-eval",  action="store_true", help="Skip eval_biomech")
    p.add_argument("--skip-dash",  action="store_true", help="Skip dashboards")
    p.add_argument("--ckpts",      nargs="+", default=None,
                   help="Override the per-run checkpoints to dashboard")
    args = p.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Skip any run dir that doesn't exist yet (e.g., not all seeds finished)
    runs = [(L, d, c) for (L, d, c) in RUNS_DEFAULT
            if (PROJECT_ROOT / d / "model.zip").exists()]
    if not runs:
        print("No runs available yet.")
        return
    print(f"Available runs: {[r[0] for r in runs]}")

    # 1. Dashboards
    if not args.skip_dash:
        for label, run_dir, ckpts in runs:
            ckpts_to_run = args.ckpts if args.ckpts else ckpts
            for ckpt in ckpts_to_run:
                # Only run if checkpoint file exists
                if ckpt == "final":
                    if not (PROJECT_ROOT / run_dir / "model.zip").exists():
                        continue
                else:
                    p = PROJECT_ROOT / run_dir / "checkpoints" / f"model_{ckpt}_steps.zip"
                    if not p.exists():
                        print(f"[skip] {label} {ckpt}: {p} missing")
                        continue
                run_dashboard(label, run_dir, ckpt)

    # 2. Eval biomech (final ckpt only). Re-use existing JSONs on disk when
    # --skip-eval, so the summary-writer can run independently.
    eval_jsons = []
    for label, run_dir, _ in runs:
        if not args.skip_eval:
            j = run_eval_biomech(label, run_dir, "final")
        else:
            j = OUT_ROOT / f"{label}_eval.json"
            if not j.exists():
                continue
        eval_jsons.append(j)

    # 3. MP4s — final ckpt for all runs
    if not args.skip_mp4:
        for label, run_dir, _ in runs:
            render_mp4(label, run_dir, "final", steps=600, seed=0)

    # 4. Hip-trace comparison panel
    hip_trace_png = hip_trace_panel(runs)

    # 5. Markdown summary
    if eval_jsons:
        write_summary(runs, eval_jsons, hip_trace_png)


if __name__ == "__main__":
    main()
