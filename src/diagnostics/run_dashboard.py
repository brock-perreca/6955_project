"""
run_dashboard.py — auto-generated 4-panel PNG that exposes whether a
trained Walker2d policy is actually walking, vs. producing sporadic
kicks that read as ROM in scalar metrics.

Panels (top-left → bottom-right):
  1. 6-panel joint angle vs phase, sim cycle overlaid on reference
     (one R-strike → R-strike window, normalised to [0, 1]).
  2. Reward decomposition over the same cycle (r_pose, r_vel, r_ee,
     r_root, ctrl_cost, weighted total).
  3. Action histograms per actuator (full rollout). Saturation at ±1
     jumps out immediately.
  4. Foot xz trajectory in root-relative frame, sim and ref overlaid,
     one subplot per leg.

The cycle window is the first complete R-foot heel-strike → R-foot
heel-strike interval (same detector as `eval_biomech.py`). If no two
strikes are detected, the joint-angle and foot-xz panels fall back to
the first 140 frames (= one nominal stride).

Usage:
    python src/diagnostics/run_dashboard.py <run_dir>:<ckpt>[:<label>]
        [--xml walker2d.xml] [--steps 600] [--seed 0]
        [--out <run_dir>/dashboard.png]

`<ckpt>` is "final" or an integer step (matches render_phase.py /
eval_biomech.py spec syntax). PPO/SAC autodetected via the same loader
fallback used by eval_biomech.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "walker2d"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "diagnostics"))
from ppo_walker2d_phase import Walker2dPhaseAware, CTRL_HZ  # noqa: E402
from eval_biomech import (  # noqa: E402
    _load_policy, _load_env_kwargs, _rising_edges, parse_spec, progress_score,
    aggregate, episode_metrics,
)

JOINT_NAMES = ["hip_r", "knee_r", "ankle_r", "hip_l", "knee_l", "ankle_l"]
NOMINAL_CYCLE_FRAMES = 140  # fallback window if heel-strike detection fails


def rollout_with_traces(env, model, max_steps, seed):
    """One deterministic episode; collect everything the dashboard needs.

    Mirrors `eval_biomech.rollout_episode` but additionally captures
    actions, reward components from `info[...]`, and root-relative foot xz.
    """
    obs, _ = env.reset(seed=seed)
    out = {
        "qpos":      [],   # joint angles (T, 6)
        "qvel":      [],   # joint velocities (T, 6)
        "action":    [],   # actuator command (T, 6)
        "vgrf_r":    [],
        "vgrf_l":    [],
        "foot_r_xz": [],   # root-relative (T, 2)
        "foot_l_xz": [],
        "r_pose":    [], "r_vel": [], "r_ee": [], "r_root": [],
        "ctrl_cost": [], "reward_total": [],
        "phase":     [],
    }
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)

        d = env.data
        root = d.body("torso").xpos
        ftr  = d.body("foot").xpos
        ftl  = d.body("foot_left").xpos

        out["qpos"].append(d.qpos[3:9].copy())
        out["qvel"].append(d.qvel[3:9].copy())
        out["action"].append(np.asarray(action, dtype=np.float32))
        out["vgrf_r"].append(abs(float(d.cfrc_ext[4, 5])))
        out["vgrf_l"].append(abs(float(d.cfrc_ext[7, 5])))
        out["foot_r_xz"].append((float(ftr[0] - root[0]), float(ftr[2] - root[2])))
        out["foot_l_xz"].append((float(ftl[0] - root[0]), float(ftl[2] - root[2])))
        out["r_pose"].append(float(info.get("r_pose", 0.0)))
        out["r_vel"].append(float(info.get("r_vel", 0.0)))
        out["r_ee"].append(float(info.get("r_ee", 0.0)))
        out["r_root"].append(float(info.get("r_root", 0.0)))
        out["ctrl_cost"].append(float(info.get("ctrl_cost", 0.0)))
        out["reward_total"].append(float(reward))
        out["phase"].append(int(info.get("phase", 0)))

        if term or trunc:
            break

    for k in out:
        out[k] = np.asarray(out[k])
    return out


def _pick_cycle_window(vgrf_r, body_weight_n, T):
    """Return (s, e) for first R-strike → R-strike window, or fallback."""
    contact_thresh = 0.05 * body_weight_n
    strikes = _rising_edges(vgrf_r, thresh=contact_thresh, min_gap=25)
    if len(strikes) >= 2:
        return int(strikes[0]), int(strikes[1]), True
    fallback_end = min(NOMINAL_CYCLE_FRAMES, T)
    return 0, fallback_end, False


def _ref_foot_traces(env, n_phase):
    """Pull (foot_r, foot_l) root-relative xz for one full ref cycle.

    `Walker2dPhaseAware._precompute_reference_kinematics` already cached
    the z components but not x — recompute both here so the ref/sim
    overlay uses one consistent FK.
    """
    ref = env._reference
    n = len(ref)
    ftr_xz = np.zeros((n, 2), dtype=np.float32)
    ftl_xz = np.zeros((n, 2), dtype=np.float32)

    qpos_save = env.data.qpos.copy()
    qvel_save = env.data.qvel.copy()
    import mujoco
    for t in range(n):
        env.data.qpos[:] = 0.0
        env.data.qpos[1] = 1.28
        env.data.qpos[3:9] = ref[t]
        env.data.qvel[:] = 0.0
        mujoco.mj_kinematics(env.model, env.data)
        root = env.data.body("torso").xpos
        ftr  = env.data.body("foot").xpos
        ftl  = env.data.body("foot_left").xpos
        ftr_xz[t] = (float(ftr[0] - root[0]), float(ftr[2] - root[2]))
        ftl_xz[t] = (float(ftl[0] - root[0]), float(ftl[2] - root[2]))
    env.data.qpos[:] = qpos_save
    env.data.qvel[:] = qvel_save
    mujoco.mj_kinematics(env.model, env.data)
    return ftr_xz, ftl_xz


def make_dashboard(traces, env, body_weight_n, title, out_path):
    qpos     = traces["qpos"]
    actions  = traces["action"]
    vgrf_r   = traces["vgrf_r"]
    foot_r_xz = traces["foot_r_xz"]
    foot_l_xz = traces["foot_l_xz"]
    T = len(qpos)
    if T == 0:
        raise RuntimeError("Empty rollout — env terminated immediately.")

    s, e, strike_based = _pick_cycle_window(vgrf_r, body_weight_n, T)
    n_cyc = e - s

    ref      = env._reference                # (N_ref, 6)
    n_ref    = len(ref)
    ref_x    = np.linspace(0, 1, n_ref)
    sim_cyc  = qpos[s:e]
    sim_x    = np.linspace(0, 1, n_cyc)

    ref_ftr_xz, ref_ftl_xz = _ref_foot_traces(env, n_ref)

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    gs  = fig.add_gridspec(3, 6)

    # Title bar.
    fig.suptitle(title, fontsize=12)

    # ── Panel 1: joint angles vs phase (6 subplots, top row) ──────────
    for i, name in enumerate(JOINT_NAMES):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(ref_x, np.rad2deg(ref[:, i]),    color="black",  lw=1.4, label="ref")
        ax.plot(sim_x, np.rad2deg(sim_cyc[:, i]), color="C0", lw=1.4, label="sim")
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
            ax.set_ylabel("angle (deg)")
        if i in (0, 3):
            pass
    cycle_lbl = ("R-strike→R-strike" if strike_based
                 else "fallback (no strikes detected)")
    fig.text(0.5, 0.665,
             f"Joint angles over one cycle  ({n_cyc} frames, {cycle_lbl})",
             ha="center", fontsize=10)

    # ── Panel 2: reward decomposition over the same cycle (mid-left) ──
    ax_r = fig.add_subplot(gs[1, 0:3])
    cyc_x = np.arange(n_cyc) / max(1, n_cyc - 1)
    for k, color in [("r_pose",  "C0"), ("r_vel",   "C1"),
                     ("r_ee",    "C2"), ("r_root",  "C3"),
                     ("reward_total", "k")]:
        ax_r.plot(cyc_x, traces[k][s:e], label=k, color=color,
                  lw=(1.6 if k == "reward_total" else 1.0))
    ax_r.set_title("Reward components over the cycle")
    ax_r.set_xlabel("phase (normalised)")
    ax_r.set_ylabel("reward")
    ax_r.legend(loc="best", fontsize=8, ncol=2)
    ax_r.grid(alpha=0.3)

    # ── Panel 3: action histograms (mid-right) ────────────────────────
    # Use the right half of the middle row; one small subplot per joint.
    sub_gs = gs[1, 3:6].subgridspec(2, 3)
    for i, name in enumerate(JOINT_NAMES):
        ax = fig.add_subplot(sub_gs[i // 3, i % 3])
        ax.hist(actions[:, i], bins=40, range=(-1.05, 1.05),
                color="C0", edgecolor="white", linewidth=0.3)
        sat_pct = 100.0 * float(np.mean(np.abs(actions[:, i]) > 0.99))
        ax.set_title(f"{name}  (sat ±1: {sat_pct:.0f}%)", fontsize=9)
        ax.set_xlim(-1.05, 1.05)
        ax.set_yticks([])
        ax.axvline(-1, color="gray", lw=0.5)
        ax.axvline( 1, color="gray", lw=0.5)
    fig.text(0.78, 0.665, f"Action histograms (full {T}-step rollout)",
             ha="center", fontsize=10)

    # ── Panel 4: foot xz, root-relative (bottom row) ──────────────────
    ax_fr = fig.add_subplot(gs[2, 0:3])
    ax_fr.plot(ref_ftr_xz[:, 0], ref_ftr_xz[:, 1], color="black", lw=1.4, label="ref")
    ax_fr.plot([p[0] for p in foot_r_xz[s:e]],
               [p[1] for p in foot_r_xz[s:e]],
               color="C0", lw=1.4, label="sim")
    ax_fr.set_title("Right foot xz (root-relative, one cycle)")
    ax_fr.set_xlabel("x rel torso (m)")
    ax_fr.set_ylabel("z rel torso (m)")
    ax_fr.legend(fontsize=8)
    ax_fr.grid(alpha=0.3)
    ax_fr.set_aspect("equal", adjustable="datalim")

    ax_fl = fig.add_subplot(gs[2, 3:6])
    ax_fl.plot(ref_ftl_xz[:, 0], ref_ftl_xz[:, 1], color="black", lw=1.4, label="ref")
    ax_fl.plot([p[0] for p in foot_l_xz[s:e]],
               [p[1] for p in foot_l_xz[s:e]],
               color="C0", lw=1.4, label="sim")
    ax_fl.set_title("Left foot xz (root-relative, one cycle)")
    ax_fl.set_xlabel("x rel torso (m)")
    ax_fl.set_ylabel("z rel torso (m)")
    ax_fl.legend(fontsize=8)
    ax_fl.grid(alpha=0.3)
    ax_fl.set_aspect("equal", adjustable="datalim")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _rom_summary_deg(qpos):
    return {n: float(np.rad2deg(qpos[:, i].max() - qpos[:, i].min()))
            for i, n in enumerate(JOINT_NAMES)}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("spec", help="run_dir:ckpt[:label]")
    p.add_argument("--xml",   default="walker2d.xml")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--seed",  type=int, default=0)
    p.add_argument("--out",   default=None,
                   help="Output PNG path (default: <run_dir>/dashboard.png)")
    args = p.parse_args()

    label, run_dir, model_path = parse_spec(args.spec)

    ref_path = Path(run_dir) / "reference.npy"
    if ref_path.exists():
        reference = np.load(ref_path).astype(np.float32)
    else:
        from ppo_walker2d_phase import load_ref_cycle
        reference = load_ref_cycle(PROJECT_ROOT / "assets" / "reference"
                                   / "gait_cycle_reference.npy")

    extras = _load_env_kwargs(run_dir)
    # Saved xml_file wins over CLI default; --xml on CLI is the override.
    xml_file = extras.pop("xml_file", args.xml)
    env = Walker2dPhaseAware(reference=reference, xml_file=xml_file, **extras)
    body_weight_n = float(np.sum(env.model.body_mass)) * abs(
        float(env.model.opt.gravity[2]))

    model = _load_policy(model_path)
    print(f"[{label}] model: {model_path}  ({type(model).__name__})")
    print(f"[{label}] body weight: {body_weight_n:.1f} N")

    traces = rollout_with_traces(env, model, args.steps, args.seed)
    T = len(traces["qpos"])
    print(f"[{label}] rolled out {T} steps "
          f"(termination at step {T} / {args.steps})")

    # Single-episode summary numbers for the title bar.
    ep_dict = {
        "qpos":   traces["qpos"],
        "vgrf_r": traces["vgrf_r"],
        "vgrf_l": traces["vgrf_l"],
    }
    metrics = episode_metrics(ep_dict, body_weight_n, reference)
    summary = aggregate([metrics])
    score = progress_score(summary, args.steps)

    # Full-rollout ROM is biased upward by sporadic kicks ("hip ROM 6.7 deg"
    # reading flexion that isn't there) — the explicit failure mode the
    # overnight sweep hit. We display per-cycle ROM in the title because that
    # is what the joint-angle panel actually shows.
    rom_full  = _rom_summary_deg(traces["qpos"])
    s_, e_, _ = _pick_cycle_window(traces["vgrf_r"], body_weight_n, T)
    rom_cycle = _rom_summary_deg(traces["qpos"][s_:e_])
    title = (
        f"{label}\n"
        f"steps={T}  strides={metrics.get('n_strides_detected', 0)}  "
        f"score={score:.2f}/4   "
        f"per-cycle ROM (deg): "
        f"hip_r={rom_cycle['hip_r']:.1f}  knee_r={rom_cycle['knee_r']:.1f}  "
        f"hip_l={rom_cycle['hip_l']:.1f}  knee_l={rom_cycle['knee_l']:.1f}"
    )

    out_path = Path(args.out) if args.out else Path(run_dir) / "dashboard.png"
    make_dashboard(traces, env, body_weight_n, title, out_path)
    env.close()

    # Print both ROM tables. Per-cycle is the honest one; full-rollout is the
    # number scalar metrics produce, kept here so the trap is visible.
    print(f"[{label}] joint ROM, per-cycle vs full-rollout (deg):")
    print(f"    {'joint':8s}  {'cycle':>7s}  {'full':>7s}")
    for n in JOINT_NAMES:
        print(f"    {n:8s}  {rom_cycle[n]:7.2f}  {rom_full[n]:7.2f}")


if __name__ == "__main__":
    main()
