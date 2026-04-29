"""
eval_biomech.py
───────────────
Held-out biomechanical evaluation for a Walker2d phase-imitation policy.

Rolls out N deterministic episodes and computes metrics that capture
biomechanical realism, *not* tracking error. The policy was trained against
tracking error; using tracking error to grade reward variants would be
Goodhart's-Law squared. These metrics are:

    stride_period_s        — mean R-foot heel-strike interval (s)
    cadence_steps_per_min  — derived from stride period
    double_support_frac    — fraction of stance time with both feet loaded
    peak_vgrf_bw           — peak vertical GRF, body-weight-normalised
    swing_drag_frac        — fraction of swing-phase frames with foot force >5% BW
    lr_stride_asymmetry    — |R-L| / mean of mean step interval (R→L vs L→R)
    hip_knee_dtw           — DTW distance, sim (hip,knee)-cycle vs reference cycle
    n_strides_detected     — sanity check; gait must be detected at all

Usage
─────
    python src/diagnostics/eval_biomech.py <run_dir>:<ckpt>[:<label>] [<run_dir>:...] ...
        [--xml walker2d.xml | walker2d_subject1.xml]
        [--eps 5] [--steps 2000] [--out eval.json]

    <ckpt> is "final" or an integer step (matches render_phase.py spec syntax).

The script imports the active Walker2dPhaseAware env to ensure metrics are
computed against the exact reference resampling and FK precomputation that
the trained policy saw at training time.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "walker2d"))
from ppo_walker2d_phase import Walker2dPhaseAware, load_ref_cycle, CTRL_HZ  # noqa: E402


# ── helpers ────────────────────────────────────────────────────────────────────

def _rising_edges(x: np.ndarray, thresh: float, min_gap: int = 25) -> np.ndarray:
    """Indices where x crosses thresh from below, with a min-gap debounce."""
    above = x > thresh
    edges = np.where(above[1:] & ~above[:-1])[0] + 1
    if len(edges) == 0:
        return edges
    keep = [edges[0]]
    for e in edges[1:]:
        if e - keep[-1] >= min_gap:
            keep.append(e)
    return np.asarray(keep)


def _dtw(a: np.ndarray, b: np.ndarray) -> float:
    """Plain DTW with squared-Euclidean cost. a:(N,D), b:(M,D). O(N*M)."""
    N, M = len(a), len(b)
    cost = np.full((N + 1, M + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            d = float(np.sum((a[i - 1] - b[j - 1]) ** 2))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[N, M] / (N + M))  # length-normalised


# ── per-episode collection ─────────────────────────────────────────────────────

def rollout_episode(
    env: Walker2dPhaseAware,
    model: PPO,
    max_steps: int,
) -> dict:
    """Run one deterministic episode; return per-step kinematics + GRF traces."""
    obs, _ = env.reset()
    qpos_log, vgrf_r_log, vgrf_l_log = [], [], []
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        # qpos[3:9] = [hip_r, knee_r, ankle_r, hip_l, knee_l, ankle_l]
        qpos_log.append(env.data.qpos[3:9].copy())
        # cfrc_ext[i] is (torque3, force3); index 5 is z-force in body frame.
        # For the foot, body z is roughly aligned with world z (foot doesn't
        # pitch much during stance), so abs(cfrc_ext[i,5]) is a usable proxy.
        vgrf_r_log.append(abs(float(env.data.cfrc_ext[4, 5])))
        vgrf_l_log.append(abs(float(env.data.cfrc_ext[7, 5])))
        if term or trunc:
            break
    return {
        "qpos":   np.asarray(qpos_log),       # (T, 6)
        "vgrf_r": np.asarray(vgrf_r_log),     # (T,)
        "vgrf_l": np.asarray(vgrf_l_log),     # (T,)
    }


def episode_metrics(
    ep: dict,
    body_weight_n: float,
    reference: np.ndarray,
) -> dict:
    """Reduce a single-episode trace to a flat dict of biomech metrics."""
    qpos     = ep["qpos"]
    vgrf_r   = ep["vgrf_r"]
    vgrf_l   = ep["vgrf_l"]
    T        = len(qpos)
    if T < 50:
        return {"n_strides_detected": 0, "ep_len_steps": T}

    bw       = body_weight_n
    contact_thresh = 0.05 * bw  # >5% BW = "in contact"

    r_strikes = _rising_edges(vgrf_r, thresh=contact_thresh, min_gap=25)
    l_strikes = _rising_edges(vgrf_l, thresh=contact_thresh, min_gap=25)
    n_strides = max(0, len(r_strikes) - 1)

    out: dict = {"n_strides_detected": int(n_strides), "ep_len_steps": int(T)}

    if n_strides >= 1:
        stride_frames = float(np.mean(np.diff(r_strikes)))
        out["stride_period_s"]       = stride_frames / CTRL_HZ
        out["cadence_steps_per_min"] = 2.0 * 60.0 / out["stride_period_s"]  # 2 steps/stride

    # Double-support fraction (over frames where at least one foot is loaded).
    r_in = vgrf_r > contact_thresh
    l_in = vgrf_l > contact_thresh
    any_in = r_in | l_in
    if any_in.sum() > 0:
        out["double_support_frac"] = float((r_in & l_in).sum() / any_in.sum())

    # Peak vGRF / BW, averaged across stance bouts (max within each contact run).
    def _peak_per_bout(in_contact: np.ndarray, force: np.ndarray) -> float:
        # Identify contiguous in-contact runs and take max in each.
        idx = np.where(in_contact)[0]
        if len(idx) == 0:
            return 0.0
        breaks = np.where(np.diff(idx) > 1)[0]
        starts = np.r_[idx[0], idx[breaks + 1]] if len(breaks) else np.array([idx[0]])
        ends   = np.r_[idx[breaks], idx[-1]] if len(breaks) else np.array([idx[-1]])
        peaks = [force[s:e + 1].max() for s, e in zip(starts, ends) if e > s]
        return float(np.mean(peaks)) if peaks else 0.0

    peak_r = _peak_per_bout(r_in, vgrf_r) / bw
    peak_l = _peak_per_bout(l_in, vgrf_l) / bw
    out["peak_vgrf_bw_r"] = peak_r
    out["peak_vgrf_bw_l"] = peak_l
    out["peak_vgrf_bw"]   = 0.5 * (peak_r + peak_l)

    # Swing-drag fraction: fraction of swing-side frames with detectable force.
    # "Swing side" = the side that is *not* currently in contact while the other is.
    r_only_swing = (~r_in) & l_in
    l_only_swing = r_in & (~l_in)
    drag_r = float((vgrf_r[r_only_swing] > 0.05 * bw).mean()) if r_only_swing.any() else 0.0
    drag_l = float((vgrf_l[l_only_swing] > 0.05 * bw).mean()) if l_only_swing.any() else 0.0
    out["swing_drag_frac"] = 0.5 * (drag_r + drag_l)

    # L-R asymmetry on step intervals (R→L vs L→R).
    if len(r_strikes) >= 1 and len(l_strikes) >= 1:
        # Pair each strike with the closest opposite-foot strike that follows.
        rl_intervals = []
        lr_intervals = []
        for r_idx in r_strikes:
            after = l_strikes[l_strikes > r_idx]
            if len(after):
                rl_intervals.append(after[0] - r_idx)
        for l_idx in l_strikes:
            after = r_strikes[r_strikes > l_idx]
            if len(after):
                lr_intervals.append(after[0] - l_idx)
        if rl_intervals and lr_intervals:
            mr = float(np.mean(rl_intervals))
            ml = float(np.mean(lr_intervals))
            out["lr_stride_asymmetry"] = abs(mr - ml) / (0.5 * (mr + ml))

    # Hip-knee phase-plane DTW vs reference cycle (right leg).
    if n_strides >= 1:
        s, e = int(r_strikes[0]), int(r_strikes[1])
        sim_cycle = qpos[s:e, [0, 1]]   # hip_r, knee_r
        ref_cycle = reference[:, [0, 1]]
        # subsample if very long to keep DTW O(N*M) tractable
        if len(sim_cycle) > 200:
            sim_cycle = sim_cycle[:: max(1, len(sim_cycle) // 200)]
        if len(ref_cycle) > 200:
            ref_cycle = ref_cycle[:: max(1, len(ref_cycle) // 200)]
        out["hip_knee_dtw"] = _dtw(sim_cycle.astype(np.float64),
                                   ref_cycle.astype(np.float64))

    return out


# ── per-run aggregation ────────────────────────────────────────────────────────

def aggregate(per_ep: list[dict]) -> dict:
    """Median + IQR across episodes. Skip episodes with no detected strides."""
    keys = set().union(*per_ep)
    out: dict = {"n_episodes": len(per_ep)}
    for k in sorted(keys):
        vs = [ep[k] for ep in per_ep if k in ep and ep[k] is not None]
        vs = [v for v in vs if isinstance(v, (int, float)) and np.isfinite(v)]
        if not vs:
            continue
        arr = np.asarray(vs, dtype=np.float64)
        out[f"{k}__median"] = float(np.median(arr))
        out[f"{k}__iqr"]    = float(np.percentile(arr, 75) - np.percentile(arr, 25))
        out[f"{k}__n"]      = int(len(arr))
    return out


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_spec(spec: str) -> tuple[str, str, str]:
    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError(f"Expected run_dir:checkpoint[:label], got {spec!r}")
    run_dir, ckpt = parts[0], parts[1]
    label = parts[2] if len(parts) > 2 else f"{Path(run_dir).name}:{ckpt}"
    if ckpt.lower() == "final":
        model_path = str(Path(run_dir) / "model")
    else:
        model_path = str(Path(run_dir) / "checkpoints" / f"model_{ckpt}_steps")
    return label, run_dir, model_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("specs", nargs="+", help="run_dir:ckpt[:label] (one or more)")
    p.add_argument("--xml", default="walker2d.xml",
                   help="MuJoCo model file (walker2d.xml or walker2d_subject1.xml)")
    p.add_argument("--eps", type=int, default=5, help="episodes per run")
    p.add_argument("--steps", type=int, default=2000, help="max steps per episode")
    p.add_argument("--out", default=None,
                   help="JSON output path (default: prints to stdout only)")
    args = p.parse_args()

    runs = [parse_spec(s) for s in args.specs]

    all_results: list[dict] = []
    for label, run_dir, model_path in runs:
        ref_path = Path(run_dir) / "reference.npy"
        if ref_path.exists():
            reference = np.load(ref_path).astype(np.float32)
            print(f"[{label}] reference: {ref_path} shape={reference.shape}")
        else:
            reference = load_ref_cycle(PROJECT_ROOT / "assets" / "reference"
                                       / "gait_cycle_reference.npy")
            print(f"[{label}] reference: fallback gait_cycle_reference.npy "
                  f"shape={reference.shape}")

        env = Walker2dPhaseAware(reference=reference, xml_file=args.xml)
        body_weight_n = float(np.sum(env.model.body_mass)) * abs(
            float(env.model.opt.gravity[2]))
        print(f"[{label}] body weight: {body_weight_n:.1f} N "
              f"({body_weight_n / 9.81:.2f} kg)")

        model = PPO.load(model_path, device="cpu")
        print(f"[{label}] model: {model_path}")

        per_ep: list[dict] = []
        for i in range(args.eps):
            ep   = rollout_episode(env, model, max_steps=args.steps)
            mets = episode_metrics(ep, body_weight_n, reference)
            per_ep.append(mets)
            print(f"  ep {i+1}/{args.eps}: "
                  f"len={mets.get('ep_len_steps', 0):4d}  "
                  f"strides={mets.get('n_strides_detected', 0):2d}  "
                  f"stride_s={mets.get('stride_period_s', float('nan')):.3f}  "
                  f"DS%={100*mets.get('double_support_frac', float('nan')):4.1f}  "
                  f"vGRF/BW={mets.get('peak_vgrf_bw', float('nan')):.2f}")

        env.close()
        agg = aggregate(per_ep)
        result = {"label": label, "run_dir": run_dir, "model_path": model_path,
                  "xml": args.xml, "n_eps": args.eps, "max_steps": args.steps,
                  "per_episode": per_ep, "summary": agg}
        all_results.append(result)
        print(f"[{label}] summary:")
        for k, v in agg.items():
            print(f"    {k}: {v}")
        print()

    if args.out:
        Path(args.out).write_text(json.dumps(all_results, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
