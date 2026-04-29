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
    {hip,knee,ankle}_{r,l}_rom_deg — per-joint range of motion within a stride
    vgrf_shape_dtw         — DTW distance, sim stance-phase vGRF curve vs reference
    n_strides_detected     — sanity check; gait must be detected at all

If `assets/reference/biomech_targets.json` exists (from
`extract_reference_biomech.py`), the per-run summary also gets a
`vs_reference` block with `delta` and `pct_err` for every metric that has a
measured Ulrich target. That block is what an agent should read to decide
whether progress was made: numbers, not eyeballs.

Usage
─────
    python src/diagnostics/eval_biomech.py <run_dir>:<ckpt>[:<label>] [<run_dir>:...] ...
        [--xml walker2d.xml | walker2d_subject1.xml]
        [--eps 5] [--steps 2000] [--out eval.json]
        [--targets assets/reference/biomech_targets.json]
        [--csv results/biomech_history.csv]

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


def _load_env_kwargs(run_dir: str) -> dict:
    """Read env_kwargs.json saved at training time, if any.

    preview_k changes obs_space; without these extras, PPO.load fails with
    a shape mismatch on multi-step-preview runs.
    """
    p = Path(run_dir) / "env_kwargs.json"
    if not p.exists():
        return {}
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] Failed to read {p}: {e}; using defaults")
        return {}
    out = {}
    if "preview_k" in meta:
        out["preview_k"] = int(meta["preview_k"])
    if "pose_joint_weights" in meta:
        out["pose_joint_weights"] = tuple(meta["pose_joint_weights"])
    if "product_reward" in meta:
        out["product_reward"] = bool(meta["product_reward"])
    if "min_joint_pose" in meta:
        out["min_joint_pose"] = bool(meta["min_joint_pose"])
    if "v_target" in meta:
        out["v_target"] = float(meta["v_target"])
    return out


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

STANCE_RESAMPLE_N = 100  # match extract_reference_biomech.py


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


def _stance_curve(force: np.ndarray, in_contact: np.ndarray,
                  min_len: int = 5) -> np.ndarray:
    """Mean stance-phase force curve, time-normalised to STANCE_RESAMPLE_N."""
    idx = np.where(in_contact)[0]
    if len(idx) == 0:
        return np.array([])
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]] if len(breaks) else np.array([idx[0]])
    ends   = np.r_[idx[breaks], idx[-1]] if len(breaks) else np.array([idx[-1]])
    curves = []
    for s, e in zip(starts, ends):
        if e - s + 1 < min_len:
            continue
        bout = force[s:e + 1]
        new_x = np.linspace(0, len(bout) - 1, STANCE_RESAMPLE_N)
        curves.append(np.interp(new_x, np.arange(len(bout)), bout))
    if not curves:
        return np.array([])
    return np.mean(np.stack(curves, axis=0), axis=0)


def _per_stride_rom_deg(angle_rad: np.ndarray,
                        strikes: np.ndarray) -> float | None:
    """Median per-stride ROM (degrees) across detected strides on this leg."""
    if len(strikes) < 2:
        return None
    ranges = []
    for i in range(len(strikes) - 1):
        s, e = int(strikes[i]), int(strikes[i + 1])
        if e - s < 5:
            continue
        ranges.append(float(angle_rad[s:e].max() - angle_rad[s:e].min()))
    if not ranges:
        return None
    return float(np.rad2deg(np.median(ranges)))


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

        # All-six-joint DTW (overnight 2026-04-29). hip_knee_dtw was hiding
        # stiff-hip exploits because the right hip channel could be pinned at
        # zero while right knee tracked correctly. Adding ankles + the LEFT
        # leg widens the surface that has to match. Same gait-cycle slice.
        sim_all = qpos[s:e, :6]
        ref_all = reference[:, :6]
        if len(sim_all) > 200:
            sim_all = sim_all[:: max(1, len(sim_all) // 200)]
        if len(ref_all) > 200:
            ref_all = ref_all[:: max(1, len(ref_all) // 200)]
        out["all_joints_dtw"] = _dtw(sim_all.astype(np.float64),
                                     ref_all.astype(np.float64))

    # Per-joint ROM (median per-stride range, in degrees) for both legs.
    # qpos cols: 0=hip_r, 1=knee_r, 2=ankle_r, 3=hip_l, 4=knee_l, 5=ankle_l.
    joint_names = ["hip_r", "knee_r", "ankle_r", "hip_l", "knee_l", "ankle_l"]
    leg_strikes = {"r": r_strikes, "l": l_strikes}
    for ji, jname in enumerate(joint_names):
        leg = jname.split("_")[1]
        rom = _per_stride_rom_deg(qpos[:, ji], leg_strikes[leg])
        if rom is not None:
            out[f"{jname}_rom_deg"] = rom

    # Stance-phase vGRF shape, time-normalised to STANCE_RESAMPLE_N samples,
    # BW-normalised. Stored on the episode dict (not aggregated by aggregate(),
    # which only handles scalars) so the per-run summary can compare it to
    # the reference curve via DTW.
    sim_curve_r = _stance_curve(vgrf_r, r_in) / bw if r_in.any() else np.array([])
    sim_curve_l = _stance_curve(vgrf_l, l_in) / bw if l_in.any() else np.array([])
    if sim_curve_r.size:
        out["_vgrf_curve_r"] = sim_curve_r.tolist()
    if sim_curve_l.size:
        out["_vgrf_curve_l"] = sim_curve_l.tolist()

    return out


# ── per-run aggregation ────────────────────────────────────────────────────────

def aggregate(per_ep: list[dict]) -> dict:
    """Median + IQR across episodes. Skip episodes with no detected strides.

    Keys starting with `_` are private (e.g., per-episode vGRF curves) and
    handled separately by `aggregate_curves`.
    """
    keys = set().union(*per_ep)
    out: dict = {"n_episodes": len(per_ep)}
    for k in sorted(keys):
        if k.startswith("_"):
            continue
        vs = [ep[k] for ep in per_ep if k in ep and ep[k] is not None]
        vs = [v for v in vs if isinstance(v, (int, float)) and np.isfinite(v)]
        if not vs:
            continue
        arr = np.asarray(vs, dtype=np.float64)
        out[f"{k}__median"] = float(np.median(arr))
        out[f"{k}__iqr"]    = float(np.percentile(arr, 75) - np.percentile(arr, 25))
        out[f"{k}__n"]      = int(len(arr))
    return out


def aggregate_curves(per_ep: list[dict]) -> dict:
    """Mean of per-episode stance-vGRF curves (one mean curve per leg)."""
    out: dict = {}
    for leg, key in [("r", "_vgrf_curve_r"), ("l", "_vgrf_curve_l")]:
        curves = [np.asarray(ep[key]) for ep in per_ep
                  if key in ep and len(ep[key]) == STANCE_RESAMPLE_N]
        if curves:
            out[f"vgrf_curve_{leg}"] = np.mean(np.stack(curves), axis=0).tolist()
    return out


# ── reference comparison ───────────────────────────────────────────────────────

# Maps eval_biomech metric -> path inside biomech_targets.json. Per-joint ROM
# is special-cased via `range_deg` extraction.
_TARGET_MAP = {
    "stride_period_s":       ("spatiotemporal", "stride_period_s"),
    "cadence_steps_per_min": ("spatiotemporal", "cadence_steps_per_min"),
    "double_support_frac":   ("spatiotemporal", "double_support_frac"),
    "peak_vgrf_bw":          ("vgrf",           "peak_bw"),
    "peak_vgrf_bw_r":        ("vgrf",           "peak_bw_fp1"),
    "peak_vgrf_bw_l":        ("vgrf",           "peak_bw_fp2"),
}


def _ref_target_value(targets: dict, metric: str) -> float | None:
    """Return the reference target for a sim metric, or None if not mapped."""
    if metric in _TARGET_MAP:
        a, b = _TARGET_MAP[metric]
        return targets.get(a, {}).get(b)
    if metric.endswith("_rom_deg"):
        joint = metric[:-len("_rom_deg")]  # "hip_r", "knee_r", ...
        rom = targets.get("kinematics_rom", {}).get(joint, {})
        return rom.get("full_trial_range_deg")
    return None


def vs_reference(summary: dict, curves: dict, targets: dict | None) -> dict:
    """Compute delta + pct_err for every summary metric that has a target.

    Adds `vgrf_shape_dtw_{r,l}` from the per-run mean stance curves vs the
    reference stance curves saved in `<targets>.vgrf_curves.npz`.
    """
    if not targets:
        return {}
    out: dict = {}
    for k_med, v in summary.items():
        if not k_med.endswith("__median"):
            continue
        metric = k_med[:-len("__median")]
        ref = _ref_target_value(targets, metric)
        if ref is None or not np.isfinite(v):
            continue
        delta = v - ref
        pct_err = 100.0 * delta / ref if ref else float("nan")
        out[metric] = {"sim": v, "ref": float(ref),
                       "delta": float(delta), "pct_err": float(pct_err)}

    # vGRF shape DTW
    curve_path = targets.get("vgrf", {}).get("stance_curve_path")
    if curve_path:
        full = PROJECT_ROOT / curve_path
        if full.exists():
            ref_curves = np.load(full)
            for leg, ref_key in [("r", "fp1"), ("l", "fp2")]:
                sim_curve = curves.get(f"vgrf_curve_{leg}")
                if sim_curve and ref_key in ref_curves.files:
                    a = np.asarray(sim_curve).reshape(-1, 1)
                    b = ref_curves[ref_key].reshape(-1, 1)
                    out[f"vgrf_shape_dtw_{leg}"] = {
                        "sim_dtw_vs_ref": _dtw(a.astype(np.float64),
                                               b.astype(np.float64)),
                    }
    return out


# ── progress score (mirrors scripts/overnight/rank_runs.py) ────────────────────

def progress_score(summary: dict, max_steps: int) -> float:
    """One-number "is this run good?" score in [0, 4]. Higher is better.

    Components (all clipped to [0, 1]):
      - survival             = ep_len / max_steps
      - cadence match        = 1 - |stride_period - 1.12| / 1.12
      - shape fidelity       = 1 - hip_knee_dtw / 0.30
      - symmetry             = 1 - lr_stride_asymmetry / 0.30

    The 1.12 / 0.30 reference values match scripts/overnight/rank_runs.py
    so external batch-ranking and the eval JSON agree.
    """
    def clip01(x):
        return float(max(0.0, min(1.0, x)))

    ep_len   = summary.get("ep_len_steps__median", 0.0)
    stride_s = summary.get("stride_period_s__median", float("nan"))
    dtw      = summary.get("hip_knee_dtw__median", float("nan"))
    asym     = summary.get("lr_stride_asymmetry__median", float("nan"))

    s_survive = clip01(ep_len / max_steps) if max_steps else 0.0
    s_cadence = clip01(1.0 - abs(stride_s - 1.12) / 1.12) if np.isfinite(stride_s) else 0.0
    s_shape   = clip01(1.0 - dtw / 0.30) if np.isfinite(dtw) else 0.0
    s_sym     = clip01(1.0 - asym / 0.30) if np.isfinite(asym) else 0.0
    return s_survive + s_cadence + s_shape + s_sym


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


def _load_targets(path: Path) -> dict | None:
    if not path.exists():
        print(f"[targets] {path} not found; skipping vs_reference block. "
              "Run src/diagnostics/extract_reference_biomech.py to create it.")
        return None
    print(f"[targets] {path.relative_to(PROJECT_ROOT)}")
    return json.loads(path.read_text())


def _append_csv(csv_path: Path, label: str, summary: dict, vs_ref: dict,
                score: float) -> None:
    """Append one row per run. Header is rewritten if columns change."""
    cols = ["timestamp", "label", "score"]
    row: dict = {"label": label, "score": f"{score:.3f}"}
    from datetime import datetime
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")

    for k in ["ep_len_steps", "n_strides_detected", "stride_period_s",
              "cadence_steps_per_min", "double_support_frac",
              "peak_vgrf_bw", "lr_stride_asymmetry", "hip_knee_dtw",
              "swing_drag_frac",
              "hip_r_rom_deg", "knee_r_rom_deg", "ankle_r_rom_deg",
              "hip_l_rom_deg", "knee_l_rom_deg", "ankle_l_rom_deg"]:
        cols.append(k)
        row[k] = summary.get(f"{k}__median", "")
    for k, blk in vs_ref.items():
        if "pct_err" in blk:
            cn = f"{k}__pct_err"
            cols.append(cn)
            row[cn] = f"{blk['pct_err']:.1f}"

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=cols)
        if write_header:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("specs", nargs="+", help="run_dir:ckpt[:label] (one or more)")
    p.add_argument("--xml", default="walker2d.xml",
                   help="MuJoCo model file (walker2d.xml or walker2d_subject1.xml)")
    p.add_argument("--eps", type=int, default=5, help="episodes per run")
    p.add_argument("--steps", type=int, default=2000, help="max steps per episode")
    p.add_argument("--out", default=None,
                   help="JSON output path (default: prints to stdout only)")
    p.add_argument("--targets",
                   default=str(PROJECT_ROOT / "assets" / "reference"
                               / "biomech_targets.json"),
                   help="Reference biomech targets JSON. Skipped if missing.")
    p.add_argument("--csv", default=None,
                   help="Append a one-row summary to this CSV (default: no CSV).")
    args = p.parse_args()

    runs = [parse_spec(s) for s in args.specs]
    targets = _load_targets(Path(args.targets))

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

        extras = _load_env_kwargs(run_dir)
        env = Walker2dPhaseAware(reference=reference, xml_file=args.xml, **extras)
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
        agg     = aggregate(per_ep)
        curves  = aggregate_curves(per_ep)
        vs_ref  = vs_reference(agg, curves, targets)
        score   = progress_score(agg, args.steps)
        result = {"label": label, "run_dir": run_dir, "model_path": model_path,
                  "xml": args.xml, "n_eps": args.eps, "max_steps": args.steps,
                  "per_episode": per_ep, "summary": agg,
                  "vgrf_curves": curves,
                  "vs_reference": vs_ref,
                  "progress_score": score}
        all_results.append(result)
        print(f"[{label}] summary:")
        for k, v in agg.items():
            print(f"    {k}: {v}")
        if vs_ref:
            print(f"[{label}] vs_reference (sim - ref, % err):")
            for k, blk in vs_ref.items():
                if "pct_err" in blk:
                    print(f"    {k}: sim={blk['sim']:.3f} "
                          f"ref={blk['ref']:.3f} delta={blk['delta']:+.3f} "
                          f"pct_err={blk['pct_err']:+.1f}%")
                elif "sim_dtw_vs_ref" in blk:
                    print(f"    {k}: dtw_vs_ref={blk['sim_dtw_vs_ref']:.4f}")
        print(f"[{label}] progress_score: {score:.3f} / 4.000")
        if args.csv:
            _append_csv(Path(args.csv), label, agg, vs_ref, score)
            print(f"[{label}] appended row to {args.csv}")
        print()

    if args.out:
        Path(args.out).write_text(json.dumps(all_results, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
