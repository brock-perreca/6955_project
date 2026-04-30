"""Diagnostic: do strikes the eval detects line up with what a human eye would call a stride?

Rolls out one deterministic episode of a candidate, dumps the right- and left-foot
vertical contact forces, runs the same `_rising_edges` detector eval_biomech.py
uses, and prints (a) the raw contact intervals, (b) the detected "strikes",
(c) what the COM forward velocity actually is, (d) what stride-period the eval
would report. Also computes a length-based stride period from forward distance
between same-foot strikes — a sanity check on the time-based one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "walker2d"))

from stable_baselines3 import PPO  # noqa: E402

from ppo_walker2d_phase import Walker2dPhaseAware, load_ref_cycle, CTRL_HZ  # noqa: E402


def rising_edges(x: np.ndarray, thresh: float, min_gap: int = 25) -> np.ndarray:
    above = x > thresh
    edges = np.where(above[1:] & ~above[:-1])[0] + 1
    if len(edges) == 0:
        return edges
    keep = [edges[0]]
    for e in edges[1:]:
        if e - keep[-1] >= min_gap:
            keep.append(e)
    return np.asarray(keep)


def contact_runs(in_contact: np.ndarray) -> list[tuple[int, int]]:
    if not in_contact.any():
        return []
    idx = np.where(in_contact)[0]
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]] if len(breaks) else np.array([idx[0]])
    ends   = np.r_[idx[breaks], idx[-1]] if len(breaks) else np.array([idx[-1]])
    return list(zip(starts.tolist(), ends.tolist()))


def main(run_dir: str, max_steps: int = 1000) -> None:
    rd = Path(run_dir)
    print(f"[run] {rd}")

    import json
    meta_path = rd / "env_kwargs.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    xml = meta.get("xml_file", "walker2d.xml")
    extras = {k: meta[k] for k in
              ("preview_k", "pose_joint_weights", "product_reward",
               "min_joint_pose", "v_target", "ref_root_drop") if k in meta}
    if "pose_joint_weights" in extras:
        extras["pose_joint_weights"] = tuple(extras["pose_joint_weights"])

    ref_path = rd / "reference.npy"
    reference = (np.load(ref_path).astype(np.float32) if ref_path.exists()
                 else load_ref_cycle(PROJECT_ROOT / "assets" / "reference"
                                     / "gait_cycle_reference.npy"))

    env = Walker2dPhaseAware(reference=reference, xml_file=xml, **extras)
    bw = float(np.sum(env.model.body_mass)) * abs(float(env.model.opt.gravity[2]))
    model = PPO.load(str(rd / "model.zip"), device="cpu")

    obs, _ = env.reset()
    vgrf_r, vgrf_l, x_pos, x_vel = [], [], [], []
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        vgrf_r.append(abs(float(env.data.cfrc_ext[4, 5])))
        vgrf_l.append(abs(float(env.data.cfrc_ext[7, 5])))
        x_pos.append(float(env.data.qpos[0]))
        x_vel.append(float(env.data.qvel[0]))
        if term or trunc:
            break
    env.close()

    vgrf_r = np.asarray(vgrf_r); vgrf_l = np.asarray(vgrf_l)
    x_pos = np.asarray(x_pos);   x_vel = np.asarray(x_vel)
    T = len(vgrf_r)
    thresh = 0.05 * bw
    print(f"[ep ] T={T} steps ({T/CTRL_HZ:.2f} s)  BW={bw:.1f} N  thresh={thresh:.1f} N")
    print(f"[vel] mean fwd vel = {x_vel.mean():.3f} m/s   "
          f"total fwd distance = {x_pos[-1] - x_pos[0]:.2f} m")

    r_in = vgrf_r > thresh
    l_in = vgrf_l > thresh
    r_runs = contact_runs(r_in)
    l_runs = contact_runs(l_in)
    print(f"\n[contact] R contact bouts: {len(r_runs)}   L contact bouts: {len(l_runs)}")
    if r_runs:
        durs = np.array([(e - s + 1) / CTRL_HZ for s, e in r_runs])
        gaps = np.diff([s for s, _ in r_runs]) / CTRL_HZ if len(r_runs) > 1 else np.array([])
        print(f"[contact] R bout duration  mean={durs.mean()*1000:.0f} ms  "
              f"min={durs.min()*1000:.0f}  max={durs.max()*1000:.0f}")
        if len(gaps):
            print(f"[contact] R bout-start interval  mean={gaps.mean():.3f} s  "
                  f"min={gaps.min():.3f}  max={gaps.max():.3f}")

    r_strikes = rising_edges(vgrf_r, thresh, min_gap=25)
    l_strikes = rising_edges(vgrf_l, thresh, min_gap=25)
    print(f"\n[strikes 25-frame min_gap (eval default)]  R={len(r_strikes)}   L={len(l_strikes)}")
    if len(r_strikes) >= 2:
        d_frames = np.diff(r_strikes)
        d_secs = d_frames / CTRL_HZ
        print(f"[strikes]  R-R intervals  mean={d_secs.mean():.3f} s  "
              f"median={np.median(d_secs):.3f} s  min={d_secs.min():.3f}  max={d_secs.max():.3f}")
        print(f"[strikes]  --> stride_period_s = {np.mean(d_frames)/CTRL_HZ:.3f}, "
              f"cadence = {2*60.0/(np.mean(d_frames)/CTRL_HZ):.1f} spm")

    for mg in (50, 75, 100):
        rs = rising_edges(vgrf_r, thresh, min_gap=mg)
        if len(rs) >= 2:
            sp = np.mean(np.diff(rs)) / CTRL_HZ
            print(f"[strikes {mg}-frame min_gap]  R={len(rs)}  stride={sp:.3f} s  "
                  f"cadence={2*60.0/sp:.1f} spm")

    if len(r_strikes) >= 2:
        x_at_strike = x_pos[r_strikes]
        stride_lengths = np.diff(x_at_strike)
        print(f"\n[geom]  per-stride forward distance "
              f"mean={stride_lengths.mean():.3f} m  "
              f"min={stride_lengths.min():.3f}  max={stride_lengths.max():.3f}")
        if len(r_strikes) >= 2:
            print(f"[geom]  fwd-distance-based stride: "
                  f"if stride={stride_lengths.mean():.3f} m at "
                  f"{x_vel.mean():.3f} m/s --> period={stride_lengths.mean()/x_vel.mean():.3f} s")

    # First 20 R-rising-edges raw indices and first contact-bout starts side-by-side
    print(f"\n[first 15 R rising edges (frame, t_s)]")
    for e in r_strikes[:15]:
        print(f"  frame={int(e):4d}  t={int(e)/CTRL_HZ:.3f}s  vgrf={vgrf_r[e]:.1f} N "
              f"({vgrf_r[e]/bw:.2f} BW)")


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else \
        str(PROJECT_ROOT / "results" / "restart_b4_hiprelax_s11")
    main(run)
