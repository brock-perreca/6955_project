"""
Per-step deterministic-rollout hip ROM evaluation. Single source of the
metric we trust (hip ROM from a deterministic rollout) for batch 4 — the
metric traps documented in OVERNIGHT_SUMMARY.md (xvel survival, DTW, stride
detection) all misled the agent overnight; only hip ROM with a clean
rollout is reliable.

Usage:
  python scripts/eval_hip_rom.py <result_dir>[:<checkpoint>] [<dir>:<ckpt>] ...

Outputs per run:
  - hip_r/hip_l ROM in degrees (max - min over 2 full reference cycles)
  - per-step time at upper joint limit (if MJCF caps hip flexion at 0deg)
  - mean forward velocity
  - episode survival
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning,
                        module="stable_baselines3")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "walker2d"))

from ppo_walker2d_phase import Walker2dPhaseAware  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402


def load_run(spec: str) -> dict:
    parts = spec.split(":")
    result_dir = parts[0]
    ckpt = parts[1] if len(parts) > 1 else "final"
    p = Path(result_dir)
    if ckpt == "final":
        model_path = p / "model"
    else:
        model_path = p / "checkpoints" / f"model_{ckpt}_steps"
    ref = np.load(p / "reference.npy")
    meta_path = p / "env_kwargs.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return {
        "label": p.name,
        "ckpt": ckpt,
        "model_path": str(model_path),
        "reference": ref,
        "xml_file": meta.get("xml_file", "walker2d.xml"),
        "preview_k": int(meta.get("preview_k", 1)),
    }


def eval_one(run: dict, n_eps: int = 4, n_steps: int = 1000) -> None:
    print(f"\n=== {run['label']} (ckpt={run['ckpt']}, xml={run['xml_file']}) ===")

    env = Walker2dPhaseAware(
        reference=run["reference"],
        xml_file=run["xml_file"],
        preview_k=run["preview_k"],
    )
    model = PPO.load(run["model_path"], env=None, device="cpu")

    # Read joint limit for hip from the MJCF the env loaded.
    hip_hi = env.model.joint("thigh_joint").range[1]
    hip_lo = env.model.joint("thigh_joint").range[0]

    # Reference range as the upper bound for what the policy could possibly do.
    ref = run["reference"]
    ref_hip_r_range_deg = np.rad2deg(ref[:, 0].max() - ref[:, 0].min())
    ref_hip_l_range_deg = np.rad2deg(ref[:, 3].max() - ref[:, 3].min())

    all_hip_r = []
    all_hip_l = []
    all_xvel  = []
    survivals = []
    per_ep_hip_r_max = []
    per_ep_hip_r_min = []

    for ep in range(n_eps):
        obs, _ = env.reset(seed=42 + ep)
        # Run for n_steps from a fixed RSI seed; clip metrics to first 250 steps
        # if episode dies earlier.
        hip_r = []
        hip_l = []
        xvel  = []
        for _ in range(n_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            hip_r.append(float(env.data.qpos[3]))
            hip_l.append(float(env.data.qpos[6]))
            xvel.append(float(env.data.qvel[0]))
            if term or trunc:
                break
        survivals.append(len(hip_r))
        all_hip_r.extend(hip_r)
        all_hip_l.extend(hip_l)
        all_xvel.extend(xvel)
        per_ep_hip_r_max.append(np.rad2deg(max(hip_r)))
        per_ep_hip_r_min.append(np.rad2deg(min(hip_r)))

    env.close()

    hip_r_arr = np.rad2deg(np.array(all_hip_r))
    hip_l_arr = np.rad2deg(np.array(all_hip_l))
    xvel_arr = np.array(all_xvel)

    print(f"  Episodes survived: {survivals}  (target {n_steps})")
    print(f"  Mean fwd velocity: {xvel_arr.mean():+.3f} m/s   "
          f"(target 1.25)")
    print(f"  Reference hip_r ROM: {ref_hip_r_range_deg:.2f} deg  "
          f"(min={np.rad2deg(ref[:,0].min()):+.2f}, "
          f"max={np.rad2deg(ref[:,0].max()):+.2f})")
    print(f"  XML thigh_joint range: [{np.rad2deg(hip_lo):+.2f}, "
          f"{np.rad2deg(hip_hi):+.2f}] deg")
    print(f"  Policy hip_r: ROM={hip_r_arr.max() - hip_r_arr.min():.2f} deg  "
          f"min={hip_r_arr.min():+.2f}  max={hip_r_arr.max():+.2f}")
    print(f"  Policy hip_l: ROM={hip_l_arr.max() - hip_l_arr.min():.2f} deg  "
          f"min={hip_l_arr.min():+.2f}  max={hip_l_arr.max():+.2f}")
    print(f"  per-ep hip_r max: "
          f"{[round(v,2) for v in per_ep_hip_r_max]}")
    print(f"  per-ep hip_r min: "
          f"{[round(v,2) for v in per_ep_hip_r_min]}")

    # How often the policy is pinned within 1deg of upper limit
    at_lim_r = float((hip_r_arr >= np.rad2deg(hip_hi) - 1.0).mean()) * 100
    at_lim_l = float((hip_l_arr >= np.rad2deg(env.model.joint("thigh_left_joint").range[1]) - 1.0).mean()) * 100
    print(f"  hip_r within 1 deg of upper limit: {at_lim_r:.1f}%  "
          f"hip_l: {at_lim_l:.1f}%")

    # Verdict
    if hip_r_arr.max() - hip_r_arr.min() > 20.0 and hip_l_arr.max() - hip_l_arr.min() > 20.0:
        verdict = "POSITIVE: hip ROM > 20 deg on both sides — basin escaped"
    elif hip_r_arr.max() - hip_r_arr.min() > 10.0 or hip_l_arr.max() - hip_l_arr.min() > 10.0:
        verdict = "PARTIAL: 10-20 deg ROM on at least one side"
    else:
        verdict = "NEGATIVE: hip ROM < 10 deg — same stiff-hip basin"
    print(f"  Verdict: {verdict}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for spec in sys.argv[1:]:
        eval_one(load_run(spec))


if __name__ == "__main__":
    main()
