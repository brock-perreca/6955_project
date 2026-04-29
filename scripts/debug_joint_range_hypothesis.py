"""
Verify Brock's joint-range hypothesis end-to-end.

1. Inspect the MJCF that the env actually loads.
2. Inspect the reference array per-joint ranges.
3. Run dynamics-respecting FK at the reference's peak-flexion phase: set qpos
   from reference, then step with action=0; measure how far the constraint
   solver pulls the joint back from its commanded value.
4. Compare against the trained xvel-5M policy's per-step hip trajectory at the
   same phase to confirm the policy is hard against the joint limit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mujoco

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "walker2d"))

from gymnasium.envs.mujoco.walker2d_v4 import Walker2dEnv  # noqa: E402

# ── 1. MJCF the env loads ────────────────────────────────────────────────────
print("=" * 72)
print("1. MJCF inspection")
print("=" * 72)

env = Walker2dEnv()
model = env.model

joint_names = [
    "thigh_joint", "leg_joint", "foot_joint",
    "thigh_left_joint", "leg_left_joint", "foot_left_joint",
]
print(f"  XML loaded: {model.opt.timestep=} (verifying it's a Walker2d model)")
print(f"  nq={model.nq}  nv={model.nv}")
print()
print("  Joint limits (from compiled MJCF, in radians AND degrees):")
print(f"  {'name':22s} {'lo (rad)':>10s} {'hi (rad)':>10s}   "
      f"{'lo (deg)':>10s} {'hi (deg)':>10s}")
for jn in joint_names:
    j = model.joint(jn)
    lo, hi = j.range
    print(f"  {jn:22s} {lo:10.4f} {hi:10.4f}   "
          f"{np.rad2deg(lo):10.2f} {np.rad2deg(hi):10.2f}")
print()

# ── 2. Reference per-joint ranges ────────────────────────────────────────────
print("=" * 72)
print("2. Reference array per-joint ranges")
print("=" * 72)

ref_path = PROJECT_ROOT / "assets" / "reference" / "gait_cycle_reference.npy"
ref = np.load(ref_path)
print(f"  Reference: {ref.shape}  {ref.dtype}  ({ref_path.name})")
print()
print("  Per-joint ranges across the 56-frame cycle "
      "(ordered: hip_r, knee_r, ankle_r, hip_l, knee_l, ankle_l):")
labels = ["hip_r", "knee_r", "ankle_r", "hip_l", "knee_l", "ankle_l"]
xml_lo_rad = np.array([model.joint(n).range[0] for n in joint_names])
xml_hi_rad = np.array([model.joint(n).range[1] for n in joint_names])
print(f"  {'joint':10s} {'min (rad)':>10s} {'max (rad)':>10s}   "
      f"{'min (deg)':>10s} {'max (deg)':>10s}   "
      f"{'XML lo':>8s} {'XML hi':>8s}   {'fits?':>6s}")
for j, lab in enumerate(labels):
    lo, hi = float(ref[:, j].min()), float(ref[:, j].max())
    fits = "yes" if (lo >= xml_lo_rad[j] - 1e-3 and hi <= xml_hi_rad[j] + 1e-3) else "NO"
    print(f"  {lab:10s} {lo:10.4f} {hi:10.4f}   "
          f"{np.rad2deg(lo):10.2f} {np.rad2deg(hi):10.2f}   "
          f"{np.rad2deg(xml_lo_rad[j]):8.2f} {np.rad2deg(xml_hi_rad[j]):8.2f}   "
          f"{fits:>6s}")
print()
peak_hip_r_phase = int(np.argmax(ref[:, 0]))
peak_hip_l_phase = int(np.argmax(ref[:, 3]))
print(f"  hip_r peak-flexion phase: {peak_hip_r_phase}/{len(ref)}  "
      f"value = {np.rad2deg(ref[peak_hip_r_phase, 0]):.2f} deg")
print(f"  hip_l peak-flexion phase: {peak_hip_l_phase}/{len(ref)}  "
      f"value = {np.rad2deg(ref[peak_hip_l_phase, 3]):.2f} deg")
print()

# ── 3. Dynamics-respecting FK probe ──────────────────────────────────────────
print("=" * 72)
print("3. Dynamics-respecting probe at peak-flexion phase")
print("=" * 72)
print()
print("  Procedure: set qpos[3:9] = ref[peak_hip_r_phase], qvel = 0, ctrl = 0,")
print("  then advance one Walker2d-v4 step (4 substeps × dt=0.002s = 8 ms).")
print("  If the joint limit is enforced, qpos[3] (thigh = hip_r) will return")
print("  toward 0 instead of staying at the +30° we commanded.")
print()

import gymnasium as gym  # noqa: E402

def probe_one_phase(phase_idx: int, label: str) -> dict:
    env_p = Walker2dEnv()
    env_p.reset()
    qpos = env_p.data.qpos.copy()
    qvel = env_p.data.qvel.copy()
    qpos[1] = 1.28
    qpos[2] = 0.0
    qpos[3:9] = ref[phase_idx]
    qvel[:] = 0.0
    env_p.set_state(qpos, qvel)
    mujoco.mj_forward(env_p.model, env_p.data)
    q_set = env_p.data.qpos[3:9].copy()
    # Action = 0: zero motor torques. Joint limits + gravity will dominate.
    obs, rew, term, trunc, info = env_p.step(np.zeros(env_p.action_space.shape))
    q_after = env_p.data.qpos[3:9].copy()
    env_p.close()
    return {
        "phase": phase_idx,
        "label": label,
        "ref": ref[phase_idx].copy(),
        "q_set": q_set,
        "q_after": q_after,
    }

results = []
results.append(probe_one_phase(peak_hip_r_phase, "peak hip_r flexion"))
results.append(probe_one_phase(peak_hip_l_phase, "peak hip_l flexion"))
# Also probe a few earlier/later phases to map the limit-clipping region
for ph in [0, len(ref)//4, len(ref)//2, 3*len(ref)//4]:
    results.append(probe_one_phase(ph, f"phase {ph}"))

print(f"  {'label':22s} {'joint':10s} "
      f"{'ref deg':>8s} {'set deg':>8s} {'after':>8s} {'delta':>7s}")
for r in results:
    for j, lab in enumerate(labels):
        d_set   = np.rad2deg(r["q_set"][j])
        d_after = np.rad2deg(r["q_after"][j])
        d_ref   = np.rad2deg(r["ref"][j])
        delta   = d_after - d_set
        flag = " <-LIM" if abs(delta) > 0.5 else ""
        print(f"  {r['label']:22s} {lab:10s} "
              f"{d_ref:8.2f} {d_set:8.2f} {d_after:8.2f} {delta:7.2f}{flag}")
    print()

# ── 4. Trained xvel-5M hip trajectory check ──────────────────────────────────
print("=" * 72)
print("4. Trained xvel-5M policy hip trajectory")
print("=" * 72)
print()

from stable_baselines3 import PPO  # noqa: E402

# Re-create the env the policy was trained in.
sys.path.insert(0, str(PROJECT_ROOT / "src" / "walker2d"))
from ppo_walker2d_phase import Walker2dPhaseAware  # noqa: E402

xvel_dir = PROJECT_ROOT / "results" / "restart_b2_xvel"
model_path = xvel_dir / "model"
ref_used = np.load(xvel_dir / "reference.npy")
print(f"  Loading xvel-5M model from: {model_path}")
print(f"  Reference shape: {ref_used.shape}")

trained_env = Walker2dPhaseAware(reference=ref_used, xml_file="walker2d.xml",
                                  warm_start=True)
model = PPO.load(str(model_path), env=None, device="cpu")

obs, _ = trained_env.reset()
trained_env._phase = 0  # start from phase 0 to walk through full cycle
obs = trained_env._get_obs()

phases    = []
hip_r_pol = []
hip_r_ref = []
hip_l_pol = []
hip_l_ref = []
n_cycle = trained_env._ref_len

for t in range(2 * n_cycle):
    action, _ = model.predict(obs, deterministic=True)
    obs, rew, term, trunc, info = trained_env.step(action)
    phases.append(int(info["phase"]))
    hip_r_pol.append(float(trained_env.data.qpos[3]))
    hip_l_pol.append(float(trained_env.data.qpos[6]))
    hip_r_ref.append(float(ref_used[info["phase"], 0]))
    hip_l_ref.append(float(ref_used[info["phase"], 3]))
    if term or trunc:
        break

trained_env.close()

phases    = np.array(phases)
hip_r_pol = np.array(hip_r_pol)
hip_r_ref = np.array(hip_r_ref)
hip_l_pol = np.array(hip_l_pol)
hip_l_ref = np.array(hip_l_ref)

print(f"  Survived {len(phases)} steps (target {2*n_cycle}).")
print()
print(f"  Reference hip_r:    [{np.rad2deg(hip_r_ref.min()):+.2f}, "
      f"{np.rad2deg(hip_r_ref.max()):+.2f}] deg")
print(f"  Policy    hip_r:    [{np.rad2deg(hip_r_pol.min()):+.2f}, "
      f"{np.rad2deg(hip_r_pol.max()):+.2f}] deg")
print(f"  Reference hip_l:    [{np.rad2deg(hip_l_ref.min()):+.2f}, "
      f"{np.rad2deg(hip_l_ref.max()):+.2f}] deg")
print(f"  Policy    hip_l:    [{np.rad2deg(hip_l_pol.min()):+.2f}, "
      f"{np.rad2deg(hip_l_pol.max()):+.2f}] deg")
print()
xml_thigh_hi_deg = np.rad2deg(model_path is not None and
                              ref_used.shape[1] == 6 and
                              xml_hi_rad[0])
print(f"  XML thigh_joint hi: {np.rad2deg(xml_hi_rad[0]):+.2f} deg")
print()

# How often is the policy within 1° of the joint upper limit while ref demands flexion?
hip_r_at_limit_mask = hip_r_pol >= xml_hi_rad[0] - np.deg2rad(1.0)
ref_demands_flex_r  = hip_r_ref > np.deg2rad(5.0)  # ref wants meaningful flexion
both = hip_r_at_limit_mask & ref_demands_flex_r
print(f"  hip_r within 1° of upper limit (0°): "
      f"{hip_r_at_limit_mask.mean()*100:.1f}% of steps")
print(f"  hip_r at limit AND ref demands >5° flexion: "
      f"{both.mean()*100:.1f}% of steps  "
      f"(direct evidence of clipping during flexion phase)")
hip_l_at_limit_mask = hip_l_pol >= xml_hi_rad[3] - np.deg2rad(1.0)
ref_demands_flex_l  = hip_l_ref > np.deg2rad(5.0)
both_l = hip_l_at_limit_mask & ref_demands_flex_l
print(f"  hip_l within 1° of upper limit (0°): "
      f"{hip_l_at_limit_mask.mean()*100:.1f}% of steps")
print(f"  hip_l at limit AND ref demands >5° flexion: "
      f"{both_l.mean()*100:.1f}% of steps")
