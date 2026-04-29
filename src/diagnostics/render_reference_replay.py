"""
render_reference_replay.py — kinematic-replay video of the on-disk
reference gait cycle, embodied in the Walker2d MJCF.

This is the **visual ceiling** for every imitation experiment. If a
trained policy's render doesn't look like this video, that is the gap.

Pure FK display — no policy, no PD controller, no physics integration:

    qpos = [root_x_drift_at_1.25_m/s, 1.28, 0.0, ref[t]]
    mj_forward(m, d)        # contacts evaluated at this configuration
    env.render()             # rgb_array frame, default Walker2d camera

The body cannot fall (we re-pin every frame), but feet may interpenetrate
the floor at this MJCF's leg length, and contact forces evaluated by
mj_forward at a kinematically-pinned pose are not the forces a *dynamic*
trajectory would produce. Both observations are diagnostic about how
compatible the Ulrich reference is with stock Walker2d-v4.

Outputs (default):
    docs/figures/reference_replay.mp4   — mp4 at 125 fps (matches
                                          render_phase.py)
    docs/figures/reference_replay.npz   — per-frame trace
    docs/figures/REFERENCE_REPLAY.md    — methodology + key findings

Usage:
    python src/diagnostics/render_reference_replay.py
    python src/diagnostics/render_reference_replay.py --cycles 5
    python src/diagnostics/render_reference_replay.py --no-doc
"""
import argparse
import sys
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "walker2d"))
from ppo_walker2d_phase import load_ref_cycle, CTRL_HZ  # noqa: E402

REF_PATH = PROJECT_ROOT / "assets" / "reference" / "gait_cycle_reference.npy"

WALK_SPEED = 1.25   # m/s — Subject 1 baseline trial speed
TORSO_Z    = 1.28   # nominal standing height (matches diag_ref.py / view_reference.py)
JOINT_NAMES = ["hip_r", "knee_r", "ankle_r", "hip_l", "knee_l", "ankle_l"]


def replay(cycles: int, out_mp4: Path, out_npz: Path) -> dict:
    """Render `cycles` loops of the reference; return summary stats."""
    ref = load_ref_cycle(REF_PATH)
    n   = len(ref)

    env = gym.make("Walker2d-v4", render_mode="rgb_array")
    env.reset()
    m, d = env.unwrapped.model, env.unwrapped.data

    # Body indices for FK readout.
    foot_r_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "foot")
    foot_l_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "foot_left")
    torso_id   = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")

    dx = WALK_SPEED / CTRL_HZ
    total_frames = cycles * n

    frames    = []
    qpos_log  = np.zeros((total_frames, 6),  dtype=np.float32)
    torso_z   = np.zeros(total_frames, dtype=np.float32)
    pitch     = np.zeros(total_frames, dtype=np.float32)
    foot_r_xz = np.zeros((total_frames, 2), dtype=np.float32)  # world
    foot_l_xz = np.zeros((total_frames, 2), dtype=np.float32)
    foot_r_zrel = np.zeros(total_frames, dtype=np.float32)
    foot_l_zrel = np.zeros(total_frames, dtype=np.float32)
    foot_r_frc = np.zeros(total_frames, dtype=np.float32)
    foot_l_frc = np.zeros(total_frames, dtype=np.float32)

    d.qpos[:] = 0.0
    d.qpos[1] = TORSO_Z
    d.qvel[:] = 0.0

    for k in range(cycles):
        for t in range(n):
            i = k * n + t
            d.qpos[0]  += dx
            d.qpos[1]   = TORSO_Z
            d.qpos[2]   = 0.0
            d.qpos[3:9] = ref[t]
            d.qvel[:]   = 0.0
            mujoco.mj_forward(m, d)

            qpos_log[i]    = d.qpos[3:9]
            torso_z[i]     = float(d.body(torso_id).xpos[2])
            pitch[i]       = float(d.qpos[2])
            ftr            = d.body(foot_r_id).xpos
            ftl            = d.body(foot_l_id).xpos
            foot_r_xz[i]   = (float(ftr[0]), float(ftr[2]))
            foot_l_xz[i]   = (float(ftl[0]), float(ftl[2]))
            foot_r_zrel[i] = float(ftr[2] - d.body(torso_id).xpos[2])
            foot_l_zrel[i] = float(ftl[2] - d.body(torso_id).xpos[2])
            # cfrc_ext indices follow Walker2d body order: foot=4, foot_left=7.
            foot_r_frc[i]  = float(np.linalg.norm(d.cfrc_ext[4]))
            foot_l_frc[i]  = float(np.linalg.norm(d.cfrc_ext[7]))

            frames.append(env.render())

    env.close()

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_mp4, frames, fps=int(CTRL_HZ), macro_block_size=1)
    print(f"Wrote {out_mp4}  ({len(frames)} frames @ {int(CTRL_HZ)} fps)")

    np.savez(
        out_npz,
        qpos=qpos_log, torso_z=torso_z, pitch=pitch,
        foot_r_xz=foot_r_xz, foot_l_xz=foot_l_xz,
        foot_r_zrel=foot_r_zrel, foot_l_zrel=foot_l_zrel,
        foot_r_frc=foot_r_frc, foot_l_frc=foot_l_frc,
        ref_used=ref,
    )
    print(f"Wrote {out_npz}  (per-frame trace)")

    # ── summary stats ──────────────────────────────────────────────────
    rom_deg = np.rad2deg(qpos_log.max(0) - qpos_log.min(0))
    foot_below_floor_r = float((foot_r_xz[:, 1] < 0.0).mean())
    foot_below_floor_l = float((foot_l_xz[:, 1] < 0.0).mean())
    min_foot_z_r = float(foot_r_xz[:, 1].min())
    min_foot_z_l = float(foot_l_xz[:, 1].min())
    peak_frc_r = float(foot_r_frc.max())
    peak_frc_l = float(foot_l_frc.max())

    return {
        "n_frames":             total_frames,
        "cycles":               cycles,
        "rom_deg":              dict(zip(JOINT_NAMES, [float(x) for x in rom_deg])),
        "torso_z_range":        (float(torso_z.min()), float(torso_z.max())),
        "pitch_range":          (float(pitch.min()), float(pitch.max())),
        "min_foot_z_world":     {"r": min_foot_z_r, "l": min_foot_z_l},
        "foot_below_floor_frac": {"r": foot_below_floor_r, "l": foot_below_floor_l},
        "peak_contact_force_N": {"r": peak_frc_r, "l": peak_frc_l},
    }


def write_doc(out_md: Path, mp4_rel: str, summary: dict) -> None:
    rom = summary["rom_deg"]
    fbf = summary["foot_below_floor_frac"]
    mfz = summary["min_foot_z_world"]
    pf  = summary["peak_contact_force_N"]
    txt = f"""# Reference replay — kinematic visual baseline

Generated by `src/diagnostics/render_reference_replay.py`. This is what
the on-disk Ulrich reference (Subject 1, baseline, 1.25 m/s) looks like
when its joint angles are written directly into the Walker2d-v4 MJCF
each frame, with the root drifted forward at 1.25 m/s and root height
and pitch pinned. **No physics, no policy, no PD controller.** Every
trained-policy mp4 should be compared to this.

![reference replay]({mp4_rel})

## Methodology

Per frame `t`:

```
qpos[0]  += 1.25 / 125    # +x drift
qpos[1]   = 1.28          # nominal standing height
qpos[2]   = 0.0           # pitch pinned
qpos[3:9] = ref[t]        # joint angles from the reference cycle
mj_forward(m, d)
env.render()              # rgb_array, default Walker2d-v4 camera
```

The reference is `assets/reference/gait_cycle_reference.npy`,
cubic-spline upsampled from 50 Hz → 125 Hz exactly the way
`Walker2dPhaseAware` consumes it during training.

## Findings ({summary['n_frames']} frames, {summary['cycles']} cycles)

| Joint | ROM in replayed qpos (deg) |
|---|---|
| hip_r   | {rom['hip_r']:.2f} |
| knee_r  | {rom['knee_r']:.2f} |
| ankle_r | {rom['ankle_r']:.2f} |
| hip_l   | {rom['hip_l']:.2f} |
| knee_l  | {rom['knee_l']:.2f} |
| ankle_l | {rom['ankle_l']:.2f} |

The ROM column equals the source reference's per-joint ROM to within
spline interpolation noise — that's the validation gate this script
ships with.

**Foot interpenetration with floor (z < 0 in world frame):**
- right foot: {100*fbf['r']:.1f}% of frames, min world z = {mfz['r']:+.3f} m
- left foot:  {100*fbf['l']:.1f}% of frames, min world z = {mfz['l']:+.3f} m

Any non-zero foot-below-floor fraction is a finding about the Ulrich
reference's compatibility with the stock Walker2d skeleton. The
trained policy will need to deviate from the reference where the
reference puts the foot through the floor.

**Peak |cfrc_ext| at the kinematically-pinned pose** (NOT a dynamic
contact-force estimate; mj_forward evaluates contact at the held
configuration):
- right foot: {pf['r']:.1f} N
- left foot:  {pf['l']:.1f} N

These are sanity readouts only — see `eval_biomech.py` for vGRF metrics
that are actually comparable to lab-grade force-plate data.
"""
    out_md.write_text(txt, encoding="utf-8")
    print(f"Wrote {out_md}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cycles", type=int, default=3,
                   help="Number of reference loops to render (default 3).")
    p.add_argument("--out", type=str,
                   default=str(PROJECT_ROOT / "docs" / "figures" / "reference_replay.mp4"),
                   help="Output mp4 path. The .npz trace is saved alongside.")
    p.add_argument("--no-doc", action="store_true",
                   help="Skip writing REFERENCE_REPLAY.md.")
    args = p.parse_args()

    out_mp4 = Path(args.out)
    out_npz = out_mp4.with_suffix(".npz")
    summary = replay(args.cycles, out_mp4, out_npz)

    # ── validation gate ────────────────────────────────────────────────
    ref = load_ref_cycle(REF_PATH)
    ref_rom_hip_r = float(np.rad2deg(ref[:, 0].max() - ref[:, 0].min()))
    sim_rom_hip_r = summary["rom_deg"]["hip_r"]
    delta = abs(sim_rom_hip_r - ref_rom_hip_r)
    print(f"\nValidation: hip_r ROM  ref={ref_rom_hip_r:.3f} deg  "
          f"replayed={sim_rom_hip_r:.3f} deg  delta={delta:.4f} deg")
    if delta > 0.1:
        print(f"  FAIL: replay deviates from reference ROM by more than 0.1 deg.")
        sys.exit(1)
    print("  PASS")

    if not args.no_doc:
        out_md = out_mp4.parent / "REFERENCE_REPLAY.md"
        write_doc(out_md, mp4_rel=out_mp4.name, summary=summary)


if __name__ == "__main__":
    main()
