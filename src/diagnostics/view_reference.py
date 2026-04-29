"""
view_reference.py — kinematic playback of the reference gait cycle in
MuJoCo's interactive viewer.

Plays the raw reference (Subject 1, baseline, 1.25 m/s, looped) on a
Walker2d-v4 skeleton so you can confirm what motion the imitation
pipeline is being asked to enforce. No policy, no physics — joint
angles are written directly into qpos each frame and the root is
drifted forward at the reference walking speed.

Run from the project root:
  python src/diagnostics/view_reference.py
  python src/diagnostics/view_reference.py --slow 4    # 4x slow-mo
  python src/diagnostics/view_reference.py --cycles 20

For 2D matplotlib joint-angle traces instead, see diag_cycle.py.
"""
import argparse
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REF_PATH     = PROJECT_ROOT / "assets" / "reference" / "gait_cycle_reference.npy"

CTRL_HZ   = 125.0    # Walker2d-v4 step rate
REF_HZ    = 50.0     # on-disk Ulrich rate
WALK_SPEED = 1.25    # m/s — Subject 1 baseline trial speed
TORSO_Z   = 1.28     # standing torso height (matches diag_ref.py)


def load_ref_125hz() -> np.ndarray:
    raw   = np.load(REF_PATH)
    n_in  = len(raw)
    n_out = int(round(n_in * CTRL_HZ / REF_HZ))
    x_in  = np.linspace(0, 1, n_in)
    x_out = np.linspace(0, 1, n_out)
    return np.stack(
        [np.interp(x_out, x_in, raw[:, j]) for j in range(6)], axis=1
    ).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Live kinematic playback of the reference gait cycle."
    )
    parser.add_argument("--slow",   type=float, default=1.0,
                        help="Slowdown factor (1.0=realtime, 4.0=4x slower)")
    parser.add_argument("--cycles", type=int,   default=20,
                        help="Cycles to loop before exit")
    args = parser.parse_args()

    ref = load_ref_125hz()
    n   = len(ref)
    dt  = (1.0 / CTRL_HZ) * args.slow
    dx  = WALK_SPEED / CTRL_HZ

    env = gym.make("Walker2d-v4")
    env.reset()
    m, d = env.unwrapped.model, env.unwrapped.data
    torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")

    print(f"Reference: {n} frames @ {CTRL_HZ:g} Hz "
          f"({n / CTRL_HZ:.2f} s/cycle), looping {args.cycles}× at "
          f"{args.slow}× speed.")
    print("Root is drifted at 1.25 m/s; pitch pinned to 0.")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = torso_id
        viewer.cam.distance    = 4.0
        viewer.cam.azimuth     = 90.0
        viewer.cam.elevation   = -10.0

        d.qpos[:] = 0.0
        d.qpos[1] = TORSO_Z
        d.qvel[:] = 0.0

        for _ in range(args.cycles):
            for t in range(n):
                if not viewer.is_running():
                    return
                t0 = time.perf_counter()
                # Walker2d's hip joint axis is [0,-1,0] and the reference's
                # negated opensim values place flexion at negative hip angles
                # (see ppo_walker2d.py:101-107). With those kinematics, "leg
                # forward at heel strike" points toward -x, so the body must
                # drift in -x to make the planted-foot illusion read as
                # forward walking. The data is unchanged; this is purely a
                # rendering-direction choice.
                d.qpos[0] -= dx
                d.qpos[1]  = TORSO_Z
                d.qpos[2]  = 0.0
                d.qpos[3:9] = ref[t]
                mujoco.mj_forward(m, d)
                viewer.sync()
                sleep_s = dt - (time.perf_counter() - t0)
                if sleep_s > 0:
                    time.sleep(sleep_s)

    env.close()


if __name__ == "__main__":
    main()
