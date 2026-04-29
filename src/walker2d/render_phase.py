"""
render_phase.py — visualize phase-aware imitation policy episodes.

Usage:
  python render_phase.py <result_dir>:<checkpoint_steps>[:<label>] [<result_dir>:...] ...

  checkpoint_steps: integer (e.g. 15000000) or "final" (uses model.zip)

Optional flags:
  --xml   XML model file (default: walker2d_subject1.xml)
  --eps   Episodes per run (default: 3)
  --steps Max steps per episode (default: 2000)
  --live  Use MuJoCo's interactive viewer (realtime, GPU-accelerated) instead
          of the matplotlib animation. Recommended for visual checks — the
          matplotlib path can't sustain 125 fps with rgb_array frames and
          plays back ~5–6× slow.
  --mp4 PATH  Write rgb_array frames to PATH as an mp4 (one file per run, with
              run label as suffix when more than one run is supplied). Uses
              imageio + ffmpeg.

Examples:
  python render_phase.py results/my_run:15000000:"15M checkpoint"
  python render_phase.py results/run1:final results/run2:10000000
  python render_phase.py --live results/my_run:final
  python render_phase.py --mp4 docs/figures/restart_b1.mp4 \\
      results/restart_b1_dm:final results/restart_b1_dm_bc:final
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import PPO, SAC
from ppo_walker2d_phase import Walker2dPhaseAware, CTRL_HZ


def _load_policy(model_path: str):
    """Try PPO.load, fall back to SAC.load — overnight 2026-04-29."""
    try:
        return PPO.load(model_path)
    except (TypeError, KeyError, AttributeError) as e:
        try:
            return SAC.load(model_path)
        except Exception:
            raise e


def load_env_kwargs(result_dir: str) -> dict:
    """Read env_kwargs.json saved at training time, or return defaults.

    preview_k changes obs_space, so render/eval must build the env with the
    same kwargs the trained policy was wired against. Pre-overnight runs
    don't have this file; we fall back to defaults that match current behavior.
    """
    p = Path(result_dir) / "env_kwargs.json"
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
    if "ref_root_drop" in meta:
        out["ref_root_drop"] = float(meta["ref_root_drop"])
    if "xml_file" in meta:
        out["xml_file"] = str(meta["xml_file"])
    return out


def parse_spec(spec: str, default_xml: str) -> dict:
    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError(f"Expected result_dir:checkpoint[:label], got: {spec!r}")
    result_dir = parts[0]
    ckpt_raw   = parts[1]
    label      = parts[2] if len(parts) > 2 else Path(result_dir).name + f":{ckpt_raw}"

    if ckpt_raw.lower() == "final":
        model_path = str(Path(result_dir) / "model")
    else:
        model_path = str(Path(result_dir) / "checkpoints" / f"model_{ckpt_raw}_steps")

    return {"label": label, "result_dir": result_dir,
            "model_path": model_path, "xml_file": default_xml}


def run_live(runs, args):
    """Realtime playback via MuJoCo's passive viewer (GPU-accelerated).

    Steps the env at CTRL_HZ wall-clock so playback matches sim time;
    matplotlib animation can't sustain that with rgb_array frames.
    """
    import mujoco
    import mujoco.viewer

    dt = 1.0 / CTRL_HZ
    for run in runs:
        ref      = np.load(f"{run['result_dir']}/reference.npy")
        extras   = load_env_kwargs(run["result_dir"])
        # If env_kwargs.json saved an xml_file (post-batch-4), use it; the
        # policy was trained against that MJCF and rendering it under a
        # different one (e.g. opening the hip range) gives misleading
        # visuals. CLI --xml only acts as the default if nothing was saved.
        xml_file = extras.pop("xml_file", run["xml_file"])
        env = Walker2dPhaseAware(
            reference=ref, xml_file=xml_file,
            pose_term_thresh=9999.0, ankle_term_thresh=9999.0,
            **extras,
        )
        model = _load_policy(run["model_path"])

        torso_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "torso")

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            # Track the torso COM from the side. mjCAMERA_TRACKING auto-updates
            # lookat to the body's position each frame so the walker stays
            # centered as it advances along +x.
            viewer.cam.type         = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid  = torso_id
            viewer.cam.distance     = 4.0
            viewer.cam.azimuth      = 90.0    # look from +y toward origin (side view)
            viewer.cam.elevation    = -10.0
            viewer.cam.lookat[:]    = [0.0, 0.0, 1.0]
            for ep in range(args.eps):
                start_phase = ep * len(ref) // args.eps
                obs, _ = env.reset(seed=args.seed + ep)
                env._phase = start_phase
                qpos = env.data.qpos.copy()
                qvel = env.data.qvel.copy()
                qpos[3:9] = np.clip(ref[start_phase], env._jnt_lo, env._jnt_hi)
                qvel[3:9] = 0.0
                env.set_state(qpos, qvel)
                obs = env._get_obs()

                steps_done = 0
                for _ in range(args.steps):
                    if not viewer.is_running():
                        break
                    t0 = time.perf_counter()
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = env.step(action)
                    viewer.sync()
                    steps_done += 1
                    if terminated or truncated:
                        break
                    sleep_s = dt - (time.perf_counter() - t0)
                    if sleep_s > 0:
                        time.sleep(sleep_s)

                print(f"[{run['label']}] ep {ep+1}: {steps_done} steps")
                if not viewer.is_running():
                    break

        env.close()


def main():
    parser = argparse.ArgumentParser(description="Render phase-aware Walker2d policy")
    parser.add_argument("specs", nargs="+",
                        help="result_dir:checkpoint_steps[:label]")
    parser.add_argument("--xml",   default="walker2d_subject1.xml",
                        help="MuJoCo model XML (default: walker2d_subject1.xml)")
    parser.add_argument("--eps",   type=int, default=3,  help="Episodes per run")
    parser.add_argument("--steps", type=int, default=2000, help="Max steps per episode")
    parser.add_argument("--live",  action="store_true",
                        help="Use MuJoCo's interactive viewer at realtime "
                             "(125 Hz). The default matplotlib path renders "
                             "rgb_array frames and can't keep up at 125 fps.")
    parser.add_argument("--seed",  type=int, default=0,
                        help="RNG seed for env.reset() (root-state noise). "
                             "Same seed → same rollout each invocation.")
    parser.add_argument("--mp4",   default=None,
                        help="If set, write rgb_array frames to this mp4 path. "
                             "With multiple runs, the run label is inserted as "
                             "a suffix (foo.mp4 -> foo_<label>.mp4).")
    args = parser.parse_args()

    runs = [parse_spec(s, args.xml) for s in args.specs]

    if args.live:
        run_live(runs, args)
        return

    all_frames = []
    run_labels = []
    per_run_frames: list[tuple[str, list]] = []

    for run in runs:
        ref      = np.load(f"{run['result_dir']}/reference.npy")
        extras   = load_env_kwargs(run["result_dir"])
        # Saved xml_file wins over CLI default (see run_live for the why).
        xml_file = extras.pop("xml_file", run["xml_file"])
        env = Walker2dPhaseAware(
            reference=ref, xml_file=xml_file, render_mode="rgb_array",
            pose_term_thresh=9999.0, ankle_term_thresh=9999.0,
            **extras,
        )
        model = _load_policy(run["model_path"])

        run_frames = []
        for ep in range(args.eps):
            # Trust env.reset()'s RSI — it samples a random phase and applies
            # the same warm-start the policy was trained against. Some
            # trained policies fall over within ~10 steps from particular
            # phases (e.g. b1_prod_reward dies from phase 138 but survives
            # the full 2500 from phases 1, 25, 62, 79, 80, 106, 109, …). To
            # get a representative MP4, try up to 8 seeds and pick the
            # longest rollout. This is what eval_biomech does implicitly via
            # multiple unseeded resets.
            best_frames: list = []
            for try_idx in range(8):
                seed_try = args.seed + ep + try_idx * 17
                obs, _ = env.reset(seed=seed_try)
                ep_frames: list = []
                for _ in range(args.steps):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = env.step(action)
                    ep_frames.append(env.render())
                    if terminated or truncated:
                        break
                if len(ep_frames) > len(best_frames):
                    best_frames = ep_frames
                # Good enough? Stop trying.
                if len(best_frames) >= min(args.steps, 200):
                    break
            ep_frames = best_frames

            print(f"[{run['label']}] ep {ep+1}: {len(ep_frames)} steps")
            all_frames.extend(ep_frames)
            run_frames.extend(ep_frames)
            run_labels.append((len(all_frames) - len(ep_frames),
                               len(ep_frames), run["label"], ep + 1))

        per_run_frames.append((run["label"], run_frames))
        env.close()

    print(f"\nTotal frames: {len(all_frames)}")

    if args.mp4:
        import imageio.v2 as imageio
        out = Path(args.mp4)
        out.parent.mkdir(parents=True, exist_ok=True)
        if len(per_run_frames) == 1:
            label, frames = per_run_frames[0]
            imageio.mimsave(out, frames, fps=int(CTRL_HZ), macro_block_size=1)
            print(f"Wrote {out}  ({len(frames)} frames @ {int(CTRL_HZ)} fps)")
        else:
            for label, frames in per_run_frames:
                safe = "".join(c if c.isalnum() or c in "._-" else "_"
                               for c in label)
                p = out.with_name(f"{out.stem}_{safe}{out.suffix}")
                imageio.mimsave(p, frames, fps=int(CTRL_HZ), macro_block_size=1)
                print(f"Wrote {p}  ({len(frames)} frames @ {int(CTRL_HZ)} fps)")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    im    = ax.imshow(all_frames[0])
    title = ax.set_title("")

    def get_label(i):
        for start, n, label, ep in run_labels:
            if start <= i < start + n:
                return f"{label} — ep {ep} (frame {i - start + 1}/{n})"
        return ""

    def update(i):
        im.set_data(all_frames[i])
        title.set_text(get_label(i))
        return (im, title)

    ani = animation.FuncAnimation(
        fig, update, frames=len(all_frames), interval=8, blit=True
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
