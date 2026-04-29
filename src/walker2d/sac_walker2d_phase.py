"""
sac_walker2d_phase.py
─────────────────────
Off-policy SAC variant of the phase-conditioned imitation pipeline.

This is the Phase-5 sibling of `ppo_walker2d_phase.py`. The env, reward,
RSI, and termination are identical — only the optimizer changes from PPO
(on-policy) to SAC (off-policy). Hypothesis: PPO's on-policy exploration
limits how fast the policy escapes the stiff-hip basin; SAC's stochastic
actor + replay buffer may explore more aggressively in early training.

Notes on differences from `ppo_walker2d_phase.py`:
  - 1 env (not 8). SAC's diversity comes from the replay buffer; multi-env
    rollout collection is supported by SB3 but is the wrong tool for the
    job here — too few env steps per gradient update.
  - 1M total env steps (vs 5M for PPO). SAC is sample-efficient.
  - Default SAC hyperparams from SB3 except a handful that benefit MuJoCo
    locomotion: `learning_rate=3e-4`, `batch_size=256`, `buffer_size=300000`.
  - No BC warm-start. Could be added later but BC is unnecessary if SAC's
    replay-buffer + entropy bonus does the same job (escape stand-still).

Usage:
    python src/walker2d/sac_walker2d_phase.py \\
        --ref_cycle assets/reference/gait_cycle_reference.npy \\
        --total_steps 1000000 --xvel_term 0.3
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, CallbackList,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ppo_walker2d_phase import (
    Walker2dPhaseAware, load_ref_cycle, CTRL_HZ, MJCF_ROOT,
)


class SACLogCallback(BaseCallback):
    """Console + TensorBoard logging mirroring PPO's LogCallback.

    Records per-rollout means of every reward component the env writes into
    info[...], plus per-rollout termination-cause counts.
    """

    REWARD_COMPS = ("r_pose", "r_vel", "r_ee", "r_root",
                    "contact_r", "swing_pen", "ctrl_cost", "energy_pen")
    TERM_CAUSES  = ("height", "pitch", "ankle", "hip", "pose", "xvel", "other")

    def __init__(self, log_interval: int = 5000):
        super().__init__(verbose=0)
        self._interval     = log_interval
        self._last_logged  = 0
        self._comp_buf     = {k: [] for k in self.REWARD_COMPS}
        self._term_counts  = {k: 0  for k in self.TERM_CAUSES}
        self._ep_r:    list[float] = []
        self._ep_l:    list[int]   = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for k, buf in self._comp_buf.items():
                v = info.get(k)
                if v is not None:
                    buf.append(float(v))
            cause = info.get("term_cause")
            if cause in self._term_counts:
                self._term_counts[cause] += 1
            ep = info.get("episode")
            if ep:
                self._ep_r.append(float(ep["r"]))
                self._ep_l.append(int(ep["l"]))

        if self.num_timesteps - self._last_logged >= self._interval:
            self._last_logged = self.num_timesteps
            for k, buf in self._comp_buf.items():
                if buf:
                    self.logger.record(f"reward/{k}", float(np.mean(buf)))
                buf.clear()
            for k, c in self._term_counts.items():
                self.logger.record(f"term/{k}", int(c))
                self._term_counts[k] = 0
            if self._ep_r:
                print(
                    f"[steps {self.num_timesteps:>9,}]  "
                    f"ep_r={np.mean(self._ep_r):8.1f}  "
                    f"ep_len={np.mean(self._ep_l):6.0f}  "
                    f"(n={len(self._ep_r)})"
                )
                self._ep_r.clear()
                self._ep_l.clear()
        return True


def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase-conditioned SAC imitation for Walker2d-v4 "
                    "(overnight 2026-04-29 Phase 5 sibling of ppo_walker2d_phase.py)"
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--ref_cycle", type=str)
    grp.add_argument("--ref_all",   action="store_true")
    p.add_argument("--total_steps", type=float, default=1_000_000)
    p.add_argument("--device",      default="cpu")
    p.add_argument("--seed",        type=int, default=0)

    # SAC hyperparams.
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--batch_size",    type=int,   default=256)
    p.add_argument("--buffer_size",   type=int,   default=300_000)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--tau",           type=float, default=0.005)
    p.add_argument("--learning_starts", type=int, default=10_000)
    p.add_argument("--train_freq",    type=int,   default=1,
                   help="Gradient steps per env step (SB3: train_freq=(N, 'step')).")
    p.add_argument("--gradient_steps", type=int,  default=1)

    # DeepMimic reward weights — same defaults as PPO.
    p.add_argument("--pose_weight", type=float, default=0.65)
    p.add_argument("--vel_weight",  type=float, default=0.10)
    p.add_argument("--ee_weight",   type=float, default=0.15)
    p.add_argument("--root_weight", type=float, default=0.10)
    p.add_argument("--pose_scale", type=float, default=10.0)
    p.add_argument("--vel_scale",  type=float, default=0.1)
    p.add_argument("--ee_scale",   type=float, default=40.0)
    p.add_argument("--root_scale", type=float, default=10.0)
    p.add_argument("--v_target", type=float, default=1.25)
    p.add_argument("--pitch_term", type=float, default=0.3)
    p.add_argument("--xvel_term", type=float, default=-1e9)

    p.add_argument("--no_tb",   action="store_true")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    # ── reference ─────────────────────────────────────────────────────
    reference = load_ref_cycle(Path(args.ref_cycle))
    print(f"Reference shape: {reference.shape}  "
          f"({len(reference)/CTRL_HZ:.1f}s @ {CTRL_HZ}Hz)")

    # ── output dir ────────────────────────────────────────────────────
    stamp   = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = PROJECT_ROOT / (args.out_dir
                              or f"results/sac_walker2d_phase_{stamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {log_dir}")
    np.save(log_dir / "reference.npy", reference)

    # ── env (single, not vectorized) ──────────────────────────────────
    env = Walker2dPhaseAware(
        reference          = reference,
        xml_file           = "walker2d.xml",
        pose_weight        = args.pose_weight,
        vel_weight         = args.vel_weight,
        ee_weight          = args.ee_weight,
        root_weight        = args.root_weight,
        pose_scale         = args.pose_scale,
        vel_scale          = args.vel_scale,
        ee_scale           = args.ee_scale,
        root_scale         = args.root_scale,
        v_target           = args.v_target,
        pitch_term_thresh  = args.pitch_term,
        xvel_term_thresh   = args.xvel_term,
    )

    # Save env_kwargs metadata so renderer/eval know what to build.
    (log_dir / "env_kwargs.json").write_text(json.dumps({
        "preview_k": 1, "v_target": args.v_target,
    }, indent=2), encoding="utf-8")

    tb_dir = None if args.no_tb else str(log_dir / "tb")
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate    = args.learning_rate,
        buffer_size      = args.buffer_size,
        learning_starts  = args.learning_starts,
        batch_size       = args.batch_size,
        gamma            = args.gamma,
        tau              = args.tau,
        train_freq       = args.train_freq,
        gradient_steps   = args.gradient_steps,
        policy_kwargs    = {"net_arch": [256, 256]},
        seed             = args.seed,
        device           = args.device,
        tensorboard_log  = tb_dir,
        verbose          = 0,
    )

    if tb_dir:
        print(f"TensorBoard logs: {tb_dir}")

    checkpoint_cb = CheckpointCallback(
        save_freq   = 200_000,
        save_path   = str(log_dir / "checkpoints"),
        name_prefix = "model",
        verbose     = 0,
    )

    total_steps = int(args.total_steps)
    print(f"Training SAC for {total_steps:,} env steps (1 env)...")
    model.learn(
        total_timesteps = total_steps,
        callback        = CallbackList([SACLogCallback(), checkpoint_cb]),
        progress_bar    = True,
    )
    env.close()

    save_path = str(log_dir / "model")
    model.save(save_path)
    print(f"Model saved -> {save_path}.zip")


if __name__ == "__main__":
    main()
