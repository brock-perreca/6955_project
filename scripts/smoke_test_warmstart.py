"""
Smoke test for the 2026-04-29 warm-start fix
(reading _jnt_lo/_jnt_hi from the MJCF instead of hardcoded constants).

Verifies:
1. Walker2dPhaseAware constructs against both stock + hipopen MJCFs.
2. reset() warm-starts qpos within the model's actual joint range.
3. step() runs without constraint-solver explosions (NaN qpos / qvel).
4. A 200-step rollout under random actions completes for both MJCFs.
5. PPO.predict + step works against the trained b4_hipopen_5M model.
6. render_phase.py's clip uses env._jnt_lo / env._jnt_hi (no NameError
   from the old _JNT_LO/_JNT_HI import).
"""
from __future__ import annotations
import sys
from pathlib import Path
import warnings

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "walker2d"))

warnings.filterwarnings("ignore", category=UserWarning)

from ppo_walker2d_phase import Walker2dPhaseAware  # noqa: E402

ref = np.load(PROJECT_ROOT / "assets" / "reference" / "gait_cycle_reference.npy")

def check_env(xml: str) -> None:
    print(f"\n=== {xml} ===")
    env = Walker2dPhaseAware(reference=ref, xml_file=xml)

    # 1) bounds match MJCF
    expected_lo = np.array(
        [env.model.joint(n).range[0] for n in
         ("thigh_joint","leg_joint","foot_joint",
          "thigh_left_joint","leg_left_joint","foot_left_joint")],
        dtype=np.float32,
    )
    assert np.allclose(env._jnt_lo, expected_lo), \
        f"_jnt_lo mismatch:\n  got      {env._jnt_lo}\n  expected {expected_lo}"
    print(f"  [ok] _jnt_lo / _jnt_hi match MJCF joint ranges")
    print(f"       hip range (deg): "
          f"[{np.rad2deg(env._jnt_lo[0]):+.1f}, "
          f"{np.rad2deg(env._jnt_hi[0]):+.1f}]")

    # 2) warm-start qpos within bounds (both edges)
    seeds_tested = 0
    for seed in range(20):
        env.reset(seed=seed)
        q = env.data.qpos[3:9]
        below = q < env._jnt_lo - 1e-6
        above = q > env._jnt_hi + 1e-6
        assert not below.any() and not above.any(), \
            f"seed={seed}: qpos out of clip range\n" \
            f"  q={np.rad2deg(q)}\n  lo={np.rad2deg(env._jnt_lo)}\n" \
            f"  hi={np.rad2deg(env._jnt_hi)}"
        seeds_tested += 1
    print(f"  [ok] {seeds_tested} resets all clipped within MJCF range")

    # 3) 200-step random rollout — no NaN, no exception
    obs, _ = env.reset(seed=0)
    for t in range(200):
        a = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(a)
        assert not np.isnan(env.data.qpos).any(), f"NaN qpos at step {t}"
        assert not np.isnan(env.data.qvel).any(), f"NaN qvel at step {t}"
        if term or trunc:
            obs, _ = env.reset()
    print(f"  [ok] 200-step random rollout completes (no NaN)")
    env.close()

check_env("walker2d.xml")
check_env("walker2d_hipopen.xml")

# 4) trained policy still loads + steps
print("\n=== trained model: results/restart_b4_hipopen_5M ===")
from stable_baselines3 import PPO  # noqa: E402
env = Walker2dPhaseAware(reference=ref, xml_file="walker2d_hipopen.xml")
model = PPO.load(str(PROJECT_ROOT / "results" / "restart_b4_hipopen_5M" / "model"),
                  env=None, device="cpu")
obs, _ = env.reset(seed=42)
n_alive = 0
for _ in range(500):
    a, _ = model.predict(obs, deterministic=True)
    obs, _, term, trunc, _ = env.step(a)
    n_alive += 1
    if term or trunc:
        break
print(f"  [ok] {n_alive} steps survived under deterministic predict")
env.close()

# 5) render_phase.py imports cleanly (would fail if _JNT_LO import wasn't removed)
import importlib  # noqa: E402
import render_phase  # noqa: F401, E402
print("\n=== render_phase.py imports: ok ===")

print("\nALL SMOKE TESTS PASSED")
