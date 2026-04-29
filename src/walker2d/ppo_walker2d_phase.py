"""
ppo_walker2d_phase.py
─────────────────────
Phase-conditioned PPO imitation for Walker2d-v4.

**Restart of 2026-04-28.** Rebuilt against the corrected reference
(`assets/reference/gait_cycle_reference.npy` now encodes natural forward
walking; the previous `walker = -opensim` flip on hip and ankle was
inverting the gait — see `docs/PROJECT_TIMELINE.md` § Phase 5). The
file was stripped back to the DeepMimic core: a sum of four
`exp(−k·err²)` tracking terms (pose, velocity, end-effector, root
height), RSI with treadmill-matched forward velocity, and a minimal
termination set (Walker2d's default height bound + a pitch guard).

The previous reward was layered with exploit-patch terms — per-joint
sharpness/weights, swing-foot contact penalty, stance-foot contact
alternation, per-joint pose/ankle termination thresholds. Most of those
exploits (ankle paddling, one-legged hopping, toe-walking) were partly
artifacts of training against a self-contradictory target where
DeepMimic pose-tracking pulled toward backward gait on hip/ankle while
the forward-velocity reward pulled toward +x. Those terms are now
optional, off-by-default flags. They get restored only after a *trained*
policy on the corrected reference shows the specific failure they were
meant to patch.

Reward (DeepMimic Eq. 6, simplified for Walker2d):

    r = 0.65 · r_p + 0.10 · r_v + 0.15 · r_e + 0.10 · r_c

    r_p = exp(−k_p · mean_j (q_j − q_ref_j)²)              k_p = 10
    r_v = exp(−k_v · mean_j (dq_j − dq_ref_j)²)            k_v = 0.1
    r_e = exp(−k_e · sum_foot ((Δx)² + (Δz)²))             k_e = 40
                                                          (root-relative)
    r_c = exp(−k_c · (h − h_ref)²)                         k_c = 10

Per-step reward is in [0, 1]; no `dt` scaling.

Observation (25-D): `[walker2d base obs (17) | q_ref (6) | sin φ | cos φ]`.
Phase advances at a fixed rate (one frame per env step), normalised to
`GAIT_CYCLE_FRAMES = 140` (~1.12 s @ 125 Hz).

Termination:
  - Walker2d-v4 default `[0.8, 2.0]` root height (inherited).
  - `|pitch| > 0.3 rad` — controlled-fall guard. Without it the agent
    learns to lean forward indefinitely; height only drops once the
    lean is irrecoverable.

RSI: `reset()` samples a phase uniformly, places `qpos[3:9]` at the
reference, sets joint velocities from the reference derivative, and
sets `qvel[0] = v_target` (default 1.25 m/s). The forward-velocity
warm-start is required because the reference kinematics are mid-stride
at 1.25 m/s; without it the body lags the joints for ~50 frames every
episode.

Usage:
    python src/walker2d/ppo_walker2d_phase.py \\
        --ref_cycle assets/reference/gait_cycle_reference.npy
    # add --bc_epochs 5 to PD-rollout warm-start before PPO.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from gymnasium import spaces
from gymnasium.envs.mujoco.walker2d_v4 import Walker2dEnv, DEFAULT_CAMERA_CONFIG
from gymnasium.envs.mujoco import MujocoEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, CallbackList,
)

# <repo>/src/walker2d/ppo_walker2d_phase.py → repo root is two parents up
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MJCF_ROOT    = PROJECT_ROOT / "assets" / "mjcf"
REF_ROOT     = PROJECT_ROOT / "assets" / "reference"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ulrich_loader import load_ulrich_reference  # noqa: E402

CTRL_HZ           = 125.0   # Walker2d-v4: frame_skip=4, dt=0.002s
REF_HZ            = 50.0    # Ulrich IK / extracted gait cycles
GAIT_CYCLE_FRAMES = 140     # ~1.12 s @ 125 Hz — used for sin/cos phase encoding


# ── reference loading ─────────────────────────────────────────────────────────

def load_ref_cycle(path: Path) -> np.ndarray:
    """Load a single gait-cycle .npy and resample 50 Hz → 125 Hz with cubic spline."""
    from scipy.interpolate import CubicSpline
    raw   = np.load(path).astype(np.float32)
    n_in  = len(raw)
    n_out = int(round(n_in * CTRL_HZ / REF_HZ))
    x_in  = np.linspace(0, 1, n_in)
    x_out = np.linspace(0, 1, n_out)
    ref   = np.stack(
        [CubicSpline(x_in, raw[:, j])(x_out) for j in range(raw.shape[1])],
        axis=1,
    ).astype(np.float32)
    print(f"Gait cycle: {n_in} frames @ {REF_HZ}Hz -> {n_out} frames @ {CTRL_HZ}Hz (cubic spline)")
    return ref


# ── env ───────────────────────────────────────────────────────────────────────

# RSI warm-start clipping is now read from the loaded MJCF's joint ranges
# (see Walker2dPhaseAware._jnt_lo / _jnt_hi). Hardcoded constants were
# removed 2026-04-29 — they masked the joint-range hypothesis (Batch 4):
# the previous _JNT_HI[0] = 0.550 rad (+31.5°) advertised hip flexion the
# loaded walker2d.xml's +0.000 rad (0°) limit forbade, so the warm-start
# qpos was a lie the dynamics solver immediately overruled.


class Walker2dPhaseAware(Walker2dEnv):
    """
    Walker2d-v4 with phase-conditioned obs and a DeepMimic-style imitation
    reward. See module docstring for the full reward + termination spec.

    Optional exploit-patch terms (swing-foot contact penalty, stance-foot
    contact alternation reward, per-joint pose/ankle termination
    thresholds) are accepted as kwargs but default to off. They were
    needed against the corrupted reference and may resurface as needed.
    """

    BASE_OBS = 17
    N_REF    = 6
    N_PHASE  = 2
    OBS_DIM  = BASE_OBS + N_REF + N_PHASE  # = 25

    def __init__(
        self,
        reference:           np.ndarray,
        xml_file:            str   = "walker2d.xml",
        # DeepMimic reward weights (paper Eq. 6).
        pose_weight:         float = 0.65,
        vel_weight:          float = 0.10,
        ee_weight:           float = 0.15,
        root_weight:         float = 0.10,
        # DeepMimic exp scales.
        pose_scale:          float = 10.0,
        vel_scale:           float = 0.1,
        ee_scale:            float = 40.0,
        root_scale:          float = 10.0,
        ref_root_drop:        float = 0.0,
        # Treadmill speed for the RSI warm-start.
        v_target:            float = 1.25,
        # Termination.
        pitch_term_thresh:   float = 0.3,    # rad — controlled-fall guard
        # Optional exploit-patch terms (off by default; see module docstring).
        swing_pen_weight:    float = 0.0,
        contact_weight:      float = 0.0,
        pose_term_thresh:    float = 9999.0, # rad — disabled by default
        ankle_term_thresh:   float = 9999.0,
        hip_term_thresh:     float = 9999.0, # rad — per-joint hip pose term
        xvel_term_thresh:    float = -np.inf, # disabled by default
        warm_start:          bool  = True,
        # Per-joint pose weighting + alternative aggregators (overnight 2026-04-29).
        pose_joint_weights:  tuple = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        product_reward:      bool  = False,  # geometric mean of per-joint exps
        min_joint_pose:      bool  = False,  # worst-joint floor
        # Energy / torque-squared penalty (overnight 2026-04-29).
        energy_weight:       float = 0.0,
        # Multi-step preview observation (overnight 2026-04-29).
        preview_k:           int   = 1,
        # Inert kwargs accepted for backward compat with old run scripts.
        max_phase_advance:   int   = 1,      # not used: phase advances 1/step
        imitation_weight:    float | None = None,  # mapped → pose_weight if set
        **kwargs,
    ):
        # Backward-compat alias used by some external callers.
        if imitation_weight is not None:
            pose_weight = imitation_weight

        self._reference          = reference.astype(np.float32)
        self._ref_len            = len(reference)
        self._xml_file           = xml_file

        self._w_pose             = float(pose_weight)
        self._w_vel              = float(vel_weight)
        self._w_ee               = float(ee_weight)
        self._w_root             = float(root_weight)
        self._k_pose             = float(pose_scale)
        self._k_vel              = float(vel_scale)
        self._k_ee               = float(ee_scale)
        self._k_root             = float(root_scale)
        self._ref_root_drop      = float(ref_root_drop)

        self._v_target           = float(v_target)
        self._pitch_term_thresh  = float(pitch_term_thresh)

        self._w_swing_pen        = float(swing_pen_weight)
        self._w_contact          = float(contact_weight)
        self._pose_term_thresh   = float(pose_term_thresh)
        self._ankle_term_thresh  = float(ankle_term_thresh)
        self._hip_term_thresh    = float(hip_term_thresh)
        self._xvel_term_thresh   = float(xvel_term_thresh)
        self._warm_start         = bool(warm_start)

        pjw = np.asarray(pose_joint_weights, dtype=np.float32)
        if pjw.shape != (6,):
            raise ValueError(f"pose_joint_weights must be length 6, got {pjw.shape}")
        self._pose_joint_weights = pjw
        self._product_reward     = bool(product_reward)
        self._min_joint_pose     = bool(min_joint_pose)
        self._energy_weight      = float(energy_weight)

        self._preview_k          = max(1, int(preview_k))
        self._phase              = 0

        # Per-frame velocity by central differencing with periodic wrap.
        ref_pad        = np.concatenate([self._reference[-1:],
                                         self._reference,
                                         self._reference[:1]], axis=0)
        self._ref_vel  = (np.gradient(ref_pad, 1.0 / CTRL_HZ, axis=0)
                          [1:-1]).astype(np.float32)

        # Stance side per frame (used only when contact_weight > 0).
        # ref[:, 0] = hip_r (positive = leg forward, +x); whichever hip is
        # more extended is the stance side.
        self._stance_right = self._reference[:, 0] >= self._reference[:, 3]

        # Walker2dEnv hardcodes "walker2d.xml"; replicate its attribute setup
        # so we can pass a custom MJCF.
        self._forward_reward_weight = 1.0
        self._ctrl_cost_weight      = 1e-3
        self._healthy_reward        = 1.0
        self._terminate_when_unhealthy = True
        self._healthy_z_range       = (0.8, 2.0)
        self._healthy_angle_range   = (-1.0, 1.0)
        self._reset_noise_scale     = 5e-3
        self._exclude_current_positions_from_observation = True

        if xml_file == "walker2d.xml":
            resolved_xml = xml_file
        elif Path(xml_file).is_absolute():
            resolved_xml = str(xml_file)
        elif (MJCF_ROOT / xml_file).exists():
            resolved_xml = str(MJCF_ROOT / xml_file)
        else:
            resolved_xml = str(PROJECT_ROOT / xml_file)

        _obs_space = spaces.Box(low=-np.inf, high=np.inf,
                                shape=(17,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            resolved_xml,
            4,                               # frame_skip
            observation_space=_obs_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        # Override observation_space after MujocoEnv.__init__ (which assigns it
        # directly — a property override would AttributeError). With preview_k>1
        # the obs grows by N_REF*(K-1) so we compute the dim per-instance.
        self._obs_dim = self.BASE_OBS + self.N_REF * self._preview_k + self.N_PHASE
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32,
        )

        # Read warm-start clip bounds from the actual MJCF, not hardcoded
        # constants. qpos[3:9] order matches walker2d.xml: thigh, leg, foot,
        # thigh_left, leg_left, foot_left.
        jnt_names = ("thigh_joint", "leg_joint", "foot_joint",
                     "thigh_left_joint", "leg_left_joint", "foot_left_joint")
        self._jnt_lo = np.array(
            [self.model.joint(n).range[0] for n in jnt_names], dtype=np.float32,
        )
        self._jnt_hi = np.array(
            [self.model.joint(n).range[1] for n in jnt_names], dtype=np.float32,
        )

        self._precompute_reference_kinematics()

        self._term_cause: str | None = None

    # ── observation ──────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        base    = super()._get_obs().astype(np.float32)
        if self._preview_k > 1:
            idxs = (self._phase + np.arange(self._preview_k)) % self._ref_len
            q_ref = self._reference[idxs].reshape(-1)  # (N_REF * K,)
        else:
            q_ref = self._reference[self._phase]
        # Normalize phase to GAIT_CYCLE_FRAMES, not _ref_len, so the encoding
        # remains a per-stride signal even when --ref_all is used.
        phi     = 2.0 * np.pi * (self._phase % GAIT_CYCLE_FRAMES) / GAIT_CYCLE_FRAMES
        phase_e = np.array([np.sin(phi), np.cos(phi)], dtype=np.float32)
        return np.concatenate([base, q_ref, phase_e])

    # ── reference FK pre-computation ─────────────────────────────────────

    def _precompute_reference_kinematics(self) -> None:
        """Cache reference root height and root-relative foot (x, z) per frame."""
        n = self._ref_len
        self._ref_root_height = np.zeros(n, dtype=np.float32)
        self._ref_foot_r_xrel = np.zeros(n, dtype=np.float32)
        self._ref_foot_l_xrel = np.zeros(n, dtype=np.float32)
        self._ref_foot_r_zrel = np.zeros(n, dtype=np.float32)
        self._ref_foot_l_zrel = np.zeros(n, dtype=np.float32)

        qpos_save = self.data.qpos.copy()
        qvel_save = self.data.qvel.copy()
        for t in range(n):
            self.data.qpos[:]   = 0.0
            self.data.qpos[1]   = 1.28 - self._ref_root_drop
            self.data.qpos[3:9] = self._reference[t]
            self.data.qvel[:]   = 0.0
            mujoco.mj_kinematics(self.model, self.data)

            root = self.data.body("torso").xpos
            ftr  = self.data.body("foot").xpos
            ftl  = self.data.body("foot_left").xpos

            self._ref_root_height[t] = float(root[2])
            self._ref_foot_r_xrel[t] = float(ftr[0] - root[0])
            self._ref_foot_l_xrel[t] = float(ftl[0] - root[0])
            self._ref_foot_r_zrel[t] = float(ftr[2] - root[2])
            self._ref_foot_l_zrel[t] = float(ftl[2] - root[2])
        self.data.qpos[:] = qpos_save
        self.data.qvel[:] = qvel_save
        mujoco.mj_kinematics(self.model, self.data)

    # ── phase tracking ───────────────────────────────────────────────────

    def _advance_phase(self) -> None:
        """Fixed-rate phase clock — advances exactly 1 frame per env step.

        DeepMimic uses a wall-clock-driven phase, not a state-matched one.
        Adaptive phase tracking (an earlier draft) let stiff-legged policies
        shop for extended-knee reference frames; a fixed clock forces the
        agent to be at the correct phase regardless of its current pose.
        """
        self._phase = (self._phase + 1) % self._ref_len

    # ── reset (RSI) ──────────────────────────────────────────────────────

    def reset(self, **kwargs):
        self._phase = np.random.randint(0, self._ref_len)
        self._term_cause = None
        _, info = super().reset(**kwargs)

        if self._warm_start:
            qpos = self.data.qpos.copy()
            qvel = self.data.qvel.copy()
            qpos[3:9] = np.clip(self._reference[self._phase],
                                self._jnt_lo, self._jnt_hi)
            qvel[3:9] = self._ref_vel[self._phase]
            # Treadmill speed: without this the body lags the joints for ~50
            # frames each episode (joints are mid-stride at 1.25 m/s).
            qvel[0]   = self._v_target
            self.set_state(qpos, qvel)

        return self._get_obs(), info

    # ── step ─────────────────────────────────────────────────────────────

    def step(self, action):
        _, _, terminated, truncated, info = super().step(action)
        height_term = bool(terminated)  # Walker2d-v4 default height bound

        q_sim   = self.data.qpos[3:9].astype(np.float32)
        dq_sim  = self.data.qvel[3:9].astype(np.float32)
        q_ref   = self._reference[self._phase]
        dq_ref  = self._ref_vel[self._phase]

        diff    = q_sim  - q_ref
        diff_v  = dq_sim - dq_ref

        # ── DeepMimic tracking terms ──────────────────────────────────
        weighted_diff_sq = self._pose_joint_weights * (diff ** 2)
        if self._min_joint_pose:
            # Worst-joint floor: r_pose = min_j exp(-k · w_j · diff_j²).
            # One bad joint kills the whole pose reward — hardest fix for the
            # 5-of-6-joint loophole.
            per_joint = np.exp(-self._k_pose * weighted_diff_sq)
            r_pose = float(per_joint.min())
        elif self._product_reward:
            # Geometric mean of per-joint exps — DeepMimic-style multiplicative
            # form on demand.
            per_joint = np.exp(-self._k_pose * weighted_diff_sq)
            r_pose = float(np.prod(per_joint) ** (1.0 / 6.0))
        else:
            # Default: arithmetic mean of weighted squared diffs (back-compat
            # with all pre-overnight runs when pose_joint_weights = ones).
            r_pose = float(np.exp(-self._k_pose * np.mean(weighted_diff_sq)))
        r_vel  = float(np.exp(-self._k_vel  * np.mean(diff_v ** 2)))

        root_xpos = self.data.body("torso").xpos
        ftr_xpos  = self.data.body("foot").xpos
        ftl_xpos  = self.data.body("foot_left").xpos

        foot_r_xrel = float(ftr_xpos[0] - root_xpos[0])
        foot_l_xrel = float(ftl_xpos[0] - root_xpos[0])
        foot_r_zrel = float(ftr_xpos[2] - root_xpos[2])
        foot_l_zrel = float(ftl_xpos[2] - root_xpos[2])
        ee_err_sq = (
            (foot_r_xrel - self._ref_foot_r_xrel[self._phase]) ** 2 +
            (foot_l_xrel - self._ref_foot_l_xrel[self._phase]) ** 2 +
            (foot_r_zrel - self._ref_foot_r_zrel[self._phase]) ** 2 +
            (foot_l_zrel - self._ref_foot_l_zrel[self._phase]) ** 2
        )
        r_ee = float(np.exp(-self._k_ee * ee_err_sq))

        h_err = (float(root_xpos[2]) - float(self._ref_root_height[self._phase])) ** 2
        r_root = float(np.exp(-self._k_root * h_err))

        reward = (self._w_pose * r_pose
                  + self._w_vel  * r_vel
                  + self._w_ee   * r_ee
                  + self._w_root * r_root)

        # ── optional exploit-patch terms (off by default) ─────────────
        contact_r = 0.0
        swing_pen = 0.0
        if self._w_swing_pen != 0.0 or self._w_contact != 0.0:
            foot_r_frc = float(np.linalg.norm(self.data.cfrc_ext[4]))
            foot_l_frc = float(np.linalg.norm(self.data.cfrc_ext[7]))
            if self._w_contact != 0.0:
                if self._stance_right[self._phase]:
                    c = np.tanh(foot_r_frc / 50.0) - np.tanh(foot_l_frc / 50.0)
                else:
                    c = np.tanh(foot_l_frc / 50.0) - np.tanh(foot_r_frc / 50.0)
                contact_r = float(max(c, 0.0))
                reward += self._w_contact * contact_r
            if self._w_swing_pen != 0.0:
                # Swing detection: opposite-side foot has higher reference Z
                # than the other → that side is in swing.
                # For a simple binary heuristic, use the same z threshold the
                # earlier code used on root-relative z.
                SWING_Z = -1.15
                r_swing = self._ref_foot_r_zrel[self._phase] > SWING_Z
                l_swing = self._ref_foot_l_zrel[self._phase] > SWING_Z
                if r_swing:
                    swing_pen += float(np.tanh(foot_r_frc / 50.0))
                if l_swing:
                    swing_pen += float(np.tanh(foot_l_frc / 50.0))
                reward -= self._w_swing_pen * swing_pen

        # Tiny ctrl cost (DeepMimic-style; helps the value baseline).
        ctrl_cost = -1e-3 * float(np.sum(np.square(self.data.ctrl)))
        reward += ctrl_cost

        # Optional torque-squared energy penalty (off by default).
        energy_pen = 0.0
        if self._energy_weight != 0.0:
            energy_pen = float(np.sum(np.square(np.asarray(action))))
            reward -= self._energy_weight * energy_pen

        # ── termination ────────────────────────────────────────────────
        root_pitch = float(self.data.qpos[2])
        x_vel      = float(info.get("x_velocity", self.data.qvel[0]))

        pitch_term = abs(root_pitch) > self._pitch_term_thresh
        # Optional pose/ankle/hip/xvel terms; default thresholds are sentinels
        # that make these never fire.
        ankle_dev = max(abs(diff[2]), abs(diff[5]))
        hip_dev   = max(abs(diff[0]), abs(diff[3]))
        other_dev = max(abs(diff[0]), abs(diff[1]),
                        abs(diff[3]), abs(diff[4]))
        ankle_term = ankle_dev > self._ankle_term_thresh
        hip_term   = hip_dev   > self._hip_term_thresh
        pose_term  = other_dev > self._pose_term_thresh
        xvel_term  = x_vel < self._xvel_term_thresh

        if pitch_term or ankle_term or hip_term or pose_term or xvel_term:
            terminated = True
        if terminated and self._term_cause is None:
            if   height_term: self._term_cause = "height"
            elif pitch_term:  self._term_cause = "pitch"
            elif ankle_term:  self._term_cause = "ankle"
            elif hip_term:    self._term_cause = "hip"
            elif pose_term:   self._term_cause = "pose"
            elif xvel_term:   self._term_cause = "xvel"
            else:             self._term_cause = "other"

        self._advance_phase()

        info.update(
            r_pose=r_pose, r_vel=r_vel, r_ee=r_ee, r_root=r_root,
            contact_r=contact_r, swing_pen=swing_pen, ctrl_cost=ctrl_cost,
            energy_pen=energy_pen,
            phase=self._phase,
        )
        if terminated:
            info["term_cause"] = self._term_cause
        return self._get_obs(), reward, terminated, truncated, info


# ── behavioral cloning warm-start ────────────────────────────────────────────

def compute_bc_dataset(
    env:     "Walker2dPhaseAware",
    n_steps: int   = 200_000,
    kp:      float = 200.0,
    kd:      float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out a PD tracking controller in MuJoCo and collect (obs, action) pairs.

    The PD law in torque space is

        τ_j = Kp · (q_ref_j − q_j) + Kd · (dq_ref_j − dq_j)
        action_j = clip(τ_j / gear, −1, 1)

    with `Kp=200`, `Kd=20`, `gear=100` giving `action = 2·q_err + 0.2·dq_err`.
    A 0.1-rad joint error → 0.2 of action range. The PD runs *inside* the full
    MuJoCo simulation, so the torques are physically consistent with ground
    contact — unlike `mj_inverse`, which ignores contact forces and produces
    wrong torques for ~half the gait cycle.
    """
    gear = float(env.model.actuator_gear[0, 0])

    obs_list:   list[np.ndarray] = []
    act_list:   list[np.ndarray] = []
    ep_lengths: list[int]        = []
    ep_len = 0

    obs, _ = env.reset()
    for _ in range(n_steps):
        q_ref  = env._reference[env._phase]
        dq_ref = env._ref_vel[env._phase]
        q_sim  = env.data.qpos[3:9].astype(np.float32)
        dq_sim = env.data.qvel[3:9].astype(np.float32)

        torque = kp * (q_ref - q_sim) + kd * (dq_ref - dq_sim)
        action = np.clip(torque / gear, -1.0, 1.0).astype(np.float32)

        obs_list.append(obs.copy())
        act_list.append(action)
        ep_len += 1

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            ep_lengths.append(ep_len)
            ep_len = 0
            obs, _ = env.reset()

    if ep_len > 0:
        ep_lengths.append(ep_len)

    ep_arr = np.array(ep_lengths)
    print(f"  PD episodes: {len(ep_arr)}  "
          f"length  mean={ep_arr.mean():.0f}  median={np.median(ep_arr):.0f}  "
          f"max={ep_arr.max()}  "
          f">140 frames: {(ep_arr > 140).mean()*100:.0f}%")

    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32)


def pretrain_bc(
    model:      PPO,
    obs_arr:    np.ndarray,
    act_arr:    np.ndarray,
    n_epochs:   int   = 10,
    batch_size: int   = 512,
    lr:         float = 1e-3,
) -> None:
    """Supervised MSE warm-start: minimise MSE(π_mean(obs), action)."""
    device  = model.device
    obs_t   = torch.tensor(obs_arr, device=device)
    act_t   = torch.tensor(act_arr, device=device)
    dataset = data_utils.TensorDataset(obs_t, act_t)
    loader  = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt     = optim.Adam(model.policy.parameters(), lr=lr)
    mse     = nn.MSELoss()

    print(f"BC pre-training: {n_epochs} epochs × {len(dataset):,} frames")
    for epoch in range(n_epochs):
        if epoch == n_epochs // 2:
            for g in opt.param_groups:
                g["lr"] = lr / 10
            print(f"  [lr -> {lr/10:.0e}]")
        total = 0.0
        for obs_b, act_b in loader:
            dist     = model.policy.get_distribution(obs_b)
            pred_act = dist.distribution.mean
            loss     = mse(pred_act, act_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"  epoch {epoch+1:3d}/{n_epochs}  mse={total/len(loader):.5f}")


# ── callback ─────────────────────────────────────────────────────────────────

class LogCallback(BaseCallback):
    """Console + TensorBoard logging.

    Records per-rollout means of every reward component the env writes into
    `info[...]`, plus per-rollout termination-cause counts. The reward keys
    were renamed `r_pose / r_vel / r_ee / r_root` (was `imit_r / vel_r / ee_r
    / root_r`) in the 2026-04-28 restart; the underlying signal is the same.
    """

    REWARD_COMPS = ("r_pose", "r_vel", "r_ee", "r_root",
                    "contact_r", "swing_pen", "ctrl_cost", "energy_pen")
    TERM_CAUSES  = ("height", "pitch", "ankle", "hip", "pose", "xvel", "other")

    def __init__(self, log_interval: int = 50):
        super().__init__(verbose=0)
        self._interval = log_interval
        self._rollout  = 0
        self._ep_r:    list[float] = []
        self._ep_l:    list[int]   = []
        self._comp_buf:    dict[str, list[float]] = {k: [] for k in self.REWARD_COMPS}
        self._term_counts: dict[str, int]         = {k: 0  for k in self.TERM_CAUSES}

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
        return True

    def _on_rollout_end(self) -> None:
        self._rollout += 1
        for k, buf in self._comp_buf.items():
            if buf:
                self.logger.record(f"reward/{k}", float(np.mean(buf)))
        for k, c in self._term_counts.items():
            self.logger.record(f"term/{k}", int(c))
        for k in self._comp_buf:
            self._comp_buf[k].clear()
        for k in self._term_counts:
            self._term_counts[k] = 0
        if self._rollout % self._interval == 0 and self._ep_r:
            print(
                f"[iter {self._rollout:5d} | steps {self.num_timesteps:>9,}]  "
                f"ep_r={np.mean(self._ep_r):8.1f}  "
                f"ep_len={np.mean(self._ep_l):6.0f}  "
                f"(n={len(self._ep_r)})"
            )
            self._ep_r.clear()
            self._ep_l.clear()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Phase-conditioned PPO imitation for Walker2d-v4 "
                    "(DeepMimic-faithful baseline; 2026-04-28 restart)"
    )

    # reference
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--ref_cycle", type=str,
                     help="Path to single gait-cycle .npy (recommended)")
    grp.add_argument("--ref_all",   action="store_true",
                     help="Use full concatenated Ulrich reference (all subjects/trials)")
    p.add_argument("--subjects",     type=int, nargs="+", default=None)
    p.add_argument("--trial_filter", type=str, default=None)

    # training
    p.add_argument("--num_envs",    type=int,   default=16)
    p.add_argument("--total_steps", type=float, default=5e6)
    p.add_argument("--device",      default="cpu")
    p.add_argument("--seed",        type=int,   default=0,
                   help="RNG seed for SB3 (env subproc seeds derive from this)")
    p.add_argument("--finetune",    default=None,
                   help="Pretrained .zip to finetune from "
                        "(lr→1e-5, ent→0, target_kl→0.005)")

    # BC warm-start (DeepMimic does not use this; off by default)
    p.add_argument("--bc_epochs", type=int, default=0,
                   help="BC warm-start epochs before PPO (0 = skip).")
    p.add_argument("--bc_steps",  type=int, default=200_000)
    p.add_argument("--bc_kp",     type=float, default=200.0)
    p.add_argument("--bc_kd",     type=float, default=20.0)
    p.add_argument("--bc_only",   action="store_true",
                   help="Stop after BC; skip PPO. Saves model_bc.zip + model.zip.")

    # DeepMimic reward weights (paper Eq. 6 default 0.65/0.10/0.15/0.10).
    p.add_argument("--pose_weight", type=float, default=0.65)
    p.add_argument("--vel_weight",  type=float, default=0.10)
    p.add_argument("--ee_weight",   type=float, default=0.15)
    p.add_argument("--root_weight", type=float, default=0.10)
    # DeepMimic exp scales.
    p.add_argument("--pose_scale", type=float, default=10.0)
    p.add_argument("--vel_scale",  type=float, default=0.1)
    p.add_argument("--ee_scale",   type=float, default=40.0)
    p.add_argument("--root_scale", type=float, default=10.0)
    p.add_argument("--ref_root_drop", type=float, default=0.0,
                   help="Lower the FK-derived reference root-height target "
                        "by this many meters. Use as a stock-geometry "
                        "contact-clearance ablation; default preserves "
                        "the pinned 1.28 m reference.")

    # Treadmill speed (RSI warm-start qvel[0]).
    p.add_argument("--v_target", type=float, default=1.25)

    # Termination.
    p.add_argument("--pitch_term", type=float, default=0.3,
                   help="|pitch| termination threshold (rad)")

    # Optional exploit-patch knobs (off by default; enable as ablations).
    p.add_argument("--swing_pen_weight", type=float, default=0.0)
    p.add_argument("--contact_weight",   type=float, default=0.0)
    p.add_argument("--pose_term",  type=float, default=9999.0,
                   help="Hip/knee deviation termination threshold (rad). "
                        "Default disabled.")
    p.add_argument("--ankle_term", type=float, default=9999.0,
                   help="Ankle deviation termination threshold (rad). "
                        "Default disabled.")
    p.add_argument("--xvel_term", type=float, default=-1e9,
                   help="x-velocity termination floor (m/s). Default disabled.")
    p.add_argument("--hip_term", type=float, default=9999.0,
                   help="Hip-only deviation termination threshold (rad). "
                        "Applies independently of --pose_term. Default disabled.")

    # Per-joint pose weighting + alternative aggregators (overnight 2026-04-29).
    p.add_argument("--pose_joint_weights", type=float, nargs=6,
                   default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                   help="Per-joint weights for pose tracking, ordered "
                        "(hip_r, knee_r, ankle_r, hip_l, knee_l, ankle_l). "
                        "Default [1,1,1,1,1,1].")
    p.add_argument("--product_reward", action="store_true",
                   help="Geometric mean of per-joint exps inside r_pose (DeepMimic-style).")
    p.add_argument("--min_joint_pose", action="store_true",
                   help="Worst-joint floor: r_pose = min_j exp(-k·w_j·diff_j²).")

    # Energy / torque-squared penalty (overnight 2026-04-29).
    p.add_argument("--energy_weight", type=float, default=0.0,
                   help="Coefficient for -w·||action||². Default 0.0 (off).")

    # Multi-step preview observation (overnight 2026-04-29).
    p.add_argument("--preview_k", type=int, default=1,
                   help="Concatenate K reference frames into the obs "
                        "(default 1 = current behaviour). K>1 changes the "
                        "obs_space; finetune from a K=1 model will not load.")

    p.add_argument("--scale_model", action="store_true",
                   help="Use Subject-1-scaled MJCF (assets/mjcf/walker2d_subject1.xml). "
                        "Mutually exclusive with --xml.")
    p.add_argument("--xml", type=str, default=None,
                   help="Custom MJCF filename under assets/mjcf/ (e.g. "
                        "walker2d_hipopen.xml, walker2d_hiprelax.xml). "
                        "Use 'walker2d.xml' for the gym default. Mutually "
                        "exclusive with --scale_model.")
    p.add_argument("--no_tb",   action="store_true")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    # ── load reference ────────────────────────────────────────────────
    if args.ref_cycle:
        reference = load_ref_cycle(Path(args.ref_cycle))
    else:
        print("Loading full Ulrich reference...")
        reference = load_ulrich_reference(
            subjects=args.subjects,
            trial_filter=args.trial_filter,
            control_hz=CTRL_HZ,
        )

    if args.scale_model and args.xml:
        raise SystemExit("--scale_model and --xml are mutually exclusive")
    if args.xml is not None:
        xml_path = args.xml
        if xml_path != "walker2d.xml" and not Path(xml_path).is_absolute():
            local = MJCF_ROOT / xml_path
            if not local.exists():
                raise FileNotFoundError(
                    f"--xml {xml_path!r} not found at {local}; "
                    "place a copy under assets/mjcf/."
                )
            print(f"Using custom MJCF: {local}")
        else:
            print(f"Using MJCF override: {xml_path}")
    elif args.scale_model:
        xml_path = str(MJCF_ROOT / "walker2d_subject1.xml")
        if not Path(xml_path).exists():
            print(f"[warn] --scale_model set but {xml_path} is missing; "
                  f"falling back to walker2d.xml")
            xml_path = "walker2d.xml"
        else:
            print(f"Using scaled model: {xml_path}")
    else:
        xml_path = "walker2d.xml"

    print(f"Reference shape: {reference.shape}  "
          f"({len(reference)/CTRL_HZ:.1f}s @ {CTRL_HZ}Hz)")

    # ── output dir ────────────────────────────────────────────────────
    stamp   = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag     = "cycle" if args.ref_cycle else "full"
    if args.scale_model:
        tag += "_s1scaled"
    log_dir = PROJECT_ROOT / (args.out_dir
                              or f"results/walker2d_phase_{tag}_dm_{stamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {log_dir}")
    np.save(log_dir / "reference.npy", reference)

    # ── env factory ───────────────────────────────────────────────────
    env_kwargs = dict(
        reference          = reference,
        xml_file           = xml_path,
        pose_weight        = args.pose_weight,
        vel_weight         = args.vel_weight,
        ee_weight          = args.ee_weight,
        root_weight        = args.root_weight,
        pose_scale         = args.pose_scale,
        vel_scale          = args.vel_scale,
        ee_scale           = args.ee_scale,
        root_scale         = args.root_scale,
        ref_root_drop      = args.ref_root_drop,
        v_target           = args.v_target,
        pitch_term_thresh  = args.pitch_term,
        swing_pen_weight   = args.swing_pen_weight,
        contact_weight     = args.contact_weight,
        pose_term_thresh   = args.pose_term,
        ankle_term_thresh  = args.ankle_term,
        hip_term_thresh    = args.hip_term,
        xvel_term_thresh   = args.xvel_term,
        pose_joint_weights = tuple(args.pose_joint_weights),
        product_reward     = args.product_reward,
        min_joint_pose     = args.min_joint_pose,
        energy_weight      = args.energy_weight,
        preview_k          = args.preview_k,
    )

    # Save the obs/reward-shape-affecting kwargs for downstream renderer/eval
    # (preview_k changes obs_space; render and eval need to know to build the
    # env with the same shape so PPO.load can wire its input layer).
    import json
    env_kwargs_meta = {
        "preview_k":          args.preview_k,
        "pose_joint_weights": list(args.pose_joint_weights),
        "product_reward":     args.product_reward,
        "min_joint_pose":     args.min_joint_pose,
        "v_target":           args.v_target,
        "ref_root_drop":      args.ref_root_drop,
        "xml_file":           xml_path,
    }
    (log_dir / "env_kwargs.json").write_text(
        json.dumps(env_kwargs_meta, indent=2), encoding="utf-8"
    )
    def make_env():
        def _init():
            return Walker2dPhaseAware(**env_kwargs)
        return _init

    env = SubprocVecEnv([make_env() for _ in range(args.num_envs)])
    env = VecMonitor(env)

    # ── model ─────────────────────────────────────────────────────────
    tb_dir = None if args.no_tb else str(log_dir / "tb")
    if args.finetune:
        path = str(Path(args.finetune).with_suffix(""))
        print(f"Finetuning from: {path}")
        model = PPO.load(path, env=env, device=args.device,
                         tensorboard_log=tb_dir)
        model.learning_rate = 1e-5
        model.ent_coef      = 0.0
        model.target_kl     = 0.005
    else:
        # Linear LR decay 3e-4 → 3e-5.
        def lr_schedule(progress_remaining: float) -> float:
            return 3e-5 + progress_remaining * (3e-4 - 3e-5)

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate = lr_schedule,
            n_steps       = 512,
            batch_size    = 4096,
            n_epochs      = 10,
            gamma         = 0.99,
            gae_lambda    = 0.95,
            clip_range    = 0.2,
            ent_coef      = 0.005,
            vf_coef       = 0.5,
            max_grad_norm = 0.5,
            target_kl     = 0.015,
            seed          = args.seed,
            device        = args.device,
            policy_kwargs = {"net_arch": [256, 256]},
            tensorboard_log = tb_dir,
            verbose       = 0,
        )
    if tb_dir:
        print(f"TensorBoard logs: {tb_dir}  (run: tensorboard --logdir {tb_dir})")

    # ── BC warm-start ─────────────────────────────────────────────────
    if args.bc_epochs > 0 and not args.finetune:
        tmp_env = Walker2dPhaseAware(**env_kwargs)
        print(f"Collecting PD-rollout dataset ({args.bc_steps:,} steps)...")
        obs_bc, act_bc = compute_bc_dataset(
            tmp_env, n_steps=args.bc_steps, kp=args.bc_kp, kd=args.bc_kd
        )
        tmp_env.close()
        print(f"  collected {len(obs_bc):,} (obs, action) pairs  "
              f"action range [{act_bc.min():.2f}, {act_bc.max():.2f}]")
        pretrain_bc(model, obs_bc, act_bc, n_epochs=args.bc_epochs)

        if args.bc_only:
            model.save(str(log_dir / "model_bc"))
            model.save(str(log_dir / "model"))
            print(f"BC-only model saved -> {log_dir}/model_bc.zip")
            env.close()
            return

    checkpoint_cb = CheckpointCallback(
        save_freq  = max(1_000_000 // args.num_envs, 1),
        save_path  = str(log_dir / "checkpoints"),
        name_prefix= "model",
        verbose    = 0,
    )

    total_steps = int(args.total_steps)
    print(f"Training for {total_steps:,} steps with {args.num_envs} envs...")
    model.learn(
        total_timesteps = total_steps,
        callback        = CallbackList([LogCallback(), checkpoint_cb]),
        progress_bar    = True,
    )
    env.close()

    save_path = str(log_dir / "model")
    model.save(save_path)
    print(f"Model saved -> {save_path}.zip")


if __name__ == "__main__":
    main()
