# restart_b5_min_joint — recipe + headline metrics

**Status (2026-04-29):** Batch 5 Variant B. Numerically pulls fwd
velocity to essentially target (1.231 m/s vs 1.25) and narrows hip
ROM, but visually indistinguishable from the `b4_hipopen_5M`
baseline on Brock's A/B review. Kept as a comparison point. See
[`docs/RESTART_LOG.md § Batch 5`](../../docs/RESTART_LOG.md#batch-5--2026-04-29--narrow-the-hipopen-over-flex--partial-positive-both-variants)
for the full setup/observation entry.

## Reproduce

```
python src/walker2d/ppo_walker2d_phase.py \
    --ref_cycle assets/reference/gait_cycle_reference.npy \
    --xml walker2d_hipopen.xml \
    --num_envs 8 --total_steps 5000000 \
    --xvel_term 0.3 --min_joint_pose \
    --seed 8 \
    --out_dir results/restart_b5_min_joint
```

Single-knob change vs `restart_b4_hipopen_5M`: `--min_joint_pose`
flips the pose aggregator from arithmetic-mean to worst-joint
floor: `r_pose = min_j exp(-k · w_j · diff_j²)`. One bad joint
kills the whole pose reward, directly attacking the 5-of-6-joint
loophole that lets the policy hide a single overshooting hip
behind 5 compliant joints.

## Files in this dir

- `model.zip` — final SB3 PPO policy (5M steps).
- `checkpoints/model_{1,2,3,4,5}000000_steps.zip` — intermediate.
- `reference.npy` — single-cycle reference (same as `b4_hipopen_5M`).
- `env_kwargs.json` — env config persisted at training time.
- `tb/PPO_1/events.out.tfevents.*` — TensorBoard scalars.
- `train_stdout.txt` — training console log (per-iteration ep_r,
  ep_len; same data as TB but easier to grep).
- `eval_hip_rom.txt` — `scripts/eval_hip_rom.py` output (alongside
  baseline + pose_scale20).

## Headline metrics (eval, 4 deterministic episodes × 1000 steps)

| metric | value | reference / target | vs b4_hipopen_5M |
|---|---|---|---|
| episodes survived | 1000 × 4 | (cap = 1000) | unchanged |
| mean fwd velocity (m/s) | **1.231** | **1.25** | 1.395 → 1.231 (~target) |
| hip_r ROM (deg) | 57.08 | 43.18 | 63.23 → 57.08 |
| hip_l ROM (deg) | 56.49 | 43.30 | 59.35 → 56.49 |
| hip_r min / max (deg) | -19.21 / **+37.87** | -13.49 / +29.69 | lowest peak hip flexion of the three |
| hip_l min / max (deg) | -20.46 / +36.02 | -13.44 / +29.94 | similar |
| % steps within 1° of upper hip limit | 0.0% | — | unchanged |

Final training ep_rew ~2071 / ep_len ~3389 (final iter 1200) —
lower than baseline because the worst-joint floor genuinely makes
pose tracking harder. The lower training reward is *expected and
correct*, not a regression: the loophole closed.

## Render

```
python src/walker2d/render_phase.py --eps 1 --steps 1000 \
    --mp4 docs/figures/restart_b5_min_joint.mp4 \
    results/restart_b5_min_joint:final:min_joint
```

Pre-rendered mp4: `docs/figures/restart_b5_min_joint.mp4`.
