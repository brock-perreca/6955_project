# restart_b4_hipopen_5M — recipe + headline metrics

**Status (2026-04-29):** current best policy. See
[`docs/RESTART_LOG.md § Batch 4`](../../docs/RESTART_LOG.md#batch-4--2026-04-29--joint-range-hypothesis-open-hip-mjcf--positive)
for the full setup/observation entry.

## Reproduce

```
python src/walker2d/ppo_walker2d_phase.py \
    --ref_cycle assets/reference/gait_cycle_reference.npy \
    --xml walker2d_hipopen.xml \
    --num_envs 8 --total_steps 5000000 \
    --xvel_term 0.3 \
    --seed 6 \
    --out_dir results/restart_b4_hipopen_5M
```

Single-knob change vs the pre-batch-4 baseline (`restart_b2_xvel`):
custom MJCF `assets/mjcf/walker2d_hipopen.xml` opens the
`thigh_joint` range from the gym default `[-150°, 0°]` to
`[-30°, +60°]`, making the reference's +30° hip-flexion peaks
physically reachable.

## Files in this dir

- `model.zip` — final SB3 PPO policy (5M steps).
- `checkpoints/model_{1,2,3,4,5}000000_steps.zip` — intermediate.
- `reference.npy` — single-cycle reference the policy was trained
  against (140 frames × 6 joints; resampled from the 56-frame
  Ulrich Subject-1 IK at 50 Hz to 125 Hz cubic).
- `env_kwargs.json` — env config persisted at training time so
  `render_phase.py` and `eval_hip_rom.py` build the env with the
  same shape/MJCF the policy is wired against.
- `tb/PPO_1/events.out.tfevents.*` — TensorBoard scalars.
- `eval_hip_rom.txt` — `scripts/eval_hip_rom.py` output for this
  run (alongside the two batch-5 variants).

## Headline metrics (eval, 4 deterministic episodes × 1000 steps)

| metric | value | reference / target |
|---|---|---|
| episodes survived | 1000 × 4 | (cap = 1000) |
| mean fwd velocity (m/s) | 1.395 | 1.25 |
| hip_r ROM (deg) | 63.23 | 43.18 |
| hip_l ROM (deg) | 59.35 | 43.30 |
| hip_r min / max (deg) | -23.16 / +40.07 | -13.49 / +29.69 |
| hip_l min / max (deg) | -17.70 / +41.65 | -13.44 / +29.94 |
| % steps within 1° of upper hip limit | 0.0% | — |

Over-flexes ~10° at the swing-forward peak; pose-tracking forgives
one overshooting joint when the others track. Visually walks
robustly at near-treadmill speed.

## Render

```
python src/walker2d/render_phase.py --eps 1 --steps 1000 \
    --mp4 docs/figures/restart_b4_hipopen_5M.mp4 \
    results/restart_b4_hipopen_5M:final:b4_hipopen_5M
```

`render_phase.py` auto-loads the trained-against MJCF from
`env_kwargs.json`; no `--xml` needed. Pre-rendered mp4 lives at
`docs/figures/restart_b4_hipopen_5M.mp4`.
