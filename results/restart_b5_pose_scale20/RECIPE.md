# restart_b5_pose_scale20 — recipe + headline metrics

**Status (2026-04-29):** Batch 5 Variant A. Numerically narrows hip
ROM (the tightest of the three) but visually indistinguishable from
the `b4_hipopen_5M` baseline on Brock's A/B review. Kept as a
comparison point. See
[`docs/RESTART_LOG.md § Batch 5`](../../docs/RESTART_LOG.md#batch-5--2026-04-29--narrow-the-hipopen-over-flex--partial-positive-both-variants)
for the full setup/observation entry.

## Reproduce

```
python src/walker2d/ppo_walker2d_phase.py \
    --ref_cycle assets/reference/gait_cycle_reference.npy \
    --xml walker2d_hipopen.xml \
    --num_envs 8 --total_steps 5000000 \
    --xvel_term 0.3 --pose_scale 20 \
    --seed 7 \
    --out_dir results/restart_b5_pose_scale20
```

Single-knob change vs `restart_b4_hipopen_5M`: `--pose_scale 20`
doubles the exponential sharpness of `r_pose = exp(-k·mean(diff²))`
from k=10 to k=20. A 10° overshoot now costs ~10% of pose reward
instead of ~5%. Mean aggregator preserved.

## Files in this dir

- `model.zip` — final SB3 PPO policy (5M steps).
- `checkpoints/model_{1,2,3,4,5}000000_steps.zip` — intermediate.
- `reference.npy` — single-cycle reference (same as `b4_hipopen_5M`).
- `env_kwargs.json` — env config persisted at training time.
- `tb/PPO_1/events.out.tfevents.*` — TensorBoard scalars.
- `train_stdout.txt` — training console log (per-iteration ep_r,
  ep_len; same data as TB but easier to grep).
- `eval_hip_rom.txt` — `scripts/eval_hip_rom.py` output (alongside
  baseline + min_joint).

## Headline metrics (eval, 4 deterministic episodes × 1000 steps)

| metric | value | reference / target | vs b4_hipopen_5M |
|---|---|---|---|
| episodes survived | 1000 × 4 | (cap = 1000) | unchanged |
| mean fwd velocity (m/s) | 1.354 | 1.25 | 1.395 → 1.354 (closer) |
| hip_r ROM (deg) | **56.58** | 43.18 | 63.23 → 56.58 (tightest) |
| hip_l ROM (deg) | 58.10 | 43.30 | 59.35 → 58.10 |
| hip_r min / max (deg) | -16.70 / +39.88 | -13.49 / +29.69 | tighter min, similar max |
| hip_l min / max (deg) | -21.94 / +36.16 | -13.44 / +29.94 | similar min, tighter max |
| % steps within 1° of upper hip limit | 0.0% | — | unchanged |

Final training ep_rew ~4153 / ep_len ~6211 (peak iter 1200) — the
sharper-but-still-forgiving aggregator is genuinely easier to
optimise than baseline at the same gait, hence higher headline
training numbers without a visibly different policy.

## Render

```
python src/walker2d/render_phase.py --eps 1 --steps 1000 \
    --mp4 docs/figures/restart_b5_pose_scale20.mp4 \
    results/restart_b5_pose_scale20:final:pose_scale20
```

Pre-rendered mp4: `docs/figures/restart_b5_pose_scale20.mp4`.
