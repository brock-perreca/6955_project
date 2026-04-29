# `<EXP_NAME>` — <one-line hypothesis>

| field | value |
|---|---|
| run_dir | `<results/overnight_*/exp_name/>` |
| based_on | `<parent run if any, e.g. results/restart_b2_xvel/model.zip>` |
| code change | `<branch name or "CLI flags only">` |
| train wallclock | `<seconds>` |
| seeds | `<N>` |

## Hypothesis

<2–4 sentences on what this experiment is *trying to disprove*. Be specific
about which mechanism this targets (e.g. "stiff hips because mean-of-squares
pose reward lets 5/6 joints earn full credit").>

## Setup

<List the exact CLI flags / code edits that differ from the parent. Anything
implicit is wrong here.>

## Headline numbers

| metric | value | reference / parent |
|---|---|---|
| `rollout/ep_rew_mean` (final) |  |  |
| `rollout/ep_len_mean` (final) |  |  |
| `reward/r_pose` (final)       |  |  |
| `reward/r_vel`  (final)       |  |  |
| `reward/r_ee`   (final)       |  |  |
| `reward/r_root` (final)       |  |  |

## Held-out biomech (from `eval_biomech.json`)

| metric | this run | parent / reference |
|---|---|---|
| ep_len_steps (median)         |  | 2500 cap |
| n_strides_detected (median)   |  | 17 in 20s |
| stride_period_s               |  | **1.120** (ref) |
| cadence_steps_per_min         |  | ~107 (ref) |
| double_support_frac           |  | 0.20–0.30 |
| swing_drag_frac               |  | 0.0 |
| LR_stride_asymmetry           |  | < 0.10 |
| peak_vgrf_bw                  |  | ~1.2 |
| hip_knee_dtw                  |  | lower is better |

## Anti-Goodhart check

For each of the four canonical exploits the project has hit before, state
whether the eval video shows it: **yes / no / partial / can't tell from
metrics alone**.

- [ ] Stand-and-wiggle (low x-vel, partial joint tracking, long ep_len): …
- [ ] Stiff hip (hip_r excursion < 15° while reference sweeps ~43°): …
- [ ] Ankle paddling (high ankle vel, no foot lift): …
- [ ] One-legged hopping / asymmetry (LR_stride_asymmetry > 0.5): …
- [ ] Toe-walking (permanent plantarflexion, short stride): …

## Diagnosis

<2–6 sentences. What did the change actually do? Where did the policy land?
If the headline numbers improved but a specific exploit reappeared, *say so
explicitly* — a high reward number is not a result.>

## Verdict

**KEEP / DROP / FOLLOW-UP**

- `KEEP`     — better than parent on the metric of interest, no new exploit.
- `DROP`     — worse, or a new exploit appeared.
- `FOLLOW-UP` — interesting partial result; a specific next experiment is
                worth queueing. Name it.
