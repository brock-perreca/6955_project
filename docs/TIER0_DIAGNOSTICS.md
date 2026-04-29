# Tier 0 — morphology vs reward diagnostics (2026-04-29)

**Goal.** Decisively answer "is the stiff-hip basin morphology-driven or
reward-driven?" before another reward sweep. The 2026-04-29 overnight 19-experiment sweep tried 8 reward-aggregator/termination ablations, 4
AMP/AIRL warm-starts, 3 preview-obs runs, 1 SAC, and 3 curriculum runs;
all 19 landed in the same stiff-hip basin. The next reward-tuning batch
should not start until we know whether reward changes can in principle
escape it.

**This document covers the O11 box's Tier 0 ledger** (A.1 → A.2 → C
hiprelax). The same morning, **Brock-Asus-Laptop independently
arrived at the same diagnosis** and ran a parallel single-knob
ablation with `walker2d_hipopen.xml` (`thigh_joint range="-30 60"`,
permissive both directions). That work is documented in
[`RESTART_LOG.md § Batch 4`](RESTART_LOG.md). The two ablations are
deliberate brackets: hiprelax has +5° headroom and *undershoots* the
reference peak (hip ROM 17–20° at 5M); hipopen has +60° headroom and
*overshoots* (hip ROM 91° at 2M, narrowed to 63° at 5M). Both
confirm the kinematic-ceiling hypothesis. See
[`assets/mjcf/README.md`](../assets/mjcf/README.md) for picking
between them.

**Working hypothesis (going in).** Stock Walker2d-v4 is fully planar.
Reference hip flexion peaks near +29.7°. Inter-leg collision and/or
joint-range limits in the MJCF may make the reference's hip kinematics
infeasible, in which case no reward fix can dislodge stiff hip.

**Method.** Cheap probes first (dashboard, reference replay extension,
PD ceiling). Decisive experiment last (one retrained run with the
morphology constraint actually relaxed). Every diagnostic that produces
motion saves an mp4 into `docs/figures/tier0/`. The reference replay
mp4 is mirrored there as `00_reference_replay.mp4` so all videos sit
in one folder.

**Cross-validation rule.** Where I compute a metric (e.g. hip ROM from
a rollout), I sanity-check against an independent computation — and
against the dashboard's phase-overlay panel. The overnight failed
partly because scalar metrics ("hip ROM 6.7°") read sporadic kicks as
flexion; that mistake is not repeated here.

---

## Pre-experiment static analysis (2026-04-29, before any training)

Two findings against the user's pre-experiment framing, derived from
the live MuJoCo model and the on-disk reference:

### Finding 1 — inter-leg collisions are *already filtered out*

The user's framing assumed leg-leg collision pressure was driving a
low-collision basin. The MJCF disagrees. Both `walker2d.xml` (gym
default, used by the live runs) and `assets/mjcf/walker2d_custom.xml`
have body geoms with `contype=1, conaffinity=0`; only `floor` has
`conaffinity=1`. MuJoCo's filter is
`(contype1 & conaffinity2) | (contype2 & conaffinity1)`; for any
leg–leg pair this evaluates to `(1&0)|(1&0) = 0`, so the pair is
pruned in broadphase before geometry tests.

Empirical confirmation: forcing thigh_r to +20° and thigh_l to −20°
(deliberately crossed) with `mj_forward` reports `ncon = 0`. There is
no inter-leg contact pressure in either MJCF. The "policy is sitting
in a low-collision basin" hypothesis is falsified by the contact
filter — collision was never enforced. Experiment 3 in the original
candidate list (flip `conaffinity`) is therefore a no-op against the
current state.

### Finding 2 — hip joint range is the active morphology constraint

Stock `walker2d.xml`: `thigh_joint range="-150 0"` — **the hip cannot
flex forward at all in stock Walker2d-v4**, only extend backward.
`walker2d_custom.xml`: `thigh_joint range="-150 20"` — the +20° user
mentioned, still 9.7° short of the reference peak.

| reference (degrees) | stock walker2d.xml | walker2d_custom.xml |
|---|---|---|
| hip_r ∈ [−13.39, +29.69] | range [−150, **0**] | range [−150, **20**] |
| hip_l ∈ [−13.44, +29.94] | range [−150, **0**] | range [−150, **20**] |

The active runs (`xvel-5M`, all overnight runs) use stock
`walker2d.xml`. So **the model literally cannot reach ~half of every
reference cycle** — every frame where `q_ref[hip] > 0` is outside the
joint range, and MuJoCo's `mj_jntrange` constraint forces clamp the
hip back to 0° regardless of policy command. The env clips RSI to
`_JNT_HI[hip]=+31.5°` (`ppo_walker2d_phase.py:115-116`), but those are
aspirational — joint limits are enforced by dynamics every step.

The reference-replay video looks fine because that script writes qpos
and calls `mj_forward` *without dynamics*; joint limits aren't
enforced in pure FK display, so the replay is showing a kinematic
target the trained policy can't actually produce.

This reframes the morphology hypothesis: **the policy may be sitting
in a stiff-hip basin because the only basin where it can track the
reference is the half-cycle where the reference is in-range.** It's
testable per-frame, decisive, and bypasses the "sporadic-kicks-read-
as-ROM" trap of the overnight metrics.

---

## Experiment ledger

Each entry: hypothesis → what was tested → observed result with
numbers → video filename → dashboard PNG filename → verdict.

### A.1 — Reference vs joint-limit overlay (DONE — decisive)

**Hypothesis.** If hip_r/hip_l are out-of-range a large fraction of
the cycle in stock `walker2d.xml`, morphology is decisively implicated
before any training is run.
**What it tests.** Per-frame, per-joint: is `q_ref[t, j]` inside
`m.jnt_range[j]` for the active MJCF?
**Tool.** `src/diagnostics/check_reference_jnt_range.py` — pure static
check, no policy/PD/dynamics. Runs in <1s. Plot panels show ref trace
with `jnt_range` band shaded; per-joint % outside the cycle and peak
overshoot reported in the panel and as JSON.

**Result — stock `walker2d.xml` (the active runs):**

| joint   | ref_lo  | ref_hi  | jnt_lo  | jnt_hi | % cycle outside | peak overshoot |
|---|---|---|---|---|---|---|
| **hip_r**   | −13.49° | **+29.69°** | −150.0° | **+0.0°**  | **68.6 %** | **+29.69°** above |
| knee_r  | −64.64° | +0.04°  | −150.0° | +0.0°  | 0.7 %  | +0.04° (spline noise) |
| ankle_r | −23.41° | +6.19°  | −45.0°  | +45.0° | 0.0 %  | — |
| **hip_l**   | −13.45° | **+29.97°** | −150.0° | **+0.0°**  | **67.9 %** | **+29.97°** above |
| knee_l  | −62.34° | −0.19°  | −150.0° | +0.0°  | 0.0 %  | — |
| ankle_l | −17.09° | +10.92° | −45.0°  | +45.0° | 0.0 %  | — |

**Result — `walker2d_custom.xml` (the +20° MJCF, NOT in active use):**

| joint | % outside | peak overshoot |
|---|---|---|
| hip_r | 40.0 % | +9.69° above |
| hip_l | 38.6 % | +9.97° above |

**Verdict.** Decisive. The active MJCF's hip range physically forbids
the reference for ~68 % of every cycle, with up to a +30° overshoot.
Knees and ankles fit. No reward change can dislodge the stiff-hip
basin while the joint range is `[−150°, 0°]`, because MuJoCo's
`mj_jntrange` constraint forces clamp the joint independent of the
policy command. The "stiff hip" basin is at least partially the only
half-cycle the model can reach; this matches every metric we've seen
(hip ROM 1.6–3° across all 19 overnight runs, ~5° on the xvel-5M
checkpoint with mid-cycle clipping). The custom MJCF is materially
better but still ~10° short.

**Output.**
- `docs/figures/tier0/A1_reference_vs_jnt_range_walker2d.png`
- `docs/figures/tier0/A1_reference_vs_jnt_range_walker2d.json`
- `docs/figures/tier0/A1_reference_vs_jnt_range_walker2d_custom.png`
- `docs/figures/tier0/A1_reference_vs_jnt_range_walker2d_custom.json`

### A.2 — Dashboard on xvel-5M (DONE — flat-topped at the upper limit)

**Hypothesis (refined per user 2026-04-29).** Is sim hip(t)
**flat-topped at the upper joint limit** (range expansion is the
experiment) or **oscillating well below the limit** (the wall isn't
binding and we still have a reward problem on top of any range issue)?

**Setup.** Confirmed `xvel-5M` was trained on stock `walker2d.xml`
(commit ccefe5e: "Batch-2 single-knob ablations on stock walker2d.xml";
no `--scale_model` was passed, training default is stock).
`env_kwargs.json` was empty `{}` for this run (the file pre-dates the
overnight machinery), so dashboard/eval rebuild the env at the
training defaults — same MJCF the run used.

Confirmed at the live model: `m.jnt_range[hip_r] = [-150.00°, +0.00°]`
and same for `hip_l`. Then deterministic rollouts × 5 seeds × 600 steps,
hip qpos and reference phase logged each step.

**Result.**

| metric (hip_r, 5 seeds × 600 steps = 2487 frames) | value |
|---|---|
| min                                  | −23.55° |
| max                                  | +23.05° |
| **mean**                             | **+1.23°** |
| **median**                           | **+1.39°** |
| std                                  | 2.22° |
| **frames within 0.5° of upper limit (+0°)** | **95.3 %** |
| **frames above upper limit**          | **93.45 %** |

(`hip_l` mean +1.21°, median +1.36° — same picture.)

The min/max tails (−23.55°, +23.05°) are sporadic startup transients
in the first ~10 steps (RSI sets `qpos[3:9] = ref[phase]`, and
reference hip can be ±25°; it takes the policy a handful of frames to
collapse back to the basin). The bulk distribution is a 2-3°-wide band
parked at the upper joint limit. **Per-cycle hip ROM is 0.77° (right)
and 1.86° (left)** — the dashboard PNG numbers, computed inside an
honest R-strike→R-strike window.

The phase-split is also unambiguous: during reference-positive frames
(swing/forward, q_ref > 0), sim hip mean = +1.61°; during
reference-non-positive frames (stance/extension), sim hip mean =
+1.25°. The policy is **barely modulating with the reference** — it's
parked at the wall regardless of phase.

That `mean = +1.23°` exceeds the nominal `+0°` upper limit by ~1° is
expected: MuJoCo's `mj_jntrange` is a **soft constraint** (default
solref/solimp give a small impedance band). The trained policy is
applying enough hip-flexion torque to sit slightly inside the
soft-wall band; the equilibrium is "torque vs. constraint impedance,"
not "free oscillation below the limit."

**Trace shape — picture is unambiguous.** See
`docs/figures/tier0/A2_xvel-5M_hip_trace.png`. The black ref trace
sweeps full ±15° on `hip_r` and 0–30° on `hip_l`; the blue sim trace
is a flat line glued to the green +0° upper limit, with the only
non-flat-line excursions being the first 10 RSI-decay steps.

**Verdict.** **Flat-topped at the upper limit.** The wall is binding.
**Range expansion is the right experiment for C** — the policy can't
get further forward than the joint range allows, and the reward
gradient is actively pinning it against the wall (not stuck below it).
This is the cleanest possible signal for "morphology, not reward" as
the dominant constraint right now. Once the wall is gone, we'll see
whether the reward gradient still drives the policy into the right
range — that's the secondary question C answers.

**Outputs.**
- `docs/figures/tier0/A2_xvel-5M_dashboard.png`
- `docs/figures/tier0/A2_xvel-5M_hip_trace.png`

### A.3 — Inter-leg contact instrumentation (planned)

**Hypothesis.** I expect zero inter-leg contacts (Finding 1 above);
the dashboard rollout should prove it rather than assume.
**Status.** TBD.

### B — PD-tracking ceiling (planned)

**Hypothesis.** If the joint-range ceiling is binding, even a strong
PD controller cannot break ~5° of hip ROM regardless of reward.
**Status.** TBD.

### C — Hip-relaxed MJCF retraining (DONE — mixed verdict)

**Hypothesis.** Single-knob morphology mod: relax `thigh_joint` range
to `-150 35` (covers reference +29.97° peak with ~5° headroom).
Retrain the xvel-5M recipe verbatim with the relaxed MJCF. Three
seeds in parallel (11, 12, 13) for seed-fragility insurance — the
overnight showed seed-dependent behavior is real.

**Setup.**
- New MJCF: `assets/mjcf/walker2d_hiprelax.xml`. **Only difference**
  from gym-bundled `walker2d.xml`: `thigh_joint` and
  `thigh_left_joint` `range="-150 35"`. Knees, ankles, geometry,
  actuators, contact bits all unchanged.
- Recipe: `--xvel_term 0.3 --num_envs 8 --total_steps 5000000` (the
  xvel-5M recipe verbatim) plus the new `--xml walker2d_hiprelax.xml`
  override. No reward / hyperparameter changes — this is a
  single-variable morphology ablation.
- Output dirs: `results/restart_b4_hiprelax_s{11,12,13}/`.
- Pre-A.1 reachability check on the relaxed MJCF: `0.0 %` of cycle
  out-of-range on both hips (peak overshoot 0.0°, vs +30° on stock).

**Tooling shipped for C.**
- `--xml` flag added to `ppo_walker2d_phase.py` (general MJCF
  override; was previously only `--scale_model`).
- `env_kwargs.json` now records `xml_file` at training time;
  `run_dashboard.py`, `eval_biomech.py`, and `render_phase.py` prefer
  the saved value over their CLI default. CLI `--xml` is still the
  explicit override.
- `scripts/tier0/evaluate_C.py` produces dashboards (final + 2M
  ckpts), `eval_biomech` JSON, MP4s, a 5-seed × 600-step hip-trace
  comparison panel, and a markdown summary. Runs against whatever
  seeds are present (skips missing).

**Validation plan.**
- Per-seed dashboard PNG (final + 2M ckpts).
- 5-seed × 600-step hip-trace comparison panel
  (`C_hip_trace_comparison.png`) — same probe as A.2, applied to all
  three seeds plus xvel-5M baseline.
- 600-step MP4 per seed-final, plus xvel-5M baseline + reference
  replay copies, all in `docs/figures/tier0/C_hiprelax/`.
- `eval_biomech` (6 eps × 2500 steps) per seed.

**Success signature (per user 2026-04-29).** `hip_r(t)` follows the
reference curve in **shape AND amplitude** (not just peak), and the
flat-topped clamping is gone. If trajectories sweep but stay
amplitude-truncated, reward is binding on top of range. If trajectories
fully track, this was purely morphology.

**Failure signature to watch.** Hip swings emerge but cause the
policy to fall (instability or — if it surfaces — leg interpenetration
unmasked by larger swings). That's a downstream finding worth knowing,
not a reason to abort.

**Result.** All three seeds completed (2026-04-29 13:51, ~54 min wall
under 24-env / 16-core contention). All three produce **qualitatively
identical** policies: clean periodic hip oscillation that tracks the
reference shape and frequency, but with truncated amplitude.

**Numbers (median over 6 deterministic eval episodes × 2500 steps):**

| metric | xvel-5M (stock) | hiprelax_s11 | hiprelax_s12 | hiprelax_s13 | reference |
|---|---|---|---|---|---|
| ep_len_steps        | 2500 | 2500 | 2500 | 2500 | — |
| stride_period_s     | 0.327 | 0.361 | 0.347 | 0.371 | **1.120** |
| cadence (steps/min) | 367.6 | 332.7 | 346.3 | 323.6 | **107.1** |
| double_support_frac | 0.074 | 0.018 | 0.019 | 0.010 | **0.227** |
| **hip_r ROM (deg)** | **1.77** | **19.79** | **19.92** | **16.50** | **45.4** |
| **hip_l ROM (deg)** | **1.94** | **15.27** | **16.56** | **18.52** | **45.4** |
| knee_r ROM (deg)    | 22.18 | 26.57 | 38.56 | 32.70 | **65.7** |
| LR_stride_asymmetry | 0.138 | 0.097 | 0.143 | 0.122 | < 0.10 |
| peak_vgrf_bw        | 3.28 | 3.97 | 4.02 | 4.18 | **1.10** |
| progress_score (0–4)| 2.31 | 2.41 | 2.19 | 2.11 | 4.00 |

**Hip-trace probe (5 seeds × 600 steps, same as A.2):** mean +14.7°
to +16.2° (xvel-5M was +1.4°), std 17.0°-19.4° (xvel-5M 2.2°), %
within 0.5° of the new +35° upper limit only 15-21% (xvel-5M was 95%
within 0.5° of +0°). The wall is no longer binding; it's a
soft-bumper occasionally touched, not a hard ceiling.

**Verdict: MIXED (range + reward) — both fixes needed.**

The morphology hypothesis is **strongly confirmed**:
- **Hip ROM grew ~10×** (1.8° → 17-20° across seeds).
- xvel-5M's flat-topped clamping (95% within 0.5° of +0°) is **gone**;
  only 15-21% of frames touch the new +35° limit.
- The trace tracks the reference *shape and frequency* across all
  three seeds (see `C_hip_trace_comparison.png`).
- Visual review: see `hiprelax_s{11,12,13}_final.mp4` next to
  `xvel-5M_final.mp4` and `00_reference_replay.mp4`.

But the reward is **still binding on top of morphology**:
- Hip ROM is **~40 % of reference** (17-20° vs 45.4°). The trace tracks
  shape but with truncated peaks.
- Cadence is still **~3× too fast** (~330 vs 107.1 steps/min), and
  *worse* on hiprelax than xvel-5M relative to the reference (333 ≪
  368, but the gap to 107 is similar).
- Peak vGRF/BW *increased* from 3.28 to ~4.0 — the body is hitting
  the floor harder per stride, the running-not-walking signature.
- Progress score is essentially unchanged (~2.3 across all four
  policies).
- Action saturation at ±1 stays at 60-77% per joint (dashboard panels
  3-4) — bang-bang control consistent with the engineered xvel-5M
  reward gradient with no `--energy_weight` brake.
- `r_pose` is partially saturated by tracking knee/ankle well; the
  hip's amplitude shortfall isn't expensive enough to push the policy
  to fully extend.

**This is the "Mixed" branch: Tier 1's reward reform is still the
right next step, but it must run on top of the relaxed MJCF, not
stock walker2d.xml.** With the wall gone, restoring
`forward_reward = exp(-3·(v-1.25)²)` and dropping `xvel_term` should
push cadence down toward the reference (since drift-at-0.4 m/s no
longer maxes out reward), and slower cadence should naturally pull
hip amplitude up toward reference.

**Outputs.**
- `docs/figures/tier0/C_hiprelax/`:
  - `00_reference_replay.mp4` (kinematic ceiling baseline, mirrored)
  - `xvel-5M_final.mp4`, `hiprelax_s{11,12,13}_final.mp4`
  - `xvel-5M_final_dashboard.png`,
    `hiprelax_s{11,12,13}_{final,2000000}_dashboard.png`
  - `hiprelax_s11_hip_trace.png` (3-seed overlay vs ref + new limit)
  - **`C_hip_trace_comparison.png`** (the cleanest single artifact —
    xvel-5M flat-topped at +0° next to all three relaxed seeds
    sweeping through ±20°)
  - `C_summary.md` (the table above)
  - `hiprelax_s{11,12,13}_eval.json`, `xvel-5M_eval.json`

---

## Headline finding (after A.1 + A.2, before C)

**Stock `walker2d.xml` hip range is `[-150°, +0°]`. Reference peak is
+29.97°. ~68 % of every gait cycle is outside the joint range.** Every
restart batch *and* all 19 overnight experiments trained against a
target the body literally could not reach. xvel-5M's measured `hip_r`
is parked at +0° — std 2.22°, and **95.3 % of frames within 0.5° of
the upper limit**. The policy is actively pushing into the wall.
**The stiff-hip basin was a kinematic ceiling, not a reward trap.**

**C tests range expansion to `[-150°, +35°]`** — a single-knob
morphology ablation, otherwise the verbatim xvel-5M training recipe.

This is the most important artifact of the session. Documented before
C runs so that if C confirms, the pivot conversation reaches for it
immediately; if C falsifies, the note still records a real bug that
affected every prior experiment.

## Verdict — MIXED (range + reward), both fixes needed

The 2026-04-28 → 2026-04-29 stiff-hip basin had **two stacked causes**,
not one:

1. **Morphology was binding (the dominant cause for xvel-5M).** Stock
   walker2d.xml's `thigh_joint range="-150 0"` made ~68 % of every
   reference cycle physically unreachable. xvel-5M's hip was parked
   at the +0° upper limit for 95.3 % of frames, std 2.22°. C confirms:
   relaxing the range to `-150 35` produces **clean periodic hip
   tracking across all three seeds**, hip ROM 10× xvel-5M (1.8° →
   17-20°), and the flat-topped clamping signature disappears.

2. **Reward is still binding on top of morphology.** Even with the
   wall gone, the relaxed-MJCF policies hit only ~40 % of the
   reference's hip ROM, ~3× the reference cadence, and ~4× the peak
   vGRF/BW. The hip trace tracks reference *shape and frequency* but
   amplitude is truncated. This is consistent with the analysis that
   `xvel_term=0.3` rewards survival floor at any forward velocity ≥
   0.31 m/s, so the policy converges to fast, low-amplitude strides
   (the body translates without committing to full hip excursion).

## Recommendation for Tier 1

**Both fixes, stacked.** A relaxed-hip MJCF is the new baseline;
Tier 1's "restore `forward_reward`, drop `xvel_term`" should run on
top of it, not on stock walker2d.xml. Running it on **both**
`walker2d_hiprelax.xml` *and* `walker2d_hipopen.xml` is the cleanest
experimental design — the two MJCFs bracket the residual reward
gap from below and above:

- on **hiprelax** the policy currently *undershoots* (17–20° vs 45°);
  a peaked forward-velocity reward should slow cadence, give the hip
  more time per stride, and grow amplitude *toward* reference.
- on **hipopen** the policy currently *overshoots* (63° at 5M);
  the same reward change should narrow ROM *toward* reference.

If both converge near 45° at cadence ~110 steps/min, reward was the
dominant remaining cause. If both stay where they are, the trap is
deeper than reward+range — candidates: gait-cycle frame-rate
mismatch, phase-observation rate, body inertia scaling vs subject
mass, or a missing energy/torque brake. That's a Tier 2 diagnostic,
not a Tier 1 risk.

The four current candidate "best" policies on disk (Brock has not
picked a single favorite yet) are:

- **hipopen track** — `restart_b4_hipopen_5M/`,
  `restart_b5_pose_scale20/`, `restart_b5_min_joint/`
- **hiprelax track** — `restart_b4_hiprelax_s11/`

See [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for per-run metrics.

## What every future session must know

- **Pre-2026-04-29 results (everything in `restart_b{1,2}_*` and
  `overnight_20260429-0211/`) trained against a target ~half of which
  was unreachable.** Hip-ROM-near-zero in those runs is the joint
  range, not the reward. Don't try to reward-tune those models.
- **Two new training MJCFs replace stock walker2d.xml:**
  `assets/mjcf/walker2d_hiprelax.xml` (this machine, +5° headroom) and
  `assets/mjcf/walker2d_hipopen.xml` (Asus laptop, +60° headroom).
  Pass either as `--xml <name>` to the training script (the flag
  was added during this session — both machines added it
  independently and the implementations were merged). `env_kwargs.json`
  now records `xml_file`, so render/eval/dashboard pick up the right
  MJCF automatically for runs trained after 2026-04-29.
- **`src/diagnostics/check_reference_jnt_range.py` is the
  reachability gate.** Run it whenever the MJCF or reference
  changes. `render_reference_replay.py` calls `mj_forward` without
  dynamics so it does NOT enforce joint limits and cannot catch
  this class of bug.
