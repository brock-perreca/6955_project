# Project status — current snapshot

**Purpose:** what's running *right now* — the active code, the current
best policy, what's known broken, and where the next move is queued.
**Read this when:** opening the project for the first time, or coming
back after a few days away.
**Adjacent:** [`PROJECT_TIMELINE.md`](PROJECT_TIMELINE.md) for the
chronological story · [`RESTART_LOG.md`](RESTART_LOG.md) for the most
recent batches with full setup/observation/render commands ·
[`ROADMAP.md`](ROADMAP.md) for what's queued next ·
[`../report/writeup_filled_1.docx`](../report/writeup_filled_1.docx) for
the formal joint writeup with Brian.

*Last updated: 2026-04-29 (post biomechanical-realism scorecard run
across all four candidates — see "Biomechanical-realism finding"
below).*

---

## Where we are

**Phase 5b.** The reference data was corrected on 2026-04-28
(`extract_gait_cycle.py` and `ulrich_loader.py` flip the knee only;
hip and ankle now match OpenSim signs — see
[`PROJECT_TIMELINE.md § Phase 5`](PROJECT_TIMELINE.md#phase-5--the-sign-error-discovery-2026-04-28)).
Every PPO/AMP/AIRL run on disk *before* the restart was trained
against a self-contradictory target; those checkpoints and the
pre-restart engineered-reward constants are kept only as a historical
record.

The post-restart pipeline rebuild is in progress. Four batches done
(see [`RESTART_LOG.md`](RESTART_LOG.md) for full details):

- **Batch 2** found the prior best policy (`results/restart_b2_xvel/`)
  via the `--xvel_term 0.3` floor termination. Walks, but with stiff
  hips (~2° ROM vs reference 45°) and 3× cadence as a downstream
  consequence.
- **Batch 3** (2026-04-29 overnight, 19 experiments) tested 8
  reward-aggregator/termination ablations + 4 AMP/AIRL warm-started
  runs + 3 multi-step preview runs + an SAC variant. **All 19 fall
  into the same stiff-hip basin or worse.** Headline read at the time
  was "reward-driven trap"; batch 4 superseded that.
- **Batch 4** (2026-04-29) **diagnosed the stiff-hip basin as a
  joint-range problem in the MJCF**, not a reward problem. Stock
  `walker2d.xml` constrains `thigh_joint` to `[-150°, 0°]`; the
  reference's +30° hip-flexion peaks are unreachable. Opening the
  range to `[-30°, +60°]` (`results/restart_b4_hipopen/`, 2M steps)
  jumped hip ROM from 1.8° to **91.5°**. Every previous batch was
  fighting an unreachable target. Variant B (re-invert hip column,
  keep stock MJCF) only partially escapes — the +13° extension peak
  still exceeds the 0° MJCF limit, producing a brittle deep-extension
  kicking gait. See [`RESTART_LOG.md § Batch 4`](RESTART_LOG.md#batch-4--2026-04-29--joint-range-hypothesis-open-hip-mjcf--positive).
- **Batch 5** (2026-04-29) ran the two queued aggregator-narrowing
  ablations on top of `hipopen`: `--pose_scale 20` (sharper mean
  aggregator) and `--min_joint_pose` (worst-joint floor). Both
  numerically narrowed the gait — hip ROM 63° → 57° and `min_joint`
  pulled fwd vel from 1.40 m/s to 1.231 m/s — but **visual A/B
  (Brock) found all three policies (b4_hipopen_5M, b5_pose_scale20,
  b5_min_joint) look essentially the same.** All three retained as
  comparison points; none later promoted. See
  [`RESTART_LOG.md § Batch 5`](RESTART_LOG.md#batch-5--2026-04-29--narrow-the-hipopen-over-flex--partial-positive-both-variants).
- **Batch 6** (2026-04-29) ran the held-out biomechanical-realism
  scorecard across all four post-Tier-0 candidates plus the
  pre-Tier-0 `b2_xvel` baseline. **None of the candidates produces
  a biomechanically realistic gait** (DSF ~1% vs ref 23%, peak vGRF
  3.7–4.8 BW vs ref 1.10, cadence 178–210 spm vs ref 107). After a
  same-day strike-detector fix to `eval_biomech.py` (see
  "Biomechanical-realism finding" below), **`b5_min_joint` was
  named as the lead policy** on the corrected scorecard.
  End-of-road on the engineered-reward Walker2d track. See
  [`RESTART_LOG.md § Batch 6`](RESTART_LOG.md#batch-6--2026-04-29--biomechanical-realism-scorecard--end-of-road-lead-named).

**Current lead: `results/restart_b5_min_joint/`** — named 2026-04-29
on the post-strike-detector-fix biomech-realism scorecard. min_joint
wins on the corrected scorecard: highest progress score (**2.66**),
lowest peak vGRF among the post-Tier-0 candidates (3.70 BW), lowest
double-support deviation, per-stride hip ROM 30°. It still misses
every spatiotemporal/kinetic target by 50 %+; it just misses by less
than the alternatives. Pre-fix, the same scorecard had ranked
`b4_hiprelax_s11` first — the strike-detector min_gap artifact
flattered hiprelax's chatter-counted cadence (see next section).

The other three are kept on disk as superseded comparison points,
not as live candidates:

*hipopen track (Brock-Asus-Laptop) — wide bracket
`[-30, +60]`. Superseded.*

- `results/restart_b4_hipopen_5M/` — 5M, seed 6,
  `--xvel_term 0.3`, `--xml walker2d_hipopen.xml`. Per-stride
  median hip ROM 30.6°, score 2.40. **Highest peak vGRF (4.77 BW)
  of the four post-Tier-0 candidates.**
- `results/restart_b5_pose_scale20/` — Batch-5 follow-up,
  `--pose_scale 20`, seed 7. Per-stride hip ROM 30.0°, score 2.26.

  *(Visual A/B between the three hipopen runs found no perceptible
  difference; the per-stride biomech scorecard confirms why — all
  three produce essentially the same bouncing gait at slightly
  different cadences. min_joint edges the other two by sitting at
  ref forward velocity and lowest peak vGRF.)*

*hiprelax track (Brock-O11) — minimal-headroom bracket
`[-150, +35]`. Superseded.*

- `results/restart_b4_hiprelax_s11/` — 5M, seed 11,
  `--xvel_term 0.3`, `--xml walker2d_hiprelax.xml`. xvel-5M recipe
  verbatim except for the relaxed MJCF. Per-stride hip ROM 30.5°,
  score 2.24 — last among the four post-Tier-0 candidates after
  the strike-detector fix. (Pre-fix it ranked first because the
  chatter-counted cadence happened to land closest to ref.)

**Decision (2026-04-29):** the engineered-reward Walker2d track has
been pushed to its useful limit on this stack. Further reward-knob
experiments (peaked-velocity reward, stacked aggregators) are
**deprioritised** based on the scorecard finding — see next section.

---

## Tier 0 finding (2026-04-29) — mixed: kinematic ceiling + reward binding

**Headline.** Stock `walker2d.xml` hip range is `[-150°, +0°]`.
Reference peak is +29.97°. ~68 % of every gait cycle was outside the
joint range. Every restart batch *and* all 19 overnight experiments
trained against a target ~half of which was unreachable. xvel-5M's
hip was parked at +0° — 95.3 % of frames within 0.5° of the upper
limit. Pre-Tier-0, the "stiff-hip basin" was a **kinematic ceiling,
not (only) a reward trap.**

**Two same-day, two-machine confirmations.** Brock-Asus-Laptop ran
Batch 4 with `walker2d_hipopen.xml` (`[-30, 60]`, permissive both
sides); Brock-O11 ran the Tier 0 ablation with
`walker2d_hiprelax.xml` (`[-150, 35]`, +5° headroom only). Both saw
the flat-topping disappear and hip ROM jump roughly an order of
magnitude (1.8° → 91.5° over-flexed for hipopen; 1.8° → 17–20°
under-amplitude for hiprelax). The two variants together bracket
the residual reward bottleneck: hipopen lets the policy explore far
past the reference peak, hiprelax sits just shy of it.

**Tier 0 experiment C result (3 seeds × 5M steps, 2026-04-29 —
Brock-O11).** All three seeds qualitatively identical: clean
periodic hip tracking that matches reference shape and frequency.
The flat-topped clamping disappears, hip-trace median shifts +1.4°
→ +15°. But the policy stalls at ~40 % of reference hip amplitude
with unchanged ~3× cadence and worse peak vGRF/BW.

**Verdict: MIXED — both morphology and reward were binding.**
Morphology was the dominant cause for xvel-5M; reward becomes the
remaining bottleneck once the wall is removed.

**Recommendation for next step.** Tier 1's planned reward reform
(restore `forward_reward = exp(-3·(v-1.25)²)`, drop `xvel_term`
floor) must run on top of a relaxed-hip MJCF. Running it on
**both** hipopen and hiprelax is the cleanest experimental design:
- if hipopen with the new reward narrows toward 45° while hiprelax
  with the new reward grows toward 45°, reward was the dominant
  remaining cause;
- if both stay where they are, the trap is deeper than reward+range
  and we look at gait-cycle frame rate / phase observation as
  Tier 2.

See [`TIER0_DIAGNOSTICS.md`](TIER0_DIAGNOSTICS.md) for the Tier 0
per-experiment ledger (A.1, A.2, C) and
[`docs/figures/tier0/C_hiprelax/`](figures/tier0/C_hiprelax/) for the
videos, dashboards, and the cleanest single artifact —
`C_hip_trace_comparison.png` — which shows xvel-5M's blue trace flat
against the green +0° wall next to all three relaxed seeds sweeping
through ±20°. See [`RESTART_LOG.md § Batch 4`](RESTART_LOG.md) for
the parallel hipopen ablation.

---

## Biomechanical-realism finding (2026-04-29) — end-of-road on the engineered-reward track

**Headline.** Held-out biomechanical scorecard (6 deterministic eps
× 2500 steps each, `eval_biomech.py --targets`) on all four
post-Tier-0 candidates + the pre-Tier-0 `b2_xvel` baseline finds
that **none of the candidates produces a biomechanically realistic
gait.** Every candidate misses every spatiotemporal/kinetic target
by at least 50%, and the post-Tier-0 candidates barely beat the
pre-Tier-0 baseline on the 0–4 progress score.

**Reference values (Ulrich Subject 1, baseline trial, computed by
`extract_reference_biomech.py`):** stride 1.12 s, cadence 107 spm,
double-support 22.7%, peak vGRF 1.10 BW, hip ROM ~45°, knee ROM
~66°, ankle ROM ~40°.

**Per-stride medians across the 5 runs (per-stride median = the
metric biomech papers actually use; full-rollout max−min is
biased upward by sporadic kicks):**

| Run | Score | Stride (s) | Cadence | DSF | vGRF/BW | hip_r ROM |
|---|---|---|---|---|---|---|
| reference | 4.00 | 1.12 | 107 | 0.227 | 1.10 | 45° |
| **`b5_min_joint`** (lead) | **2.66** | 0.63 | 192 | 0.007 | 3.70 | 30.2° |
| `b2_xvel` (pre-Tier-0) | 2.47 | 0.63 | 191 | 0.072 | 3.29 | 2.78° |
| `b4_hipopen_5M` | 2.40 | 0.67 | 179 | 0.017 | 4.77 | 30.6° |
| `b5_pose_scale20` | 2.26 | 0.68 | 178 | 0.021 | 4.48 | 30.0° |
| `b4_hiprelax_s11` | 2.24 | 0.57 | 210 | 0.017 | 4.05 | 30.5° |

**Eval-detector fix (2026-04-29).** The pre-fix table reported
stride ~0.36 s and cadence ~330 spm across the candidates, which
read as "3× too fast" and was the headline of the original
end-of-road framing. That number was a strike-detector artifact:
`eval_biomech._rising_edges` hardcoded `min_gap=25` frames, which
at the 125 Hz sim control rate is only 0.2 s — short enough that
the high-impact contact chatter (4–5 BW slamming, 30 ms bouts)
in these stiff-legged gaits registered as 2–3 separate strikes
per real stride. The reference-extractor used a 0.5-s debounce
(`int(0.5 * grf_hz)` at 50 Hz GRF), so the two sides of the
comparison weren't apples-to-apples. The fix scales the eval's
debounce to `int(0.5 * CTRL_HZ) = 62` frames so both detectors
share the same 0.5-s temporal window. Cached pre-fix artifacts
are kept at `results/biomech_candidates_eval.pre-mingap-fix.json`
and `results/biomech_history.pre-mingap-fix.csv`.

**Three diagnoses fall out of the corrected table:**

1. **Cadence is ~1.7–2.0× too fast, not 3×.** Stride ~0.57–0.68 s
   vs reference 1.12 s; cadence 178–210 spm vs 107 spm. Still off,
   but a real (and smaller) gap, not the order-of-magnitude
   mismatch the pre-fix numbers suggested.
2. **Double-support fraction and peak vGRF are the real smoking
   guns.** Walking has substantial double-support (~23 %); running
   has zero. Every candidate sits at 1–2 % DSF with peak vGRF 3–5
   BW (running peaks ~2.5 BW; walking ~1.1 BW). These are
   bouncing/skipping gaits, not walking — and that's exactly what
   the strike chatter was a symptom of. The kinematics now overlay
   reasonably (hip ROM 30° vs ref 45°, knee ROM tracks within 10%);
   the gait *style* is what's broken.
3. **The lead candidate is `b5_min_joint`, not `b4_hiprelax_s11`.**
   With the corrected detector min_joint scores 2.66 (highest),
   beats both hipopen variants on stride period, double-support,
   and peak vGRF, and matches their hip ROM. `b4_hiprelax_s11` —
   which had been flagged the lead based on the pre-fix score —
   drops to last (2.24): its narrower thigh range (35° headroom
   vs 60°) couples with the stiff-leg slamming to produce shorter
   stride and higher peak vGRF.

**Conclusion.** Phase-conditioned PPO + DeepMimic-style engineered
reward on Walker2d, even with the kinematic-ceiling fix and three
rounds of aggregator/termination tuning, **does not recover human
walking biomechanics**. The reward shape that would close this gap
(double-support incentive, vGRF cap, peaked-velocity target)
amounts to engineering a walking gait by hand — the opposite of
the "imitation-only realistic locomotion" goal in the narrative
arc. Further reward-knob experiments on this stack are
**deprioritised**; the path to biomech-realistic gait runs through
either (a) the AMP/MJX dream (writeup §7.1), or (b) a measured
return to the musculoskeletal track (`src/legacy/musculoskeletal/`).

**Tooling that produced this finding (all under `scripts/` /
`src/diagnostics/`, see [`scripts/README.md`](../scripts/README.md)
and [`src/diagnostics/README.md`](../src/diagnostics/README.md)):**

- `src/diagnostics/extract_reference_biomech.py` — measured Ulrich
  targets (existing).
- `src/diagnostics/eval_biomech.py` — per-policy biomech scorecard
  with `vs_reference` block + 0–4 progress score (existing).
- **`scripts/biomech_realism_dashboard.py` (new, 2026-04-29)** —
  consumes a multi-run `eval_biomech` JSON and emits a single
  comparison figure with L+R kinematics overlay, both-leg vGRF
  stance-phase curves, hip-knee phase plane (R + L), progress-
  score bars, and a ±20% credible-band scorecard. Sister tool to
  `biomech_report.py` (which only covers the right leg).
- **`scripts/biomech_report.py` (fixed, 2026-04-29)** —
  `--rerollout` now reads each run's training MJCF from
  `env_kwargs.json` instead of forcing a single CLI `--xml`.
  Pre-fix it would silently mis-render hipopen/hiprelax models
  under stock `walker2d.xml`.
- **`results/biomech_candidates_eval.json`** + **`results/biomech_history.csv`**
  carry the canonical per-run measurements.
- **`docs/figures/biomech_realism_dashboard.{png,md}`** — the
  writeup-ready artifact.

---

## What this project is

A reinforcement learning study of **gait imitation on the MuJoCo
Walker2d-v4 planar biped**, conditioned on inverse-kinematics reference
data from the Ulrich treadmill walking dataset (Subject 1, 1.25 m/s).

Two complementary imitation methods are studied:

1. **Phase-conditioned PPO + DeepMimic 4-term reward** — primary
   working track. Active code in [`../src/walker2d/ppo_walker2d_phase.py`](../src/walker2d/ppo_walker2d_phase.py).
   Off-policy SAC sibling at [`../src/walker2d/sac_walker2d_phase.py`](../src/walker2d/sac_walker2d_phase.py).
2. **Adversarial Motion Priors (AMP) + AIRL** — comparison track
   ([`../src/walker2d/amp_walker2d.py`](../src/walker2d/amp_walker2d.py),
   [`../src/walker2d/airl_walker2d.py`](../src/walker2d/airl_walker2d.py),
   cherry-picked from upstream `bk-37/6955_Project@3e4c3fa` on
   2026-04-28). AMP collapses at 8 CPU envs (writeup §6.3); the
   recommended workflow today is to finetune from a working
   PPO+DeepMimic checkpoint, but the warm-started Batch 3 runs found
   the discriminator gradient pushes the policy out of stiff-hip into
   asymmetric kicks rather than natural gait. The full unblock is the
   GPU/MJX port — see [`ROADMAP.md § 1`](ROADMAP.md#1-mujoco-mjx--gpu-port-for-amp-writeup-71).

Three top-line scientific contributions (from the writeup):

- A working phase-conditioned imitation policy on real human IK data.
- A mechanistic taxonomy of reward-hacking failure modes (ankle
  paddling, one-legged hopping, toe-walking, plus the post-restart
  stiff-hip trap) framed as canonical Goodhart's-Law cases. See
  [`REWARD_DESIGN.md`](REWARD_DESIGN.md).
- A characterization of AMP's discriminator collapse at small env
  counts (writeup §6.3) and the mechanism that explains it.

---

## How to validate progress

`eval_biomech.py` compares every checkpoint against a *measured*
reference (computed from Subject 1's force plates and IK by
`extract_reference_biomech.py`), and `scripts/biomech_report.py`
renders a writeup-ready markdown table + 6-panel figure. After any
training batch:

```
python src/diagnostics/eval_biomech.py --xml walker2d.xml --eps 6 --steps 2500 \
    results/<run>:final:<label> --out results/<run>_eval.json --csv results/biomech_history.csv
python scripts/biomech_report.py results/<run>_eval.json --rerollout
```

The eval JSON's `vs_reference` block carries `delta` and `pct_err` per
metric, plus a single `progress_score` in [0, 4]. See
[`METHODS.md § Held-out biomechanical evaluation`](METHODS.md#held-out-biomechanical-evaluation-the-two-tool-flow).
**Anti-Goodhart caveat from Batch 3:** `progress_score` and
`hip_knee_dtw` flatter stand-and-wiggle policies; pair with
`hip_r_rom_deg` and visual review.

---

## What's currently running

- **Active training script:** `src/walker2d/ppo_walker2d_phase.py` —
  the DeepMimic-faithful 4-term reward (sum of `exp(−k·err²)` on pose,
  velocity, end-effector, root height). All exploit-patch terms
  (swing_pen, contact_r, per-joint sharpness/weights, per-joint
  termination thresholds, BC) are off-by-default kwargs/CLI flags. See
  [`METHODS.md § Reward`](METHODS.md#reward--deepmimic-four-term-sum)
  and [`RESTART_LOG.md`](RESTART_LOG.md).
- **Active reference:** `assets/reference/gait_cycle_reference.npy` —
  one clean stride from Ulrich Subject 1 baseline (56 frames @ 50 Hz,
  resampled to 140 frames @ 125 Hz inside the env). FK-verified after
  the 2026-04-28 sign fix to encode forward walking.
- **Current lead policy (single):**
  - **`results/restart_b5_min_joint/`** — Batch-5 narrowing variant
    on hipopen, seed 8, `--min_joint_pose`, `--xml
    walker2d_hipopen.xml` (`[-30°, +60°]`). **Lead as of 2026-04-29**
    (post strike-detector fix) on the biomech-realism scorecard:
    highest progress score (**2.66**), lowest peak vGRF among the
    four post-Tier-0 candidates (3.70 BW), lowest double-support
    deviation (0.7 % vs ref 22.7 %), per-stride hip ROM 30°. Cadence
    is 192 spm vs reference 107 (~1.8× too fast); gait is still
    bouncing-not-walking, but bounces more like the reference than
    any alternative on disk does.
- **Superseded comparison policies (kept on disk, not the lead):**
  - `results/restart_b4_hipopen_5M/` — 5M steps, seed 6, 8 envs,
    `--xvel_term 0.3`, `--xml walker2d_hipopen.xml` (`[-30°, +60°]`).
    Per-stride median hip ROM 30.6°, score 2.40. Highest peak vGRF
    of the four post-Tier-0 candidates (4.77 BW).
  - `results/restart_b5_pose_scale20/` — Batch-5 narrowing variant
    on hipopen, seed 7, `--pose_scale 20`. Per-stride hip ROM 30.0°;
    score 2.26.
  - `results/restart_b4_hiprelax_s11/` — 5M steps, seed 11, 8 envs,
    `--xvel_term 0.3`, `--xml walker2d_hiprelax.xml` (`thigh_joint
    range="-150 35"`, +5° headroom). xvel-5M recipe verbatim except
    for the relaxed MJCF. Per-stride hip ROM 30.5°, score **2.24**
    (last among the four post-Tier-0 candidates after the
    strike-detector fix; pre-fix it had ranked first because the
    chatter-counted cadence happened to land closest to ref).
- **Pre-Tier-0 superseded baseline (kept for reference):**
  `results/restart_b2_xvel/` — 5M steps, stock `walker2d.xml`,
  single-cycle reference, 8 envs. Same recipe minus the relaxed MJCF.
  Stiff-hip basin (hip ROM ~2.8° vs reference 45°) — the joint
  range literally couldn't reach the reference. Kept as the
  "before" policy for showing what opening the hip range did; do
  not branch new work off it.
- **Pre-restart canonical** (kept for historical comparison only —
  trained on the inverted reference; do not branch new work off
  these): `results/walker2d_phase_cycle_s1scaled_sum_20260423-213031/`
  (100M, scaled MJCF, engineered reward).

## Comparison runs on disk

Sorted by training date; `restart_*` runs are post-restart on the
corrected reference, `walker2d_phase_*` runs are pre-restart on the
inverted reference and require the missing `walker2d_subject1.xml`
MJCF to render.

| Result dir | Steps | Notes |
|---|---|---|
| **`results/restart_b5_min_joint/`** | **5M** | **Lead (named 2026-04-29 on the post-strike-detector-fix biomech-realism scorecard).** hipopen + `--min_joint_pose`, seed=8. Highest progress score (2.66), lowest peak vGRF (3.70 BW), lowest double-support deviation, per-stride hip ROM 30°, mean fwd vel 1.23 m/s (essentially ref). |
| `results/restart_b4_hipopen_5M/` | 5M | Superseded comparison run (hipopen track). DeepMimic 4-term + `--xvel_term 0.3` + `--xml walker2d_hipopen.xml`. seed=6. Per-stride hip ROM 30.6°, score 2.40, highest peak vGRF (4.77 BW). 1000×4 eval survival. |
| `results/restart_b5_pose_scale20/`  | 5M     | Superseded comparison: hipopen + `--pose_scale 20`. seed=7. Per-stride hip ROM 30.0°, fwd vel 1.35 m/s, score 2.26. Visually indistinguishable from b4_hipopen_5M. |
| `results/restart_b4_hiprelax_s11/` | 5M | Superseded (was lead pre-strike-detector-fix). xvel-5M recipe + `--xml walker2d_hiprelax.xml` (`thigh_joint range="-150 35"`). seed=11. Per-stride hip ROM 30.5°, score 2.24 — last among the four post-Tier-0 candidates after the fix. |
| `results/restart_b4_hiprelax_s12/` | 5M | Tier 0 C seed 12. Higher knee ROM (38.6°) but worse LR symmetry (0.143) and DTW (0.641). Kept for the 3-seed comparison artifact. |
| `results/restart_b4_hiprelax_s13/` | 5M | Tier 0 C seed 13. Slightly slower cadence than s11 but worst progress score (2.11) and highest peak vGRF among the hiprelax seeds (4.18). Kept for the 3-seed comparison artifact. |
| `results/restart_b4_hipopen/`       | 2M     | Pre-5M `b4_hipopen` checkpoint. seed=4. Hip ROM 91° (under-trained). |
| `results/restart_b4_hipinvert/`     | 2M     | Batch 4 Variant B (re-invert hip ref, stock MJCF). seed=5. Brittle deep-extension kicking gait, episodes die at 47-250 steps. |
| `results/restart_b2_xvel/` | 5M | Pre-Tier-0 / pre-batch-4 stiff-hip baseline. DeepMimic 4-term + `--xvel_term 0.3`. Stock walker2d.xml, seed=2. ep_len 2120, all-episode 2500-step survival on eval, but hip ROM ~2° vs ref 45° because the joint range literally couldn't reach the reference. Visual review (Brock, 2026-04-29): "looks pretty bad" relative to the relaxed-MJCF runs; do not branch new work off this. |
| `results/restart_b2_k30/` | 5M | DeepMimic + `--pose_scale 30`. Unstable; 4/6 eval episodes fall in <120 steps. |
| `results/restart_b1_dm/` | 2M (killed) | Pure DeepMimic baseline. Stand-and-wiggle exploit. |
| `results/restart_b1_dm_bc/` | 2M (killed) | DeepMimic + 5-epoch BC. Same exploit, marginally varied across seeds. |
| `results/walker2d_phase_cycle_s1scaled_sum_20260423-213031/` | 100M | Pre-restart canonical (inverted reference, scaled MJCF). |
| `results/walker2d_phase_cycle_s1scaled_sum_20260422-175117/` | 60M | Earlier pre-restart canonical, scaled MJCF. |
| `results/walker2d_phase_full_sum_20260410-124935/` | 18M | Stock Walker2d, full-trial ref, uniform-k=8 (pre-restart). |
| `results/walker2d_phase_full_sum_20260410-105306/` | 45M | Earlier pre-restart DeepMimic-reward run. |
| `results/walker2d_phase_cycle_sum_20260409-211537/` | 10.5M | First single-cycle reference run (pre-restart). |
| `results/walker2d_pretrain_symmetry_20260407-172719/` | 5M | Symmetry-pretrain ankle-paddling demo (legacy). |

The 2026-04-29 overnight sweep produced 19 additional runs under
`results/overnight_<TIMESTAMP>/` (not all retained on every checkout —
see `RESTART_LOG.md § Batch 3`).

Render the lead policy with:

```
python src/walker2d/render_phase.py --live results/restart_b5_min_joint:final
python src/walker2d/render_phase.py --mp4 docs/figures/foo.mp4 results/restart_b5_min_joint:final
```

The other three (the two remaining hipopen variants and `b4_hiprelax_s11`)
are kept for visual comparison only.

Since the 2026-04-29 merge, `render_phase.py` automatically reads
`xml_file` from each run's `env_kwargs.json`, so the trained-against
MJCF is used without a `--xml` flag (the policy was trained against
that MJCF, and rendering it under a different one — e.g. opening the
hip range — gives misleading visuals). Pre-2026-04-29 runs that
predate the `xml_file` field still need `--xml walker2d.xml` for stock
Walker2d (or `--xml walker2d_subject1.xml` for the missing scaled
MJCF). Render multiple runs back-to-back by passing multiple specs:

```
python src/walker2d/render_phase.py --live \
    results/restart_b4_hipopen_5M:final \
    results/restart_b4_hiprelax_s11:final \
    results/restart_b5_min_joint:final \
    results/restart_b5_pose_scale20:final
```

(In PowerShell put it all on one line; the backslash-continuations
above are bash syntax.)

---

## Tools at a glance — what to reach for when

Comprehensive tooling docs live in
[`scripts/README.md`](../scripts/README.md) and
[`src/diagnostics/README.md`](../src/diagnostics/README.md). Common
flows:

| Want to... | Tool |
|---|---|
| Train a new policy from scratch | `python src/walker2d/ppo_walker2d_phase.py --ref_cycle assets/reference/gait_cycle_reference.npy --xml walker2d_hipopen.xml --xvel_term 0.3 --num_envs 8 --total_steps 5e6` |
| Render a trained policy live | `python src/walker2d/render_phase.py --live results/<run>:final` |
| Pre-render an mp4 | `python src/walker2d/render_phase.py --mp4 out.mp4 results/<run>:final` |
| **Single-source-of-truth hip ROM metric** (4-ep deterministic rollout) | `python scripts/eval_hip_rom.py results/<run>` |
| Reachability gate: does the reference fit a given MJCF's joint ranges? | `python src/diagnostics/check_reference_jnt_range.py --xml walker2d_hipopen.xml` |
| End-to-end joint-range hypothesis verification (MJCF + ref + dynamics probe + trained-policy probe) | `python scripts/debug_joint_range_hypothesis.py` |
| Tier 0 experiment-C panel (3-seed dashboards + eval + mp4s + comparison plot + summary) | `python scripts/tier0/evaluate_C.py` |
| Held-out biomech eval vs measured Subject-1 targets | `python src/diagnostics/eval_biomech.py results/<run>:final --out results/<run>_eval.json` |
| Writeup-ready biomech table + 6-panel figure (R leg) | `python scripts/biomech_report.py results/<run>_eval.json --rerollout` |
| **Multi-run biomech-realism dashboard** (L+R kinematics, both-leg vGRF, scorecard with ±20% credible band) | `python scripts/biomech_realism_dashboard.py results/biomech_candidates_eval.json` |
| Smoke-test BC warm-start | `python scripts/smoke_test_warmstart.py` |
| Re-render every run dir to mp4 (PowerShell) | `scripts/render_all_results.ps1` |
| Pretty-print TensorBoard scalars side-by-side | `python src/diagnostics/compare_tb.py results/<run-A>/tb results/<run-B>/tb` |
| Live-view the on-disk reference cycle on a Walker2d skeleton | `python src/diagnostics/view_reference.py` |

---

## What still needs to happen

For the **current writeup-driven scope**, see [`ROADMAP.md`](ROADMAP.md).
After the 2026-04-29 biomech-realism finding above, the priority
order has been reshuffled:

0. ~~**Close the residual reward gap on relaxed-hip MJCFs**~~
   **DEPRIORITISED.** The biomech scorecard says all post-Tier-0
   candidates are at the same biomech-quality floor as the
   pre-Tier-0 baseline. The reward shape that would close the
   double-support / vGRF / cadence gap is engineering walking
   by hand, which is the opposite of the imitation-only goal.
1. **Write up the negative result** as the primary new contribution
   — engineered-reward + relaxed-MJCF + 5M PPO is not enough on
   Walker2d. Use `docs/figures/biomech_realism_dashboard.png` as
   the central artifact.
2. **MJX/GPU port** to make AMP function (4,000-env parallelism) —
   now the *first* high-value experimental step, not the second.
3. **Multi-cycle / multi-subject reference** for temporal smoothness
   — useful regardless of which method is on top.
4. **Possible measured return to the musculoskeletal track**
   (`src/legacy/musculoskeletal/`) — the original scope's
   biomech question is more naturally answered with muscle
   actuators that have to load the leg the way a human does.

For **revisiting the original 3D / musculoskeletal scope**, see
[`LEGACY_TRACKS.md`](LEGACY_TRACKS.md). Code is preserved in
`src/legacy/musculoskeletal/`.

---

## Known gaps in this checkout

- **`assets/mjcf/walker2d_subject1.xml` is missing.** The pre-restart
  canonical run was trained against it; required for `--scale_model`
  and is the default `--xml` for `render_phase.py`. Must be
  regenerated or copied from the user's other machine before training
  / rendering with scaled geometry. Stock-Walker2d runs (the entire
  `restart_*` family and most pre-restart full-trial runs) load and
  roll out fine without it — pass `--xml walker2d.xml` to
  `render_phase.py`.
- **`amp_walker2d.py` and `airl_walker2d.py`** were cherry-picked from
  upstream commit `3e4c3fa` on 2026-04-28; relocated from the repo
  root into `src/walker2d/` and rewired to import the active loader.
  `--ref_cycle` works out-of-the-box; `--ref_all` does not receive
  per-trial segment lengths from the loader, so boundary transitions
  are not filtered out of the expert buffer.
- **`Ulrich_Treadmill_Data/`** is gitignored. Users supply their own
  copy at `<repo>/Ulrich_Treadmill_Data/Subject{1..10}/IK/walking_*/output/results_ik.sto`.
  See [`DATA_SOURCES.md`](DATA_SOURCES.md).

### Per-machine setup notes

- **Current laptop (no GPU, Python 3.13):** `Ulrich_Treadmill_Data/` is
  a directory junction to `CoordinationRetrainingData/forSimTK/`. Venv
  at `.venv/` was built with the CPU build of PyTorch 2.7.0 and
  `requirements/windows_cpu.txt`. MyoSuite is **not** in this venv —
  it pins `gymnasium==1.2.3` / `mujoco==3.6.0` (incompatible with the
  active Walker2d stack) and requires Python 3.10–3.12. For the
  legacy `src/legacy/musculoskeletal/` track, a sibling venv lives at
  `.venv-myo/` (Python 3.12, MyoSuite 2.12.1, verified
  `myoLegWalk-v0` reset+step). See `README.md` "MyoAssist (legacy
  musculoskeletal track)" for the recipe.
- **numpy must be 2.x** to load existing checkpoints — their pickle
  blobs reference `numpy._core` (a 2.x-only path). The requirements
  files now pin `numpy>=2.0,<3.0`.
