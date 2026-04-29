# Project status — current snapshot

**Purpose:** what's running *right now* — the active code, the current
best policy, what's known broken, and where the next move is queued.
**Read this when:** opening the project for the first time, or coming
back after a few days away.
**Adjacent:** [`PROJECT_TIMELINE.md`](PROJECT_TIMELINE.md) for the
chronological story · [`RESTART_LOG.md`](RESTART_LOG.md) for the most
recent batches with full setup/observation/render commands ·
[`ROADMAP.md`](ROADMAP.md) for what's queued next ·
[`reports/writeup_filled_1.docx`](reports/writeup_filled_1.docx) for
the formal joint writeup with Brian.

*Last updated: 2026-04-29.*

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
  b5_min_joint) look essentially the same.** No variant promoted to
  new current best; all three retained as comparison points. See
  [`RESTART_LOG.md § Batch 5`](RESTART_LOG.md#batch-5--2026-04-29--narrow-the-hipopen-over-flex--partial-positive-both-variants).

**Two leading current-best candidates** (one per machine — both
post-Tier-0 ablations of the same kinematic-ceiling fix):

- `results/restart_b4_hipopen_5M/` (Brock-Asus-Laptop, 5M, seed 6,
  `walker2d_hipopen.xml` `[-30, 60]`). Hip ROM 63.2° vs reference 43°,
  mean fwd vel 1.40 m/s vs target 1.25, all 4 deterministic eval
  episodes survive 1000 steps. Over-flexes by ~10° at the swing
  peak. Batch-5 narrowing variants (`pose_scale20`, `min_joint`)
  retained as comparison points; visual A/B against the baseline
  found no perceptible difference.
- `results/restart_b4_hiprelax_s11/` (Brock-O11, 5M, seed 11,
  `walker2d_hiprelax.xml` `[-150, 35]` — minimal +5° headroom over the
  reference peak). Best of three Tier 0 C seeds on LR symmetry, DTW,
  progress score, and vGRF. Hip ROM 17–20° (under the 45° target)
  but the trace tracks reference *shape and frequency*. Visual
  review (Brock, 2026-04-29): "all the videos look great" relative
  to xvel-5M.

The two ablations bracket the answer: hipopen *overshoots* the hip
ROM target, hiprelax *undershoots*. Both confirm morphology was the
dominant cause of pre-Tier-0 stiff-hip; both leave a residual
amplitude/cadence gap that points at reward as the secondary
bottleneck.

**Top-priority next steps** (see [`ROADMAP.md`](ROADMAP.md)):

1. **Structural reward reform** — restore
   `forward_reward = exp(-3·(v-1.25)²)`, drop the `xvel_term` floor.
   Run on **both** `walker2d_hipopen.xml` and `walker2d_hiprelax.xml`
   so we can attribute the residual gap to reward vs morphology.
2. **Stack the two batch-5 knobs** (`--pose_scale 20
   --min_joint_pose`) on hipopen — neither single-knob run tested
   the combination, and both moved the gait in the same direction.
   Lower expectations for visible change given Batch 5's outcome.
3. **Re-run AMP/AIRL warm-started from `b4_hipopen_5M`** — batch-3
   AMP failed partly because the underlying PPO couldn't produce
   reference-like hip flexion; that constraint is now removed. The
   discriminator may now have a learnable signal that aggregator
   tweaks alone don't provide.

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
- **Current best policies (two leading candidates, one per machine):**
  - `results/restart_b4_hipopen_5M/` — 5M steps, seed 6, 8 envs,
    `--xvel_term 0.3`, `--xml walker2d_hipopen.xml` (custom MJCF with
    hip range opened to `[-30°, +60°]`). Hip ROM 63.2° vs reference
    43°; mean fwd vel 1.40 m/s vs target 1.25; all 4 deterministic
    eval episodes survive 1000 steps. *Over-flexes ~10° at swing peak.*
    Batch-5 narrowing variants (`restart_b5_pose_scale20/`,
    `restart_b5_min_joint/`) numerically narrowed ROM to ~57° and
    `min_joint` pulled fwd vel to 1.231 m/s — but visual A/B against
    this baseline showed no perceptible difference, so all three are
    kept as comparison points.
  - `results/restart_b4_hiprelax_s11/` — 5M steps, seed 11, 8 envs,
    `--xvel_term 0.3`, `--xml walker2d_hiprelax.xml` (`thigh_joint
    range="-150 35"`, +5° headroom). xvel-5M recipe verbatim except
    for the relaxed MJCF. Visual review: clearly best of the Tier 0
    runs (Brock, 2026-04-29 — "all the videos look great"; s11 also
    leads on LR symmetry, DTW, progress score, vGRF). *Hip ROM still
    ~40 % of reference (~17-20° vs 45°)* — reward is binding on top
    of the now-removed kinematic ceiling. Cadence ~3× too fast
    (stride 0.36 s vs reference 1.12 s).
- **Pre-Tier-0 superseded baseline (kept for reference):**
  `results/restart_b2_xvel/` — 5M steps, stock `walker2d.xml`,
  single-cycle reference, 8 envs. Same recipe minus the relaxed MJCF.
  Stiff-hip basin (hip ROM ~2° vs reference 45°, cadence 3× too
  fast) — the joint range literally couldn't reach the reference.
  Kept as the "before" policy for showing what opening the hip range
  did; do not branch new work off it.
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
| **`results/restart_b4_hipopen_5M/`** | **5M** | **Leading current-best (hipopen track, Brock-Asus-Laptop).** DeepMimic 4-term + `--xvel_term 0.3` + `--xml walker2d_hipopen.xml`. seed=6. Hip ROM 63° (over-flexed by ~20° vs ref 43°), fwd vel 1.40 m/s, 1000×4 eval survival. |
| **`results/restart_b4_hiprelax_s11/`** | **5M** | **Leading current-best (hiprelax track, Brock-O11, canonical Tier 0 C seed).** Same xvel-5M recipe + `--xml walker2d_hiprelax.xml` (`thigh_joint range="-150 35"`). Best of 3 seeds on LR symmetry (0.097, only seed under 0.10), all-joints DTW (0.532, lowest), progress score (2.41), peak vGRF (3.97, lowest). hip_r ROM 19.8° / hip_l 15.3°. Tracks ref shape+frequency; amplitude ~40 % of reference, leaving Tier 1 reward reform as the next step. |
| `results/restart_b4_hiprelax_s12/` | 5M | Tier 0 C seed 12. Higher knee ROM (38.6°) but worse LR symmetry (0.143) and DTW (0.641). Kept for the 3-seed comparison artifact. |
| `results/restart_b4_hiprelax_s13/` | 5M | Tier 0 C seed 13. Slightly slower cadence (323.6 vs 332.7) but worst progress score (2.11) and highest peak vGRF (4.18). Kept for the 3-seed comparison artifact. |
| `results/restart_b5_min_joint/`     | 5M     | Batch 5 Variant B: hipopen + `--min_joint_pose`. seed=8. Hip ROM 57°, fwd vel 1.23 m/s. Visually indistinguishable from b4_hipopen_5M. |
| `results/restart_b5_pose_scale20/`  | 5M     | Batch 5 Variant A: hipopen + `--pose_scale 20`. seed=7. Hip ROM 56.6°, fwd vel 1.35 m/s. Visually indistinguishable from b4_hipopen_5M. |
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

Render any of them with:

```
python src/walker2d/render_phase.py --xml walker2d.xml --live results/restart_b2_xvel:final
python src/walker2d/render_phase.py --xml walker2d.xml --mp4 docs/figures/foo.mp4 results/restart_b2_xvel:final
```

`render_phase.py` defaults to `--xml walker2d_subject1.xml`; for stock
Walker2d runs (all post-restart and most early pre-restart), override
with `--xml walker2d.xml`.

---

## What still needs to happen

For the **current writeup-driven scope**, see [`ROADMAP.md`](ROADMAP.md).
Top items:

0. **Reward reform** (NEW priority): restore `forward_reward` peaked
   term, drop `xvel_term` floor.
1. **MJX/GPU port** to make AMP function (4,000-env parallelism).
2. **Multi-step future context** in the observation — implemented as
   `--preview_k`; ineffective on the broken reward (Batch 3); worth
   one more pass after item 0.
3. **DTW-based shape-fidelity evaluation** — `hip_knee_dtw` and
   `all_joints_dtw` shipped in `eval_biomech.py`; pair with hip ROM
   per Batch 3 caveat.
4. **Multi-cycle / multi-subject reference** for temporal smoothness
   and robustness.

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
