# `scripts/` — wrappers, sweep harnesses, and one-shot tools

**Purpose:** what each script in `scripts/` does and when to reach
for it. Companion to [`src/diagnostics/README.md`](../src/diagnostics/README.md)
(scripts here are *outside* `src/`, are usually multi-step pipelines
or one-off helpers, and may shell out to `src/walker2d/` and
`src/diagnostics/` modules).

**Read this when:** you need a tool you didn't write yourself, or
you're picking the right metric/diagnostic for a question.

Run everything from the project root (each script does
`Path(__file__).resolve().parents[1]` to find the repo root).

---

## Top-level scripts

| Script | Purpose | Typical use |
|---|---|---|
| [`biomech_report.py`](biomech_report.py) | Render a writeup-ready markdown table + 6-panel matplotlib figure from one or more `eval_biomech` JSONs. Overlays sim curves on the Ulrich reference (hip / knee / ankle / vGRF / hip-knee phase plane / stride-period bars). **Right leg only.** As of 2026-04-29 `--rerollout` reads each run's training MJCF from `env_kwargs.json` (pre-fix it would silently mis-render hipopen/hiprelax models under stock walker2d.xml). | Every writeup pass: `python scripts/biomech_report.py results/<run>_eval.json --rerollout` |
| [`biomech_realism_dashboard.py`](biomech_realism_dashboard.py) | **Multi-run biomechanical-realism dashboard.** Consumes a multi-run `eval_biomech` JSON (e.g. `results/biomech_candidates_eval.json`) and emits a single comparison figure: 6 joint-angle panels (hip/knee/ankle × R/L) overlaid on the reference, both-leg vGRF stance curves, hip-knee phase plane (R + L), progress-score bars, and a ±20%-credible-band scorecard with per-metric % error. Sister to `biomech_report.py` (which is R-leg only). The 2026-04-29 end-of-road biomech finding ([`PROJECT_STATUS.md`](../docs/PROJECT_STATUS.md#biomechanical-realism-finding-2026-04-29--end-of-road-on-the-engineered-reward-track)) was produced by this tool. | When comparing several candidates side-by-side: `python scripts/biomech_realism_dashboard.py results/biomech_candidates_eval.json` |
| [`eval_hip_rom.py`](eval_hip_rom.py) | **Single-source-of-truth hip ROM metric.** 4 deterministic episodes × 1000 steps from RSI seeds 42–45. Reports per-leg ROM, % steps at upper joint limit, mean fwd vel, episode survival. The metric we trust after Batch 3 burned us on `progress_score`/`hip_knee_dtw` flattering stand-and-wiggle. | After every training run: `python scripts/eval_hip_rom.py results/<run>` |
| [`debug_joint_range_hypothesis.py`](debug_joint_range_hypothesis.py) | End-to-end joint-range hypothesis verification: (1) MJCF inspection, (2) reference per-joint range, (3) dynamics-respecting FK probe at each peak (set qpos from ref → step with action=0 → measure constraint-solver pullback), (4) trained xvel-5M policy probe (% time at upper limit). Companion to [`check_reference_jnt_range.py`](../src/diagnostics/check_reference_jnt_range.py): the diagnostic script is the *static* check (PNG + JSON for the writeup); this is the *dynamic* verification (console output for triage). | When you suspect a new MJCF doesn't match a reference, or before changing a joint range. |
| [`make_hipinvert_reference.py`](make_hipinvert_reference.py) | Build `assets/reference/gait_cycle_reference_hipinvert.npy` from the corrected reference by re-inverting hip columns 0 and 3. Used by Batch 4 Variant B (`results/restart_b4_hipinvert/`). One-time helper; run once and commit the output. | Only when reproducing the hipinvert ablation. |
| [`smoke_test_warmstart.py`](smoke_test_warmstart.py) | Smoke test for the 2026-04-29 MJCF-read warm-start fix. Checks: (1) `Walker2dPhaseAware` constructs against stock + hipopen + hiprelax MJCFs, (2) `reset()` warm-starts within the model's actual joint range, (3) `step()` doesn't NaN, (4) 200-step random rollout completes, (5) PPO.predict + step works on a trained model, (6) `render_phase.py`'s clip uses `env._jnt_lo / env._jnt_hi` (no `_JNT_LO/_JNT_HI` NameError). | After any change to env construction, MJCF selection, or warm-start clipping. |
| [`render_all_results.ps1`](render_all_results.ps1) | PowerShell driver that renders one mp4 per saved run under `results/` to `docs/figures/results_overview/`. Skips overnight sweeps + runs that lack `reference.npy`. Each render is independent so one crash doesn't stop the rest. | When you want a fresh batch of overview videos for visual triage. |

## `scripts/tier0/` — Tier 0 morphology-vs-reward harness

The 2026-04-29 Tier 0 diagnostic (see
[`docs/TIER0_DIAGNOSTICS.md`](../docs/TIER0_DIAGNOSTICS.md)) ships
its full validation pipeline as a single script:

| Script | Purpose | Output |
|---|---|---|
| [`tier0/evaluate_C.py`](tier0/evaluate_C.py) | Tier 0 experiment-C end-to-end. For each of `restart_b4_hiprelax_s11/s12/s13` and the `restart_b2_xvel/` baseline: produces a per-checkpoint dashboard PNG, a 6-ep × 2500-step `eval_biomech` JSON, a 600-step mp4 at the reference-replay camera, and a 5-seed × 600-step hip-trace plot. Then assembles the `C_hip_trace_comparison.png` panel + `C_summary.md`. | Everything under `docs/figures/tier0/C_hiprelax/` |

Run with `python scripts/tier0/evaluate_C.py`. Idempotent — safe to
re-run; it overwrites the comparison artifacts but skips checkpoints
that already produced their dashboard / eval JSON / mp4.

## `scripts/overnight/` — multi-experiment sweep scaffolding

Used by [`docs/RESTART_LOG.md § Batch 3`](../docs/RESTART_LOG.md)
(the 19-experiment overnight sweep that initially read as a
"reward-driven trap" before Tier 0 / Batch 4 found the kinematic
ceiling).

| File | Purpose |
|---|---|
| `overnight/run_experiment.py` | Per-experiment driver: train → eval_biomech → preview.mp4 → write meta.json. Designed for parallel invocations from a sweep file. |
| `overnight/rank_runs.py` | Read all experiment metas, compute a composite score, write `RANKING.md`. |
| `overnight/write_report.py` | Fill `REPORT_TEMPLATE.md` from each run's eval JSON to produce `<run>/REPORT.md`. |
| `overnight/REPORT_TEMPLATE.md` | Per-run report template the harness fills in. |
| `overnight/STATUS_TEMPLATE.md` | Top-level overnight summary template. |

A new sweep is a new YAML/JSON file driving `run_experiment.py`;
results land under `results/overnight_<TIMESTAMP>/`. The harness was
designed for the 2026-04-29 sweep; reuse it for any future
multi-experiment session that wants ranked, per-run reports.

---

## How this directory relates to `src/diagnostics/`

| | `src/diagnostics/` | `scripts/` |
|---|---|---|
| **Lives under `src/`** | yes | no |
| **Granularity** | atomic checks (one figure or one metric) | pipelines + sweeps + one-shots |
| **Imports from `src/walker2d/`** | sometimes | usually |
| **Outputs into `docs/figures/`** | by default | sometimes (mostly `docs/figures/tier0/`, `results_overview/`) |
| **Run frequency** | every checkout / every reference change / every training run | every batch / every writeup pass / one-time |

Rule of thumb: if it produces *one* artifact you'd cite in the
writeup (a static plot or a single JSON metric), it belongs in
`src/diagnostics/`. If it orchestrates several of those into a
batch report or sweep ranking, it belongs in `scripts/`.
