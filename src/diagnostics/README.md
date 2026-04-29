# `src/diagnostics/` — sanity-check scripts

Standalone scripts for inspecting the reference data and the Walker2d
model. **Not** on the training path — none of these are imported by
`src/walker2d/`. Run them by hand when something feels off about the
reference, the joint sign convention, or the scaling.

| Script | Purpose | Output |
|---|---|---|
| `diag_cycle.py` | Plot 3 looped gait cycles + measure per-joint discontinuity at the seam. | `docs/figures/cycle_continuity.png` |
| `diag_ref.py` | Print per-joint reference ranges and run open-loop FK at fixed pitch to confirm the reference stays upright. | stdout |
| `diag_walker_mass.py` | Dump Walker2d per-body masses (total ≈ 23.68 kg) and the scale factor for comparing to a 75 kg subject. | stdout |
| `extract_osim_mass.py` | Parse `*.osim` XML files to pull per-subject total body mass. Used for BW-normalized GRF comparison. | stdout |
| `extract_reference_biomech.py` | Compute *measured* biomech targets from a Subject's GRF .mot + IK .sto + scaled .osim: stride period, cadence, double-support, peak vGRF/BW, per-joint ROM, plus a normalised stance-phase vGRF curve. **Run this once per subject/trial; the output drives `eval_biomech.py --targets` and `scripts/biomech_report.py`.** | `assets/reference/biomech_targets.json` + `.vgrf_curves.npz` |
| `eval_biomech.py` | Held-out biomech metrics for a checkpoint. With targets present (default), emits `vs_reference` (delta, pct_err) and a `progress_score` (0–4) per run. Use `--csv` to append one row per run to a history file. | JSON (+ optional CSV append) |

Run all from the project root, e.g.:

```bash
python src/diagnostics/diag_cycle.py
python src/diagnostics/diag_ref.py
```

`diag_cycle.py` and `diag_ref.py` read
`assets/reference/gait_cycle_reference.npy` — make sure
`src/walker2d/extract_gait_cycle.py` has been run first.

## Validating progress against real biomechanics

The two-tool flow that lets an AI agent grade a run *without eyeballing
video*:

```bash
# 1. (once per subject/trial) measured reference targets from Ulrich data
python src/diagnostics/extract_reference_biomech.py        # Subject1, walking_baseline1

# 2. (per run / per checkpoint) sim biomech + delta vs ref + 0–4 score
python src/diagnostics/eval_biomech.py --xml walker2d.xml --eps 6 --steps 2500 \
    results/<run>:final:<label> --out results/<run>_eval.json \
    --csv results/biomech_history.csv

# 3. (per writeup pass) render markdown table + 6-panel comparison figure
python scripts/biomech_report.py results/<run>_eval.json --rerollout
```

After step 2 the per-run JSON has a `vs_reference` block (`delta`,
`pct_err` for every metric with a measured Ulrich target) and a
`progress_score` in [0, 4]. After step 3 you get
`docs/figures/biomech_report.{md,png}` ready to drop into the writeup
or `RESTART_LOG.md`. The `biomech_history.csv` accumulates one row per
eval run so you can plot any metric across batches without parsing
JSON.
