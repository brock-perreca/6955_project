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

Run all from the project root, e.g.:

```bash
python src/diagnostics/diag_cycle.py
python src/diagnostics/diag_ref.py
```

`diag_cycle.py` and `diag_ref.py` read
`assets/reference/gait_cycle_reference.npy` — make sure
`src/walker2d/extract_gait_cycle.py` has been run first.
