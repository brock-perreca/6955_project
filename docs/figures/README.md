# Figure outputs

**Purpose:** what each figure is, and which script wrote it.
**Read this when:** you need to regenerate a figure or you're
considering committing a new one.

These are **not part of the build**. They're tools you run by hand
when something feels off about the reference, or curated artifacts
captured for the writeup / a `RESTART_LOG.md` entry.

| File | Written by | What it shows |
|---|---|---|
| `gait_cycle_check.png` | `src/walker2d/extract_gait_cycle.py` | Six-panel plot (one per joint) of the extracted single stride, with joint-limit lines. Sanity-check for the cycle picked by the heel-strike detector. |
| `cycle_continuity.png` | `src/diagnostics/diag_cycle.py` | Three looped gait cycles per joint; red lines mark cycle boundaries. Visual check for the seam discontinuity at the wrap-around. |
| `biomech_report.png` / `biomech_report.md` | `scripts/biomech_report.py` | 6-panel comparison figure + markdown table from one or more `eval_biomech` JSONs. The drop-in for the writeup. |
| `restart_b1_preview_*.mp4`, `restart_b2_*.mp4` | `scripts/overnight/run_experiment.py` (or hand-rendered) | Curated preview clips of post-restart batches. Referenced from `RESTART_LOG.md`. |
| `reference_check.png` | older diagnostic | Reference joint trajectories, kept as historical artifact. |
