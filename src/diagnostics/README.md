# `src/diagnostics/` — sanity-check scripts

**Purpose:** what each diagnostic script does and when to run it.
**Read this when:** something feels off about the reference, the
joint sign convention, or the scaling — or you need to grade a
training run against measured biomechanics.

Standalone scripts for inspecting the reference data and the Walker2d
model. **Not** on the training path — none of these are imported by
`src/walker2d/`.

| Script | Purpose | Output |
|---|---|---|
| `diag_cycle.py` | Plot 3 looped gait cycles + measure per-joint discontinuity at the seam. | `docs/figures/cycle_continuity.png` |
| `diag_ref.py` | Print per-joint reference ranges and run open-loop FK at fixed pitch to confirm the reference stays upright. | stdout |
| `diag_walker_mass.py` | Dump Walker2d per-body masses (total ≈ 23.68 kg) and the scale factor for comparing to a 75 kg subject. | stdout |
| `extract_osim_mass.py` | Parse `*.osim` XML files to pull per-subject total body mass. Used for BW-normalized GRF comparison. | stdout |
| `view_reference.py` | MuJoCo-viewer playback of the on-disk gait cycle (`qpos[3:9] = ref[t]`, body translated at 1.25 m/s). The original sign-error discovery tool. | live viewer |
| `compare_tb.py` | Side-by-side TensorBoard scalar comparison across runs. | stdout / matplotlib |
| `extract_reference_biomech.py` | Compute *measured* biomech targets from a Subject's GRF .mot + IK .sto + scaled .osim: stride period, cadence, double-support, peak vGRF/BW, per-joint ROM, plus a normalised stance-phase vGRF curve. **Run this once per subject/trial; the output drives `eval_biomech.py --targets` and `scripts/biomech_report.py`.** | `assets/reference/biomech_targets.json` + `.vgrf_curves.npz` |
| `eval_biomech.py` | Held-out biomech metrics for a checkpoint. With targets present (default), emits `vs_reference` (delta, pct_err) and a `progress_score` (0–4) per run. Use `--csv` to append one row per run to a history file. | JSON (+ optional CSV append) |
| `render_reference_replay.py` | Kinematic replay of `gait_cycle_reference.npy` driven into the Walker2d MJCF (no policy, no PD, no physics integration). The **visual ceiling** every trained-policy mp4 should be compared to. Logs torso z, pitch, foot xz + contact forces; validates hip ROM matches reference within 0.1°. | `docs/figures/reference_replay.{mp4,npz}` + `REFERENCE_REPLAY.md` |
| `run_dashboard.py` | Auto-generated 4-panel PNG per trained run: 6-joint angle vs phase (sim/ref overlaid, one cycle), reward decomposition, action histograms, foot xz trajectory. Title prints **per-cycle** ROM (the joint-angle panel's actual content) — full-rollout max−min is reported separately because sporadic kicks bias it upward (the trap that hid stiff-hip basins in the overnight sweep). | `<run_dir>/dashboard.png` |

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

## Visual track — the same question, eyeball-driven

```powershell
# 1. (run once) the visual ceiling: kinematic replay of the reference,
#    embodied in the Walker2d MJCF, at the same camera as render_phase.py.
python src/diagnostics/render_reference_replay.py --cycles 3
# -> docs/figures/reference_replay.mp4 + .npz + REFERENCE_REPLAY.md

# 2. (per run) one PNG that exposes whether the policy is walking or
#    producing sporadic kicks that read as ROM in scalar metrics.
python src/diagnostics/run_dashboard.py results/<run>:final --steps 600
# -> results/<run>/dashboard.png

# 3. (per checkpoint, when you want to see it move) policy mp4 at the
#    same camera as the reference replay above:
python src/walker2d/render_phase.py --mp4 docs/figures/<run>.mp4 \
    results/<run>:final
```

The dashboard is the 30-second triage — open the PNG, check whether
sim hip/knee curves overlay the reference. If they don't, the run
isn't walking, regardless of what `progress_score` says.
