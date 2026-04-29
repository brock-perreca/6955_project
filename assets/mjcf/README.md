# MuJoCo MJCF assets

| File | Status |
|---|---|
| `walker2d_hiprelax.xml` | **Active (Tier 0, 2026-04-29).** Stock walker2d.xml with `thigh_joint range="-150 35"` (instead of `-150 0`). Single-knob morphology fix — the reference asks for hip flexion up to +29.97° but stock caps at +0°. See [`docs/TIER0_DIAGNOSTICS.md`](../../docs/TIER0_DIAGNOSTICS.md) for the discovery and the `restart_b4_hiprelax_*` runs that test it. Pass via `--xml walker2d_hiprelax.xml` to `ppo_walker2d_phase.py`. |
| `walker2d_subject1.xml` | **Missing.** Subject-1-scaled Walker2d, required for `--scale_model` runs and the default `--xml` for `src/walker2d/render_phase.py`. The current canonical run was trained against this file. Regenerate or copy from the user's other machine before running anything that needs it. |
| `walker2d_custom.xml` | Legacy. Predates the decision to stick with stock Walker2d. Don't reference from active code. |

Stock `walker2d.xml` is pulled from gymnasium's bundled assets at
runtime — not stored in this repo.

## Verifying a reference vs. an MJCF's joint ranges

Use `python src/diagnostics/check_reference_jnt_range.py --xml <file>`
to confirm a reference is reachable under a given MJCF's joint
limits. It writes a per-joint plot + JSON to `docs/figures/tier0/`.

## How active code finds these files

`Walker2dPhaseAware` resolves an `xml_file=` argument like so:

1. Literal `"walker2d.xml"` → look up in gymnasium's MuJoCo asset dir.
2. Absolute path → use as-is.
3. Bare filename (e.g. `"walker2d_subject1.xml"`) → look in
   `assets/mjcf/` first, then fall back to the repo root for backward
   compatibility with older runs.
