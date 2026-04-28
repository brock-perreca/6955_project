# MuJoCo MJCF assets

| File | Status |
|---|---|
| `walker2d_subject1.xml` | **Missing.** Subject-1-scaled Walker2d, required for `--scale_model` runs and the default `--xml` for `src/walker2d/render_phase.py`. The current canonical run was trained against this file. Regenerate or copy from the user's other machine before running anything that needs it. |
| `walker2d_custom.xml` | Legacy. Predates the decision to stick with stock Walker2d. Don't reference from active code. |

Stock `walker2d.xml` is pulled from gymnasium's bundled assets at
runtime — not stored in this repo.

## How active code finds these files

`Walker2dPhaseAware` resolves an `xml_file=` argument like so:

1. Literal `"walker2d.xml"` → look up in gymnasium's MuJoCo asset dir.
2. Absolute path → use as-is.
3. Bare filename (e.g. `"walker2d_subject1.xml"`) → look in
   `assets/mjcf/` first, then fall back to the repo root for backward
   compatibility with older runs.
