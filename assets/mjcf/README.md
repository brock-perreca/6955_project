# MuJoCo MJCF assets

| File | Status |
|---|---|
| `walker2d_hipopen.xml` | **Active (Batch 4 / Brock-Asus-Laptop, 2026-04-29).** Stock walker2d.xml with `thigh_joint range="-30 60"` (permissive both directions: also lets the hip extend a bit past the original 0° lower bound, and flex well past the reference's +30° peak). Single-knob morphology fix — the reference asks for hip flexion up to +29.97° but stock caps at +0°. Used by `restart_b4_hipopen*` and `restart_b5_*` runs. Pass via `--xml walker2d_hipopen.xml` to `ppo_walker2d_phase.py`. See [`docs/RESTART_LOG.md § Batch 4`](../../docs/RESTART_LOG.md). |
| `walker2d_hiprelax.xml` | **Active (Batch 4b / Tier 0, Brock-O11, 2026-04-29).** Same stock walker2d.xml with `thigh_joint range="-150 35"` (instead of `-150 0`) — *minimal-headroom* variant: only +5° over the reference peak, no looser, on the hypothesis that an overly-permissive limit creates new bad basins (overswing/kicking — exactly what hipopen does). Used by `restart_b4_hiprelax_s11..s13` runs. Pass via `--xml walker2d_hiprelax.xml`. See [`docs/TIER0_DIAGNOSTICS.md`](../../docs/TIER0_DIAGNOSTICS.md) and [`docs/RESTART_LOG.md § Batch 4b`](../../docs/RESTART_LOG.md). |
| `walker2d_subject1.xml` | **Missing.** Subject-1-scaled Walker2d, required for `--scale_model` runs and the default `--xml` for `src/walker2d/render_phase.py`. The pre-restart canonical run was trained against this file. Regenerate or copy from the user's other machine before running anything that needs it. |
| `walker2d_custom.xml` | Legacy. Predates the decision to stick with stock Walker2d. Don't reference from active code. |

### `hipopen` vs `hiprelax` — when to use which

The two MJCFs are deliberately bracketing variants that came out of
the same-day, two-machine kinematic-ceiling diagnosis:

- **`walker2d_hipopen.xml`** (`thigh_joint range="-30 60"`) gives
  the policy *room to overshoot*. Hip ROM at 5M
  (`results/restart_b4_hipopen_5M/`) is **63°** vs reference 43° —
  over-flexed but stable, all eval episodes survive 1000 steps. Use
  this MJCF when you want the policy to fully escape the wall and
  explore aggressive flexion, and rely on a sharper aggregator or a
  peaked forward reward to pull it back toward reference. The Asus
  laptop's primary Batch 5 follow-up runs are on this MJCF.
- **`walker2d_hiprelax.xml`** (`thigh_joint range="-150 35"`) gives
  the policy *just enough* to express the reference. Hip ROM at 5M
  (`results/restart_b4_hiprelax_s11/`) is **17–20°** — under the
  target but the trace tracks reference shape and frequency
  cleanly. Use this MJCF when you want to test whether reward
  changes alone can pull amplitude *up* toward 45°. The O11 box's
  primary Tier 1 follow-up runs are on this MJCF.

Running the same reward-reform recipe against *both* MJCFs
(hipopen narrows down toward 45°, hiprelax grows up toward 45°) is
the cleanest experimental design for attributing the residual gap
to reward vs morphology. See [`docs/ROADMAP.md § 0`](../../docs/ROADMAP.md).

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
