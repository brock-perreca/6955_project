# `src/legacy/` — frozen earlier work

Two earlier project tracks, kept for historical reference. **Do not
extend without confirming with the user first** — the active pipeline
under [`../walker2d/`](../walker2d/) is almost certainly the right place
for any new feature.

| Subdirectory | What it is |
|---|---|
| [`walker2d_v1/`](walker2d_v1/) | Earlier 2D Walker2d attempts: phase-blind imitation (Phase 1), symmetry-reward pretraining (Phase 2), GAIL. Superseded by the active phase-conditioned pipeline. |
| [`musculoskeletal/`](musculoskeletal/) | Original 3D 80-muscle MyoLeg + OpenCap proposal. Out of scope for the current writeup but **preserved for return** — the user has indicated they may revisit some of these ideas. |

For details on why each track is here, what each file does, and what to
verify before re-running, see
[`../../docs/LEGACY_TRACKS.md`](../../docs/LEGACY_TRACKS.md).

For the chronological story (why the project pivoted away from these),
see [`../../docs/PROJECT_TIMELINE.md`](../../docs/PROJECT_TIMELINE.md).
