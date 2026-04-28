# `src/legacy/walker2d_v1/` — earlier 2D Walker2d attempts

Frozen Walker2d code from before phase conditioning was adopted. Kept on
disk for historical reference. **Do not extend without confirming with
the user first.**

| File | Status |
|---|---|
| `ppo_walker2d.py` | Phase-blind imitation. Failed for three reasons (no resampling, phase-blind obs, 413k-frame concatenated reference). Loaders extracted to `src/walker2d/ulrich_loader.py`; everything else preserved verbatim. |
| `pretrain_walker2d.py` | Symmetry reward-shaping pretrainer (no reference). Hit four canonical local optima — see `docs/RUN_LOG.md`. |
| `gail_walker2d.py` | GAIL approach for Walker2d. Not part of the writeup; superseded by AMP / AIRL on the adversarial track. |
| `render_walker.py` | Renderer for the legacy `Walker2dImitation` env and vanilla Walker2d-v4 (`--vanilla`). Use for legacy checkpoints (e.g. `results/walker2d_pretrain_symmetry_*/`). |

**For the active pipeline, use [`../../walker2d/`](../../walker2d/).**

For the chronology, see
[`../../../docs/PROJECT_TIMELINE.md`](../../../docs/PROJECT_TIMELINE.md).
For curated demo runs from this group with reproduce/render commands,
see [`../../../docs/RUN_LOG.md`](../../../docs/RUN_LOG.md).
For per-file rationale, see
[`../../../docs/LEGACY_TRACKS.md`](../../../docs/LEGACY_TRACKS.md).
