# `src/` — code

Code is organized by **status**, not by topic:

- [`walker2d/`](walker2d/) — **active** phase-conditioned imitation pipeline. Modify these.
- [`diagnostics/`](diagnostics/) — standalone sanity-check scripts (not on the training path).
- [`legacy/`](legacy/) — frozen earlier work. *Don't extend without confirming with the user.*

For the full directory map and how the pieces wire together, see
[`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md).
For implementation details, see [`../docs/METHODS.md`](../docs/METHODS.md).
For the chronology of *why* this layout exists, see
[`../docs/PROJECT_TIMELINE.md`](../docs/PROJECT_TIMELINE.md).

## Where to put new code

| New code is… | Put it in |
|---|---|
| A new reward term, env tweak, or training option | `walker2d/ppo_walker2d_phase.py` |
| A new render mode or comparison tool | `walker2d/` (new file) |
| A new diagnostic / sanity-check script | `diagnostics/` |
| A revival of the 3D musculoskeletal track | new `musculoskeletal/` (NOT under `legacy/`) — see [`../docs/LEGACY_TRACKS.md`](../docs/LEGACY_TRACKS.md) |
| A new AMP / AIRL / DTW / MJX experiment | new sibling dir under `src/` |

The default rule is "extend the active pipeline." Reach into `legacy/`
only after confirming with the user that the legacy track is being
revisited.
