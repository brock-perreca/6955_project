# Reports

**Purpose:** index of formal write-ups.
**Read this when:** you need the canonical methods/results description
for a writeup section, or to compare current scope against the
original proposal.

The on-disk layout is:

| File | What it is | Status |
|---|---|---|
| `Advanced_AI_Project_Report.pdf` | **Original proposal**: "Lab-to-Field Transfer in Musculoskeletal Reinforcement Learning." 6-condition 3D 80-muscle MyoLeg study comparing OpenCap markerless vs lab-grade reference. | Historical motivation only. The project pivoted; the current scope is in `writeup_filled_1.docx`. See [`../PROJECT_TIMELINE.md`](../PROJECT_TIMELINE.md) for the pivot story. |
| `writeup_filled_1.docx` | **Current authoritative writeup** — "Learning Human-Like Bipedal Gait from Mocap Demonstrations via Phase-Conditioned Imitation and Adversarial Style Rewards" (Keller & Pereca). Includes problem statement, methods (DeepMimic + AMP + AIRL), three hypotheses (H1/H2/H3), experimental design, and results. | The thing to read for scope, methods, and results. |
| `writeup_extracted.txt` | Plain-text extraction of `writeup_filled_1.docx` for grep / quick reading. | Auto-generated; regenerate after edits to the .docx. |
| `methods_analysis.docx` | Earlier methods analysis document. | Pre-pivot. |

## Hypothesis labels (writeup §5.1)

These appear repeatedly in the docs and code review:

- **H1.** Phase-conditioned observation + multi-term imitation reward
  produces qualitatively better gait than a phase-blind policy with only
  forward-velocity reward.
- **H2.** The multi-term reward is exploitable: each unconstrained DoF
  produces a characteristic degenerate strategy (Goodhart's-Law cases —
  see [`../REWARD_DESIGN.md`](../REWARD_DESIGN.md)).
- **H3.** AMP with fewer than ~100 parallel envs produces discriminator
  collapse. Confirmed at 8 CPU envs.

## What if the docx and the code disagree?

The code is the source of truth for *what is currently running*. The
writeup is the source of truth for *scope, methods, and results
narrative*. If they disagree on a constant or a flag default:

- If it's a default value or a switch (e.g. swing-foot penalty weight,
  pitch termination threshold), the code is right — it has been tuned
  past the writeup snapshot.
- If it's a method description (e.g. "we use cubic spline resampling"),
  the writeup is right — the code might have a follow-up tweak that
  isn't worth amending the writeup over.

Update the writeup if the disagreement is material to the narrative;
update the code's docstrings/comments if the disagreement is operational.
