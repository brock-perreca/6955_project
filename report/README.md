# Report

**Purpose:** all materials for the formal CS 6955 final write-up live
here — past write-ups, the in-flight writeup, the assignment rubric,
and the Overleaf template/style files used to compile the final PDF.

**Read this when:** you need the canonical methods/results description
for a write-up section, are about to edit the final write-up, or need
to compare current scope against the original proposal.

## On-disk layout

| File | What it is | Status |
|---|---|---|
| `Final_Project_Report.pdf` | **Assignment rubric** for the CS 6955 final report (Canvas handout). Describes the required sections (Abstract, Introduction, Related Work, Problem Statement, Method, Experimental Design, Results & Discussion, Conclusion & Future Work, Code link), 5–8 pages excluding references, Overleaf template required. | Authoritative spec for what the final PDF must contain. |
| `template.tex` | **Overleaf LaTeX template** referenced by the rubric. Top-level `.tex` file for the final write-up. | The thing to compile. |
| `project.sty` | LaTeX style file used by `template.tex`. | Required for compilation. |
| `sample.bib` | BibTeX bibliography stub for the template. | Replace/extend with the project's actual references (see [`../docs/papers/papers.md`](../docs/papers/papers.md)). |
| `writeup_filled_1.docx` | **Current authoritative narrative writeup** — "Learning Human-Like Bipedal Gait from Mocap Demonstrations via Phase-Conditioned Imitation and Adversarial Style Rewards" (Keller & Pereca). Includes problem statement, methods (DeepMimic + AMP + AIRL), three hypotheses (H1/H2/H3), experimental design, and results. | The thing to read for scope, methods, and results. To be migrated into `template.tex` for the final submission. |
| `writeup_extracted.txt` | Plain-text extraction of `writeup_filled_1.docx` for grep / quick reading. | Auto-generated; regenerate after edits to the .docx. |
| `methods_analysis.docx` | Earlier methods analysis document. | Pre-pivot. |
| `Advanced_AI_Project_Report.pdf` | **Original proposal**: "Lab-to-Field Transfer in Musculoskeletal Reinforcement Learning." 6-condition 3D 80-muscle MyoLeg study comparing OpenCap markerless vs lab-grade reference. | Historical motivation only. The project pivoted; the current scope is in `writeup_filled_1.docx`. See [`../docs/PROJECT_TIMELINE.md`](../docs/PROJECT_TIMELINE.md) for the pivot story. |

## Workflow for the final submission

1. Read `Final_Project_Report.pdf` to confirm the required sections.
2. Migrate the narrative content from `writeup_filled_1.docx` into
   `template.tex`, section by section, updating with the latest
   Tier-0 / hipopen / hiprelax / `b5_min_joint` results from
   [`../docs/PROJECT_STATUS.md`](../docs/PROJECT_STATUS.md) and
   [`../docs/RESTART_LOG.md`](../docs/RESTART_LOG.md).
3. Populate `sample.bib` with the references in
   [`../docs/papers/papers.md`](../docs/papers/papers.md).
4. Compile (Overleaf or local `pdflatex`) and review.

## Hypothesis labels (writeup §5.1)

These appear repeatedly in the docs and code review:

- **H1.** Phase-conditioned observation + multi-term imitation reward
  produces qualitatively better gait than a phase-blind policy with only
  forward-velocity reward.
- **H2.** The multi-term reward is exploitable: each unconstrained DoF
  produces a characteristic degenerate strategy (Goodhart's-Law cases —
  see [`../docs/REWARD_DESIGN.md`](../docs/REWARD_DESIGN.md)).
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
