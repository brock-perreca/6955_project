# Kickoff prompt

Paste this verbatim into a fresh Claude Code session in this repo
when you're ready to start the overnight run.

---

You are the master agent for an autonomous overnight research sweep
on the Walker2d gait imitation project. Read the full charter at
`docs/overnight_2026-04-28/HANDOFF.md` end-to-end before doing
anything — it specifies the constraints, the artefact contract,
the 6-phase plan, the sub-agent prompting pattern, and the exit
criteria.

You have ~10 hours of wall-clock budget on a CPU-only Windows 11
box (16 cores, no CUDA). Run two trainings in parallel where the
plan calls for it; never three. Your goal is to produce ~22 ranked
candidate policies with metrics, MP4s, and per-experiment reports
for the user (Brock) to review in the morning. You are NOT picking
the winner — you are producing a pre-screened slate.

Scaffolding is already in place:

- `scripts/overnight/run_experiment.py` — the only way to launch a
  training run. Wraps train + eval_biomech + preview.mp4 + meta.
- `scripts/overnight/REPORT_TEMPLATE.md` — fill this in for every
  experiment.
- `scripts/overnight/STATUS_TEMPLATE.md` — copy into each overnight
  root and update live.
- `scripts/overnight/rank_runs.py` — composite-score ranking.
  Re-run after each phase.

**Start by:**

1. Reading `CLAUDE.md`, `docs/RESTART_LOG.md`, `docs/REWARD_DESIGN.md`,
   and the HANDOFF charter in full.
2. Skimming `src/walker2d/ppo_walker2d_phase.py` to confirm the CLI
   flags the plan assumes (`--xvel_term`, `--ee_weight`, `--ee_scale`,
   `--vel_weight`, `--out_dir`, `--finetune`, `--num_envs`,
   `--total_steps`, `--ref_cycle`) all exist. The plan calls out
   which flags need to be added (`--pose_joint_weights`,
   `--product_reward` wiring, `--min_joint_pose`, `--hip_term_thresh`,
   `--energy_weight`, `--preview_k`, possibly `--v_target`).
3. Setting up `results/overnight_<YYYYMMDD-HHMM>/` with the
   STATUS template populated.
4. Running the Phase 0 smoke test (~3 min) before committing to real
   experiments.
5. Then executing Phases 1 → 6 in sequence, keeping STATUS.md
   updated.

Be honest. Headline numbers lie — this morning we discovered that
`r_pose ≈ 0.56` was hiding a stiff-hip exploit because mean-of-squares
over 6 joints lets one bad joint hide. Every report you write must
include the per-exploit anti-Goodhart checklist. Brock will review
your top 5 MP4s in the morning, so the ranking is a starting point
for visual review, not a verdict.

Check the tools we have, and feel free to come up with new tools to validate your work. More feedback loops for agents to improve future development is always helpful. If you start hitting my Claude code limit or extra usage, please try to figure out some way to pause and continue when my 5-hour limit resets. 

Begin.
