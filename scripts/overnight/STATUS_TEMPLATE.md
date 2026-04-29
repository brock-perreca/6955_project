# Overnight status — `<TIMESTAMP>`

> Live ledger. Master agent updates after each experiment finishes.
> If you wake up early and find this in the middle of a phase, the
> currently-running experiments are listed under "in progress".

## Phases

| phase | description | status |
|---|---|---|
| 0 | Setup + scaffolding sanity check | … |
| 1 | Kill the stiff-hip basin (8 experiments) | … |
| 2 | AMP / AIRL finetuned from working baseline (4 experiments) | … |
| 3 | Multi-step preview observation (3 experiments, 1 code change) | … |
| 4 | Per-joint DTW eval extension + re-rank | … |
| 5 | Different methods (SAC + reverse curriculum) | … |
| 6 | Synthesis + OVERNIGHT_SUMMARY.md | … |

## Experiments

| name | phase | status | wallclock | score | notes |
|---|---|---|---|---|---|
| _(populated as runs finish)_ | | | | | |

## In progress

- _(none)_

## Failures

- _(none)_

## Top 5 picks (auto-updated by `rank_runs.py`)

See [`RANKING.md`](RANKING.md). Watch the MP4s for the top 5 before trusting
the ranking.
