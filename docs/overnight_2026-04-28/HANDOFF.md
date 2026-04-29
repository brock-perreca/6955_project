# Overnight handoff — 2026-04-28

> **You are the master agent for an autonomous overnight run.** This
> file is your charter. Read it end-to-end before doing anything. The
> total budget is **~10 hours of wall-clock time** on a CPU-only Windows
> 11 box with 16 cores. The user (Brock) goes to bed when you start
> and reviews your output in the morning.

---

## 1. Read this first — the project state

Read these in order. Don't skim.

1. [`CLAUDE.md`](../../CLAUDE.md) — orientation hub. The "narrative
   arc" is the framing for what you're trying to accomplish.
2. [`docs/RESTART_LOG.md`](../RESTART_LOG.md) — the corrected-reference
   restart. **Batches 1 and 2 are done; you are starting from there.**
   Batch 2's `xvel-5M` is the current best policy and the parent for
   most of your experiments.
3. [`docs/REWARD_DESIGN.md`](../REWARD_DESIGN.md) — every reward term
   and the exploit it closes. **Read the warning at the top.**
4. [`docs/METHODS.md § Reward`](../METHODS.md#reward-weighted-sum-per-joint-scaled),
   §RSI, §Termination, §Adversarial-imitation tracks.
5. [`src/walker2d/ppo_walker2d_phase.py`](../../src/walker2d/ppo_walker2d_phase.py)
   constructor of `Walker2dPhaseAware` (lines ~135–250) and the `step`
   method (~328–432). The reward and exploit-patch knobs live here.
6. [`src/walker2d/amp_walker2d.py`](../../src/walker2d/amp_walker2d.py)
   and [`src/walker2d/airl_walker2d.py`](../../src/walker2d/airl_walker2d.py)
   docstrings + CLI sections.
7. [`docs/papers/papers.md`](../papers/papers.md) — when an experiment
   is paper-faithful (DeepMimic Eq. 6, AMP LSGAN, AIRL shaping
   potential), open the actual PDF in `docs/papers/` and verify the
   formula before coding.

## 2. The research question (do not lose sight of this)

The user's stated goal: **a 2D Walker2d that walks from imitation data
and produces biomechanically reasonable emergent behaviour, with
minimal hand-crafted reward shaping.** The current best policy
(`results/restart_b2_xvel/`) walks but with two visible defects:

1. **Stiff hips.** Hip excursion ~14° vs reference ~43°. The
   mean-of-squares pose reward over 6 joints is hiding this — `r_pose
   ≈ 0.56` while one joint sits at ~0°, because the other 5 satisfy
   the per-step mean.
2. **3× cadence.** Stride period 0.32s vs reference 1.12s. Downstream
   of stiff hips: foot can't reach the reference x-excursion without
   hip flexion, so the body strides ~3× per phase cycle.

Your job overnight is to produce a wide variety of candidate policies
that attack these two defects (Phases 1, 3, 5) AND make a real attempt
at the imitation-only "dream" path with AMP/AIRL (Phase 2).

**You will not pick the winner.** Brock reviews the MP4s in the
morning. Your job is to produce ~22 candidates with metrics, videos,
and per-experiment reports, ranked into a starting-point order.

## 3. The hard constraints

- **CPU-only.** No CUDA on this machine. Don't try to spawn an MJX
  port (it's a multi-day code change). 8 envs is the standard; 16 is
  the upper bound for one experiment.
- **The current-best policy is sacred.** Do not retrain into
  `results/restart_b2_xvel/` or any other pre-existing dir. Everything
  goes under `results/overnight_<TIMESTAMP>/<exp_name>/`.
- **Do not push to a remote.** Do not merge any branch. Do not
  delete or move any existing file under `results/`.
- **Do not edit `src/legacy/`.** Stay in `src/walker2d/`,
  `src/diagnostics/`, and `scripts/`.
- **No `git add -A`.** When committing experiment branches locally,
  enumerate the files explicitly.
- **Run scripts from the project root.** Path resolution depends on it
  (`PROJECT_ROOT = parents[2]`).
- **Logging discipline.** Every training run goes through
  `scripts/overnight/run_experiment.py`. Don't invent new wrappers.
- **Anti-Goodhart.** Headline `ep_rew_mean` and `r_pose` lied to us
  in batch 1. **Do not declare an experiment a success based on
  training scalars alone.** Inspect `eval_biomech.json` and inspect
  the structure of `preview.mp4` (file size, frame count). Better
  metrics are below; worse are detected by the per-exploit checklist
  in the report template.
- **Time budget per training run.** 5M steps × 8 envs ≈ 30 minutes
  *if alone on the box*. With two trainings in parallel, expect
  35–45 min each. Plan accordingly.

## 4. Compute layout

- **CPU:** 16 logical cores. 1 training run uses ~8 (one per env
  worker). You can run **2 trainings in parallel**; do not run 3.
- **Disk:** put everything under `results/overnight_<TIMESTAMP>/`.
  Each experiment subdir is ~50–200 MB (model, checkpoints, TB,
  preview.mp4). Budget < 5 GB total.
- **Worktrees:** OK to create as needed for code-change experiments.
  Brock will not look at them; they exist so your experiments can
  modify code in isolation without stepping on each other. Keep them
  under `../overnight_worktrees/<exp_name>/` (sibling to project
  root). Do **not** delete them at the end — Brock may want to
  inspect what you did.

## 5. What to produce — the artefact contract

For every experiment, the wrapper guarantees these files in
`results/overnight_<TS>/<exp>/`:

```
run.log                full stdout/stderr of training
train_cmd.txt          exact command line
train_meta.json        {start, end, exit_code, wallclock_s, ...}
model.zip              from training script
checkpoints/           periodic snapshots from training script
reference.npy          from training script
tb/                    TensorBoard scalars
eval_biomech.json      held-out biomech metrics on N deterministic eps
preview.mp4            ~12s deterministic rollout
REPORT.md              ← YOU fill this in (template provided)
```

The wrapper does not write `REPORT.md`. **Sub-agents must fill it from
the template at [`scripts/overnight/REPORT_TEMPLATE.md`](../../scripts/overnight/REPORT_TEMPLATE.md).**

Master maintains:

```
results/overnight_<TS>/STATUS.md     live status table (use STATUS_TEMPLATE.md)
results/overnight_<TS>/RANKING.md    auto-generated by rank_runs.py
results/overnight_<TS>/RANKING.json  same data, machine-readable
results/overnight_<TS>/OVERNIGHT_SUMMARY.md  ← final synthesis
```

## 6. Tools you have

| script | what it does |
|---|---|
| `scripts/overnight/run_experiment.py` | Train + eval_biomech + preview.mp4. **Always use this.** |
| `scripts/overnight/REPORT_TEMPLATE.md` | One-page report template. Sub-agents fill it. |
| `scripts/overnight/STATUS_TEMPLATE.md` | Master's live ledger. |
| `scripts/overnight/rank_runs.py` | Composite-score ranking of all runs. Re-run after each phase. |
| `src/diagnostics/eval_biomech.py` | Held-out biomech metrics. Already invoked by `run_experiment.py`. |
| `src/walker2d/render_phase.py` | MP4 rendering. Already invoked by `run_experiment.py`. |

`run_experiment.py` usage:

```
python scripts/overnight/run_experiment.py \
    --name <exp_name> \
    --out_dir results/overnight_<TS>/<exp_name> \
    --xml walker2d.xml \
    --eval_eps 6 --eval_steps 2500 \
    --preview_steps 1500 \
    -- \
    python src/walker2d/ppo_walker2d_phase.py \
        --ref_cycle assets/reference/gait_cycle_reference.npy \
        --num_envs 8 --total_steps 5000000 \
        --xvel_term 0.3
```

Everything after the literal `--` is the training command. The wrapper
appends `--out_dir <path>` automatically. **Use stock geometry**
(`--xml walker2d.xml`) — `walker2d_subject1.xml` is missing on this
checkout (see `docs/PROJECT_STATUS.md § Known gaps`).

## 7. The plan — 6 phases

You drive the schedule. Aim for 22 experiments + 1 code change in
Phase 3. Run two in parallel (background two `Bash` tool invocations).
Update `STATUS.md` after each finishes.

### Phase 0 — Setup (~30 min)

1. Create `results/overnight_<TIMESTAMP>/` (use ISO date+time).
2. Copy `STATUS_TEMPLATE.md` into it as `STATUS.md`. Fill in the
   timestamp. Add `EXPERIMENTS_QUEUED.md` with your full plan.
3. **Smoke test the wrapper** with a 100k-step run *of the proven
   recipe* to make sure all paths/flags work BEFORE you commit to
   real experiments. If the smoke test fails, fix the wrapper before
   continuing — a broken wrapper would silently invalidate the night.
   Smoke test command:
   ```
   python scripts/overnight/run_experiment.py \
       --name b0_smoke \
       --out_dir results/overnight_<TS>/b0_smoke \
       --xml walker2d.xml --eval_eps 1 --eval_steps 200 --preview_steps 100 \
       -- \
       python src/walker2d/ppo_walker2d_phase.py \
           --ref_cycle assets/reference/gait_cycle_reference.npy \
           --num_envs 8 --total_steps 100000 --xvel_term 0.3
   ```
   Expect this to take ~3 minutes. Verify the output files exist.

### Phase 1 — Kill the stiff-hip basin (~3.5 hours, 8 experiments)

All 8 experiments start from the **proven `xvel-5M` recipe** (vanilla
DeepMimic 4-term reward + `--xvel_term 0.3`) and change exactly ONE
knob. Single-knob design = readable result.

| name | change | code work? |
|---|---|---|
| `b1_hip2x` | `pose_joint_weights=(2,1,1,2,1,1)` | YES — see §7.1 below |
| `b1_hip4x` | `pose_joint_weights=(4,1,1,4,1,1)` | YES (same flag) |
| `b1_prod_reward` | Geometric mean over per-joint exps inside `r_pose` | YES — see §7.2 |
| `b1_min_joint`   | `r_pose = min_j exp(-k·diff_j²)` | YES — see §7.3 |
| `b1_ee30`        | `--ee_weight 0.30 --ee_scale 20` | NO (CLI flags exist) |
| `b1_velw5`       | `--vel_weight 0.50` | NO (CLI flag exists) |
| `b1_hipterm`     | Per-joint hip pose termination at 0.4 rad | YES — see §7.4 |
| `b1_energy`      | Add `-w·||torque||²` term, w=1e-3 | YES — see §7.5 |

Run pairs in parallel: `(hip2x, hip4x)`, `(prod_reward, min_joint)`,
`(ee30, velw5)`, `(hipterm, energy)`. Each pair ~40 min wall-clock.
Total ~3 hours training + 30 min eval/render overhead.

After Phase 1: re-run `rank_runs.py`. The top 1–2 by composite score
are the **Phase 1 winner(s)**, used as parents in Phase 2 and 3.

#### §7.1 Adding a `--pose_joint_weights` CLI flag

The constructor of `Walker2dPhaseAware` already accepts `pose_joint_weights`
(see `ppo_walker2d_phase.py:153`). It is NOT yet wired into the env's
`step` method (the current `r_pose = exp(-k_pose * mean(diff²))` line
ignores it) and is NOT exposed as a CLI flag.

You need:
1. In `Walker2dPhaseAware.__init__`: store `self._pose_joint_weights`
   as a `np.array(..., dtype=np.float32)` of length 6.
2. In `Walker2dPhaseAware.step`: change
   `r_pose = exp(-self._k_pose * np.mean(diff**2))`
   to
   `r_pose = exp(-self._k_pose * np.mean(self._pose_joint_weights * diff**2))`.
3. In the CLI, add `--pose_joint_weights` accepting 6 floats, default
   `(1,1,1,1,1,1)`. Pass it through `make_env`.

Use a worktree for this — multiple Phase 1 experiments will use the
flag. Branch: `overnight/per_joint_pose_weights`.

#### §7.2 `--product_reward` (geometric mean across joints)

The CLI flag exists but is currently a no-op (see METHODS.md
"--product_reward switches imit_r from arithmetic to geometric mean of
per-joint exps" — the comment in the file explicitly notes the env's
`product_reward` kwarg is not used). Wire it up:

```python
if self._product_reward:
    per_joint = np.exp(-self._k_pose * (diff ** 2))   # 6-vector
    r_pose = float(np.prod(per_joint) ** (1.0/6.0))   # geometric mean
else:
    r_pose = float(np.exp(-self._k_pose * np.mean(diff ** 2)))
```

Branch: `overnight/product_reward`.

#### §7.3 `--min_joint_pose` (worst-joint floor)

New flag. When set, `r_pose = min_j exp(-k_pose · diff_j²)`. This is
the most aggressive fix for the loophole — one bad joint kills the
whole pose reward. Likely produces unstable training (the policy may
not be able to escape the cliff edge); document the failure mode if
so. Branch: `overnight/min_joint_pose`.

#### §7.4 Per-joint hip pose termination

Currently `pose_term_thresh` applies to *all* of hip/knee. Add
`--hip_term_thresh` that *only* terminates on hip deviation. The hip
indices in `diff` are 0 and 3.

```python
hip_dev = max(abs(diff[0]), abs(diff[3]))
hip_term = hip_dev > self._hip_term_thresh
```

Add `hip_term` to the termination logic and the cause string. Default
threshold sentinel `9999.0` (off). For this experiment use `0.4` rad.
Branch: `overnight/hip_term`.

#### §7.5 Energy / torque-squared penalty

Add `--energy_weight` (default 0.0). When >0, subtract
`energy_weight · sum(action²)` from the reward (action is in normalized
torque space, ±1). The default ctrl_cost is 1e-3; try 1e-2 and 5e-3.
Make this exp use `--energy_weight 5e-3` and the ctrl_cost left in
place. Branch: `overnight/energy_penalty`.

### Phase 2 — AMP / AIRL from a working baseline (~2.5 hours, 4 experiments)

The collapse-at-8-envs problem is *cold-start collapse*: the
discriminator achieves near-perfect separation before the policy
produces walking transitions, so the gradient vanishes. Hypothesis:
**warm-starting from a working policy puts the policy on the expert
manifold from step 1**, so the discriminator can't trivially separate
and has to actually learn a useful style signal.

| name | command (abbreviated; full setup = `--num_envs 8 --total_steps 5000000`) |
|---|---|
| `b2_amp_ft_xvel`   | `amp_walker2d.py … --finetune results/restart_b2_xvel/model.zip` |
| `b2_amp_ft_winner` | `amp_walker2d.py … --finetune results/overnight_<TS>/<phase1_winner>/model.zip` |
| `b2_airl_ft_winner`| `airl_walker2d.py … --finetune results/overnight_<TS>/<phase1_winner>/model.zip` |
| `b2_amp_16env`     | `amp_walker2d.py … --num_envs 16 --total_steps 5000000` (no finetune; pushes the env-count threshold up) |

Run `(b2_amp_ft_xvel, b2_amp_16env)` in parallel, then
`(b2_amp_ft_winner, b2_airl_ft_winner)` in parallel. ~80 min × 2 = ~3
hours.

For each AMP/AIRL run, **record `frac_expert` over time from the
TensorBoard log** in REPORT.md. If it crashes to 0 within the first
10% of training, the discriminator collapsed — say so. If it stays in
[0.3, 0.7] the discriminator is healthy.

If any AMP run holds the working gait for the full 5M steps without
degrading, **flag it as `MUST_WATCH` in your final summary**. That's
the writeup money shot.

### Phase 3 — Multi-step preview observation (~1.5 hours, 3 experiments)

This is the only phase with a substantive code change. Hypothesis:
giving the policy a preview window of upcoming reference frames lets
it *anticipate* hip flexion and stop being late.

Code change in worktree `overnight/preview_obs`:

1. Add `--preview_k` CLI flag (default 1; 1 = current behaviour).
2. In `Walker2dPhaseAware.__init__`, store `self._preview_k`. Update
   `OBS_DIM = BASE_OBS + N_REF * preview_k + N_PHASE`.
3. In `_get_obs`, replace `q_ref = self._reference[self._phase]` with
   ```python
   idxs = (self._phase + np.arange(self._preview_k)) % self._ref_len
   q_ref_window = self._reference[idxs].reshape(-1)   # (N_REF * K,)
   ```
   and concatenate that into the obs.
4. Update the `observation_space` shape accordingly.
5. CRITICAL: render_phase.py and eval_biomech.py construct
   `Walker2dPhaseAware` — they need to know about preview_k too. Save
   the preview_k value to `<run_dir>/env_kwargs.json` at training time
   and have the renderer/eval read it. (See `eval_biomech.py:253` and
   `render_phase.py:155` — both currently call
   `Walker2dPhaseAware(reference=..., xml_file=...)` with no extras.)
6. PPO MLP input layer auto-resizes to the obs_space, so no SB3 change
   is needed. Verify with the smoke test that obs shape changed.

| name | preview_k | parent |
|---|---|---|
| `b3_preview_k4`        | 4 | xvel-recipe |
| `b3_preview_k4_winner` | 4 | Phase 1 winner |
| `b3_preview_k8`        | 8 | xvel-recipe |

### Phase 4 — DTW eval extension + re-rank (~30 min, no training)

`eval_biomech.py` already computes hip-knee DTW. Extend it to compute
DTW over all 6 joints (`hip_r, knee_r, ankle_r, hip_l, knee_l, ankle_l`)
on the same gait-cycle slice, name the new key `all_joints_dtw`.
Re-run eval on every Phase 1–3 result and re-run `rank_runs.py`.
Update `RANKING.md`.

(Optional: include `all_joints_dtw` in the composite score in
`rank_runs.py`. If you do, document the new formula in `RANKING.md`.)

### Phase 5 — Different methods (~2.5 hours, 2 experiments)

Two experiments. Run them in parallel.

#### `b5_sac` — SAC instead of PPO (~1.5–2.5 hours)

SAC is off-policy and sample-efficient. Hypothesis: PPO's on-policy
exploration limits how fast the policy can escape stiff-hip; SAC may
explore more aggressively. New script
`src/walker2d/sac_walker2d_phase.py`. Use SB3's `SAC`, same env, same
reward. 1M steps total (SAC needs fewer env steps; warm replay buffer
fills quickly). Default SB3 SAC hyperparameters except `learning_rate=3e-4`,
`batch_size=256`, `buffer_size=300000`. Use the proven `--xvel_term 0.3`
recipe. Branch: `overnight/sac`.

If SAC gets unstable (NaNs, value blowup), document and move on. Don't
spend more than 2 hours on this single experiment.

#### `b5_reverse_curriculum` — slow then fast (~2 hours, two stages)

Stage A: train with `--v_target 0.6` (slower treadmill) for 3M steps
on the xvel recipe. Slower target gives the policy more time per
phase frame to flex the hip.

Stage B: finetune from Stage A with `--v_target 1.25` for 2M steps.

Both stages go in **the same out_dir** with stage-A checkpoints
preserved. Use `b5_reverse_curriculum_a` and `b5_reverse_curriculum_b`
naming so each stage has its own sub-dir.

Note: `--v_target` is in the env constructor (see
`ppo_walker2d_phase.py:155`) but you must verify a CLI flag exists.
Grep the script. If absent, add it. Branch: `overnight/v_target_cli`
(if needed).

### Phase 6 — Synthesis (~30 min)

1. Re-run `rank_runs.py` over the whole night.
2. Write `results/overnight_<TS>/OVERNIGHT_SUMMARY.md`:
   - **Top 5 picks** with thumbnail summaries and links to MP4.
   - **3 must-watch videos**: the highest composite score, the most
     interesting failure mode, and the AMP/AIRL run that came closest
     to holding gait.
   - **Per-phase diagnosis**: did Phase 1 escape the stiff hip? Did
     Phase 2 confirm or refute the discriminator-from-good-baseline
     hypothesis? Did Phase 3's preview obs help?
   - **What you would queue next**, given another 10 hours.
3. Mark all phases ✅ in `STATUS.md`.

## 8. Sub-agent prompting pattern

Use the `Agent` tool with `subagent_type="general-purpose"` for each
experiment. The sub-agent prompt template:

> You are running ONE experiment as part of an overnight research
> sweep. The master agent's plan is in
> `docs/overnight_2026-04-28/HANDOFF.md` — read §3 (constraints) and
> §5 (artefact contract) before doing anything.
>
> Your experiment: **`<exp_name>`**.
>
> **Hypothesis:** <one sentence>
>
> **Setup:** <code changes if any, branch name, exact CLI command>
>
> **Out dir:** `results/overnight_<TS>/<exp_name>/`
>
> **Steps:**
> 1. (If code change) `git worktree add ../overnight_worktrees/<exp_name>
>    -b overnight/<branch_name>` from project root. Make the code
>    change. Verify it works with a 50k-step smoke run before
>    launching the full 5M-step training.
> 2. Launch training via `scripts/overnight/run_experiment.py` (do NOT
>    invoke `ppo_walker2d_phase.py` directly — the wrapper handles
>    eval + render + meta).
> 3. After training finishes, READ the `eval_biomech.json`. Compare
>    against batch-2's `xvel-5M` numbers (in `RESTART_LOG.md`). Run
>    the per-exploit checklist in `REPORT_TEMPLATE.md`.
> 4. Fill `<out_dir>/REPORT.md` from the template. Be honest:
>    `r_pose` improvements without hip excursion improvements are
>    NOT a success.
> 5. Reply to me with: status (DONE/FAILED), the verdict
>    (KEEP/DROP/FOLLOW-UP), the headline biomech numbers (cadence,
>    DTW, ep_len), and a one-line summary.
>
> Time budget: <X minutes wall-clock for this experiment>. Do not
> retry on failure — if the training crashes, write
> `<out_dir>/FAILED.md` with the traceback and report failure.

Spawn at most 2 sub-agents in parallel (compute constraint).

## 9. Failure-mode playbook

- **Wrapper smoke test fails** → fix the wrapper. Most likely cause:
  CLI flag mismatch with the underlying script. Check the underlying
  script's `--help`.
- **Training crashes mid-run** → mark experiment FAILED, move on. Do
  not retry. Note the crash in `STATUS.md § Failures`.
- **Sub-agent spawns unrelated work** → ignore its output, mark its
  experiment as needing rerun, queue it again with a more constrained
  prompt.
- **Two experiments collide on the GPU/disk** → you have neither GPU
  nor a busy disk. CPU contention will just slow each by ~30%. That's
  expected and budgeted for.
- **An experiment looks perfect but has weird metrics** → trust the
  metrics. The lesson from this morning's stiff-hip discovery is
  that "perfect headline number" can hide gross failure. Note the
  discrepancy in REPORT.md and flag for Brock's review.
- **You finish the plan in <10 hours** → great. Spend the remainder
  on (a) running 2 more seeds of the top Phase 1 / Phase 2
  experiments (single-seed results are noisy), or (b) trying a
  combined experiment (`hip2x + preview_k4 + ee30` together). Do NOT
  start a new architectural change you can't finish in the remaining
  time.

## 10. End-of-night exit criteria

You're done when:

- [ ] `STATUS.md` shows all 6 phases marked ✅ or with explicit notes
      about what was skipped and why.
- [ ] `RANKING.md` exists with at least 15 ranked experiments.
- [ ] `OVERNIGHT_SUMMARY.md` exists with 3 named must-watch videos
      and per-phase diagnosis.
- [ ] All MP4s are renderable (file size > 0, frames > 0).
- [ ] `git status` is clean on the `brock` branch (your work is on
      `overnight/*` branches only).
- [ ] Every experiment dir has either `REPORT.md` (success) or
      `FAILED.md` (failure). No silent skips.

Good luck. Be honest in your reports. Brock would rather see "this
didn't work and here's why" for 22 experiments than "here are 22 wins"
that turn out to all be stand-and-wiggle.
