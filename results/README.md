# `results/` — training outputs

One directory per training run. Each contains:

```
<run-name>/
  reference.npy           # the reference array used at training time (re-loaded at render)
  model.zip               # final SB3 PPO policy
  checkpoints/
    model_<N>_steps.zip   # periodic snapshots, ~every 5M env steps
```

The `reference.npy` is saved per-run so `render_phase.py` can re-create
the *same* env (with the *same* gait cycle) the policy was trained
against. This is why the renderer takes a `result_dir` and not just a
model path.

## Notable runs

For the canonical / canonical-failure runs and what each one shows, see
[`../docs/PROJECT_STATUS.md`](../docs/PROJECT_STATUS.md) ("Comparison
runs on disk") and [`../docs/RUN_LOG.md`](../docs/RUN_LOG.md).

## Render any run

```bash
# By model.zip (final)
python src/walker2d/render_phase.py results/<run-name>:final

# By integer checkpoint step (looks under checkpoints/)
python src/walker2d/render_phase.py results/<run-name>:60000000:"60M"

# Compare back-to-back
python src/walker2d/render_phase.py \
    results/<run-A>:50000000:"A 50M" \
    results/<run-B>:final
```

Default `--xml` is `walker2d_subject1.xml` (the Subject-1-scaled MJCF in
`assets/mjcf/`). For runs trained against stock Walker2d, override with
`--xml walker2d.xml`.

## What's gitignored

Most run directories *are* checked in (they include `model.zip` files,
which are small SB3 pickles), but stage outputs like TensorBoard event
files and per-rollout dumps may be gitignored. Check `.gitignore` if you
expected something to be committed but isn't.
