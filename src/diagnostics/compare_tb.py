"""
compare_tb.py — print a side-by-side comparison of TensorBoard scalars
across multiple runs.

Reads `<run>/tb/PPO_1/events.*` (the SB3 default layout) and prints one
row per scalar tag, one column per run, showing the *last* recorded
value (with the step). Useful as a fast "did training go anywhere?"
check after a batch finishes — open this instead of TB for the headline
numbers.

Usage:
    python src/diagnostics/compare_tb.py results/run_A results/run_B
    python src/diagnostics/compare_tb.py --tags rollout/ep_rew_mean reward/r_pose results/*/

By default shows a curated set of headline scalars
(rollout/{ep_rew_mean,ep_len_mean}, reward/{r_pose,r_vel,r_ee,r_root},
term/{pitch,height,ankle,pose,xvel,other}).
"""
import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


_DEFAULT_TAGS = [
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "reward/r_pose",
    "reward/r_vel",
    "reward/r_ee",
    "reward/r_root",
    "reward/contact_r",
    "reward/swing_pen",
    "term/height",
    "term/pitch",
    "term/ankle",
    "term/pose",
    "term/xvel",
    "term/other",
]


def _last(ea: EventAccumulator, tag: str) -> tuple[int, float] | None:
    if tag not in ea.Tags().get("scalars", []):
        return None
    evs = ea.Scalars(tag)
    if not evs:
        return None
    return evs[-1].step, float(evs[-1].value)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Side-by-side TB scalar comparison across runs."
    )
    p.add_argument("runs", nargs="+",
                   help="Run dirs (each contains tb/PPO_1/events.*)")
    p.add_argument("--tags", nargs="*", default=None,
                   help="Tag list (defaults to a curated headline set)")
    args = p.parse_args()

    tags = args.tags or _DEFAULT_TAGS
    accs: list[tuple[str, EventAccumulator]] = []
    for r in args.runs:
        rp = Path(r)
        tb_dir = rp / "tb" / "PPO_1"
        if not tb_dir.exists():
            print(f"[skip] {r}: no tb/PPO_1 dir")
            continue
        ea = EventAccumulator(str(tb_dir))
        ea.Reload()
        accs.append((rp.name, ea))

    if not accs:
        print("No runs with TB data found.")
        return

    name_w = max(20, max(len(n) for n, _ in accs) + 2)
    tag_w  = max(len(t) for t in tags) + 2

    header = "tag".ljust(tag_w) + "".join(n.ljust(name_w) for n, _ in accs)
    print(header)
    print("-" * len(header))
    for tag in tags:
        row = tag.ljust(tag_w)
        for _, ea in accs:
            r = _last(ea, tag)
            if r is None:
                row += "—".ljust(name_w)
            else:
                step, val = r
                row += f"{val:>9.3f} @ {step//1000:>5}k".ljust(name_w)
        print(row)


if __name__ == "__main__":
    main()
