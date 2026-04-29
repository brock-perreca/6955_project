"""
run_experiment.py
─────────────────
Wrapper used by the overnight master agent and its sub-agents.

A single experiment is: train one PPO/AMP/AIRL run with a given CLI
spec, evaluate the resulting policy with eval_biomech, render a short
deterministic MP4, and write a one-page markdown report.

The wrapper exists so every sub-agent produces *identical* on-disk
artefacts in the same layout, regardless of which method or knobs the
experiment is testing.

Layout written under <out_dir>:
    <out_dir>/
        run.log              full stdout/stderr of the training command
        train_cmd.txt        the exact command line that was executed
        train_meta.json      {start_time, end_time, exit_code, ...}
        model.zip            (from training)
        reference.npy        (from training)
        checkpoints/         (from training)
        tb/                  (from training)
        eval_biomech.json    biomech metrics on N deterministic episodes
        preview.mp4          ~30s deterministic rollout
        REPORT.md            one-page markdown summary (filled by sub-agent)

Usage
─────
    python scripts/overnight/run_experiment.py \\
        --name b1_xvel_baseline \\
        --out_dir results/overnight_<ts>/b1_xvel_baseline \\
        --xml walker2d.xml \\
        --eval_eps 6 --eval_steps 2500 \\
        --preview_steps 1500 \\
        -- \\
        python src/walker2d/ppo_walker2d_phase.py \\
            --ref_cycle assets/reference/gait_cycle_reference.npy \\
            --num_envs 8 --total_steps 5000000 \\
            --xvel_term 0.3

Everything after the bare `--` is the training command. The wrapper
appends `--out_dir <out_dir>` to that command automatically (so
sub-agents don't have to remember the convention).

If the training command contains the literal token `__OUT_DIR__`,
it's replaced with the resolved out_dir instead of appended — useful
for non-PPO scripts that take the output path elsewhere.

The wrapper does NOT swallow training failures: it writes
train_meta.json with exit_code != 0, skips eval/render, and exits
non-zero. The sub-agent is expected to inspect the failure and write
a FAILED.md before reporting back.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_train_cmd(raw: list[str], out_dir: Path) -> list[str]:
    """Inject --out_dir or substitute __OUT_DIR__ as appropriate."""
    if not raw:
        raise SystemExit("No training command supplied after `--`.")
    if any(tok == "__OUT_DIR__" for tok in raw):
        return [str(out_dir) if tok == "__OUT_DIR__" else tok for tok in raw]
    # Default: append --out_dir <out_dir>. Most active scripts accept this.
    return list(raw) + ["--out_dir", str(out_dir)]


def run_training(cmd: list[str], out_dir: Path) -> dict:
    """Execute training; tee stdout/stderr to run.log; return meta dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    cmd_path = out_dir / "train_cmd.txt"
    cmd_path.write_text(" ".join(shlex.quote(c) for c in cmd) + "\n",
                        encoding="utf-8")

    meta = {
        "name":       out_dir.name,
        "out_dir":    str(out_dir),
        "cmd":        cmd,
        "start_time": _utc_now(),
        "host":       os.environ.get("COMPUTERNAME", "unknown"),
    }

    t0 = time.time()
    with open(log_path, "wb") as logf:
        proc = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )
        rc = proc.wait()
    meta["exit_code"]    = rc
    meta["wallclock_s"]  = round(time.time() - t0, 1)
    meta["end_time"]     = _utc_now()
    return meta


def run_eval(out_dir: Path, xml: str, eps: int, steps: int) -> int:
    eval_json = out_dir / "eval_biomech.json"
    spec      = f"{out_dir}:final:{out_dir.name}"
    cmd = [
        sys.executable, str(PROJECT_ROOT / "src" / "diagnostics" / "eval_biomech.py"),
        spec, "--xml", xml,
        "--eps",   str(eps),
        "--steps", str(steps),
        "--out",   str(eval_json),
    ]
    log = open(out_dir / "eval.log", "wb")
    rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT), stdout=log, stderr=subprocess.STDOUT)
    log.close()
    return rc


def run_render(out_dir: Path, xml: str, steps: int) -> int:
    mp4_path = out_dir / "preview.mp4"
    spec     = f"{out_dir}:final:{out_dir.name}"
    cmd = [
        sys.executable, str(PROJECT_ROOT / "src" / "walker2d" / "render_phase.py"),
        spec, "--xml", xml,
        "--eps",   "1",
        "--steps", str(steps),
        "--mp4",   str(mp4_path),
    ]
    log = open(out_dir / "render.log", "wb")
    rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT), stdout=log, stderr=subprocess.STDOUT)
    log.close()
    return rc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name",     required=True, help="Experiment slug.")
    ap.add_argument("--out_dir",  required=True,
                    help="Output dir; created if missing. Becomes results/overnight_*/<name>.")
    ap.add_argument("--xml",      default="walker2d.xml",
                    help="MuJoCo XML for eval+render (default: walker2d.xml — stock geometry).")
    ap.add_argument("--eval_eps",     type=int, default=6)
    ap.add_argument("--eval_steps",   type=int, default=2500)
    ap.add_argument("--preview_steps", type=int, default=1500,
                    help="Frames in the preview MP4 (~12s @ 125Hz for 1500).")
    ap.add_argument("--skip_eval",   action="store_true")
    ap.add_argument("--skip_render", action="store_true")
    ap.add_argument("train_cmd", nargs=argparse.REMAINDER,
                    help="Everything after `--` is the training command.")
    args = ap.parse_args()

    raw = args.train_cmd
    if raw and raw[0] == "--":
        raw = raw[1:]
    if not raw:
        raise SystemExit("Pass the training command after `--`.")

    out_dir = Path(args.out_dir).resolve()
    cmd     = _resolve_train_cmd(raw, out_dir)

    print(f"[{args.name}] training: {' '.join(shlex.quote(c) for c in cmd)}")
    meta = run_training(cmd, out_dir)
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2),
                                             encoding="utf-8")
    if meta["exit_code"] != 0:
        print(f"[{args.name}] TRAINING FAILED (exit={meta['exit_code']}); "
              f"see {out_dir / 'run.log'}")
        sys.exit(meta["exit_code"])

    print(f"[{args.name}] training done in {meta['wallclock_s']}s")

    if not args.skip_eval:
        rc = run_eval(out_dir, args.xml, args.eval_eps, args.eval_steps)
        print(f"[{args.name}] eval_biomech rc={rc} -> {out_dir / 'eval_biomech.json'}")
    if not args.skip_render:
        rc = run_render(out_dir, args.xml, args.preview_steps)
        print(f"[{args.name}] preview rc={rc} -> {out_dir / 'preview.mp4'}")

    print(f"[{args.name}] DONE")


if __name__ == "__main__":
    main()
