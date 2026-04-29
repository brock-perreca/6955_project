"""
rank_runs.py
────────────
Rank every overnight experiment that has an eval_biomech.json on disk.

Composite score (higher is better):

    score = clip(ep_len_steps__median / 2500, 0, 1)            # survival
          + clip(1 - |stride_period_s - 1.12| / 1.12, 0, 1)    # cadence match
          + clip(1 - hip_knee_dtw / 0.30, 0, 1)                # shape fidelity
          + clip(1 - lr_stride_asymmetry / 0.30, 0, 1)         # symmetry
          + clip(1 - swing_drag_frac, 0, 1)                    # no toe-drag

Each term is in [0, 1]; max score = 5.0. Reference cycle is the Subject 1
baseline (1.25 m/s, ~1.12 s stride period, ~17 strides per 20s window).

Usage
─────
    python scripts/overnight/rank_runs.py results/overnight_<ts>/
        # or, omit arg to scan results/overnight_*/

Writes:
    <root>/RANKING.md         human-readable table sorted by composite
    <root>/RANKING.json       same data, machine-readable
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _score(summary: dict) -> tuple[float, dict]:
    """Composite score + per-term breakdown."""
    g = lambda k, default=math.nan: float(summary.get(k, default))

    survival = _clip(g("ep_len_steps__median", 0) / 2500.0, 0, 1)

    sp = g("stride_period_s__median", math.nan)
    if math.isnan(sp):
        cadence = 0.0
    else:
        cadence = _clip(1.0 - abs(sp - 1.12) / 1.12, 0, 1)

    dtw = g("hip_knee_dtw__median", math.nan)
    shape = 0.0 if math.isnan(dtw) else _clip(1.0 - dtw / 0.30, 0, 1)

    asym = g("lr_stride_asymmetry__median", math.nan)
    sym  = 0.0 if math.isnan(asym) else _clip(1.0 - asym / 0.30, 0, 1)

    drag = g("swing_drag_frac__median", 0.0)
    no_drag = _clip(1.0 - drag, 0, 1)

    breakdown = {
        "survival":  round(survival, 3),
        "cadence":   round(cadence,  3),
        "shape":     round(shape,    3),
        "symmetry":  round(sym,      3),
        "no_drag":   round(no_drag,  3),
    }
    return round(survival + cadence + shape + sym + no_drag, 3), breakdown


def collect(root: Path) -> list[dict]:
    out = []
    for run_dir in sorted(root.glob("*/")):
        eval_json = run_dir / "eval_biomech.json"
        if not eval_json.exists():
            continue
        try:
            data = json.loads(eval_json.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[skip] {run_dir.name}: bad JSON ({e})")
            continue
        for entry in data:
            summary = entry.get("summary", {})
            score, terms = _score(summary)
            out.append({
                "name":      run_dir.name,
                "label":     entry.get("label", run_dir.name),
                "score":     score,
                "terms":     terms,
                "summary":   summary,
                "run_dir":   str(run_dir),
                "preview":   str(run_dir / "preview.mp4"),
                "report":    str(run_dir / "REPORT.md"),
            })
    out.sort(key=lambda r: r["score"], reverse=True)
    return out


def write_markdown(rows: list[dict], path: Path) -> None:
    lines = ["# Overnight ranking", ""]
    lines.append(
        "Composite score is a sum of 5 terms in [0,1]: survival, "
        "cadence-match, hip-knee DTW, L-R symmetry, no-toe-drag. "
        "Max = 5.0. Higher is better. **This is a starting point for "
        "your visual review, not a verdict.** Watch the top 5 MP4s "
        "before trusting the ranking."
    )
    lines.append("")
    lines.append("| rank | name | score | survival | cadence | shape | sym | no_drag | preview |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        t = r["terms"]
        lines.append(
            f"| {i} | `{r['name']}` | **{r['score']:.2f}** | {t['survival']:.2f} | "
            f"{t['cadence']:.2f} | {t['shape']:.2f} | {t['symmetry']:.2f} | "
            f"{t['no_drag']:.2f} | [mp4]({r['preview']}) |"
        )
    lines.append("")
    lines.append("## Detailed metrics")
    lines.append("")
    for r in rows:
        s = r["summary"]
        lines.append(f"### `{r['name']}` (score {r['score']:.2f})")
        lines.append("")
        for key in (
            "ep_len_steps__median", "n_strides_detected__median",
            "stride_period_s__median", "cadence_steps_per_min__median",
            "double_support_frac__median", "swing_drag_frac__median",
            "lr_stride_asymmetry__median", "peak_vgrf_bw__median",
            "hip_knee_dtw__median",
        ):
            if key in s:
                lines.append(f"- `{key}`: {s[key]}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=None,
                    help="Overnight root dir; default scans results/overnight_*/")
    args = ap.parse_args()

    if args.root:
        roots = [Path(args.root).resolve()]
    else:
        roots = sorted((PROJECT_ROOT / "results").glob("overnight_*/"))

    for root in roots:
        rows = collect(root)
        if not rows:
            print(f"[{root.name}] no eval_biomech.json found yet")
            continue
        (root / "RANKING.json").write_text(json.dumps(rows, indent=2),
                                           encoding="utf-8")
        write_markdown(rows, root / "RANKING.md")
        print(f"[{root.name}] ranked {len(rows)} runs -> {root / 'RANKING.md'}")


if __name__ == "__main__":
    main()
