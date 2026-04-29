"""
check_reference_jnt_range.py — does the active MJCF's joint range
contain the on-disk reference?

Pure static check; no policy, no PD, no physics integration. For each
joint of the gait-cycle reference, intersect the per-frame angle with
the active MJCF's `jnt_range` and report:

  - per-joint range vs. reference range (degrees)
  - fraction of cycle with q_ref outside the joint range (per side)
  - peak overshoot above the upper limit and below the lower limit

The motivation is that `render_reference_replay.py` calls `mj_forward`
without dynamics, so it shows the reference even when the joint range
forbids it. `mj_jntrange` constraint forces only enforce the limit
during *integration* — i.e. when a trained policy or PD controller is
running. So a kinematically-pretty replay can hide a reference the
trained model cannot reach.

Outputs (default):
    docs/figures/tier0/A1_reference_vs_jnt_range.png
    docs/figures/tier0/A1_reference_vs_jnt_range.json   (per-joint stats)

Usage:
    python src/diagnostics/check_reference_jnt_range.py
    python src/diagnostics/check_reference_jnt_range.py --xml walker2d_custom.xml
    python src/diagnostics/check_reference_jnt_range.py --xml walker2d_hiprelax.xml
"""
import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "walker2d"))
from ppo_walker2d_phase import load_ref_cycle  # noqa: E402

REF_PATH    = PROJECT_ROOT / "assets" / "reference" / "gait_cycle_reference.npy"
MJCF_ROOT   = PROJECT_ROOT / "assets" / "mjcf"
JOINT_NAMES = ["hip_r", "knee_r", "ankle_r", "hip_l", "knee_l", "ankle_l"]
# The joint names in the MJCF (in the same order qpos[3:9] uses).
MJCF_JOINTS = ["thigh_joint", "leg_joint", "foot_joint",
               "thigh_left_joint", "leg_left_joint", "foot_left_joint"]


def _resolve_xml(xml_arg: str) -> str:
    """Mirror Walker2dPhaseAware's resolution: bare filename → MJCF_ROOT
    if present, else fall through to the gym-bundled stock MJCF."""
    if xml_arg == "walker2d.xml":
        return xml_arg
    if Path(xml_arg).is_absolute():
        return xml_arg
    if (MJCF_ROOT / xml_arg).exists():
        return str(MJCF_ROOT / xml_arg)
    return xml_arg


def _load_jnt_ranges(xml_file: str) -> tuple[np.ndarray, str]:
    """Return (range[6, 2] in radians, friendly_path_label)."""
    if xml_file == "walker2d.xml":
        # Gym resolves bare 'walker2d.xml' from its bundled assets.
        env = gym.make("Walker2d-v4")
        env.reset()
        m = env.unwrapped.model
        label = "walker2d.xml (gym default)"
    else:
        m = mujoco.MjModel.from_xml_path(xml_file)
        label = xml_file
    rng = np.zeros((6, 2), dtype=np.float32)
    for i, name in enumerate(MJCF_JOINTS):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Joint {name!r} not in {label}")
        if not bool(m.jnt_limited[jid]):
            rng[i] = (-np.inf, np.inf)
        else:
            rng[i] = m.jnt_range[jid]
    return rng, label


def analyse(ref: np.ndarray, jnt_rng: np.ndarray) -> dict:
    n = len(ref)
    out = {}
    for i, name in enumerate(JOINT_NAMES):
        q   = ref[:, i]
        lo, hi = float(jnt_rng[i, 0]), float(jnt_rng[i, 1])
        out_above = q > hi
        out_below = q < lo
        out[name] = {
            "ref_lo_deg":    float(np.degrees(q.min())),
            "ref_hi_deg":    float(np.degrees(q.max())),
            "ref_rom_deg":   float(np.degrees(q.max() - q.min())),
            "jnt_lo_deg":    float(np.degrees(lo)) if np.isfinite(lo) else None,
            "jnt_hi_deg":    float(np.degrees(hi)) if np.isfinite(hi) else None,
            "frac_above":    float(out_above.mean()),
            "frac_below":    float(out_below.mean()),
            "frac_outside":  float((out_above | out_below).mean()),
            "peak_overshoot_above_deg": (
                float(np.degrees((q - hi)[out_above].max()))
                if out_above.any() else 0.0
            ),
            "peak_overshoot_below_deg": (
                float(np.degrees((lo - q)[out_below].max()))
                if out_below.any() else 0.0
            ),
        }
    return out


def plot(ref: np.ndarray, jnt_rng: np.ndarray, label: str,
         stats: dict, out_png: Path) -> None:
    n = len(ref)
    x = np.arange(n) / n  # phase ∈ [0, 1)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    fig.suptitle(
        f"Reference vs joint range  —  MJCF: {label}\n"
        f"Shaded band = MJCF jnt_range; black trace = reference q_ref(phase). "
        f"Anything outside the band is unreachable under physics.",
        fontsize=11,
    )

    for i, name in enumerate(JOINT_NAMES):
        ax = axes[i // 3, i % 3]
        q_deg = np.degrees(ref[:, i])
        ax.plot(x, q_deg, color="black", lw=1.4, label="ref")

        lo, hi = float(jnt_rng[i, 0]), float(jnt_rng[i, 1])
        if np.isfinite(lo) and np.isfinite(hi):
            ax.axhspan(np.degrees(lo), np.degrees(hi),
                       color="C2", alpha=0.18, label="jnt_range")
            ax.axhline(np.degrees(lo), color="C2", lw=0.8)
            ax.axhline(np.degrees(hi), color="C2", lw=0.8)

        s = stats[name]
        msg = (f"ref [{s['ref_lo_deg']:+.1f}, {s['ref_hi_deg']:+.1f}]°"
               f"\nrom {s['ref_rom_deg']:.1f}°"
               f"\noutside {100*s['frac_outside']:.1f}% of cycle")
        if s["frac_above"] > 0:
            msg += (f"\n↑overshoot {s['peak_overshoot_above_deg']:.1f}°"
                    f" ({100*s['frac_above']:.0f}% frames)")
        if s["frac_below"] > 0:
            msg += (f"\n↓overshoot {s['peak_overshoot_below_deg']:.1f}°"
                    f" ({100*s['frac_below']:.0f}% frames)")
        ax.set_title(name, fontsize=10)
        ax.text(0.02, 0.98, msg, transform=ax.transAxes,
                fontsize=8, va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"))
        ax.set_xlabel("phase")
        ax.set_ylabel("angle (deg)")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="lower right", fontsize=7)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_png}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--xml", default="walker2d.xml",
                   help="MJCF file to test against (default: gym-bundled "
                        "walker2d.xml). Bare filenames also resolve against "
                        "assets/mjcf/.")
    p.add_argument("--ref", default=str(REF_PATH),
                   help="Reference .npy (default: assets/reference/gait_cycle_reference.npy)")
    p.add_argument("--out", default=None,
                   help="Output PNG path (default: docs/figures/tier0/"
                        "A1_reference_vs_jnt_range_<xml>.png)")
    args = p.parse_args()

    ref = load_ref_cycle(Path(args.ref))
    xml = _resolve_xml(args.xml)
    jnt_rng, label = _load_jnt_ranges(xml)

    stats = analyse(ref, jnt_rng)

    print(f"\n=== Reference reachability check ===")
    print(f"reference: {args.ref}  ({len(ref)} frames)")
    print(f"MJCF:      {label}")
    print(f"\n{'joint':9s}  {'ref_lo':>7s}  {'ref_hi':>7s}  "
          f"{'jnt_lo':>7s}  {'jnt_hi':>7s}  {'%out':>6s}  "
          f"{'over+':>6s}  {'over-':>6s}")
    for name in JOINT_NAMES:
        s = stats[name]
        jl = f"{s['jnt_lo_deg']:+.1f}" if s['jnt_lo_deg'] is not None else "  inf"
        jh = f"{s['jnt_hi_deg']:+.1f}" if s['jnt_hi_deg'] is not None else "  inf"
        print(f"{name:9s}  {s['ref_lo_deg']:+7.2f}  {s['ref_hi_deg']:+7.2f}  "
              f"{jl:>7s}  {jh:>7s}  {100*s['frac_outside']:6.1f}  "
              f"{s['peak_overshoot_above_deg']:6.2f}  "
              f"{s['peak_overshoot_below_deg']:6.2f}")

    if args.out:
        out_png = Path(args.out)
    else:
        slug = Path(args.xml).stem
        out_png = (PROJECT_ROOT / "docs" / "figures" / "tier0"
                   / f"A1_reference_vs_jnt_range_{slug}.png")

    plot(ref, jnt_rng, label, stats, out_png)

    out_json = out_png.with_suffix(".json")
    out_json.write_text(json.dumps({"mjcf": label, "stats": stats}, indent=2),
                        encoding="utf-8")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
