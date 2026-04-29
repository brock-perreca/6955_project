"""
Build a re-inverted-hip variant of the gait-cycle reference for batch-4
Variant B. Negates columns 0 (hip_r) and 3 (hip_l). Knee and ankle columns
are preserved.

This reverts only the hip portion of the 2026-04-28 sign-convention fix.
After re-inverting, hip_r and hip_l peaks are negative (flexion in the
old/inverted sign convention). Note: peak EXTENSION is now ~+13deg, which
still exceeds the stock walker2d.xml upper limit of 0deg, so this variant
alone does NOT fully fit the reference inside the joint range -- it's a
sanity-check ablation, not a guaranteed fix. See docs/RESTART_LOG.md
Batch 4.
"""
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
src = PROJECT_ROOT / "assets" / "reference" / "gait_cycle_reference.npy"
dst = PROJECT_ROOT / "assets" / "reference" / "gait_cycle_reference_hipinvert.npy"

ref = np.load(src)
print(f"loaded {src.name}: shape {ref.shape}")
print("  before: hip_r [{:+.2f}, {:+.2f}] deg  hip_l [{:+.2f}, {:+.2f}] deg".format(
    np.rad2deg(ref[:, 0].min()), np.rad2deg(ref[:, 0].max()),
    np.rad2deg(ref[:, 3].min()), np.rad2deg(ref[:, 3].max())))

inv = ref.copy()
inv[:, 0] = -inv[:, 0]
inv[:, 3] = -inv[:, 3]

print("  after : hip_r [{:+.2f}, {:+.2f}] deg  hip_l [{:+.2f}, {:+.2f}] deg".format(
    np.rad2deg(inv[:, 0].min()), np.rad2deg(inv[:, 0].max()),
    np.rad2deg(inv[:, 3].min()), np.rad2deg(inv[:, 3].max())))

np.save(dst, inv)
print(f"wrote {dst.name}")
