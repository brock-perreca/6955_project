# `requirements/` — pip requirements per platform

Pinned dependency lists, separated by platform / hardware. Pick the one
matching your machine.

| File | Use when… |
|---|---|
| `windows_5090.txt` | NVIDIA RTX 5090 (CUDA 12.8). Install PyTorch from the cu128 wheels first; see `../README.md` Option A. |
| `windows_cpu.txt` | Windows machine without a working CUDA install — CPU PyTorch. |
| `macos.txt` | macOS (Apple Silicon or Intel). |

The active Walker2d pipeline runs on CPU comfortably — PPO with an MLP
policy is env-step-bound, not GPU-bound. The 5090 build matters mostly
for the planned MJX / GPU port (see
[`../docs/ROADMAP.md`](../docs/ROADMAP.md)).

For the **musculoskeletal** legacy track
([`../src/legacy/musculoskeletal/`](../src/legacy/musculoskeletal/)),
MyoSuite lives in a **separate** Python 3.12 venv (`.venv-myo`) — it
pins `gymnasium==1.2.3` and `mujoco==3.6.0`, which would break the
active Walker2d stack if installed alongside it. See the project
[`../README.md`](../README.md) "MyoAssist (legacy musculoskeletal
track)" section for the setup recipe.
