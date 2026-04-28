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
([`../src/legacy/musculoskeletal/`](../src/legacy/musculoskeletal/)) some
extra system tools are needed (CMake, Bazelisk). See the project
[`../README.md`](../README.md) "MyoAssist" section for that build path.
