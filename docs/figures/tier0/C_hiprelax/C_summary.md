# Tier 0 — experiment C summary

Hip-trace comparison: `docs\figures\tier0\C_hiprelax\C_hip_trace_comparison.png`

All values are medians over 6 deterministic eval episodes × 2500 max steps (eval_biomech.py).

| label | ep_len | strides | stride s | cadence | DS frac | hip_r ROM | hip_l ROM | knee_r ROM | LR asym | vGRF/BW | progress |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xvel-5M | 2500 | 60 | 0.327 | 367.6 | 0.074 | 1.77 | 1.94 | 22.18 | 0.138 | 3.28 | 2.31 |
| hiprelax_s11 | 2500 | 54 | 0.361 | 332.7 | 0.018 | 19.79 | 15.27 | 26.57 | 0.097 | 3.97 | 2.41 |
| hiprelax_s12 | 2500 | 56 | 0.347 | 346.3 | 0.019 | 19.92 | 16.56 | 38.56 | 0.143 | 4.02 | 2.19 |
| hiprelax_s13 | 2500 | 53 | 0.371 | 323.6 | 0.010 | 16.50 | 18.52 | 32.70 | 0.122 | 4.18 | 2.11 |

**Reference (measured Subject 1, baseline 1.25 m/s):** stride 1.120 s, cadence 107.1, DS 0.227, hip_r ROM 45.4°, hip_l ROM 45.4°, knee_r ROM 65.7°, LR asym < 0.10, peak vGRF/BW 1.10. Progress score is in [0, 4].
