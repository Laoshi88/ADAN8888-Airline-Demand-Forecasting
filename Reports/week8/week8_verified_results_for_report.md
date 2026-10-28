# Week 8 — Empirical Results (See written report)

| Model | Hidden layers | Parameters | Epochs | Train MAE | Val MAE | Train RMSE | Val RMSE | Train R² | Val R² |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP_A | 32-16 | 641 | 20 | 1,171.37 | 1,171.51 | 2,236.71 | 2,019.43 | 0.9169 | 0.9380 |
| MLP_B | 64-32 | 2,305 | 17 | 1,161.63 | 1,143.20 | 2,236.20 | 1,988.78 | 0.9169 | 0.9399 |
| MLP_C | 64-32-16 | 2,817 | 17 | 1,159.01 | 1,149.90 | 2,230.23 | 1,999.78 | 0.9174 | 0.9392 |

**Week 8 selected: MLP_B**, validation MAE **1,143.20 passengers**.
Week 7 HGB_C reference validation MAE: 1,161.07 passengers.
**Test partition evaluated: NO.**


See `week8_figures/` for generated figures.
