# Week 9 verified values for the written report

Source: Week 6–8 actual validation artifacts; Week 9 test is available only after running locally.
Selection occurs before any test labels are accessed.

| Model | Week | Train MAE | Val MAE | Train RMSE | Val RMSE | Train R² | Val R² |
|---|---|---:|---:|---:|---:|---:|---:|
| Ridge_C | Week6 | 3,395.14 | 3,418.39 | 5,985.65 | 6,163.44 | 0.4047 | 0.4224 |
| Ridge_B | Week6 | 3,332.43 | 3,259.03 | 5,789.18 | 5,890.28 | 0.4431 | 0.4725 |
| Ridge_A | Week6 | 3,331.96 | 3,257.18 | 5,787.13 | 5,887.10 | 0.4435 | 0.4731 |
| HGB_A | Week7 | 1,180.22 | 1,179.31 | 2,263.18 | 2,060.22 | 0.9149 | 0.9355 |
| HGB_B | Week7 | 1,164.08 | 1,163.47 | 2,254.86 | 2,050.41 | 0.9155 | 0.9361 |
| HGB_C | Week7 | 1,159.02 | 1,161.07 | 2,247.90 | 2,051.79 | 0.9160 | 0.9360 |
| MLP_A | Week8 | 1,171.37 | 1,171.51 | 2,236.71 | 2,019.43 | 0.9169 | 0.9380 |
| MLP_B | Week8 | 1,161.63 | 1,143.20 | 2,236.20 | 1,988.78 | 0.9169 | 0.9399 |
| MLP_C | Week8 | 1,159.01 | 1,149.90 | 2,230.23 | 1,999.78 | 0.9174 | 0.9392 |

**Validation winner: MLP_B**, MAE = 1,143.20 passengers.
**Important:** Cross-family capacity is only approximately ordered. Training–validation loss gap is a diagnostic, not a formal bias–variance decomposition.

## Final selected-model test results

Test MAE: **1,122.50** passengers
Test RMSE: **1,935.24** passengers
Test R²: **0.9390**
