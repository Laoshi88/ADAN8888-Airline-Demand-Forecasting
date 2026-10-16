# Week 6 verified results — paste into written report only after reviewing

Results based on actual execution of Week 6 on the repository dataset.

Selected Ridge_A (Ridge alpha = 1) because it attained the lowest validation MAE.

| Variant | Alpha | Train MAE | Val MAE | Train RMSE | Val RMSE | Train R² | Val R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge_A | 1 | 3,331.96 | 3,257.18 | 5,787.13 | 5,887.10 | 0.4435 | 0.4731 |
| Ridge_B | 10,000 | 3,332.43 | 3,259.03 | 5,789.18 | 5,890.28 | 0.4431 | 0.4725 |
| Ridge_C | 1,000,000 | 3,395.14 | 3,418.39 | 5,985.65 | 6,163.44 | 0.4047 | 0.4224 |

Figures saved under Reports/week6/week6_figures/.
The test partition was NOT evaluated, and must not be used to choose the Week 6 winner.
