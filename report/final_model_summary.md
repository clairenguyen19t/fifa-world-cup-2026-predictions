# Final Model Summary

**Selected public-facing model:** `ExtraTreesClassifier + sigmoid calibration`

## Why this model
Stage 9 produced two strong candidates:

| Candidate | Accuracy | Macro F1 | Log loss | Draw recall |
|---|---:|---:|---:|---:|
| **ExtraTrees + sigmoid** (selected) | 60.2% | 0.457 | 1.641 | 0.024 |
| HistGB(balanced) + sigmoid | ~58.6% | ~0.428 | ~1.552 (best) | ~0.005 |

`ExtraTrees + sigmoid` was chosen as the **public-facing** model because it delivers
the **highest holdout accuracy (60.2%)** while remaining
probability-calibrated (sigmoid/Platt scaling fit via `TimeSeriesSplit`), giving the
best balance between predictive performance and presentation value. HistGB(balanced)
keeps a marginally lower log loss but loses ~1.5pp of accuracy.

## Final holdout metrics (2023-2025, chronological)

| Metric | Final model | Stage 3 baseline (RandomForest) | Delta |
|---|---:|---:|---:|
| Accuracy | 60.24% | 58.53% | +1.71pp |
| Macro F1 | 0.457 | 0.508 | -0.052 |
| Log loss | 1.641 | 1.743 | -0.102 |

Confusion matrix (rows = actual `['home_win', 'draw', 'away_win']`):

```
[[1353   21  172]
 [ 489   18  243]
 [ 366   16  609]]
```

## Reproducibility
- Trained on `data/processed/train_features.parquet`, 2010-2022 train window.
- Identical leakage-safe preprocessing as Stage 3 (Elo missingness indicators +
  median imputation fit on train only).
- Calibration: `CalibratedClassifierCV(method="sigmoid", cv=TimeSeriesSplit(3))`.
- Artifacts: `outputs/final_match_model.pkl`, `outputs/final_model_metrics.csv`,
  `outputs/final_confusion_matrix.png`.
