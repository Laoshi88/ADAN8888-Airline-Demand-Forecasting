# Week 7 model output

After executing Week 7, this folder contains:
- `week7_selected_hist_gradient_boosting.joblib` — model selected using validation MAE.
- `week7_model_manifest.json` — selected hyperparameters, validation/training metrics, and provenance.

The estimator expects **Week 5 PC1 and PC2 values** computed using the original training-fitted preparation pipeline. It is not a deployable raw BTS ZIP predictor without those transformations.
