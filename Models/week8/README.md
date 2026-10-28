# Week 8 Saved Neural Network

After running Week 8 **on real BTS records**, this folder should contain:
- `week8_selected_mlp_neural_network.joblib`: selected trained `TransformedTargetRegressor`, including Week 8's training-fitted PC1/PC2 scaler, fitted neural network, and training-fitted target scaler.
- `week8_model_manifest.json`: parameters, train/validation scores, input provenance, and explicit `EvaluatedTestSet=false`.

**Important:** The joblib object expects `PC1` and `PC2` as inputs. It does *not* include the earlier Week 2–5 raw-data cleaning, feature creation, and PCA transformation. Do not feed raw BTS fields directly into it.
