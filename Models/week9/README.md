# Week 9 — Final Selected Model

**The frozen model is not retrained here.** Week 9 selects MLP_B using the nine validation results and loads the existing fitted model at:

`Models/week8/week8_selected_mlp_neural_network.joblib`

The Week 9 script writes `Models/week9/week9_final_model_manifest.json` once the actual final test evaluation succeeds. This manifest identifies the selected model, SHA-256 of the unchanged model artifact, test metrics, and required upstream PCA preprocessing. Do **not** load joblib files from untrusted sources.

The selected model expects `PC1` and `PC2` from the Week 5 *training-fitted* feature and PCA transformations, and retains the Week 8 input and target standardizers. It cannot ingest raw BTS ZIP archives directly.
