# Week 10 model outputs

On real BTS execution, `week10_final_model_manifest.json` records the validation-selected model and its fitted preprocessing. When an enhanced stage wins, `week10_selected_enhanced_mlp_bundle.joblib` is generated. If the original MLP_B remains selected, its already-validated Week 8 saved estimator is **reused** and no new Week 10 binary is required.

Do not load untrusted `.joblib` files. Do not regenerate the already-viewed test scores by repeatedly rerunning the model. The pipeline caches the completed Week 10 results.
