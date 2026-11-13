ADAN8888 Week 10 — verified results from local execution

Selected on validation: MLP_B_temporal
Train rows: 1,062,603; validation rows: 239,395; test rows: 264,291

Training and validation comparisons:
                 Model      Split    Rows  InputFeatures  TrainableParameters       MAE      RMSE     R2  MeanResidual  WAPE_Percent
    MLP_B_original_PCA   Training 1062603              2                 2305 1161.6278 2236.2033 0.9169     -102.7270       23.0088
    MLP_B_original_PCA Validation  239395              2                 2305 1143.1952 1988.7780 0.9399       94.3514       19.7833
        MLP_B_temporal   Training 1062603             16                 3201  808.1664 1434.7893 0.9658     -140.4477       16.0076
        MLP_B_temporal Validation  239395             16                 3201  809.3316 1299.8096 0.9743      -95.6100       14.0057
MLP_B_temporal_quality   Training 1062603             30                 4097  801.5727 1428.7769 0.9661      111.1713       15.8770
MLP_B_temporal_quality Validation  239395             30                 4097  819.8029 1330.5446 0.9731      180.7649       14.1869
  MLP_B_enhanced_refit   Training 1062603             38                 4609  704.1294 1203.0290 0.9760      112.9542       13.9469
  MLP_B_enhanced_refit Validation  239395             38                 4609  870.3522 1466.9507 0.9673      161.9055       15.0617

Incremental data-improvement evidence:
                 Stage  Validation_MAE  Validation_RMSE  InputFeatures  TrainableParameters  MAE_Improvement_vs_Original  MAE_Improvement_vs_Previous
    MLP_B_original_PCA       1143.1952        1988.7780              2                 2305                       0.0000                          NaN
        MLP_B_temporal        809.3316        1299.8096             16                 3201                     333.8636                     333.8636
MLP_B_temporal_quality        819.8029        1330.5446             30                 4097                     323.3923                     -10.4714
  MLP_B_enhanced_refit        870.3522        1466.9507             38                 4609                     272.8429                     -50.5493

Selected model historical test metrics:
         Model Split   Rows  InputFeatures  TrainableParameters      MAE      RMSE     R2  MeanResidual  WAPE_Percent
MLP_B_temporal  Test 264291             16                 3201 790.2854 1283.1248 0.9732     -177.7251        14.444

Test period already examined in Week 9; it is not a fresh independent holdout.
Data availability: BTS reference-month input publication lag not resolved for live deployment.
A business acceptance threshold has not been specified; operational readiness is undetermined.
