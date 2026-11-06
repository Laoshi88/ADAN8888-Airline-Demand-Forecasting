"""Self-contained Week 9 lifecycle test (uses synthetic data for API checks ONLY).

Not a source of research metrics. Writes only into a temporary project.
Tests the 9-model validation lock, one-time final test, and idempotent rerun.
"""
from pathlib import Path
import importlib.util
import json
import shutil
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

PACKAGE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('w9_smoke_module',PACKAGE/'Src'/'week9_src_select_and_final_test.py')
w9=importlib.util.module_from_spec(spec);sys.modules[spec.name]=w9;spec.loader.exec_module(w9)
with tempfile.TemporaryDirectory() as temp:
    root=Path(temp)
    for d in ('Src','Notebooks','Models/week8','Reports/week9/week9_reference_metrics'):
        (root/d).mkdir(parents=True,exist_ok=True)
    (root/'Src/week8_src_mlp_neural_network.py').write_text('# test file for root detection\n')
    ref='Reports/week9/week9_reference_metrics/week9_verified_prior_week6_to_week8_metrics.csv'
    shutil.copy(PACKAGE/ref,root/ref)
    selected=w9.select_and_lock(root)
    assert len(selected.comparison)==9 and selected.winner=='MLP_B'
    assert selected.ranking.iloc[0].Model=='MLP_B'
    assert selected.lock_path.is_file()
    # Synthesize validation target with observed MAE/RMSE compatible with the
    # *historical reference*, solely to exercise the final-test gate.
    mae=float(selected.ranking.iloc[0].Validation_MAE)
    rmse=float(selected.ranking.iloc[0].Validation_RMSE)
    nval=5_000
    nonzero=int(round(nval*(mae/rmse)**2))
    val_y=np.r_[np.repeat(mae*nval/nonzero,nonzero),np.zeros(nval-nonzero)]
    xv=np.zeros((nval,2),dtype='float64')
    xt=np.zeros((200,2),dtype='float64')
    test_y=np.linspace(0,4000,len(xt))
    reg=DummyRegressor(strategy='constant',constant=0).fit(np.zeros((20,2)),np.zeros(20))
    joblib.dump(reg,root/'Models/week8/week8_selected_mlp_neural_network.joblib')
    w9.COUNT_EXPECTATIONS={'Training':100,'Validation':nval,'Test':len(xt)}
    def synthetic_pca(_root):
        return {'Validation':(xv,val_y,pd.Series(pd.date_range('2023-01-01',periods=nval,freq='h'))),
                'Test':(xt,test_y,pd.Series(pd.date_range('2024-09-01',periods=len(xt),freq='D')))},pd.DataFrame()
    w9._prepare_for_final_test=synthetic_pca
    first=w9.final_test_once(root,selected)
    assert first['SelectedModel']=='MLP_B'
    assert first['ValidationConsistencyCheckPassed']
    assert first['TestEvaluatedAfterSelectionLock']
    assert (root/'Reports/week9/week9_tables/week9_test_winner_summary.json').exists()
    assert (root/'Reports/week9/week9_tables/week9_winner_train_validation_test_metrics.csv').exists()
    assert len(list((root/'Reports/week9/week9_figures').glob('*.png'))) == 4
    assert not (root/'Reports/week9/week9_tables/week9_test_evaluation_started.marker').exists()
    # A second invocation must NOT access the test data again.
    def must_not_be_called(_root):
        raise AssertionError('Test preparation was repeated, violating idempotence')
    w9._prepare_for_final_test=must_not_be_called
    second=w9.final_test_once(root,selected)
    assert first==second
    w9.write_report_values(root,selected,second)
    assert (root/'Reports/week9/week9_verified_results_for_report.md').exists()
print('Week 9 synthetic smoke test PASSED: 9-model lock; validation gate; one test score; idempotence.')
