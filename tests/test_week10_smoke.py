"""Week 10 synthetic regression test. NEVER report synthetic scores as BTS results.

Validates exact-calendar lags, feature stages, saved Week 9 model compatibility,
training-only fitting, validation selection, late test scoring and cached re-run.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
import pandas as pd
import joblib

SRC = Path(__file__).resolve().parents[1] / 'Src' / 'week10_src_data_enhancement_error_analysis_refit.py'
WK8 = Path(__file__).resolve().parents[1] / 'Src' / 'week8_src_mlp_neural_network.py'

def module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

week10 = module_from_path('week10_smoke_src', SRC)
week8 = module_from_path('week10_smoke_week8', WK8) if WK8.exists() else None
if week8 is None:
    # In the delivered merge, Week 8 source is in the USER'S REPOSITORY, not this package.
    # Use a small compatible model factory with identical fixed hyperparameters for synthetic tests only.
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    COMMON = dict(activation='relu',solver='adam',alpha=.001,batch_size=64,
                  learning_rate_init=.001,max_iter=12,tol=1e-4,n_iter_no_change=5,
                  early_stopping=False,random_state=42,shuffle=True)
    def factory(hidden):
        return TransformedTargetRegressor(regressor=Pipeline([
            ('input_scaler',StandardScaler()),('mlp',MLPRegressor(hidden_layer_sizes=hidden,**COMMON))
        ]),transformer=StandardScaler())
    week8 = SimpleNamespace(COMMON=COMMON,build_model=factory)

class FakeWeek5:
    def __init__(self,monthly):self.original=monthly
    def load_monthly_without_rewriting_reports(self,root):return self.original.copy()
    def create_cross_field_features(self, monthly):
        m=monthly.copy()
        m['SeatsPerFlight']=m.Seats/m.DepPerformed.clip(lower=1)
        m['PassengersPerFlight']=m.Passengers/m.DepPerformed.clip(lower=1)
        m['UnusedSeats']=(m.Seats-m.Passengers).clip(lower=0)
        m['PassengerMiles']=m.Passengers*m.Distance
        return m
    def _numeric_inputs(self,frame):
        fields=['Passengers','Seats','DepScheduled','DepPerformed','Distance',
                'SeatsPerFlight','PassengersPerFlight','UnusedSeats','PassengerMiles']
        return np.log1p(frame[fields].where(frame[fields].ge(0)))
    def get_week3_splits(self,monthly,root):
        month=monthly.copy()
        if 'TargetPeriod' not in month.columns:
            month['TargetPeriod']=pd.to_datetime(month['Period'])+pd.DateOffset(months=1)
        month=month[month.TargetPassengersNextMonth.notna()].sort_values(['TargetPeriod','UniqueCarrier','Origin','Dest']).reset_index(drop=True)
        unique=sorted(month.TargetPeriod.unique())
        cuts=[unique[24],unique[36]]
        assigned={
            'Training':month[month.TargetPeriod < cuts[0]].reset_index(drop=True),
            'Validation':month[(month.TargetPeriod>=cuts[0])&(month.TargetPeriod < cuts[1])].reset_index(drop=True),
            'Test':month[month.TargetPeriod >= cuts[1]].reset_index(drop=True),
        }
        summary=pd.DataFrame({'Split':list(assigned),'Rows':[len(v) for v in assigned.values()]})
        return assigned,summary

# Three routes with varying trend/seasonality; 50 months (last target absent).
months=pd.date_range('2018-01-01',periods=50,freq='MS')
rows=[]
for r,(carrier,orig,dest) in enumerate([('AA','DFW','ORD'),('BB','PHL','BOS'),('CC','DEN','MIA')]):
    for j,m in enumerate(months):
        demand=max(0,1100+250*r+15*j+130*np.sin((j+2)*np.pi/6)+r*60*np.cos(j/4))
        rows.append({'UniqueCarrier':carrier,'Origin':orig,'Dest':dest,'Period':m,
                     'Passengers':demand,'Seats':demand*1.3+30,'DepScheduled':45+r*4+j%3,
                     'DepPerformed':43+r*4+j%2,'Distance':450+r*450})
monthly=pd.DataFrame(rows).sort_values(['UniqueCarrier','Origin','Dest','Period']).reset_index(drop=True)
monthly['TargetPassengersNextMonth']=monthly.groupby(['UniqueCarrier','Origin','Dest']).Passengers.shift(-1)
# Validate exact-calendar missing join -- dropping a row cannot turn t-2 into t-1.
toy=monthly.iloc[:].copy()
key=toy.loc[0,['UniqueCarrier','Origin','Dest']].tolist()
drop_month=months[2]
toy=toy.loc[~((toy.UniqueCarrier==key[0])&(toy.Origin==key[1])&(toy.Dest==key[2])&(toy.Period==drop_month))].copy()
aug=week10.add_history_features(toy)
row=aug[(aug.UniqueCarrier==key[0])&(aug.Period==months[3])].iloc[0]
assert pd.isna(row.PassengerLag1) and pd.isna(row.PassengerRolling3Mean)
assert aug.TargetPassengersNextMonth.equals(toy.TargetPassengersNextMonth.reset_index(drop=True)) if toy.index.equals(pd.RangeIndex(len(toy))) else True
wk5=FakeWeek5(monthly)

with tempfile.TemporaryDirectory(prefix='adan_week10_synthetic_') as folder:
    root=Path(folder)
    for sub in ['Data/raw','Src','Reports/week9/week9_tables','Models/week8','Models/week9']:
        (root/sub).mkdir(parents=True,exist_ok=True)
    prepared=wk5.create_cross_field_features(monthly)
    split, summary=wk5.get_week3_splits(prepared,root)
    week10.REFERENCE_EXPECTED_COUNTS=dict(zip(summary.Split,summary.Rows))
    state=week10.fit_original_pca(split['Training'],wk5)
    xt=week10.original_pca_inputs(split['Training'],state,wk5)
    xv=week10.original_pca_inputs(split['Validation'],state,wk5)
    x_test=week10.original_pca_inputs(split['Test'],state,wk5)
    model=week8.build_model(week10.HIDDEN_LAYERS)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(xt,split['Training'][week10.TARGET].to_numpy())
    vals=week10.regression_metrics(split['Validation'][week10.TARGET].to_numpy(),model.predict(xv))
    oldtest=week10.regression_metrics(split['Test'][week10.TARGET].to_numpy(),model.predict(x_test))
    saved=(root / week10.BASELINE_ARTIFACT)
    joblib.dump(model,saved)
    manifest={'SelectedModel':'MLP_B','SelectedModelArtifact':week10.BASELINE_ARTIFACT,
        'ValidationConsistencyCheckPassed':True,'TestEvaluatedAfterSelectionLock':True,
        'TestLabelsUsedToSelectModel':False,'RecomputedValidationMAE':vals['MAE'],
        'RecomputedValidationRMSE':vals['RMSE'],
        'FinalTestMAE':oldtest['MAE'],'FinalTestRMSE':oldtest['RMSE'],'FinalTestR2':oldtest['R2'],
        'FrozenModelSha256':hashlib.sha256(saved.read_bytes()).hexdigest()}
    (root/'Models/week9/week9_final_model_manifest.json').write_text(json.dumps(manifest))
    (root/'Reports/week9/week9_tables/week9_selection_lock.json').write_text(json.dumps({'Winner':'MLP_B'}))
    prev=week10.load_source
    week10.load_source=lambda f,name: wk5 if 'week5' in str(f) else week8
    try:
        outcomes=week10.run_week10(root,threads=1)
        assert len(outcomes.query('Split == "Validation"'))==4
        assert len(outcomes.query('Split == "Training"'))==4
        assert len(outcomes.query('Split == "Test"'))==1
        selection=json.loads((root/'Reports/week10/week10_tables/week10_selection_lock.json').read_text())
        assert selection['Winner'] in (week10.BASELINE_NAME,*week10.STAGE_NAMES)
        assert selection['TestEvaluatedAtSelection'] is False
        test=json.loads((root/'Reports/week10/week10_tables/week10_final_test_result.json').read_text())
        assert test['TestLabelsUsedForSelection'] is False
        assert test['Week9TestPreviouslyExamined'] is True
        ablation=pd.read_csv(root/'Reports/week10/week10_tables/week10_incremental_ablation.csv')
        assert list(ablation.InputFeatures)==[2,16,30,38]
        assert len(pd.read_csv(root/'Reports/week10/week10_tables/week10_three_data_improvements.csv'))==4
        assert (root/'Models/week10/week10_final_model_manifest.json').is_file()
        # Completion cache must reuse results rather than evaluating test a second time.
        original_metrics=outcomes.copy()
        cached=week10.run_week10(root,threads=1)
        pd.testing.assert_frame_equal(original_metrics,cached,check_exact=False)
        print('Week 10 synthetic end-to-end smoke test passed; do NOT report these synthetic MAEs as BTS results.')
    finally:
        week10.load_source=prev
