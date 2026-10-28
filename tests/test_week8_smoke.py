"""Synthetic smoke test ONLY; not mean to generate / report genuine BTS metrics."""
import importlib.util
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

src=Path(__file__).resolve().parents[1]/'Src'/'week8_src_mlp_neural_network.py'
spec=importlib.util.spec_from_file_location('week8_smoke',src)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
rng=np.random.default_rng(88)
def synth(months,rows_per_month):
    x=rng.normal(size=(len(months)*rows_per_month,2))
    y=np.maximum(3800+850*x[:,0]**2+1200*np.sin(x[:,1])+rng.normal(0,280,len(x)),0)
    return pd.DataFrame({'PC1':x[:,0],'PC2':x[:,1],'ForecastMonth':np.repeat(months,rows_per_month),'TargetPassengersNextMonth':y})
train=synth(pd.date_range('2020-01-01',periods=10,freq='MS'),90)
valid=synth(pd.date_range('2020-11-01',periods=4,freq='MS'),90)
summary=pd.DataFrame([{'Split':'Training','Rows':len(train)},{'Split':'Validation','Rows':len(valid)}])
result=mod.fit_three_variations(train,valid,summary,'Synthetic smoke test only')
assert len(result.comparison)==3 and len(result.metrics_long)==6
assert result.selected_name==result.ranking.iloc[0].Model
assert list(result.comparison.TrainableParameters)==sorted(result.comparison.TrainableParameters)
with tempfile.TemporaryDirectory() as path:
    root=Path(path)
    mod.save_results(result,root)
    tables=root/'Reports'/'week8'/'week8_tables'
    assert len(list((root/'Reports'/'week8'/'week8_figures').glob('*.png')))==5
    assert len(pd.read_csv(tables/'week8_three_model_comparison.csv'))==3
    import json
    d=json.loads((tables/'week8_winner_summary.json').read_text())
    assert d['EvaluatedTestSet'] is False
    assert (root/'Models'/'week8'/'week8_selected_mlp_neural_network.joblib').is_file()
print('Week 8 synthetic smoke test passed. DO NOT present its scores as BTS results.')
