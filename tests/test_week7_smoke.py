"""Fast, synthetic-data-only smoke test of Week 7 fitting/export.

This test creates no synthetic datasets in the user's research pipeline. It uses
small artificial data only to verify code executes and artifacts have correct shape.
"""
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / 'Src' / 'week7_src_hist_gradient_boosting.py'
spec = importlib.util.spec_from_file_location('week7_smoke_module', p)
w7 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = w7
spec.loader.exec_module(w7)
rng = np.random.default_rng(827)

def sample(name, months, rows_per_month):
    x = rng.normal(size=(len(months) * rows_per_month, 2))
    y = np.maximum(3500 + 950 * np.sin(2*x[:,0]) + 850*x[:,1]**2 + rng.normal(0, 220,len(x)), 0)
    return pd.DataFrame({'PC1': x[:,0], 'PC2': x[:,1],
                         'TargetPassengersNextMonth': y,
                         'ForecastMonth': np.repeat(months,rows_per_month)})

train = sample('train', pd.date_range('2020-01-01', periods=8, freq='MS'), 150)
val = sample('validation', pd.date_range('2020-09-01', periods=2, freq='MS'), 120)
splits = pd.DataFrame([{'Split':'Training','Rows':len(train)}, {'Split':'Validation','Rows':len(val)}])
r = w7.fit_three_variations(train,val,splits,'synthetic smoke test only')
assert len(r.comparison) == 3 and len(r.metrics_long) == 6
assert r.winning_model_name in {c['Model'] for c in w7.CANDIDATES}
assert r.comparison.Validation_MAE.ge(0).all()
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    w7.save_week7(r,root)
    assert (root/'Models/week7/week7_selected_hist_gradient_boosting.joblib').exists()
    assert len(list((root/'Reports/week7/week7_figures').glob('*.png'))) == 4
    assert (root/'Reports/week7/week7_tables/week7_three_model_comparison.csv').exists()
print('Week 7 smoke test passed (synthetic QA only — not actual BTS results).')
