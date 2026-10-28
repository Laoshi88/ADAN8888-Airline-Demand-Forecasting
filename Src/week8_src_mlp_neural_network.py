"""ADAN8888 Week 8: third modeling approach, neural-network (MLP) regression.

Train exactly three increasing-capacity feedforward neural networks on the
FULL Week 5 PC1/PC2 chronological training partition; rank using validation MAE.
No held-out test scores or test-based hyperparameter selection are produced.

Run from repository root: python Src/week8_src_mlp_neural_network.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import json
import sys
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

FEATURES = ("PC1", "PC2")
TARGET = "TargetPassengersNextMonth"
EXPECTED_COUNTS = {"Training": 1_062_603, "Validation": 239_395}
# Planned, not dynamically optimized after viewing validation outcomes.
CANDIDATES = (
    {"Model": "MLP_A", "hidden_layer_sizes": (32, 16)},
    {"Model": "MLP_B", "hidden_layer_sizes": (64, 32)},
    {"Model": "MLP_C", "hidden_layer_sizes": (64, 32, 16)},
)
COMMON = dict(activation="relu", solver="adam", alpha=0.001,
              batch_size=4096, learning_rate_init=0.001,
              max_iter=45, tol=1e-4, n_iter_no_change=10,
              early_stopping=False, random_state=42, shuffle=True)
WEEK7_VALIDATION_MAE_REFERENCE = 1161.0682061257894


@dataclass
class Week8Results:
    split_summary: pd.DataFrame
    comparison: pd.DataFrame
    metrics_long: pd.DataFrame
    ranking: pd.DataFrame
    selected_name: str
    selected_model: TransformedTargetRegressor
    fitted_models: dict
    settings: dict
    source: str
    x_validation: np.ndarray
    y_validation: np.ndarray
    validation_months: pd.Series


def detect_project_root(start: Path | None = None) -> Path:
    path = (start or Path.cwd()).resolve()
    for folder in (path, *path.parents):
        if (folder / 'Src' / 'week6_src_ridge_models.py').is_file() and (folder / 'Notebooks').is_dir():
            return folder
    raise FileNotFoundError('Open the original repository with Src/week6_src_ridge_models.py.')


def load_full_training_validation(root: Path):
    """Reuse the prior-week FULL-data adapter; discard any returned test frame.

    If the earlier pipeline needs raw BTS ZIPs, it reads its existing Data/raw
    sources and uses the earlier training-fitted PCA transformations.
    """
    file = root / 'Src' / 'week6_src_ridge_models.py'
    spec = importlib.util.spec_from_file_location('week8_prior_week6', file)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load prior-week module: {file}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    data, summary, descriptor = mod._load_week5_data(root)
    assert all(k in data for k in ('Training','Validation')), 'Missing partitions'
    train = data['Training']; valid = data['Validation']
    for key, frame in [('Training',train),('Validation',valid)]:
        if len(frame) != EXPECTED_COUNTS[key]:
            raise AssertionError(f'{key}: expected {EXPECTED_COUNTS[key]:,} rows, got {len(frame):,}')
    if pd.to_datetime(train['ForecastMonth']).max() >= pd.to_datetime(valid['ForecastMonth']).min():
        raise AssertionError('Forecast-month chronological overlap!')
    return {'Training': train, 'Validation': valid}, summary, str(descriptor)


def features_and_target(frame: pd.DataFrame):
    expected = set(FEATURES) | {TARGET, 'ForecastMonth'}
    if missing := expected.difference(frame.columns):
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    x = frame[list(FEATURES)].to_numpy(dtype=np.float64, copy=True)
    y = pd.to_numeric(frame[TARGET],errors='raise').to_numpy(dtype=np.float64,copy=True)
    months = pd.to_datetime(frame['ForecastMonth'],errors='raise').reset_index(drop=True)
    if not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all() or (y<0).any():
        raise ValueError('Invalid/nonfinite or empty model arrays')
    return x, y, months


def parameter_count(dims: tuple[int,...]) -> int:
    shape = (len(FEATURES), *dims, 1)
    return int(sum((a + 1) * b for a, b in zip(shape[:-1],shape[1:])))


def build_model(hidden: tuple[int,...]) -> TransformedTargetRegressor:
    """Both the feature scaler and target scaler fit exclusively on training data."""
    reg = Pipeline(steps=[
        ('input_scaler', StandardScaler()),
        ('mlp', MLPRegressor(hidden_layer_sizes=hidden, **COMMON)),
    ])
    return TransformedTargetRegressor(regressor=reg, transformer=StandardScaler())


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str,float]:
    pred=np.clip(np.asarray(pred,dtype=np.float64),0,None)
    return {'MAE':float(mean_absolute_error(y,pred)),
            'RMSE':float(np.sqrt(mean_squared_error(y,pred))),
            'R2':float(r2_score(y,pred))}


def fit_three_variations(training: pd.DataFrame, validation: pd.DataFrame,
                         split_summary: pd.DataFrame, source: str = 'Week 5 PCA full dataset') -> Week8Results:
    xtr,ytr,ttr = features_and_target(training)
    xv,yv,tv = features_and_target(validation)
    if ttr.max()>=tv.min(): raise AssertionError('Chronological partitions overlap')
    if len(split_summary) and {'Split','Rows'}.issubset(split_summary.columns):
        counts=dict(zip(split_summary['Split'],split_summary['Rows']))
        for name,n in [('Training',len(xtr)),('Validation',len(xv))]:
            if name in counts and int(counts[name])!=n:
                raise AssertionError(f'Count drift for {name}')
    results=[]; fitted={}
    with threadpool_limits(limits=4):
        for cfg in CANDIDATES:
            model=build_model(cfg['hidden_layer_sizes'])
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always',ConvergenceWarning)
                model.fit(xtr,ytr)
            fitted[cfg['Model']]=model
            nnet=model.regressor_.named_steps['mlp']
            row={'Model':cfg['Model'],
                 'HiddenLayers':'-'.join(map(str,cfg['hidden_layer_sizes'])),
                 'TrainableParameters':parameter_count(cfg['hidden_layer_sizes']),
                 'FittedEpochs':int(nnet.n_iter_),
                 'TrainingConverged':not any(issubclass(w.category,ConvergenceWarning) for w in captured)}
            for part,x,y in [('Training',xtr,ytr),('Validation',xv,yv)]:
                row.update({f'{part}_{m}':score for m,score in metrics(y,model.predict(x)).items()})
            row['MAE_Gap_ValidationMinusTraining']=row['Validation_MAE']-row['Training_MAE']
            results.append(row)
            print(f"Finished {cfg['Model']} ({row['HiddenLayers']}); validation MAE={row['Validation_MAE']:,.2f}; epochs={row['FittedEpochs']}",flush=True)
    comp=pd.DataFrame(results)
    score_cols=[f'{part}_{m}' for part in ('Training','Validation') for m in ('MAE','RMSE','R2')]
    if len(comp)!=3 or not np.isfinite(comp[score_cols]).all().all():raise AssertionError('Invalid results')
    ranked=comp.sort_values(['Validation_MAE','Validation_RMSE','TrainableParameters','Model'],kind='mergesort').reset_index(drop=True)
    name=str(ranked.iloc[0].Model)
    setting=next(c for c in CANDIDATES if c['Model']==name)
    long=[]
    for _,r in comp.iterrows():
        for part in ('Training','Validation'):
            long.append({'Model':r.Model,'HiddenLayers':r.HiddenLayers,'Split':part,
                         'Rows':len(xtr) if part=='Training' else len(xv),
                         'MAE':r[f'{part}_MAE'],'RMSE':r[f'{part}_RMSE'],'R2':r[f'{part}_R2']})
    return Week8Results(split_summary.copy(),comp,pd.DataFrame(long),ranked,
                        name,fitted[name],fitted,dict(setting),source,xv,yv,tv)


def make_figures(result: Week8Results, dest: Path) -> pd.DataFrame:
    dest.mkdir(parents=True,exist_ok=True)
    comp=result.comparison
    for metric in ('MAE','RMSE'):
        fig,ax=plt.subplots(figsize=(8,4.7))
        ax.plot(comp.Model,comp[f'Training_{metric}'],marker='o',label=f'Training {metric}')
        ax.plot(comp.Model,comp[f'Validation_{metric}'],marker='o',label=f'Validation {metric}')
        ax.set(title=f'Week 8 neural network: {metric} comparison',xlabel='MLP architecture',ylabel=f'{metric} (passengers)')
        ax.legend();fig.tight_layout();fig.savefig(dest/f'week8_figure_{metric.lower()}_comparison.png',dpi=160);plt.close(fig)
    # Only the scatter is sampled; metrics and monthly diagnostics use ALL validation observations.
    rg=np.random.default_rng(42)
    idx=np.sort(rg.choice(len(result.y_validation),size=min(4000,len(result.y_validation)),replace=False))
    observed=result.y_validation[idx]; predictions=np.clip(result.selected_model.predict(result.x_validation[idx]),0,None)
    bound=max(observed.max(),predictions.max(),1)
    fig,ax=plt.subplots(figsize=(6.6,5.4))
    ax.scatter(observed,predictions,s=7,alpha=.24)
    ax.plot([0,bound],[0,bound],ls='--',label='Perfect prediction')
    ax.set(title=f'{result.selected_name}: validation predictions (display sample)',xlabel='Actual passengers',ylabel='Predicted passengers')
    ax.legend();fig.tight_layout();fig.savefig(dest/'week8_figure_validation_actual_vs_predicted.png',dpi=160);plt.close(fig)
    full_pred=np.clip(result.selected_model.predict(result.x_validation),0,None)
    monthly=pd.DataFrame({'ForecastMonth':result.validation_months,
                          'AbsoluteError':np.abs(full_pred-result.y_validation)}).groupby('ForecastMonth',as_index=False).agg(MonthlyMAE=('AbsoluteError','mean'),Rows=('AbsoluteError','size'))
    fig,ax=plt.subplots(figsize=(8.3,4.7))
    ax.plot(monthly.ForecastMonth,monthly.MonthlyMAE,marker='o',markersize=3)
    ax.set(title=f'{result.selected_name}: mean absolute error by validation month',xlabel='Forecast month',ylabel='Monthly MAE (passengers)')
    fig.autofmt_xdate();fig.tight_layout();fig.savefig(dest/'week8_figure_monthly_validation_mae.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.0,4.7))
    for name,model in result.fitted_models.items():
        curve=model.regressor_.named_steps['mlp'].loss_curve_
        ax.plot(np.arange(1,len(curve)+1),curve,label=name)
    ax.set(title='Week 8: training optimization loss by epoch',xlabel='Epoch',ylabel='Scaled-target training objective (not passenger MAE)')
    ax.legend();fig.tight_layout();fig.savefig(dest/'week8_figure_training_loss_curves.png',dpi=160);plt.close(fig)
    return monthly


def save_results(result: Week8Results,root:Path)->None:
    r=root/'Reports'/'week8'; t=r/'week8_tables';f=r/'week8_figures';m=root/'Models'/'week8'
    for folder in (t,f,m):folder.mkdir(parents=True,exist_ok=True)
    result.split_summary.to_csv(t/'week8_split_summary.csv',index=False)
    result.comparison.to_csv(t/'week8_three_model_comparison.csv',index=False)
    result.metrics_long.to_csv(t/'week8_metrics_long.csv',index=False)
    result.ranking.to_csv(t/'week8_validation_ranking.csv',index=False)
    monthly=make_figures(result,f);monthly.to_csv(t/'week8_monthly_validation_mae.csv',index=False)
    winner=result.ranking.iloc[0]
    manifest={
        'SelectedModel':result.selected_name,
        'SelectionRule':'Minimum validation MAE; tie: minimum validation RMSE; then fewer trainable parameters',
        'ModelFamily':'Multilayer perceptron neural-network regressor',
        'InputFeatures':list(FEATURES),
        'SelectedHiddenLayers':list(result.settings['hidden_layer_sizes']),
        'FixedHyperparameters':COMMON,
        'TargetScaler':'StandardScaler fitted using training targets only',
        'InputScaler':'StandardScaler fitted on training PC1 and PC2 only',
        'TrainingRows':EXPECTED_COUNTS['Training'],'ValidationRows':EXPECTED_COUNTS['Validation'],
        'TrainingMAE':float(winner.Training_MAE),'ValidationMAE':float(winner.Validation_MAE),
        'TrainingRMSE':float(winner.Training_RMSE),'ValidationRMSE':float(winner.Validation_RMSE),
        'TrainingR2':float(winner.Training_R2),'ValidationR2':float(winner.Validation_R2),
        'NegativePredictionPolicy':'Clip negative predictions to 0 after inverse target transform',
        'InputProvenance':result.source,'EvaluatedTestSet':False,
        'Week7ValidationMAEReference':WEEK7_VALIDATION_MAE_REFERENCE,
        'ReferenceUse':'Context only; no Week 7 metrics used for Week 8 model selection',
    }
    text=json.dumps(manifest,indent=2)
    (t/'week8_winner_summary.json').write_text(text,encoding='utf8')
    (m/'week8_model_manifest.json').write_text(text,encoding='utf8')
    path=m/'week8_selected_mlp_neural_network.joblib'
    joblib.dump(result.selected_model,path)
    check=joblib.load(path)
    if not np.allclose(check.predict(result.x_validation[:12]),result.selected_model.predict(result.x_validation[:12])):
        raise AssertionError('Serialized model failed prediction round trip')
    lines=['# Week 8 — Empirical Results (See written report)','',
        '| Model | Hidden layers | Parameters | Epochs | Train MAE | Val MAE | Train RMSE | Val RMSE | Train R² | Val R² |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _,row in result.comparison.iterrows():
        lines.append(f'| {row.Model} | {row.HiddenLayers} | {int(row.TrainableParameters):,} | {int(row.FittedEpochs)} | {row.Training_MAE:,.2f} | {row.Validation_MAE:,.2f} | {row.Training_RMSE:,.2f} | {row.Validation_RMSE:,.2f} | {row.Training_R2:.4f} | {row.Validation_R2:.4f} |')
    lines += ['',f"**Week 8 selected: {result.selected_name}**, validation MAE **{winner.Validation_MAE:,.2f} passengers**.",
              f"Week 7 HGB_C reference validation MAE: {WEEK7_VALIDATION_MAE_REFERENCE:,.2f} passengers.",
              '**Test partition evaluated: NO.**','']
    lines.append("\nSee `week8_figures/` for generated figures.\n")
    (r/'week8_verified_results_for_report.md').write_text('\n'.join(lines),encoding='utf8')


def run_week8(root:Path|None=None)->Week8Results:
    root=root or detect_project_root()
    data,summary,source=load_full_training_validation(root)
    result=fit_three_variations(data['Training'],data['Validation'],summary,source)
    save_results(result,root)
    return result


def print_summary(result:Week8Results)->None:
    print('Week 8 model comparison:')
    print(result.comparison.round(3).to_string(index=False))
    print(f'Week 8 provisional winner: {result.selected_name}')
    print('Test dataset scored: NO')
    print('All Week 8 checks passed.')
    print('===== WEEK 8 END =====')

if __name__=='__main__':
    print('===== WEEK 8 START =====',flush=True)
    print_summary(run_week8())
