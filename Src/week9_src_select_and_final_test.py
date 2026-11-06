"""ADAN8888 Week 9: Select from 9 PRIOR validation models, then final-test the winner.

Run from the repository root:
    python Src/week9_src_select_and_final_test.py

Preselection: archived, comparable full-validation scores from Weeks 6-8.
Final test: exactly one test evaluation AFTER selection is locked and validated.
The held-out test partition is never used to choose models or parameters.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FAMILY = {"Ridge": "Week6", "HGB": "Week7", "MLP": "Week8"}
WINNER_ARTIFACTS = {
    "Ridge_A": "Models/week6/week6_selected_ridge.joblib",
    "HGB_C": "Models/week7/week7_selected_hist_gradient_boosting.joblib",
    "MLP_B": "Models/week8/week8_selected_mlp_neural_network.joblib",
}
COUNT_EXPECTATIONS = {"Training": 1_062_603, "Validation": 239_395, "Test": 264_291}
ORDER = ["Ridge_C", "Ridge_B", "Ridge_A", "HGB_A", "HGB_B", "HGB_C", "MLP_A", "MLP_B", "MLP_C"]
REQUIRED_METRICS = [f"{s}_{m}" for s in ("Training", "Validation") for m in ("MAE", "RMSE", "R2")]
BASELINE_CSV = "Reports/week9/week9_reference_metrics/week9_verified_prior_week6_to_week8_metrics.csv"
MODEL_CSVS = {
    "Ridge": "Reports/week6/week6_tables/week6_models_metrics_comparison.csv",
    "HGB": "Reports/week7/week7_tables/week7_three_model_comparison.csv",
    "MLP": "Reports/week8/week8_tables/week8_three_model_comparison.csv",
}


@dataclass
class Selection:
    comparison: pd.DataFrame
    ranking: pd.DataFrame
    winner: str
    lock_path: Path
    lock: dict[str, Any]


def root_from(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for p in (here, *here.parents):
        if (p / "Src" / "week8_src_mlp_neural_network.py").exists() and (p / "Notebooks").is_dir():
            return p
    raise FileNotFoundError("Open the existing Git repo root containing Week 8 source and Notebooks.")


def _hash(df: pd.DataFrame) -> str:
    # All nine validation scores and labels, stable to source ordering.
    data = df.sort_values("Model")[['Model', 'Validation_MAE', 'Validation_RMSE']].to_csv(index=False, float_format="%.10f")
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def _clip_eval(actual, prediction):
    y = np.asarray(actual, dtype="float64")
    predicted = np.maximum(np.asarray(prediction, dtype="float64"), 0.0)
    if len(y) == 0 or len(y) != len(predicted) or not np.isfinite(y).all() or not np.isfinite(predicted).all() or (y < 0).any():
        raise ValueError("Invalid actual or predicted passenger counts")
    return {"MAE": float(mean_absolute_error(y, predicted)),
            "RMSE": float(np.sqrt(mean_squared_error(y, predicted))),
            "R2": float(r2_score(y, predicted))}


def load_prior_comparison(root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Prefer files actually committed during Weeks 6-8; fallback to audited uploaded references.

    Week 6 reference values were printed to three decimal places in the
    executed notebook, so their R² reference precision is limited accordingly.
    """
    ref = root / BASELINE_CSV
    if not ref.exists():
        raise FileNotFoundError(f"Missing included historical comparison reference: {ref}")
    reference = pd.read_csv(ref)
    required = {'Model','Week','Family','Configuration',*REQUIRED_METRICS}
    if required.difference(reference.columns):
        raise ValueError("Reference metrics missing fields")
    if len(reference) != 9 or reference.Model.nunique() != 9 or set(reference.Model) != set(ORDER):
        raise ValueError("Historical reference must contain exactly the nine known configurations")
    source_by_family = {}
    selected = []
    for fam in ('Ridge', 'HGB', 'MLP'):
        original_path = root / MODEL_CSVS[fam]
        prior = reference.loc[reference.Family.eq(fam)].copy()
        if original_path.exists():
            original = pd.read_csv(original_path)
            if set(original.Model) != set(prior.Model) or len(original) != 3:
                raise ValueError(f"Unexpected model labels/count in {original_path}")
            if not set(REQUIRED_METRICS).issubset(original.columns):
                raise ValueError(f"Missing train/validation metric columns in {original_path}")
            old = prior.set_index('Model')
            fresh = original.set_index('Model')
            # Detect stale or unrelated datasets; tolerate Week 6 PDF rounding.
            for key in ('Validation_MAE', 'Validation_RMSE'):
                if not np.allclose(old.loc[fresh.index, key].to_numpy(dtype=float),
                                   fresh[key].to_numpy(dtype=float), rtol=0, atol=.06):
                    raise ValueError(f"{original_path} disagrees with verified prior {key}; STOP before test.")
            for col in REQUIRED_METRICS:
                old.loc[fresh.index,col] = fresh[col]
            prior = old.reset_index()
            source_by_family[fam] = str(MODEL_CSVS[fam])
        else:
            source_by_family[fam] = str(BASELINE_CSV) + ' (historical reference fallback)'
        selected.append(prior)
    out = pd.concat(selected,ignore_index=True)
    out = out.set_index('Model').loc[ORDER].reset_index()
    nums = out[REQUIRED_METRICS].to_numpy(dtype=float)
    if not np.isfinite(nums).all() or (out[['Training_MAE','Validation_MAE','Training_RMSE','Validation_RMSE']] < 0).any().any():
        raise ValueError('Invalid prior score table')
    out['Validation_MSE'] = out['Validation_RMSE']**2
    out['Training_MSE'] = out['Training_RMSE']**2
    out['MAE_Gap_ValidationMinusTraining'] = out['Validation_MAE']-out['Training_MAE']
    out['ComplexityOrder'] = np.arange(1,10)
    return out, source_by_family


def _bias_variance_plot(comparison: pd.DataFrame, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.5,5.5))
    xs = np.arange(len(comparison))
    ax.plot(xs, comparison.Training_MSE / 1e6, marker='o', label='Training MSE')
    ax.plot(xs, comparison.Validation_MSE / 1e6, marker='o', label='Validation MSE')
    ax.axvline(2.5, color='gray', linestyle=':', alpha=.7)
    ax.axvline(5.5, color='gray', linestyle=':', alpha=.7)
    ax.set_xticks(xs, comparison.Model.tolist(), rotation=30)
    ax.set_xlabel('Nominal model capacity (within-family ordering; cross-family ordering illustrative)')
    ax.set_ylabel('Mean squared error (million passengers²)')
    ax.set_title('Week 9: Training and validation loss across nine configurations')
    ax.legend();fig.tight_layout()
    fig.savefig(dest/'week9_figure_nine_model_bias_variance_diagnostic.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5,4.6))
    ax.plot(comparison.Model, comparison.Validation_MAE, marker='o')
    ax.set_ylabel('Validation MAE (passengers)')
    ax.set_xlabel('Nine candidate configurations (grouped by family)')
    ax.set_title('Week 9: Model selection using the same validation MAE')
    ax.tick_params(axis='x',labelrotation=25)
    fig.tight_layout();fig.savefig(dest/'week9_figure_nine_model_validation_mae.png',dpi=180)
    plt.close(fig)


def select_and_lock(root: Path) -> Selection:
    """Choose from VALIDATION ONLY and write a persistent lock before touching test labels."""
    out = root/'Reports'/'week9'
    tables = out/'week9_tables'; figs = out/'week9_figures'
    tables.mkdir(parents=True,exist_ok=True)
    comparison, sources = load_prior_comparison(root)
    ranking = comparison.sort_values(['Validation_MAE','Validation_RMSE','ComplexityOrder','Model'],kind='mergesort').reset_index(drop=True)
    winner = str(ranking.iloc[0].Model)
    # A saved, already-fitted model is needed for defensible one-time test scoring.
    if winner not in WINNER_ARTIFACTS:
        raise RuntimeError(f"{winner} is not one of the three saved prior-week winners. Stop: cannot silently refit.")
    if winner != 'MLP_B':
        raise RuntimeError(f"Unexpected winner {winner}. Reconcile previous-week artifacts before final test.")
    digest = _hash(comparison)
    lock_path = tables/'week9_selection_lock.json'
    lock = {
        'Winner':winner,
        'SelectionRule':'Lowest validation MAE; then validation RMSE; then predeclared simple-to-complex ordering',
        'ComparisonDigest': digest,
        'SelectedValidationMAE':float(ranking.iloc[0].Validation_MAE),
        'SelectedValidationRMSE':float(ranking.iloc[0].Validation_RMSE),
        'Sources':sources,
        'TestDataViewedForSelection':False,
        'TestDataEvaluatedAtSelection':False,
        'ParametersChangedAfterSelection':False,
        'SelectedSavedModel': WINNER_ARTIFACTS[winner],
    }
    if lock_path.exists():
        old = json.loads(lock_path.read_text(encoding='utf-8'))
        if old['Winner'] != winner or old['ComparisonDigest'] != digest:
            raise RuntimeError('Existing Week 9 selection lock conflicts with current prior-week results. Stop.')
    else:
        lock_path.write_text(json.dumps(lock,indent=2),encoding='utf-8')
    comparison.to_csv(tables/'week9_nine_model_comparison.csv',index=False)
    ranking.to_csv(tables/'week9_validation_ranking.csv',index=False)
    _bias_variance_plot(comparison,figs)
    return Selection(comparison,ranking,winner,lock_path,lock)


def _load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(f"Required prior-week source missing: {path}")
    spec = importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def _prepare_for_final_test(root: Path):
    """Use the exact Week 5 training-fitted PCA recipe; never fit transformations on test."""
    wk5 = _load_module(root/'Src'/'week5_src_features_dimensionality.py','week9_week5_preparation')
    monthly = wk5.load_monthly_without_rewriting_reports(root)
    engineered = wk5.create_cross_field_features(monthly)
    partitions, summary = wk5.get_week3_splits(engineered,root)
    if set(['Training','Validation','Test']) - set(partitions):
        raise ValueError('All three historical partitions are required')
    observed_counts = {k:len(partitions[k]) for k in COUNT_EXPECTATIONS}
    if observed_counts != COUNT_EXPECTATIONS:
        raise ValueError(f'Split counts changed! {observed_counts} vs {COUNT_EXPECTATIONS}')
    timecol = 'TargetPeriod' if 'TargetPeriod' in partitions['Validation'] else 'ForecastMonth'
    times = {k:pd.to_datetime(partitions[k][timecol],errors='raise') for k in COUNT_EXPECTATIONS}
    if times['Training'].max() >= times['Validation'].min() or times['Validation'].max() >= times['Test'].min():
        raise ValueError('Chronological train/validation/test boundaries are invalid')
    fitted = wk5.fit_feature_reduction(partitions,summary)
    data = {}
    # Prepare validation first; the test target is not used to select or refit any model.
    for name in ('Validation','Test'):
        source=partitions[name]
        logged=wk5._numeric_inputs(source)
        filled=logged.fillna(fitted.medians)
        matrix=fitted.scaler.transform(filled[fitted.kept_features].to_numpy(dtype='float64'))
        reduced=fitted.pca.transform(matrix)
        if reduced.shape != (len(source),2) or not np.isfinite(reduced).all():
            raise ValueError(f'Bad {name} PCA projection')
        y=pd.to_numeric(source['TargetPassengersNextMonth'], errors='raise').to_numpy(dtype='float64')
        if not np.isfinite(y).all() or (y<0).any():
            raise ValueError(f'Invalid {name} labels')
        data[name] = (reduced, y, times[name].reset_index(drop=True))
    return data,summary


def _test_plots(test_y, test_pred, months, figures: Path) -> pd.DataFrame:
    figures.mkdir(parents=True,exist_ok=True)
    pred=np.maximum(np.asarray(test_pred,dtype=float),0)
    sample=np.sort(np.random.default_rng(42).choice(len(test_y),size=min(4000,len(test_y)),replace=False))
    y=test_y[sample];yhat=pred[sample]
    upper=max(float(y.max()),float(yhat.max()),1.0)
    fig,ax=plt.subplots(figsize=(6.8,5.6));ax.scatter(y,yhat,s=8,alpha=.25)
    ax.plot([0,upper],[0,upper],'--',label='Perfect prediction')
    ax.set(xlabel='Actual next-month passengers',ylabel='Predicted next-month passengers',title='Week 9 selected MLP_B: Test actual vs predicted (display sample)')
    ax.legend();fig.tight_layout();fig.savefig(figures/'week9_figure_test_actual_vs_predicted.png',dpi=180);plt.close(fig)
    monthly=pd.DataFrame({'ForecastMonth':months,'AbsoluteError':np.abs(test_y-pred)}).groupby('ForecastMonth',as_index=False).agg(MonthlyMAE=('AbsoluteError','mean'),Rows=('AbsoluteError','size'))
    fig,ax=plt.subplots(figsize=(9,4.7));ax.plot(monthly.ForecastMonth,monthly.MonthlyMAE,marker='o')
    ax.set(xlabel='Test forecast month',ylabel='Monthly MAE (passengers)',title='Week 9: Locked winner test MAE by forecast month')
    fig.autofmt_xdate();fig.tight_layout();fig.savefig(figures/'week9_figure_test_monthly_mae.png',dpi=180);plt.close(fig)
    return monthly


def final_test_once(root: Path, selected: Selection) -> dict:
    """One post-selection test evaluation, with validation-reconstruction integrity gate."""
    tables=root/'Reports'/'week9'/'week9_tables'
    summary_path=tables/'week9_test_winner_summary.json'
    if summary_path.exists():
        existing=json.loads(summary_path.read_text(encoding='utf-8'))
        if existing.get('ComparisonDigest') != selected.lock['ComparisonDigest'] or existing.get('SelectedModel') != selected.winner:
            raise RuntimeError('A different test summary already exists; DO NOT repeat test evaluation.')
        print('Existing final test result found; reusing saved result, not scoring test twice.')
        return existing
    started = tables / 'week9_test_evaluation_started.marker'
    if started.exists():
        raise RuntimeError('A previous final test evaluation began but did not finish. Reconcile this marker and saved outputs before retrying; do not silently rescore.')
    model_path = root / WINNER_ARTIFACTS[selected.winner]
    if not model_path.exists():
        raise FileNotFoundError(f'Cannot evaluate test until the original fitted winner is available: {model_path}')
    model = joblib.load(model_path)  # Load only trusted artifacts you created in your repo.
    data, split_summary = _prepare_for_final_test(root)
    xval,yval,tval=data['Validation']
    prior_row=selected.comparison.set_index('Model').loc[selected.winner]
    val_pred=model.predict(xval)
    val_scores=_clip_eval(yval,val_pred)
    # Essential compatibility check: prevents evaluating an incompatible PCA/model pairing.
    if abs(val_scores['MAE']-prior_row.Validation_MAE)>0.50 or abs(val_scores['RMSE']-prior_row.Validation_RMSE)>0.75:
        raise RuntimeError(
            'The saved MLP_B model does NOT reproduce its previously recorded validation performance. '
            f'Current validation MAE/RMSE={val_scores["MAE"]:.2f}/{val_scores["RMSE"]:.2f}; '
            f'prior={prior_row.Validation_MAE:.2f}/{prior_row.Validation_RMSE:.2f}. '
            'STOP: investigate source version, PCA transformations and model artifact. TEST NOT SCORED.'
        )
    # Selection and all compatibility checks have now been completed using validation only.
    x_test,y_test,months = data['Test']
    started.write_text('Final held-out evaluation was initiated after the selection lock and validation reproduction check.\n', encoding='utf-8')
    test_pred=model.predict(x_test)  # The only newly scored final held-out test prediction.
    scores=_clip_eval(y_test,test_pred)
    metrics=pd.DataFrame([
        {'Partition':'Training','Rows':COUNT_EXPECTATIONS['Training'],
         'MAE':float(prior_row.Training_MAE),'RMSE':float(prior_row.Training_RMSE),'R2':float(prior_row.Training_R2)},
        {'Partition':'Validation','Rows':COUNT_EXPECTATIONS['Validation'],
         'MAE':float(prior_row.Validation_MAE),'RMSE':float(prior_row.Validation_RMSE),'R2':float(prior_row.Validation_R2)},
        {'Partition':'Test','Rows':COUNT_EXPECTATIONS['Test'],**scores},
    ])
    metrics.to_csv(tables/'week9_winner_train_validation_test_metrics.csv',index=False)
    monthly=_test_plots(y_test,test_pred,months,root/'Reports'/'week9'/'week9_figures')
    monthly.to_csv(tables/'week9_test_monthly_mae.csv',index=False)
    record={
        'SelectedModel':selected.winner,
        'SelectedModelArtifact':WINNER_ARTIFACTS[selected.winner],
        'ComparisonDigest':selected.lock['ComparisonDigest'],
        'TrainingRows':COUNT_EXPECTATIONS['Training'],
        'ValidationRows':COUNT_EXPECTATIONS['Validation'],
        'TestRows':COUNT_EXPECTATIONS['Test'],
        'ValidationConsistencyCheckPassed':True,
        'RecomputedValidationMAE':val_scores['MAE'],
        'RecomputedValidationRMSE':val_scores['RMSE'],
        'FinalTestMAE':scores['MAE'],
        'FinalTestRMSE':scores['RMSE'],
        'FinalTestR2':scores['R2'],
        'TestEvaluatedAfterSelectionLock':True,
        'TestLabelsUsedToSelectModel':False,
        'RefitOnTestData':False,
        'ModelRefittedOnTrainPlusValidation':False,
        'TestEvaluationPolicy':'Evaluate locked, prior-training-fitted MLP_B on test only once',
    }
    summary_path.write_text(json.dumps(record,indent=2),encoding='utf-8')
    model_hash=hashlib.sha256(model_path.read_bytes()).hexdigest()
    final_model_folder=root/'Models'/'week9'
    final_model_folder.mkdir(parents=True,exist_ok=True)
    (final_model_folder/'week9_final_model_manifest.json').write_text(
        json.dumps({**record,
            'FrozenModelSha256':model_hash,
            'InputFeatures':['PC1','PC2'],
            'FeatureProvenance':'Week 5 training-fitted transformations; Week 8 training-only scalers',
            'ModelRetainedFromWeek8':True},indent=2),encoding='utf-8')
    started.unlink(missing_ok=True)
    return record


def write_report_values(root: Path, selection: Selection, test: dict | None):
    out = root/'Reports'/'week9'
    lines = [
        '# Week 9 verified values for the written report', '',
        'Source: Week 6–8 actual validation artifacts; Week 9 test is available only after running locally.',
        'Selection occurs before any test labels are accessed.', '',
        '| Model | Week | Train MAE | Val MAE | Train RMSE | Val RMSE | Train R² | Val R² |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for _,r in selection.comparison.iterrows():
        lines.append(f'| {r.Model} | {r.Week} | {r.Training_MAE:,.2f} | {r.Validation_MAE:,.2f} | {r.Training_RMSE:,.2f} | {r.Validation_RMSE:,.2f} | {r.Training_R2:.4f} | {r.Validation_R2:.4f} |')
    lines+=['',f'**Validation winner: {selection.winner}**, MAE = {selection.ranking.iloc[0].Validation_MAE:,.2f} passengers.',
             '**Important:** Cross-family capacity is only approximately ordered. Training–validation loss gap is a diagnostic, not a formal bias–variance decomposition.','']
    if test:
        lines += ['## Final selected-model test results','',
                  f"Test MAE: **{test['FinalTestMAE']:,.2f}** passengers",
                  f"Test RMSE: **{test['FinalTestRMSE']:,.2f}** passengers",
                  f"Test R²: **{test['FinalTestR2']:.4f}**"]
    else:
        lines += ['## Test results not yet executed','',
                  'After the selection lock is written, run the full Week 9 pipeline against the official source data. Do not invent test metrics.']
    (out/'week9_verified_results_for_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run_week9(root: Path | None = None, validation_only: bool = False):
    root=root or root_from()
    print('===== WEEK 9 START =====')
    selected=select_and_lock(root)
    print('Historical 9-model validation ranking:')
    print(selected.ranking[['Model','Validation_MAE','Validation_RMSE']].round(3).to_string(index=False))
    print('Selection LOCKED before final test:', selected.winner)
    test= None if validation_only else final_test_once(root,selected)
    write_report_values(root,selected,test)
    if test:
        print(f"Final held-out test: MAE {test['FinalTestMAE']:,.2f}, RMSE {test['FinalTestRMSE']:,.2f}, R² {test['FinalTestR2']:.4f}")
    else:
        print('Validation-only audit complete; no test metrics computed.')
    print('All Week 9 checks passed.' if (test or validation_only) else 'Week 9 incomplete')
    print('===== WEEK 9 END =====')
    return selected,test


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-only',action='store_true',help='Audit nine validation scores WITHOUT accessing test.')
    args=parser.parse_args()
    run_week9(validation_only=args.validation_only)
