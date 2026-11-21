"""Week 11. Reconstruct the published baseline and audit a fixed mitigation."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from threadpoolctl import threadpool_limits
import week5_src_features_dimensionality as w5
import week10_src_data_enhancement_error_analysis_refit as w10

FLAG = 'PassengerLag11Missing'
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'Reports/week11/tables'
DATA = ROOT / 'Data/processed/week11'
BUNDLE = ROOT / 'Models/week10/week10_selected_enhanced_mlp_bundle.joblib'
ARCHIVE = ROOT / 'Reports/week10/week10_tables/week10_train_validation_test_metrics.csv'
LABEL = w10.TARGET


def build_partitions(bundle):
    """Apply the saved preprocessing, retaining only audit columns after each split."""
    monthly = w5.create_cross_field_features(w5.load_monthly_without_rewriting_reports(ROOT))
    enriched = w10.add_history_features(monthly)
    del monthly
    splits, summary = w5.get_week3_splits(enriched, ROOT)
    del enriched
    w10.assert_split_integrity(splits, summary)
    if {k: len(v) for k, v in splits.items()} != w10.REFERENCE_EXPECTED_COUNTS:
        raise ValueError('Split counts differ from Week 10.')
    frames = {}
    for split in w10.SPLITS:
        raw = splits.pop(split)
        pcs = w10.original_pca_inputs(raw, bundle['original_pca'], w5)
        x, _ = w10.transform_enhanced(raw, pcs, bundle['enhancement'])
        frames[split] = pd.concat([raw[w10.KEYS + [w10.FORECAST_TIME, LABEL]],
                                   x[bundle['feature_order']]], axis=1).reset_index(drop=True)
    return frames


def evaluate(bundle, frames):
    """Resample the training data once; retain every original row for evaluation."""
    train = frames['Training']
    cols = bundle['feature_order']
    scarce = np.flatnonzero(train[FLAG].to_numpy() == 1)
    rng = np.random.default_rng(111)
    extra = rng.choice(scarce, size=len(scarce)//2, replace=True)
    indices = np.r_[np.arange(len(train)), extra]
    refit = clone(bundle['model'])
    with threadpool_limits(limits=2):
        refit.fit(train[cols].to_numpy()[indices], train[LABEL].to_numpy()[indices])
    metrics, groups = [], []
    archive = pd.read_csv(ARCHIVE).query("Model == 'MLP_B_temporal'").set_index('Split')
    for split, f in frames.items():
        for name, model in [('Week 10', bundle['model']), ('Reweighted', refit)]:
            y = f[LABEL].to_numpy()
            pred = np.maximum(0, model.predict(f[cols].to_numpy()))
            m = w10.regression_metrics(y, pred)
            if name == 'Week 10':
                for key in ['MAE', 'RMSE', 'R2', 'MeanResidual', 'WAPE_Percent']:
                    np.testing.assert_allclose(m[key], archive.loc[split,key], atol=1e-5)
            metrics.append(dict(Split=split, Model=name, Rows=len(f), **m))
            for missing, part in f.groupby(FLAG):
                ix = part.index.to_numpy()
                groups.append(dict(Split=split, Model=name, History='missing' if missing else 'observed',
                                   Rows=len(ix), Share=len(ix)/len(f), MeanActual=y[ix].mean(),
                                   MeanForecast=pred[ix].mean(), UnderRate=(y[ix]>pred[ix]).mean(),
                                   **w10.regression_metrics(y[ix],pred[ix])))
    write('metrics', metrics)
    write('subgroups', groups)
    write('training_treatment',[dict(Original=len(train), OriginalMissing=len(scarce),
          Added=len(extra), Treated=len(indices), Epochs=refit.regressor_.named_steps['mlp'].n_iter_)])


def write(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / (name+'.csv'), index=False)

def feature_groups(cols):
    """Preserve temporal dependencies within each block during attribution."""
    return {'PC1':[0], 'PC2':[1],
            'Recent history':[i for i,c in enumerate(cols) if any(x in c for x in ('Lag1Log','Lag1Missing','Lag2','Rolling','Monthly'))],
            'Prior target season':[i for i,c in enumerate(cols) if 'Lag11' in c],
            'Annual history':[i for i,c in enumerate(cols) if 'Lag12' in c or 'Annual' in c]}


def forecast(model, x):
    return np.maximum(0., model.predict(x))


def importance(bundle, frames):
    f = frames['Validation'].sample(n=8192, random_state=111)
    cols, model = bundle['feature_order'], bundle['model']
    x, y = f[cols].to_numpy(), f[LABEL].to_numpy()
    baseline = np.abs(y-forecast(model,x)).mean()
    groups = feature_groups(cols)
    assert sorted(i for block in groups.values() for i in block) == list(range(len(cols)))
    results = []
    for kind, blocks in [('Feature', {c:[i] for i,c in enumerate(cols)}), ('Block', groups)]:
        for name, index in blocks.items():
            rng = np.random.default_rng(110)
            effects = []
            for _ in range(12):
                shuffled = x.copy()
                shuffled[:,index] = x[rng.permutation(len(x))][:,index]
                effects.append(np.abs(y-forecast(model,shuffled)).mean()-baseline)
            results.append(dict(Type=kind, Predictor=name, Increase=np.mean(effects),
                                RepeatSD=np.std(effects,ddof=1), SampleMAE=baseline))
    write('importance', sorted(results,key=lambda row:-row['Increase']))


def local_cases(bundle, frames):
    """Exact five-block Shapley values relative to 64 observed training profiles."""
    cols, model = bundle['feature_order'], bundle['model']
    groups = feature_groups(cols)
    background = frames['Training'].sample(64,random_state=112)[cols].to_numpy()
    cases = frames['Test'].sample(5,random_state=113).reset_index(drop=True)
    DATA.mkdir(parents=True,exist_ok=True)
    cases.to_csv(DATA/'five_random_test_rows.csv',index=False)
    ix = cols.index('PassengerLag11Log1p')
    flag = cols.index(FLAG)
    historical = frames['Training'].loc[lambda d:d[FLAG]==0,cols[ix]]
    grid = np.unique(np.rint(np.expm1(historical.quantile(np.linspace(0,1,1001)))).astype(int))
    results, details = [], []
    for n,row in cases.iterrows():
        x = row[cols].to_numpy(dtype=float)
        values = []
        for mask in range(2**len(groups)):
            z = background.copy()
            for j, index in enumerate(groups.values()):
                if mask & (1<<j):
                    z[:,index] = x[index]
            values.append(forecast(model,z).mean())
        effects = {}
        for j,name in enumerate(groups):
            terms = []
            for mask in range(32):
                if mask & (1<<j):
                    continue
                size = mask.bit_count()
                terms.append((values[mask | (1<<j)]-values[mask])/(5*math.comb(4,size)))
            effects[name] = sum(terms)
        p = forecast(model,x[None,:])[0]
        np.testing.assert_allclose(values[0]+sum(effects.values()),p,atol=1e-7)
        details.append(dict(Case=n+1,Reference=values[0],**effects))
        z = np.tile(x,(len(grid),1))
        z[:,ix], z[:,flag] = np.log1p(grid), 0
        changes = forecast(model,z)-p
        threshold = max(50.,.1*p)
        valid = np.flatnonzero(abs(changes)>=threshold)
        if not len(valid):
            raise ValueError('Lag 11 scenario cannot reach the specified materiality threshold.')
        anchor = np.expm1(x[ix])
        best = valid[np.argmin(abs(grid[valid]-anchor))]
        results.append(dict(Case=n+1,Carrier=row.UniqueCarrier,Route=row.Origin+'–'+row.Dest,
               Month=str(row.TargetPeriod)[:7],Actual=row[LABEL],Forecast=p,Threshold=threshold,
               PriorCount=np.nan if x[flag] else round(anchor), ScenarioCount=grid[best],
               ScenarioForecast=p+changes[best],Change=changes[best]))
    write('cases',results)
    write('attributions',details)

def airports(bundle, frames):
    """Group by exposure available before validation; never infer passenger demographics."""
    counts = frames['Training'].Origin.value_counts()
    cutoff = int(frames['Training'].Origin.map(counts).quantile(.25))
    f = frames['Test']
    counts_at_origin = f.Origin.map(counts).fillna(0)
    pred = forecast(bundle['model'],f[bundle['feature_order']].to_numpy())
    rows = []
    for name,mask in [('Lower exposure',counts_at_origin<=cutoff),('Higher exposure',counts_at_origin>cutoff)]:
        rows.append(dict(Group=name,UpperCutoffForLowerGroup=cutoff,Rows=mask.sum(),
                         Origins=f.loc[mask,'Origin'].nunique(), MissingHistoryPct=100*f.loc[mask,FLAG].mean(),
                         **w10.regression_metrics(f.loc[mask,LABEL],pred[mask])))
    write('origin_exposure',rows)


def run():
    with threadpool_limits(limits=2):
        bundle = joblib.load(BUNDLE)
        frames = build_partitions(bundle)
        importance(bundle, frames)
        local_cases(bundle, frames)
        airports(bundle, frames)
        evaluate(bundle, frames)


if __name__ == '__main__':
    run()
