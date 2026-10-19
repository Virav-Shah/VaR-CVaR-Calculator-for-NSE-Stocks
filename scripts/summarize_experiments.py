"""Export README evidence from trusted local training artifacts, without refitting ML.

Run after both pipelines: python -m scripts.summarize_experiments
Training scores are reconstructed diagnostics for the first saved OOS refit.
"""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from src.numerical_var import pinball_loss
from src.portfolio import min_variance_portfolio
from src.portfolio_diagnostics import portfolio_diagnostics
from src.ru_optimization import ru_portfolio_optimization


def read(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)


def trial_records(path, model, ticker, unit, quantiles):
    trials = pd.read_csv(path)
    complete = trials[trials.state == 'COMPLETE']
    best = complete.loc[complete.value.idxmin()]
    records = []
    for _, row in trials.iterrows():
        params = {k.removeprefix('params_'):row[k] for k in trials if k.startswith('params_')}
        iteration = row.get('user_attrs_refit_estimators', row.get('user_attrs_frozen_iterations', np.nan))
        records.append(dict(Model=model,Ticker=ticker,Trial=int(row.number),Selected=bool(row.number==best.number),
                            ValidationPinball=float(row.value),ScoreUnit=unit,Quantiles=json.dumps(quantiles),
                            FrozenIterations=iteration,TrialParameters=json.dumps(params),State=row.state))
    return best, records


def summarize():
    destination = Path('results/tables')
    destination.mkdir(parents=True, exist_ok=True)
    prices = read('data/raw/prices.csv')
    losses = -np.log(prices/prices.shift(1))
    rows, trials, manifests, validation_folds = [], [], {}, None
    for ticker in prices:
        name = ticker.replace('.NS','')
        folder = Path('results/ml')/name
        manifest = json.loads((folder/'manifest.json').read_text())
        manifests[ticker] = manifest
        X = read(folder/'features.csv')
        target = losses[ticker].shift(-1).reindex(X.index)
        if validation_folds is None:
            validation_folds = pd.read_csv(folder/'validation_folds.csv')
        for kind, label in [('linear','Linear V1'),('lightgbm','LightGBM V1')]:
            folds = pd.read_csv(folder/f'{kind}_folds.csv')
            first = folds.iloc[0]
            train = X.loc[first.TrainStart:first.TrainEnd]
            assert len(train) == manifest['config']['train_size']
            model = joblib.load(Path('results/models')/name/kind/'block_000.joblib')
            train_pred = model.predict(train)
            quantiles = np.array(manifest['quantiles'])
            train_loss = pinball_loss(target.loc[train.index].to_numpy()[:,None]-train_pred,quantiles).mean()
            pred = read(folder/f'{kind}_predictions.csv')
            q = pred[[f'q_{a:g}' for a in quantiles]].to_numpy()
            test_loss = pinball_loss(pred.Loss.to_numpy()[:,None]-q,quantiles).mean()
            best, records = trial_records(folder/f'{kind}_trials.csv',label,ticker,'log loss',quantiles.tolist())
            trials.extend(records)
            rows.append(dict(Model=label,Ticker=ticker,Features=len(X.columns),TrainRows=len(train),
                             ScoreUnit='log loss',Quantiles=json.dumps(quantiles.tolist()),
                             FirstRefitTrainPinball=float(train_loss),BestValidationPinball=float(best.value),
                             FullOOSPinball=float(test_loss),BestTrial=int(best.number),
                             TrainStart=first.TrainStart,TrainEnd=first.TrainEnd,
                             ValidationOriginStart=validation_folds.TestStart.iloc[0],
                             ValidationOriginEnd=validation_folds.TestEnd.iloc[-1],
                             OOSStart=str(pred.index.min().date()),OOSEnd=str(pred.index.max().date()),
                             OOSRows=len(pred),FrozenParameters=json.dumps(manifest['frozen_parameters'][kind])))
    v2 = json.loads(Path('results/ml_v2/manifest.json').read_text())
    panel = pd.read_csv('results/ml_v2/panel.csv',parse_dates=['Origin','TargetDate']).set_index(['Origin','Ticker'])
    folds = pd.read_csv('results/ml_v2/pooled_folds.csv')
    first = folds.iloc[0]
    origins = panel.index.get_level_values('Origin')
    train = panel[(origins>=pd.Timestamp(first.TrainStart)) & (origins<=pd.Timestamp(first.TrainEnd))]
    model = joblib.load('results/models_v2/block_000.joblib')
    quantiles = np.array(v2['quantiles'])
    error = (train.Loss.to_numpy()[:,None]-model.predict(train))/train.Scale.to_numpy()[:,None]
    train_score = float(pinball_loss(error,quantiles).mean())
    predictions = pd.concat([read(Path('results/ml_v2')/t.replace('.NS','')/'hybrid_raw.csv') for t in prices])
    q = predictions[[f'q_{a:g}' for a in quantiles]].to_numpy()
    error = (predictions.Loss.to_numpy()[:,None]-q)/predictions.Scale.to_numpy()[:,None]
    test_score = float(pinball_loss(error,quantiles).mean())
    best, records = trial_records(Path('results/ml_v2/pooled_trials.csv'),'Pooled hybrid V2','ALL',
                                  'volatility-normalized loss',quantiles.tolist())
    trials.extend(records)
    rows.append(dict(Model='Pooled hybrid V2',Ticker='ALL',Features=len(v2['features']),TrainRows=len(train),
                     ScoreUnit='volatility-normalized loss',Quantiles=json.dumps(quantiles.tolist()),
                     FirstRefitTrainPinball=train_score,BestValidationPinball=float(best.value),FullOOSPinball=test_score,
                     BestTrial=int(best.number),TrainStart=str(pd.Timestamp(first.TrainStart).date()),
                     TrainEnd=str(pd.Timestamp(first.TrainEnd).date()),
                     ValidationOriginStart=validation_folds.TestStart.iloc[0],ValidationOriginEnd=validation_folds.TestEnd.iloc[-1],
                     OOSStart=str(predictions.index.min().date()),OOSEnd=str(predictions.index.max().date()),
                     OOSRows=len(predictions),FrozenParameters=json.dumps(v2['frozen_parameters'])))
    development = pd.DataFrame(rows)
    development.to_csv(destination/'model_development_scores.csv',index=False)
    pd.DataFrame(trials).to_csv(destination/'validation_trials.csv',index=False)
    validation_folds.to_csv(destination/'validation_folds.csv',index=False)
    folds.to_csv(destination/'pooled_training_folds.csv',index=False)
    # Reproduce the CURRENT dashboard's simple-return portfolio metrics.
    returns = np.log(prices/prices.shift(1)).dropna()
    portfolio_train = returns.loc[returns.index<='2015-12-31']
    portfolio_test = returns.loc[returns.index>'2015-12-31']
    portfolio_scores, allocation_rows = [], []
    for alpha in (.95,.99):
        min_cvar,_,_ = ru_portfolio_optimization(portfolio_train,alpha=alpha)
        weights = pd.DataFrame({'Equal Weight':np.full(len(prices.columns),1/len(prices.columns)),
                                'Min Variance':min_variance_portfolio(portfolio_train),
                                'Min CVaR':min_cvar},index=prices.columns)
        _,_,_,metrics = portfolio_diagnostics(portfolio_test,weights,alpha)
        metrics = metrics.reset_index()
        metrics.insert(0,'Alpha',alpha)
        metrics['Start'],metrics['End'],metrics['Sessions'] = str(portfolio_test.index.min().date()),str(portfolio_test.index.max().date()),len(portfolio_test)
        portfolio_scores.append(metrics)
        allocation = weights.rename_axis('Ticker').reset_index()
        allocation.insert(0,'Alpha',alpha)
        allocation_rows.append(allocation)
    portfolio = pd.concat(portfolio_scores,ignore_index=True)
    portfolio.to_csv(destination/'portfolio_dashboard_metrics.csv',index=False)
    pd.concat(allocation_rows,ignore_index=True).to_csv(destination/'portfolio_dashboard_weights.csv',index=False)
    importance = []
    bundle = Path('results/dashboard_v2')
    for version,filename in [('LightGBM V1','v1_shap_values.csv.gz'),('Pooled hybrid V2 residual','shap_values.csv.gz')]:
        values = pd.concat([read(bundle/t.replace('.NS','')/filename) for t in prices])
        for feature,value in values.abs().mean().sort_values(ascending=False).items():
            importance.append(dict(Model=version,Feature=feature,MeanAbsoluteSHAP=float(value),ForecastRows=len(values)))
    pd.DataFrame(importance).to_csv(destination/'global_shap_importance.csv',index=False)
    example_folder,day = bundle/'BHARTIARTL',pd.Timestamp('2020-03-23')
    values = read(example_folder/'shap_values.csv.gz').loc[day]
    meta = read(example_folder/'shap_metadata.csv.gz').loc[day]
    pred = read(example_folder/'hybrid_calibrated.csv.gz').loc[day]
    top = values.abs().nlargest(5).index
    components = {'Student-t anchor':meta.StudentBase,'Tree expected residual':meta.TreeBase,
                  **{f'SHAP: {f}':values[f] for f in top},'SHAP: all remaining features':values.drop(top).sum(),
                  'Raw quantile ordering':meta.RawOrderAdjustment,'Calibration buffer':meta.CalibrationBuffer,
                  'Final quantile ordering':meta.FinalOrderAdjustment}
    np.testing.assert_allclose(sum(components.values()),meta.FinalVaR99,atol=1e-7)
    pd.DataFrame([dict(Ticker='BHARTIARTL.NS',Date=str(day.date()),Component=k,Contribution=v,
                       FinalVaR99=meta.FinalVaR99,RealizedLogLoss=pred.Loss) for k,v in components.items()]).to_csv(destination/'shap_crash_example.csv',index=False)
    record = dict(
        prices=dict(rows=len(prices),stocks=len(prices.columns),start=str(prices.index.min().date()),end=str(prices.index.max().date()),
                    missing_cells=int(prices.isna().sum().sum()),sha256=hashlib.sha256(Path('data/raw/prices.csv').read_bytes()).hexdigest()),
        v1_manifests=manifests,v2_manifest=v2,
        development_score_notes=['Training scores were reconstructed from the first serialized OOS refit, without refitting.',
                                 'Validation scores are the winning saved Optuna objectives across three rolling folds.',
                                 'Full OOS scores use 3181 dates per stock before the calibration comparison warm-up exclusion.',
                                 'V1 averages six quantiles in log-loss units; V2 averages two quantiles in volatility-normalized units.',
                                 'These scores are not directly comparable across V1 and V2.',
                                 'The first refit training window overlaps the earlier tuning/validation era; this is not a disjoint random train/validation/test split.',
                                 'No historical per-iteration training loss curves were saved.'],
        portfolio=dict(train_start=str(portfolio_train.index.min().date()),train_end=str(portfolio_train.index.max().date()),
                       train_sessions=len(portfolio_train),test_start=str(portfolio_test.index.min().date()),
                       test_end=str(portfolio_test.index.max().date()),test_sessions=len(portfolio_test)))
    Path('results/experiment_record.json').write_text(json.dumps(record,indent=2))
    print(development.groupby('Model')[['FirstRefitTrainPinball','BestValidationPinball','FullOOSPinball']].mean().to_string())
    print(portfolio.to_string(index=False))
    print('Exported development scores, validation trials, fold timing, portfolio metrics, SHAP evidence and experiment provenance.')


if __name__=='__main__':
    summarize()
