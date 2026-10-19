"""V1-preserving calibration experiment and pooled V2 research benchmark."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import joblib
import numpy as np
import pandas as pd
from src.hybrid_var import (LEVELS, student_context, calibrate_frame, build_panel,
                            panel_slice, PooledHybrid, student_tail_es)
from src.ml_var import WalkForwardConfig, validation_splits
from src.numerical_var import pinball_loss
from src.backtest import evaluate_forecasts

VERSIONS = ['Student-t baseline','LightGBM V1','V1 + calibration','Pooled hybrid V2','V2 + calibration']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(scores):
    rows = []
    for (alpha,version), group in scores.groupby(['Alpha','Version'],sort=False):
        coverage = group['Kupiec p-value'] > .05
        independence = group['Christoffersen p-value'] > .05
        both = coverage & independence
        n = len(group)
        status = ('Coverage needs improvement' if coverage.sum() < n else
                  'Dependence needs improvement' if independence.sum() < n else
                  'Statistical gates met; independent validation pending')
        rows.append(dict(Alpha=alpha, Version=version, Assets=n,
                         CoveragePasses=int(coverage.sum()), IndependencePasses=int(independence.sum()),
                         BothPasses=int(both.sum()), MeanBreachRate=group['Exception Rate (%)'].mean(),
                         MeanPinball=group['Pinball Loss'].mean(), MeanVaR=group['Mean VaR'].mean(),
                         MeanES=group['Mean ES'].mean(), Governance=status))
    return pd.DataFrame(rows)


def compare_versions(v1_root, root, tickers):
    scores = []
    for ticker in tickers:
        name = ticker.replace('.NS','')
        old, new = v1_root/name, root/name
        calibrated = pd.read_csv(new/'v1_calibrated.csv',index_col=0,parse_dates=True).dropna(subset=['q_0.95','q_0.99'])
        dates = calibrated.index
        variants = {'LightGBM V1':pd.read_csv(old/'lightgbm_predictions.csv',index_col=0,parse_dates=True),
                    'V1 + calibration':calibrated}
        for label,filename in [('Pooled hybrid V2','hybrid_raw.csv'),('V2 + calibration','hybrid_calibrated.csv')]:
            if (new/filename).exists():
                variants[label] = pd.read_csv(new/filename,index_col=0,parse_dates=True)
        if 'V2 + calibration' in variants:
            valid = variants['V2 + calibration'].dropna(subset=['q_0.95','q_0.99']).index
            if not dates.equals(valid):
                raise ValueError('Calibration evaluation dates differ between V1 and V2')
        for a in LEVELS:
            baseline = pd.read_csv(old/f'student_t_{a:g}_predictions.csv',index_col=0,parse_dates=True).loc[dates]
            entries = [('Student-t baseline',baseline, 'VaR','CVaR')]
            entries += [(version,frame.loc[dates],f'q_{a:g}',f'cvar_{a:g}') for version,frame in variants.items()]
            for version,frame,varcol,escol in entries:
                if not np.isfinite(frame[['Loss',varcol,escol]].to_numpy()).all():
                    raise ValueError('Comparison contains missing forecasts')
                result = evaluate_forecasts(frame.Loss,frame[varcol],frame[escol],a)
                result['Mean ES'] = frame[escol].mean()
                scores.append(dict(Ticker=ticker,Version=version,Alpha=a,Start=str(dates[0].date()),End=str(dates[-1].date()),**result))
    result = pd.DataFrame(scores)
    result.to_csv(root/'version_comparison.csv',index=False)
    summarize(result).to_csv(root/'headline_summary.csv',index=False)
    return result


def prepare(v1_root=Path('results/ml'), root=Path('results/ml_v2')):
    root.mkdir(parents=True,exist_ok=True)
    prices = pd.read_csv('data/raw/prices.csv',index_col=0,parse_dates=True)
    price_hash = digest('data/raw/prices.csv')
    snapshots, contexts = {}, {}
    for ticker in prices:
        name = ticker.replace('.NS','')
        old, new = v1_root/name, root/name
        new.mkdir(exist_ok=True)
        for path in [old/'lightgbm_predictions.csv',old/'manifest.json',old/'features.csv',old/'backtest_results.csv']:
            snapshots[str(path)] = digest(path)
        cache = new/'student_context.csv'
        metadata = new/'context_metadata.json'
        if cache.exists() and metadata.exists() and json.loads(metadata.read_text()).get('prices_sha256') == price_hash:
            context = pd.read_csv(cache,index_col=0,parse_dates=['Origin','TargetDate'])
        else:
            print(f'Student-t context: {ticker}',flush=True)
            returns = np.log(prices[ticker]/prices[ticker].shift(1)).dropna()
            context = student_context(returns)
            context.to_csv(cache)
            metadata.write_text(json.dumps({'prices_sha256':price_hash,'window':250,'refit':5,'volatility_window':20},indent=2))
        contexts[ticker] = context
        oldpred = pd.read_csv(old/'lightgbm_predictions.csv',index_col=0,parse_dates=True)
        origins = pd.DatetimeIndex(pd.to_datetime(oldpred.Origin))
        oldpred['Scale'] = context.loc[origins,'Scale'].to_numpy()
        oldpred['df'] = context.loc[origins,'df'].to_numpy()
        # Confirm statistical anchors match the existing benchmark exactly.
        for a in LEVELS:
            benchmark = pd.read_csv(old/f'student_t_{a:g}_predictions.csv',index_col=0,parse_dates=True)
            np.testing.assert_allclose(context.loc[origins,f't_var_{a:g}'],benchmark.loc[oldpred.index,'VaR'],atol=1e-9)
        calibrate_frame(oldpred).to_csv(new/'v1_calibrated.csv')
    (root/'v1_snapshot.json').write_text(json.dumps(snapshots,indent=2))
    compare_versions(v1_root,root,list(prices.columns))
    print('Saved calibration-only comparison without retraining V1.',flush=True)
    return prices, contexts


def tune_pooled(panel, features, dates, config, trials=10):
    import optuna
    folds = list(validation_splits(len(dates),config))
    def objective(trial):
        depth = trial.suggest_int('max_depth',2,4)
        params = dict(max_depth=depth,num_leaves=trial.suggest_int('num_leaves',4,min(15,2**depth)),
                      min_child_samples=trial.suggest_int('min_child_samples',300,1000),
                      learning_rate=trial.suggest_float('learning_rate',.01,.06,log=True),
                      n_estimators=trial.suggest_int('n_estimators',80,250))
        losses, iterations = [], []
        for tr,va in folds:
            train,valid = panel_slice(panel,dates[tr]),panel_slice(panel,dates[va])
            if train.TargetDate.max() >= dates[va[0]]:
                raise ValueError('Training labels reach the validation origin')
            model = PooledHybrid(params,config.seed).fit(train,features,valid)
            predictions = model.predict(valid)/valid.Scale.to_numpy()[:,None]
            target = (valid.Loss/valid.Scale).to_numpy()[:,None]
            # Every date has ten stocks, so each date receives equal total weight.
            losses.extend(pinball_loss(target-predictions,np.array(LEVELS)).mean(axis=1))
            iterations.extend(m.best_iteration_ or params['n_estimators'] for m in model.models.values())
        trial.set_user_attr('frozen_iterations',max(1,int(np.median(iterations))))
        return float(np.mean(losses))
    study = optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=config.seed))
    study.optimize(objective,n_trials=trials)
    params = study.best_params.copy()
    params['n_estimators'] = study.best_trial.user_attrs['frozen_iterations']
    return params,study.trials_dataframe()


def run_hybrid(trials=10, calibration_only=False, v1_root=Path('results/ml'), root=Path('results/ml_v2')):
    prices,contexts = prepare(v1_root,root)
    if calibration_only:
        return
    # Match the V1 temporal design exactly; never split by pooled row number.
    manifests = [json.loads((v1_root/t.replace('.NS','')/'manifest.json').read_text()) for t in prices]
    if any(m['config'] != manifests[0]['config'] for m in manifests):
        raise ValueError('V1 assets must share the same temporal design')
    config = WalkForwardConfig(**manifests[0]['config'])
    panel,features,dates = build_panel(prices,v1_root,contexts)
    params,trials_table = tune_pooled(panel,features,dates,config,trials)
    trials_table.to_csv(root/'pooled_trials.csv',index=False)
    panel.to_csv(root/'panel.csv')
    models = root.parent/'models_v2'
    models.mkdir(exist_ok=True)
    predictions, attribution, explanation_meta, folds = [], [], [], []
    for block,start in enumerate(range(config.test_start,len(dates),config.refit_every)):
        end = min(start+config.refit_every,len(dates))
        train_dates = dates[max(0,start-config.gap-config.train_size):start-config.gap]
        test_dates = dates[start:end]
        train,test = panel_slice(panel,train_dates),panel_slice(panel,test_dates)
        if train.TargetDate.max() >= test_dates[0]:
            raise ValueError('Pooled train labels overlap forecast origins')
        model = PooledHybrid(params,config.seed).fit(train,features)
        joblib.dump(model,models/f'block_{block:03}.joblib')
        native,ordered = model.predict_native(test),model.predict(test)
        result = test[['TargetDate','Scale','Loss','df','t_var_0.95','t_var_0.99']].copy()
        result['Block'] = block
        for j,a in enumerate(LEVELS):
            result[f'native_q_{a:g}'] = native[:,j]
            result[f'q_{a:g}'] = ordered[:,j]
        es95,es99,_,_ = student_tail_es(ordered[:,0],ordered[:,1],test['df'])
        result['cvar_0.95'],result['cvar_0.99'] = es95,es99
        predictions.append(result)
        values,bases = model.explain(test)
        attribution.append(pd.DataFrame(values,index=test.index,columns=features))
        explanation_meta.append(pd.DataFrame({'StudentBase':test['t_var_0.99'], 'TreeBase':bases,
                                               'NativeVaR99':native[:,1], 'RawOrderAdjustment':ordered[:,1]-native[:,1]},index=test.index))
        folds.append(dict(Block=block,TrainStart=str(train_dates[0]),TrainEnd=str(train_dates[-1]),
                          LastLabel=str(train.TargetDate.max()),TestStart=str(test_dates[0]),TestEnd=str(test_dates[-1]),
                          TrainDates=len(train_dates),TrainRows=len(train),TestDates=len(test_dates)))
        print(f'Pooled block {block+1}: {len(train_dates)} dates / {len(train)} training rows',flush=True)
    allpred, allshap, allmeta = pd.concat(predictions),pd.concat(attribution),pd.concat(explanation_meta)
    pd.DataFrame(folds).to_csv(root/'pooled_folds.csv',index=False)
    for ticker in prices:
        folder = root/ticker.replace('.NS','')
        raw = allpred.xs(ticker,level='Ticker').copy()
        origin = raw.index
        raw['Origin'] = origin
        raw = raw.set_index('TargetDate').rename_axis('Date')
        old = pd.read_csv(v1_root/ticker.replace('.NS','')/'lightgbm_predictions.csv',index_col=0,parse_dates=True)
        if not raw.index.equals(old.index):
            raise ValueError('V1/V2 forecast realization dates differ')
        raw.to_csv(folder/'hybrid_raw.csv')
        calibrated = calibrate_frame(raw)
        calibrated.to_csv(folder/'hybrid_calibrated.csv')
        sv,meta = allshap.xs(ticker,level='Ticker').copy(),allmeta.xs(ticker,level='Ticker').copy()
        sv.index,meta.index = raw.index,raw.index
        meta['CalibrationBuffer'] = calibrated['buffer_0.99']
        meta['FinalOrderAdjustment'] = calibrated['order_adjustment_0.99']
        meta['FinalVaR99'] = calibrated['q_0.99']
        meta['Block'] = raw.Block
        reconstructed = meta.StudentBase+meta.TreeBase+sv.sum(axis=1)+meta.RawOrderAdjustment+meta.CalibrationBuffer+meta.FinalOrderAdjustment
        np.testing.assert_allclose(reconstructed.dropna(),meta.FinalVaR99.dropna(),atol=1e-7)
        sv.to_csv(folder/'shap_values.csv')
        meta.to_csv(folder/'shap_metadata.csv')
    scores = compare_versions(v1_root,root,list(prices.columns))
    snapshots = json.loads((root/'v1_snapshot.json').read_text())
    assert all(digest(p) == h for p,h in snapshots.items()), 'V1 artifacts changed'
    manifest = dict(config=asdict(config),frozen_parameters=params,features=features,
                    prices_sha256=digest('data/raw/prices.csv'),panel_sha256=digest(root/'panel.csv'),
                    source_hashes={str(p):digest(p) for p in [Path('src/hybrid_var.py'),Path('src/hybrid_pipeline.py')]},
                    package_versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','lightgbm','shap','optuna']},
                    quantiles=list(LEVELS),calibration_window=250,calibration_rank={'0.95':239,'0.99':249},
                    train_dates=750,train_rows=750*len(prices.columns),assets=list(prices.columns),
                    oos_start=str(allpred.TargetDate.min()),oos_end=str(allpred.TargetDate.max()),
                    evaluation_start=scores.Start.iloc[0],evaluation_end=scores.End.iloc[0],
                    interpretation='Retrospective V2 experiment: V1 OOS results influenced this redesign; not an untouched final test.',
                    tail_assumption='Student-t df from historical fit; location and scale match calibrated 95% and 99% VaR.',
                    limitations=['Correlated stocks do not provide independent crash observations.',
                                 'Rolling residual calibration has no IID finite-sample guarantee here.',
                                 'ES is a parametric tail proxy and has not passed an independent ES validation.',
                                 'Statistical test passes alone do not establish production readiness.'])
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('V2 completed; V1 artifacts unchanged.',flush=True)
