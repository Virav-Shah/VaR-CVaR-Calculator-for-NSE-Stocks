"""Purged chronological quantile models with auditable block-level provenance."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
from src.numerical_var import pinball_loss

QUANTILES = np.array([.95, .96, .97, .98, .99, .995])


@dataclass(frozen=True)
class WalkForwardConfig:
    train_size: int = 750
    validation_size: int = 375
    gap: int = 5
    refit_every: int = 60
    validation_block: int = 125
    seed: int = 42

    def __post_init__(self):
        if min(self.train_size, self.validation_size, self.refit_every, self.validation_block) < 1 or self.gap < 1:
            raise ValueError('Positive windows and at least one purged session required')

    @property
    def test_start(self):
        return self.train_size + self.gap + self.validation_size + self.gap


def validation_splits(n, config):
    end = config.train_size + config.gap + config.validation_size
    if n <= end + config.gap:
        raise ValueError('Insufficient complete observations for train, validation and holdout')
    for start in range(config.train_size + config.gap, end, config.validation_block):
        stop = start - config.gap
        yield np.arange(max(0, stop - config.train_size), stop), np.arange(start, min(start + config.validation_block, end))


def quantile_cvar(predictions, alpha, quantiles=QUANTILES):
    """Piecewise-linear integral plus flat terminal tail to 1.

    A finite grid cannot identify ES. Flat continuation above .995 is an
    explicitly conservative-in-information (usually downward-biased) estimate,
    not an exact, assumption-free CVaR. No strict VaR/ES inequality is imposed.
    """
    qs = np.asarray(quantiles)
    if not np.any(np.isclose(qs, alpha)):
        raise ValueError('alpha must be on the fitted grid')
    pred = np.sort(np.asarray(predictions), axis=-1)
    mask = qs >= alpha - 1e-12
    tail = pred[..., mask]
    return (np.trapezoid(tail, qs[mask], axis=-1) + (1 - qs[-1]) * tail[..., -1]) / (1 - alpha)


class QuantileMLModel:
    def __init__(self, kind='lightgbm', params=None, quantiles=QUANTILES, seed=42):
        if kind not in ('linear', 'lightgbm'):
            raise ValueError('kind must be linear or lightgbm')
        self.kind, self.params, self.seed = kind, dict(params or {}), seed
        self.quantiles = np.asarray(quantiles, float)
        if np.any(np.diff(self.quantiles) <= 0) or not np.all((self.quantiles > 0) & (self.quantiles < 1)):
            raise ValueError('Quantiles must be strictly increasing in (0, 1)')

    def fit(self, X, y, validation=None):
        self.columns = list(X.columns)
        self.scaler = StandardScaler().fit(X) if self.kind == 'linear' else None
        data = self.scaler.transform(X) if self.scaler else X
        self.models = []
        for q in self.quantiles:
            if self.kind == 'linear':
                model = QuantileRegressor(quantile=q, alpha=self.params.get('penalty', .0001), solver='highs')
                model.fit(data, y)
            else:
                from lightgbm import LGBMRegressor, early_stopping
                params = dict(n_estimators=200, learning_rate=.03, max_depth=3, num_leaves=7,
                              min_child_samples=50, subsample=.8, subsample_freq=1,
                              colsample_bytree=.8, verbosity=-1, n_jobs=2, device_type='cpu')
                params.update(self.params)
                model = LGBMRegressor(objective='quantile', alpha=q, random_state=self.seed, **params)
                kwargs = {}
                if validation is not None:
                    kwargs = dict(eval_set=[validation], callbacks=[early_stopping(25, verbose=False)])
                model.fit(data, y, **kwargs)
            self.models.append(model)
        return self

    def predict_raw(self, X):
        X = X[self.columns]
        data = self.scaler.transform(X) if self.scaler else X
        return np.column_stack([m.predict(data) for m in self.models])

    def predict(self, X):
        return np.sort(self.predict_raw(X), axis=1)

    def explain(self, X, alpha=.99):
        """Explain the actual sorted output via its selected source quantile.

        Sorting can select a different model per row; SHAP is conditional on
        that selection. Save source quantile and raw prediction for audit.
        """
        if self.kind != 'lightgbm':
            raise ValueError('Tree SHAP requires LightGBM')
        import shap
        raw = self.predict_raw(X)
        position = np.flatnonzero(np.isclose(self.quantiles, alpha))[0]
        sources = np.argsort(raw, axis=1, kind='stable')[:, position]
        values = np.zeros(X.shape)
        bases = np.zeros(len(X))
        for j in np.unique(sources):
            mask = sources == j
            explanation = shap.TreeExplainer(self.models[j])(X.loc[mask, self.columns])
            values[mask] = explanation.values
            bases[mask] = np.asarray(explanation.base_values).reshape(-1)
        predicted = raw[np.arange(len(X)), sources]
        if not np.allclose(bases + values.sum(axis=1), predicted, atol=1e-7):
            raise ValueError('SHAP reconstruction failed')
        return values, bases, self.quantiles[sources]


def tune_model(X, y, kind, config, n_trials=10):
    import optuna
    splits = list(validation_splits(len(X), config))
    def objective(trial):
        if kind == 'linear':
            params = {'penalty': trial.suggest_float('penalty', 1e-6, 1e-3, log=True)}
        else:
            depth = trial.suggest_int('max_depth', 2, 4)
            params = dict(max_depth=depth, num_leaves=trial.suggest_int('num_leaves', 4, min(15, 2**depth)),
                          min_child_samples=trial.suggest_int('min_child_samples', 30, 100),
                          subsample=trial.suggest_float('subsample', .6, .9),
                          colsample_bytree=trial.suggest_float('colsample_bytree', .6, .9),
                          learning_rate=trial.suggest_float('learning_rate', .01, .08, log=True),
                          n_estimators=trial.suggest_int('n_estimators', 100, 400))
        losses, iterations = [], []
        for train, valid in splits:
            model = QuantileMLModel(kind, params, seed=config.seed).fit(
                X.iloc[train], y.iloc[train], validation=(X.iloc[valid], y.iloc[valid]))
            pred = model.predict(X.iloc[valid])
            losses.extend(pinball_loss(y.iloc[valid].to_numpy()[:, None] - pred, QUANTILES).mean(axis=1))
            if kind == 'lightgbm':
                iterations.extend(m.best_iteration_ or params['n_estimators'] for m in model.models)
        if iterations:
            trial.set_user_attr('refit_estimators', max(1, int(np.median(iterations))))
        return float(np.mean(losses))
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=config.seed))
    study.optimize(objective, n_trials=n_trials)
    params = study.best_params.copy()
    if kind == 'lightgbm':
        params['n_estimators'] = study.best_trial.user_attrs['refit_estimators']
    return params, study.trials_dataframe()


def walk_forward(X, y, target_dates, kind, params, config, on_block=None):
    if not X.index.equals(y.index) or not X.index.equals(target_dates.index):
        raise ValueError('Features, labels and target dates must align')
    if not X.index.is_unique or not X.index.is_monotonic_increasing:
        raise ValueError('Chronological unique origins required')
    list(validation_splits(len(X), config))
    predictions, folds = [], []
    for block, start in enumerate(range(config.test_start, len(X), config.refit_every)):
        stop = start - config.gap
        train = np.arange(max(0, stop - config.train_size), stop)
        test = np.arange(start, min(start + config.refit_every, len(X)))
        if target_dates.iloc[train].max() >= X.index[test[0]]:
            raise ValueError('Training label overlaps the forecast information set')
        model = QuantileMLModel(kind, params, seed=config.seed).fit(X.iloc[train], y.iloc[train])
        pred = pd.DataFrame(model.predict(X.iloc[test]), index=pd.DatetimeIndex(target_dates.iloc[test]),
                            columns=[f'q_{q:g}' for q in QUANTILES])
        pred.index.name = 'Date'
        pred['Loss'] = y.iloc[test].to_numpy()
        pred['Origin'] = X.index[test]
        pred['Block'] = block
        for alpha in (.95, .99):
            pred[f'cvar_{alpha:g}'] = quantile_cvar(pred.iloc[:, :len(QUANTILES)].to_numpy(), alpha)
        predictions.append(pred)
        folds.append(dict(Block=block, Stage='OOS', TrainStart=X.index[train[0]],
                          TrainEnd=X.index[train[-1]], LastLabel=target_dates.iloc[train[-1]],
                          TestStart=X.index[test[0]], TestEnd=X.index[test[-1]], Gap=config.gap))
        if on_block:
            on_block(block, model, X.iloc[train], X.iloc[test], pred)
    return pd.concat(predictions), pd.DataFrame(folds)
