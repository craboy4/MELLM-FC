"""Corrected evaluator; original gates and expression semantics remain in fraud_alpha."""
from __future__ import annotations
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import threading
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from fraud_alpha.calculator import FraudCalculator
from fraud_alpha.expression import Feature, Constant, DeltaTime
from fraud_alpha.factor_analyzer import FactorAnalyzer
from common import BASE, RUN, save, identity
from feature_parallel import FeatureValueBackend

def expression_meta(expr):
    if isinstance(expr, Feature):
        return dict(depth=1, nodes=1, variables=[expr._feature], windows=[])
    if isinstance(expr, Constant):
        return dict(depth=1, nodes=1, variables=[], windows=[])
    if isinstance(expr, DeltaTime):
        return dict(depth=1, nodes=1, variables=[], windows=[expr._delta_time])
    children = [expression_meta(x) for x in expr.operands]
    return dict(depth=1 + max((x['depth'] for x in children)), nodes=1 + sum((x['nodes'] for x in children)), variables=sorted(set((v for x in children for v in x['variables']))), windows=sorted(set((v for x in children for v in x['windows']))))

def metrics(y, p):
    return dict(AUC=float(roc_auc_score(y, p)), AP=float(average_precision_score(y, p)), n=len(y), positives=int(np.sum(y)))

class Evaluator(FraudCalculator):

    def __init__(self, raw, config, persist=True, parallel_values=False):
        assert raw['year'].between(2010, 2021).all()
        assert not raw.duplicated(['firm_id', 'year']).any()
        self.raw = raw.copy().sort_values(['firm_id', 'year'], kind='stable').reset_index(drop=True)
        self.frame = self.raw.copy()
        self.train_mask = self.frame['year'].le(2018).to_numpy()
        self.valid_mask = self.frame['year'].between(2019, 2021).to_numpy()
        self.medians = self.frame.loc[self.train_mask, BASE].replace([np.inf, -np.inf], np.nan).median()
        assert self.medians.notna().all()
        self.frame[BASE] = self.frame[BASE].replace([np.inf, -np.inf], np.nan).fillna(self.medians)
        super().__init__(self.frame.loc[self.train_mask].reset_index(drop=True), self.frame.loc[self.valid_mask].reset_index(drop=True), base_features=BASE, model_config=config['model'], feature_validation_config=config['feature_validation'])
        self.config = config
        self.feature_arrays = {}
        self.feature_audit = {}
        self.scores = {}
        self.models = OrderedDict()
        self.trial_log = []
        self.fit_count = 0
        self._guard = threading.RLock()
        self._key_locks = {}
        self.value_backend = FeatureValueBackend(self.frame, self.train_data, self.train_mask, BASE, workers=config['numeric_workers']) if parallel_values else None
        if persist:
            save(RUN / 'preprocessing.json', dict(medians=self.medians.to_dict(), fit_years=[2010, 2018], apply_years=[2010, 2021], training_rows=int(self.train_mask.sum()), validation_rows=int(self.valid_mask.sum()), test_loaded=False))

    def _lock(self, key):
        with self._guard:
            return self._key_locks.setdefault(str(key), threading.RLock())

    def values(self, expr):
        key = str(expr)
        with self._lock(('feature', key)):
            if key in self.feature_arrays:
                return self.feature_arrays[key]
            meta = expression_meta(expr)
            if not set(meta['variables']) <= set(BASE) or not meta['variables']:
                raise ValueError('Invalid feature whitelist')
            if any((w <= 0 for w in meta['windows'])):
                raise ValueError('Nonpositive/future window')
            if self.value_backend is not None:
                values = self.value_backend.evaluate(expr)
            else:
                with warnings.catch_warnings(), np.errstate(all='ignore'):
                    warnings.simplefilter('ignore')
                    values = np.asarray(expr.evaluate(self.frame), float)
                    before = np.asarray(expr.evaluate(self.train_data), float)
                if values.shape != (len(self.frame),):
                    raise ValueError('Expression shape mismatch')
                if not np.allclose(values[self.train_mask], before, equal_nan=True):
                    raise ValueError('Future dependency or row alignment failure')
                values = np.where(np.isfinite(values), values, np.nan)
            self.feature_arrays[key] = values
            self.feature_audit[key] = dict(**meta, prefix_invariance=True, train_missing=float(np.isnan(values[self.train_mask]).mean()), validation_missing=float(np.isnan(values[self.valid_mask]).mean()))
            return values

    def prefetch_values(self, exprs):
        if self.value_backend is not None and exprs:
            with ThreadPoolExecutor(max_workers=self.config['numeric_workers']) as ex:
                list(ex.map(self.values, exprs))

    def close(self):
        if self.value_backend is not None:
            self.value_backend.close()

    def get_cached_train_feature(self, expr):
        values = self.values(expr)[self.train_mask]
        return values if self._is_valid_feature(values) else None

    def get_cached_valid_feature(self, expr):
        values = self.values(expr)[self.valid_mask]
        return values if self._is_valid_feature(values) else None

    def remove_feature_cache(self, expr_str):
        return None

    def matrix(self, exprs, base=None):
        base = BASE if base is None else base
        parts = [self.frame[base].to_numpy(float)]
        parts += [self.values(e)[:, None] for e in exprs]
        return np.column_stack(parts)

    def parameters(self):
        return dict(self.ensemble_params, objective='binary', random_state=self.config['seed'], verbosity=-1, n_jobs=self.config['threads_per_model'], deterministic=True, force_col_wise=True)

    def score(self, exprs, base=None, model_required=False, record=True):
        base = BASE if base is None else list(base)
        key = (tuple(base), tuple((str(e) for e in exprs)))
        with self._lock(('model', key)):
            with self._guard:
                cached = self.models.get(key)
            if key not in self.scores or (model_required and cached is None):
                X = self.matrix(exprs, base)
                model = lgb.LGBMClassifier(**self.parameters())
                model.fit(X[self.train_mask], self.y_train, eval_set=[(X[self.valid_mask], self.y_valid)], eval_metric='auc', callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)])
                train_pred = model.predict_proba(X[self.train_mask])[:, 1]
                val_pred = model.predict_proba(X[self.valid_mask])[:, 1]
                result = dict(training=metrics(self.y_train, train_pred), validation=metrics(self.y_valid, val_pred), best_iteration=int(model.best_iteration_ or self.ensemble_params['n_estimators']), pool_version=identity(key))
                cached = (model, train_pred, val_pred)
                with self._guard:
                    self.fit_count += 1
                    self.scores[key] = result
                    self.models[key] = cached
                    while len(self.models) > 48:
                        self.models.popitem(last=False)
            result = self.scores[key]
            if record:
                with self._guard:
                    self.trial_log.append(dict(expressions=list(key[1]), **result))
            return (result, cached)

    def calc_pool_auc_with_lgbm(self, exprs, selected_base_features=None, return_model=False, use_cv=False):
        if use_cv:
            raise ValueError('Random cross-validation is not part of this temporal protocol')
        result, cached = self.score(exprs, selected_base_features, model_required=return_model)
        return (result['validation']['AUC'], cached[0] if return_model else None)

    def calc_pool_auc(self, exprs, use_valid=True, **kwargs):
        result, _ = self.score(exprs)
        return result['validation' if use_valid else 'training']['AUC']

    def calc_raw_auc(self, expr, use_valid=True):
        key = ('raw', str(expr), use_valid)
        if key in self._cache:
            return self._cache[key]
        train = self.get_cached_train_feature(expr)
        target = self.get_cached_valid_feature(expr) if use_valid else train
        if train is None or target is None:
            return 0.5
        median = float(np.nanmedian(train))
        y = self.y_valid if use_valid else self.y_train
        auc = float(roc_auc_score(y, np.where(np.isnan(target), median, target)))
        self._cache[key] = max(auc, 1 - auc)
        return self._cache[key]

    def calc_single_auc_fast(self, expr, use_valid=True):
        key = ('fast', str(expr), use_valid)
        if key in self._cache:
            return self._cache[key]
        train = self.get_cached_train_feature(expr)
        valid = self.get_cached_valid_feature(expr)
        if train is None or valid is None:
            return 0.0
        med = float(np.nanmedian(train))
        train = np.where(np.isnan(train), med, train).reshape(-1, 1)
        valid = np.where(np.isnan(valid), med, valid).reshape(-1, 1)
        model = LogisticRegression(max_iter=100, solver='lbfgs')
        model.fit(train, self.y_train)
        for role, X, y in [(False, train, self.y_train), (True, valid, self.y_valid)]:
            self._cache['fast', str(expr), role] = float(roc_auc_score(y, model.predict_proba(X)[:, 1]))
        return self._cache[key]

    def calc_single_auc(self, expr, use_valid=True):
        key = (str(expr), use_valid)
        if key in self._cache:
            return self._cache[key]
        train = self.get_cached_train_feature(expr)
        valid = self.get_cached_valid_feature(expr)
        if train is None or valid is None:
            return 0.0
        model = lgb.LGBMClassifier(**self.single_params, objective='binary', random_state=self.config['seed'], verbosity=-1, n_jobs=self.config['threads_per_model'], deterministic=True, force_col_wise=True)
        model.fit(train[:, None], self.y_train)
        for role, X, y in [(False, train, self.y_train), (True, valid, self.y_valid)]:
            self._cache[str(expr), role] = float(roc_auc_score(y, model.predict_proba(X[:, None])[:, 1]))
        return self._cache[key]

class Analyzer(FactorAnalyzer):

    def compute_marginal_contributions(self):
        exprs = list(self.pool.exprs)
        if not exprs:
            return {}
        full, _ = self.calculator.score(exprs, record=False)

        def one(i):
            score, _ = self.calculator.score(exprs[:i] + exprs[i + 1:], record=False)
            return (str(exprs[i]), {r: full[r]['AUC'] - score[r]['AUC'] for r in ['training', 'validation']})
        with ThreadPoolExecutor(max_workers=self.calculator.config['numeric_workers']) as ex:
            self.paired_marginals = dict(ex.map(one, range(len(exprs))))
        return {k: v['validation'] for k, v in self.paired_marginals.items()}

    def compute_all_stats(self):
        self._stats_cache.clear()
        result = super().compute_all_stats()
        for s in result:
            s['marginal_training'] = self.paired_marginals[s['expr_str']]['training']
            s['marginal_validation'] = self.paired_marginals[s['expr_str']]['validation']
            s['marginal_source'] = 'same evaluator; full pool minus leave-one-out; single factor minus true Base'
        return result
