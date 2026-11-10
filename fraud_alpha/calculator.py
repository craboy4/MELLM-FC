"""Evaluate individual factors and factor combinations on fixed development splits."""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import lightgbm as lgb
import warnings
from fraud_alpha.expression import Expression, RollingOperator, PairRollingOperator
from fraud_alpha.data_loader import get_base_feature_columns

def _has_rolling_ops(expr: Expression) -> bool:
    if isinstance(expr, (RollingOperator, PairRollingOperator)):
        return True
    if hasattr(expr, 'operands'):
        return any((_has_rolling_ops(op) for op in expr.operands))
    return False

def _is_numeric_feature_series(series: pd.Series, threshold: float=0.95) -> bool:
    numeric = pd.to_numeric(series, errors='coerce')
    if len(series) == 0:
        return False
    return float(numeric.notna().mean()) >= threshold
warnings.filterwarnings('ignore')

class FraudCalculator:

    def __init__(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, target_col: str='Fraud', use_cache: bool=True, model_config: Optional[Dict[str, Any]]=None, feature_validation_config: Optional[Dict[str, Any]]=None, base_features: Optional[List[str]]=None):
        self.train_data = train_data.copy()
        self.valid_data = valid_data.copy()
        self.target_col = target_col
        self.use_cache = use_cache
        mc = model_config or {}
        ensemble_cfg = mc.get('ensemble', {})
        single_cfg = mc.get('single', {})
        self.ensemble_params = {'n_estimators': ensemble_cfg.get('n_estimators', 200), 'max_depth': ensemble_cfg.get('max_depth', 5), 'num_leaves': ensemble_cfg.get('num_leaves', 24), 'learning_rate': ensemble_cfg.get('learning_rate', 0.03), 'colsample_bytree': ensemble_cfg.get('colsample_bytree', 0.7), 'subsample': ensemble_cfg.get('subsample', 0.7), 'reg_alpha': ensemble_cfg.get('reg_alpha', 0.1), 'reg_lambda': ensemble_cfg.get('reg_lambda', 1.0), 'min_child_samples': ensemble_cfg.get('min_child_samples', 30)}
        self.early_stopping_rounds = ensemble_cfg.get('early_stopping_rounds', 20)
        self._initial_ensemble_params = dict(self.ensemble_params)
        self._initial_early_stopping_rounds = self.early_stopping_rounds
        self.single_params = {'n_estimators': single_cfg.get('n_estimators', 50), 'max_depth': single_cfg.get('max_depth', 3), 'learning_rate': single_cfg.get('learning_rate', 0.1)}
        fv = feature_validation_config or {}
        self.max_nan_ratio = fv.get('max_nan_ratio', 0.2)
        self.min_valid_values = fv.get('min_valid_values', 10)
        self.min_variance = fv.get('min_variance', 1e-10)
        self._cache = {} if use_cache else None
        self._feature_cache_train: Dict[str, np.ndarray] = {}
        self._feature_cache_valid: Dict[str, np.ndarray] = {}
        self.y_train = self.train_data[target_col].values
        self.y_valid = self.valid_data[target_col].values
        self.base_features = list(base_features or get_base_feature_columns(train_data.columns))
        print(f'Calculator initialization:')
        print(f'  Training set: {len(self.train_data)} samples, fraud rate={self.y_train.mean():.4f}')
        print(f'  Validation set: {len(self.valid_data)} samples, fraud rate={self.y_valid.mean():.4f}')
        print(f'  Base features: {len(self.base_features)} items')

    def update_ensemble_params(self, new_params: Dict[str, Any]) -> Dict[str, Any]:
        old_params = dict(self.ensemble_params)
        for k, v in new_params.items():
            if k in self.ensemble_params:
                self.ensemble_params[k] = v
        return old_params

    def _sanitize_feature(self, values: np.ndarray) -> np.ndarray:
        return np.where(np.isinf(values), np.nan, values)

    def calc_single_auc(self, expr: Expression, use_valid: bool=True) -> float:
        cache_key = (str(expr), use_valid)
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        try:
            if use_valid:
                feature_values = self.get_cached_valid_feature(expr)
                y = self.y_valid
            else:
                feature_values = self.get_cached_train_feature(expr)
                y = self.y_train
            if feature_values is None:
                return 0.0
            feature_df = pd.DataFrame({'feature': feature_values})
            model = lgb.LGBMClassifier(objective='binary', n_estimators=self.single_params['n_estimators'], max_depth=self.single_params['max_depth'], learning_rate=self.single_params['learning_rate'], random_state=42, verbose=-1)
            if use_valid:
                train_feature = self.get_cached_train_feature(expr)
                if train_feature is None:
                    return 0.0
                train_df = pd.DataFrame({'feature': train_feature})
                model.fit(train_df, self.y_train)
                y_pred = model.predict_proba(feature_df)[:, 1]
            else:
                model.fit(feature_df, y)
                y_pred = model.predict_proba(feature_df)[:, 1]
            auc = roc_auc_score(y, y_pred)
            if self.use_cache:
                self._cache[cache_key] = auc
            return auc
        except Exception as e:
            print(f'AUC calculation failed [{expr}]: {e}')
            return 0.0

    def calc_raw_auc(self, expr: Expression, use_valid: bool=True) -> float:
        cache_key = ('raw', str(expr), use_valid)
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        try:
            if use_valid:
                feature_values = self.get_cached_valid_feature(expr)
                y = self.y_valid
            else:
                feature_values = self.get_cached_train_feature(expr)
                y = self.y_train
            if feature_values is None:
                return 0.5
            nan_mask = np.isnan(feature_values)
            if nan_mask.all():
                return 0.5
            clean = feature_values.copy()
            if nan_mask.any():
                clean[nan_mask] = np.nanmedian(feature_values)
            auc = roc_auc_score(y, clean)
            auc = max(auc, 1.0 - auc)
            if self.use_cache:
                self._cache[cache_key] = auc
            return auc
        except Exception:
            return 0.5

    def calc_single_auc_fast(self, expr: Expression, use_valid: bool=True) -> float:
        from sklearn.linear_model import LogisticRegression
        cache_key = ('fast', str(expr), use_valid)
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        try:
            train_feature = self._sanitize_feature(expr.evaluate(self.train_data))
            if not self._is_valid_feature(train_feature):
                return 0.0
            X_train = train_feature.reshape(-1, 1)
            X_train = np.nan_to_num(X_train, nan=0.0)
            model = LogisticRegression(max_iter=100, solver='lbfgs')
            model.fit(X_train, self.y_train)
            if use_valid:
                valid_feature = self._sanitize_feature(expr.evaluate(self.valid_data))
                X_eval = np.nan_to_num(valid_feature.reshape(-1, 1), nan=0.0)
                y_eval = self.y_valid
            else:
                X_eval = X_train
                y_eval = self.y_train
            y_pred = model.predict_proba(X_eval)[:, 1]
            auc = roc_auc_score(y_eval, y_pred)
            if self.use_cache:
                self._cache[cache_key] = auc
            return auc
        except Exception:
            return 0.0

    def get_cached_train_feature(self, expr: Expression) -> Optional[np.ndarray]:
        key = str(expr)
        if key in self._feature_cache_train:
            return self._feature_cache_train[key]
        try:
            values = self._sanitize_feature(expr.evaluate(self.train_data))
            if self._is_valid_feature(values):
                self._feature_cache_train[key] = values
                return values
            return None
        except Exception:
            return None

    def get_cached_valid_feature(self, expr: Expression) -> Optional[np.ndarray]:
        key = str(expr)
        if key in self._feature_cache_valid:
            return self._feature_cache_valid[key]
        try:
            if _has_rolling_ops(expr):
                combined = pd.concat([self.train_data, self.valid_data], ignore_index=True)
                all_values = self._sanitize_feature(expr.evaluate(combined))
                values = all_values[len(self.train_data):]
            else:
                values = self._sanitize_feature(expr.evaluate(self.valid_data))
            if self._is_valid_feature(values):
                self._feature_cache_valid[key] = values
                return values
            return None
        except Exception:
            return None

    def remove_feature_cache(self, expr_str: str):
        self._feature_cache_train.pop(expr_str, None)
        self._feature_cache_valid.pop(expr_str, None)

    def calc_mutual_corr(self, expr1: Expression, expr2: Expression) -> float:
        try:
            values1 = self.get_cached_train_feature(expr1)
            values2 = self.get_cached_train_feature(expr2)
            if values1 is None or values2 is None:
                return 0.0
            corr = np.corrcoef(values1, values2)[0, 1]
            return abs(corr) if not np.isnan(corr) else 0.0
        except Exception as e:
            print(f'Correlation calculation failed: {e}')
            return 0.0

    def calc_pool_auc(self, exprs: List[Expression], weights: Optional[List[float]]=None, use_valid: bool=True) -> float:
        if len(exprs) == 0:
            return 0.0
        try:
            if use_valid:
                data = self.valid_data
                y = self.y_valid
                train_data = self.train_data
                y_train = self.y_train
            else:
                data = self.train_data
                y = self.y_train
                train_data = None
                y_train = None
            generated_features = []
            valid_exprs = []
            for expr in exprs:
                try:
                    if use_valid:
                        feature_values = self.get_cached_valid_feature(expr)
                    else:
                        feature_values = self.get_cached_train_feature(expr)
                    if feature_values is not None:
                        generated_features.append(feature_values)
                        valid_exprs.append(expr)
                except:
                    continue
            if len(generated_features) == 0:
                return 0.0
            X = data[self.base_features].values
            X_generated = np.column_stack(generated_features)
            X_combined = np.hstack([X, X_generated])
            model = lgb.LGBMClassifier(objective='binary', n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
            if use_valid and train_data is not None:
                train_generated = []
                for expr in valid_exprs:
                    try:
                        train_feat = self.get_cached_train_feature(expr)
                        if train_feat is not None:
                            train_generated.append(train_feat)
                    except:
                        pass
                if len(train_generated) != len(generated_features):
                    return self.calc_single_auc(exprs[0], use_valid=True) if len(exprs) > 0 else 0.0
                X_train = train_data[self.base_features].values
                X_train_generated = np.column_stack(train_generated)
                X_train_combined = np.hstack([X_train, X_train_generated])
                model.fit(X_train_combined, y_train)
                y_pred = model.predict_proba(X_combined)[:, 1]
            else:
                model.fit(X_combined, y)
                y_pred = model.predict_proba(X_combined)[:, 1]
            auc = roc_auc_score(y, y_pred)
            return auc
        except Exception as e:
            print(f'Combined AUC calculation failed: {e}')
            return 0.0

    def calc_pool_auc_with_lgbm(self, exprs: List[Expression], selected_base_features: Optional[List[str]]=None, return_model: bool=False, use_cv: bool=False) -> Tuple[float, Optional[lgb.LGBMClassifier]]:
        if len(exprs) == 0:
            return (0.0, None)
        try:
            base_features_to_use = selected_base_features if selected_base_features is not None else self.base_features
            train_features = []
            valid_features = []
            valid_exprs = []
            for expr in exprs:
                try:
                    train_feat = self.get_cached_train_feature(expr)
                    valid_feat = self.get_cached_valid_feature(expr)
                    if train_feat is not None and valid_feat is not None:
                        train_features.append(train_feat)
                        valid_features.append(valid_feat)
                        valid_exprs.append(expr)
                except:
                    continue
            if len(train_features) == 0:
                return (0.0, None)
            X_train = self.train_data[base_features_to_use].values
            X_train_generated = np.column_stack(train_features)
            X_train_combined = np.hstack([X_train, X_train_generated])
            X_valid = self.valid_data[base_features_to_use].values
            X_valid_generated = np.column_stack(valid_features)
            X_valid_combined = np.hstack([X_valid, X_valid_generated])
            lgbm_params = dict(objective='binary', n_estimators=self.ensemble_params['n_estimators'], max_depth=self.ensemble_params['max_depth'], num_leaves=self.ensemble_params['num_leaves'], learning_rate=self.ensemble_params['learning_rate'], colsample_bytree=self.ensemble_params['colsample_bytree'], subsample=self.ensemble_params['subsample'], reg_alpha=self.ensemble_params['reg_alpha'], reg_lambda=self.ensemble_params['reg_lambda'], min_child_samples=self.ensemble_params['min_child_samples'], random_state=42, verbose=-1)
            if use_cv:
                X_all = np.vstack([X_train_combined, X_valid_combined])
                y_all = np.concatenate([self.y_train, self.y_valid])
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_aucs = []
                for train_idx, val_idx in skf.split(X_all, y_all):
                    fold_model = lgb.LGBMClassifier(**lgbm_params)
                    fold_model.fit(X_all[train_idx], y_all[train_idx], eval_set=[(X_all[val_idx], y_all[val_idx])], eval_metric='auc', callbacks=[lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False)])
                    y_pred_fold = fold_model.predict_proba(X_all[val_idx])[:, 1]
                    cv_aucs.append(roc_auc_score(y_all[val_idx], y_pred_fold))
                cv_auc = float(np.mean(cv_aucs))
                model = lgb.LGBMClassifier(**lgbm_params)
                model.fit(X_train_combined, self.y_train, eval_set=[(X_valid_combined, self.y_valid)], eval_metric='auc', callbacks=[lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False)])
                if return_model:
                    return (cv_auc, model)
                else:
                    return (cv_auc, None)
            else:
                model = lgb.LGBMClassifier(**lgbm_params)
                model.fit(X_train_combined, self.y_train, eval_set=[(X_valid_combined, self.y_valid)], eval_metric='auc', callbacks=[lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False)])
                y_pred = model.predict_proba(X_valid_combined)[:, 1]
                auc = roc_auc_score(self.y_valid, y_pred)
                if return_model:
                    return (auc, model)
                else:
                    return (auc, None)
        except Exception as e:
            print(f'LightGBM training failed: {e}')
            return (0.0, None)

    def calc_comparison_auc(self, exprs: List[Expression], selected_base_features: Optional[List[str]]=None) -> Dict[str, Any]:
        base_features_to_use = selected_base_features if selected_base_features is not None else self.base_features
        result = {'only_base': 0.0, 'only_generated': 0.0, 'combined': 0.0, 'n_base_features': len(base_features_to_use), 'n_generated_features': 0}
        try:
            lgbm_params = dict(objective='binary', n_estimators=self.ensemble_params['n_estimators'], max_depth=self.ensemble_params['max_depth'], num_leaves=self.ensemble_params['num_leaves'], learning_rate=self.ensemble_params['learning_rate'], colsample_bytree=self.ensemble_params['colsample_bytree'], subsample=self.ensemble_params['subsample'], reg_alpha=self.ensemble_params['reg_alpha'], reg_lambda=self.ensemble_params['reg_lambda'], min_child_samples=self.ensemble_params['min_child_samples'], random_state=42, verbose=-1)
            es_callback = [lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False)]
            if len(base_features_to_use) > 0:
                X_train_base = self.train_data[base_features_to_use].values
                X_valid_base = self.valid_data[base_features_to_use].values
                model_base = lgb.LGBMClassifier(**lgbm_params)
                model_base.fit(X_train_base, self.y_train, eval_set=[(X_valid_base, self.y_valid)], eval_metric='auc', callbacks=es_callback)
                y_pred_base = model_base.predict_proba(X_valid_base)[:, 1]
                result['only_base'] = roc_auc_score(self.y_valid, y_pred_base)
            train_features = []
            valid_features = []
            for expr in exprs:
                try:
                    train_feat = self.get_cached_train_feature(expr)
                    valid_feat = self.get_cached_valid_feature(expr)
                    if train_feat is not None and valid_feat is not None:
                        train_features.append(train_feat)
                        valid_features.append(valid_feat)
                except:
                    continue
            result['n_generated_features'] = len(train_features)
            if len(train_features) > 0:
                X_train_gen = np.column_stack(train_features)
                X_valid_gen = np.column_stack(valid_features)
                model_gen = lgb.LGBMClassifier(**lgbm_params)
                model_gen.fit(X_train_gen, self.y_train, eval_set=[(X_valid_gen, self.y_valid)], eval_metric='auc', callbacks=es_callback)
                y_pred_gen = model_gen.predict_proba(X_valid_gen)[:, 1]
                result['only_generated'] = roc_auc_score(self.y_valid, y_pred_gen)
                if len(base_features_to_use) > 0:
                    X_train_combined = np.hstack([X_train_base, X_train_gen])
                    X_valid_combined = np.hstack([X_valid_base, X_valid_gen])
                    model_combined = lgb.LGBMClassifier(**lgbm_params)
                    model_combined.fit(X_train_combined, self.y_train, eval_set=[(X_valid_combined, self.y_valid)], eval_metric='auc', callbacks=es_callback)
                    y_pred_combined = model_combined.predict_proba(X_valid_combined)[:, 1]
                    result['combined'] = roc_auc_score(self.y_valid, y_pred_combined)
                else:
                    result['combined'] = result['only_generated']
            return result
        except Exception as e:
            print(f'Comparison experiment failed: {e}')
            return result

    def _is_valid_feature(self, values: np.ndarray) -> bool:
        if values is None or len(values) == 0:
            return False
        values = np.where(np.isinf(values), np.nan, values)
        nan_ratio = np.isnan(values).sum() / len(values)
        if nan_ratio > self.max_nan_ratio:
            return False
        valid_values = values[~np.isnan(values)]
        if len(valid_values) < self.min_valid_values:
            return False
        if np.var(valid_values) < self.min_variance:
            return False
        return True

    def clear_cache(self) -> None:
        if self._cache is not None:
            self._cache.clear()

    def compute_dataset_profile(self) -> Dict[str, Any]:
        if hasattr(self, '_dataset_profile_cache') and self._dataset_profile_cache:
            return self._dataset_profile_cache
        from scipy import stats as sp_stats
        profile = {'n_train': len(self.train_data), 'n_valid': len(self.valid_data), 'fraud_rate_train': float(self.y_train.mean()), 'fraud_rate_valid': float(self.y_valid.mean()), 'variable_stats': {}, 'significant_vars': [], 'high_corr_pairs': []}
        train = self.train_data
        y = self.y_train
        fraud_mask = y == 1
        normal_mask = y == 0
        numeric_vars = [v for v in self.base_features if _is_numeric_feature_series(train[v])]
        for var in self.base_features:
            raw_series = train[var]
            numeric_series = pd.to_numeric(raw_series, errors='coerce')
            numeric_ratio = float(numeric_series.notna().mean()) if len(raw_series) > 0 else 0.0
            vals = numeric_series.values.astype(float)
            valid_vals = vals[~np.isnan(vals)]
            vs = {'missing_rate': float(raw_series.isna().mean()), 'numeric_ratio': numeric_ratio}
            if len(valid_vals) >= 10:
                vs['mean'] = float(np.mean(valid_vals))
                vs['std'] = float(np.std(valid_vals))
                vs['skew'] = float(sp_stats.skew(valid_vals, nan_policy='omit'))
                vs['kurtosis'] = float(sp_stats.kurtosis(valid_vals, nan_policy='omit'))
                vs['q25'] = float(np.percentile(valid_vals, 25))
                vs['q75'] = float(np.percentile(valid_vals, 75))
                fraud_vals = vals[fraud_mask]
                normal_vals = vals[normal_mask]
                fraud_valid = fraud_vals[~np.isnan(fraud_vals)]
                normal_valid = normal_vals[~np.isnan(normal_vals)]
                if len(fraud_valid) > 5 and len(normal_valid) > 5:
                    try:
                        stat, pval = sp_stats.mannwhitneyu(fraud_valid, normal_valid, alternative='two-sided')
                        vs['mannwhitney_p'] = float(pval)
                        vs['fraud_mean'] = float(np.mean(fraud_valid))
                        vs['normal_mean'] = float(np.mean(normal_valid))
                        pooled_std = float(np.std(valid_vals))
                        if pooled_std > 1e-10:
                            vs['cohens_d'] = float((np.mean(fraud_valid) - np.mean(normal_valid)) / pooled_std)
                        else:
                            vs['cohens_d'] = 0.0
                        if pval < 0.01:
                            profile['significant_vars'].append({'var': var, 'p': pval, 'd': vs['cohens_d']})
                    except Exception:
                        pass
            else:
                valid_raw = raw_series.dropna()
                vs['n_unique'] = int(valid_raw.nunique()) if len(valid_raw) > 0 else 0
            profile['variable_stats'][var] = vs
        if len(numeric_vars) > 1:
            try:
                corr_matrix = train[numeric_vars].corr(method='spearman')
                for i in range(len(numeric_vars)):
                    for j in range(i + 1, len(numeric_vars)):
                        c = abs(corr_matrix.iloc[i, j])
                        if c > 0.7:
                            profile['high_corr_pairs'].append({'var1': numeric_vars[i], 'var2': numeric_vars[j], 'corr': float(corr_matrix.iloc[i, j])})
                profile['high_corr_pairs'].sort(key=lambda x: abs(x['corr']), reverse=True)
            except Exception:
                pass
        profile['significant_vars'].sort(key=lambda x: abs(x['d']), reverse=True)
        self._dataset_profile_cache = profile
        return profile

    def compute_shap_interactions(self, model, n_top: int=20) -> List[Dict]:
        try:
            import shap
            sample_size = min(500, len(self.valid_data))
            X_sample = self.valid_data[self.base_features].values[:sample_size]
            explainer = shap.TreeExplainer(model)
            shap_interaction = explainer.shap_interaction_values(X_sample)
            if isinstance(shap_interaction, list):
                shap_interaction = shap_interaction[1]
            n_features = shap_interaction.shape[1]
            interactions = []
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    strength = float(np.abs(shap_interaction[:, i, j]).mean())
                    if strength > 1e-06:
                        interactions.append({'var1': self.base_features[i], 'var2': self.base_features[j], 'strength': strength})
            interactions.sort(key=lambda x: x['strength'], reverse=True)
            return interactions[:n_top]
        except Exception:
            return []

    def compute_temporal_stability(self, expr: Expression) -> Dict[str, float]:
        result = {'auc_early': 0.5, 'auc_late': 0.5, 'auc_gap': 0.0, 'stable': True}
        try:
            if 'year' not in self.train_data.columns:
                return result
            years = self.train_data['year'].values
            median_year = np.median(years)
            early_mask = years <= median_year
            late_mask = years > median_year
            feature_values = self.get_cached_train_feature(expr)
            if feature_values is None:
                return result
            for mask, key in [(early_mask, 'auc_early'), (late_mask, 'auc_late')]:
                y_sub = self.y_train[mask]
                f_sub = feature_values[mask]
                valid = ~np.isnan(f_sub)
                if valid.sum() < 20 or len(np.unique(y_sub[valid])) < 2:
                    continue
                try:
                    auc = float(roc_auc_score(y_sub[valid], f_sub[valid]))
                    if auc < 0.5:
                        auc = 1.0 - auc
                    result[key] = auc
                except Exception:
                    pass
            result['auc_gap'] = abs(result['auc_early'] - result['auc_late'])
            result['stable'] = result['auc_gap'] <= 0.05
        except Exception:
            pass
        return result

    def calc_pool_auc_temporal_cv(self, exprs: List[Expression], selected_base_features: Optional[List[str]]=None, temporal_folds: Optional[List[Dict[str, Any]]]=None, robust_std_penalty: float=0.0, model_params: Optional[Dict[str, Any]]=None, return_model: bool=False):
        if len(exprs) == 0:
            if return_model:
                return (0.0, 0.0, {}, None)
            return (0.0, 0.0, {})
        params_source = model_params if model_params is not None else self.ensemble_params
        if temporal_folds is None:
            temporal_folds = [{'train_years': list(range(2010, 2018)), 'valid_years': [2018]}, {'train_years': list(range(2010, 2019)), 'valid_years': [2019]}, {'train_years': list(range(2010, 2020)), 'valid_years': [2020]}, {'train_years': list(range(2010, 2021)), 'valid_years': [2021]}]
        base_features_to_use = selected_base_features if selected_base_features is not None else self.base_features
        all_data = pd.concat([self.train_data, self.valid_data], ignore_index=True)
        all_y = np.concatenate([self.y_train, self.y_valid])
        if 'year' not in all_data.columns:
            auc, model = self.calc_pool_auc_with_lgbm(exprs, selected_base_features=selected_base_features, return_model=return_model)
            result = (auc, 0.0, {'fallback': auc})
            if return_model:
                return (auc, 0.0, {'fallback': auc}, model)
            return result
        years = all_data['year'].values
        fold_aucs = {}
        fold_models = []
        for fold_idx, fold_cfg in enumerate(temporal_folds):
            train_years = set(fold_cfg['train_years'])
            valid_years = set(fold_cfg['valid_years'])
            train_mask = np.array([y in train_years for y in years])
            valid_mask = np.array([y in valid_years for y in years])
            fold_train_data = all_data[train_mask]
            fold_valid_data = all_data[valid_mask]
            fold_y_train = all_y[train_mask]
            fold_y_valid = all_y[valid_mask]
            if len(fold_y_train) < 500 or len(fold_y_valid) < 200:
                continue
            if len(np.unique(fold_y_train)) < 2 or len(np.unique(fold_y_valid)) < 2:
                continue
            try:
                train_gen_features = []
                valid_gen_features = []
                for expr in exprs:
                    try:
                        train_vals = self._sanitize_feature(expr.evaluate(fold_train_data))
                        valid_vals = self._sanitize_feature(expr.evaluate(fold_valid_data))
                        if self._is_valid_feature(train_vals) and self._is_valid_feature(valid_vals):
                            train_gen_features.append(train_vals)
                            valid_gen_features.append(valid_vals)
                    except Exception:
                        continue
                if len(train_gen_features) == 0:
                    continue
                X_train = fold_train_data[base_features_to_use].values
                X_train_gen = np.column_stack(train_gen_features)
                X_train_combined = np.hstack([X_train, X_train_gen])
                X_valid = fold_valid_data[base_features_to_use].values
                X_valid_gen = np.column_stack(valid_gen_features)
                X_valid_combined = np.hstack([X_valid, X_valid_gen])
                lgbm_params = dict(objective='binary', n_estimators=params_source['n_estimators'], max_depth=params_source['max_depth'], num_leaves=params_source['num_leaves'], learning_rate=params_source['learning_rate'], colsample_bytree=params_source['colsample_bytree'], subsample=params_source['subsample'], reg_alpha=params_source['reg_alpha'], reg_lambda=params_source['reg_lambda'], min_child_samples=params_source['min_child_samples'], random_state=42, verbose=-1)
                model = lgb.LGBMClassifier(**lgbm_params)
                model.fit(X_train_combined, fold_y_train, eval_set=[(X_valid_combined, fold_y_valid)], eval_metric='auc', callbacks=[lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False)])
                y_pred = model.predict_proba(X_valid_combined)[:, 1]
                auc = roc_auc_score(fold_y_valid, y_pred)
                max_train_year = max(train_years)
                fold_key = f'fold{fold_idx + 1}_{min(train_years)}_{max_train_year}'
                fold_aucs[fold_key] = auc
                fold_models.append(model)
            except Exception as e:
                print(f'Temporal CV fold {fold_idx + 1} failed: {e}')
                continue
        if len(fold_aucs) == 0:
            auc, model = self.calc_pool_auc_with_lgbm(exprs, selected_base_features=selected_base_features, return_model=return_model)
            if return_model:
                return (auc, 0.0, {'fallback': auc}, model)
            return (auc, 0.0, {'fallback': auc})
        auc_values = list(fold_aucs.values())
        mean_auc = float(np.mean(auc_values))
        std_auc = float(np.std(auc_values))
        score_auc = float(mean_auc - float(robust_std_penalty) * std_auc)
        if return_model and fold_models:
            return (score_auc, std_auc, fold_aucs, fold_models[-1])
        if return_model:
            return (score_auc, std_auc, fold_aucs, None)
        return (score_auc, std_auc, fold_aucs)
