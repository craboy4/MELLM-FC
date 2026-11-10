"""Compute factor statistics, dependence, and incremental predictive contribution."""
import logging
from typing import Dict, List, Optional
import numpy as np
from fraud_alpha.calculator import FraudCalculator
from fraud_alpha.alpha_pool import AlphaPool
from fraud_alpha.expression import Expression, Feature
from fraud_alpha.operator_catalog import build_operator_regex, get_documented_operator_names, normalize_operator_name
logger = logging.getLogger(__name__)

class FactorAnalyzer:

    def __init__(self, calculator: FraudCalculator, pool: AlphaPool):
        self.calculator = calculator
        self.pool = pool
        self._stats_cache: Dict[str, Dict] = {}

    def compute_single_stats(self, expr: Expression, expr_str: str) -> Dict:
        default_stats = {'expr_str': expr_str, 'auc': 0.0, 'weight': 0.0, 'max_corr': 0.0, 'mean': 0.0, 'std': 0.0, 'missing_rate': 1.0, 'zero_rate': 0.0, 'effect_size': 0.0, 'valid': False}
        try:
            feature_values = self.calculator.get_cached_train_feature(expr)
            if feature_values is None:
                logger.warning(f'Empty factor values; marked invalid: {expr_str}')
                return default_stats
            if np.all(np.isnan(feature_values)):
                logger.warning(f'All factor values are NaN; marked invalid: {expr_str}')
                return default_stats
            total = len(feature_values)
            nan_mask = np.isnan(feature_values)
            missing_rate = float(np.sum(nan_mask) / total) if total > 0 else 1.0
            valid_values = feature_values[~nan_mask]
            if len(valid_values) == 0:
                return default_stats
            mean_val = float(np.mean(valid_values))
            std_val = float(np.std(valid_values))
            zero_rate = float(np.sum(valid_values == 0) / total) if total > 0 else 0.0
            from scipy import stats as sp_stats
            skewness = float(sp_stats.skew(valid_values, nan_policy='omit')) if len(valid_values) > 2 else 0.0
            kurtosis = float(sp_stats.kurtosis(valid_values, nan_policy='omit')) if len(valid_values) > 2 else 0.0
            q25 = float(np.percentile(valid_values, 25))
            q75 = float(np.percentile(valid_values, 75))
            auc = self.calculator.calc_single_auc(expr, use_valid=True)
            weight = 0.0
            pool_exprs_str = [str(e) for e in self.pool.exprs]
            if expr_str in pool_exprs_str:
                idx = pool_exprs_str.index(expr_str)
                if idx < len(self.pool.weights):
                    weight = float(self.pool.weights[idx])
            max_corr = self._compute_max_correlation(expr, expr_str)
            effect_size = self._compute_effect_size(feature_values)
            ttest_p = 1.0
            fraud_mean = 0.0
            fraud_median = 0.0
            non_fraud_mean = 0.0
            non_fraud_median = 0.0
            try:
                y = self.calculator.y_train
                fraud_mask = (y == 1) & ~nan_mask
                non_fraud_mask = (y == 0) & ~nan_mask
                fraud_vals = feature_values[fraud_mask]
                non_fraud_vals = feature_values[non_fraud_mask]
                if len(fraud_vals) > 5 and len(non_fraud_vals) > 5:
                    fraud_mean = float(np.mean(fraud_vals))
                    fraud_median = float(np.median(fraud_vals))
                    non_fraud_mean = float(np.mean(non_fraud_vals))
                    non_fraud_median = float(np.median(non_fraud_vals))
                    stat, pval = sp_stats.mannwhitneyu(fraud_vals, non_fraud_vals, alternative='two-sided')
                    ttest_p = float(pval)
            except Exception:
                pass
            min_val = float(np.min(valid_values))
            max_val = float(np.max(valid_values))
            stats = {'expr_str': expr_str, 'auc': float(auc) if np.isfinite(auc) else 0.0, 'weight': float(weight) if np.isfinite(weight) else 0.0, 'max_corr': float(max_corr) if np.isfinite(max_corr) else 0.0, 'mean': float(mean_val) if np.isfinite(mean_val) else 0.0, 'std': float(std_val) if np.isfinite(std_val) else 0.0, 'missing_rate': float(missing_rate) if np.isfinite(missing_rate) else 1.0, 'zero_rate': float(zero_rate) if np.isfinite(zero_rate) else 0.0, 'effect_size': float(effect_size) if np.isfinite(effect_size) else 0.0, 'skewness': float(skewness) if np.isfinite(skewness) else 0.0, 'kurtosis': float(kurtosis) if np.isfinite(kurtosis) else 0.0, 'q25': float(q25) if np.isfinite(q25) else 0.0, 'q75': float(q75) if np.isfinite(q75) else 0.0, 'ttest_p': float(ttest_p) if np.isfinite(ttest_p) else 1.0, 'fraud_mean': float(fraud_mean) if np.isfinite(fraud_mean) else 0.0, 'fraud_median': float(fraud_median) if np.isfinite(fraud_median) else 0.0, 'non_fraud_mean': float(non_fraud_mean) if np.isfinite(non_fraud_mean) else 0.0, 'non_fraud_median': float(non_fraud_median) if np.isfinite(non_fraud_median) else 0.0, 'min_val': float(min_val) if np.isfinite(min_val) else 0.0, 'max_val': float(max_val) if np.isfinite(max_val) else 0.0, 'valid': True}
            return stats
        except Exception as e:
            logger.error(f'Factor statistics calculation failed [{expr_str}]: {e}')
            return default_stats

    def _compute_max_correlation(self, expr: Expression, expr_str: str) -> float:
        max_corr = 0.0
        for other_expr in self.pool.exprs:
            other_str = str(other_expr)
            if other_str == expr_str:
                continue
            corr = self.calculator.calc_mutual_corr(expr, other_expr)
            if corr > max_corr:
                max_corr = corr
        return max_corr

    def _compute_effect_size(self, feature_values: np.ndarray) -> float:
        try:
            y = self.calculator.y_train
            if len(feature_values) != len(y):
                return 0.0
            nan_mask = np.isnan(feature_values)
            fraud_mask = (y == 1) & ~nan_mask
            non_fraud_mask = (y == 0) & ~nan_mask
            fraud_values = feature_values[fraud_mask]
            non_fraud_values = feature_values[non_fraud_mask]
            if len(fraud_values) == 0 or len(non_fraud_values) == 0:
                return 0.0
            fraud_mean = np.mean(fraud_values)
            non_fraud_mean = np.mean(non_fraud_values)
            fraud_std = np.std(fraud_values)
            non_fraud_std = np.std(non_fraud_values)
            pooled_std = np.sqrt((fraud_std ** 2 + non_fraud_std ** 2) / 2)
            if pooled_std < 1e-10:
                return 0.0
            effect = (fraud_mean - non_fraud_mean) / pooled_std
            return float(effect) if np.isfinite(effect) else 0.0
        except Exception as e:
            logger.warning(f'Effect size calculation failed: {e}')
            return 0.0

    def compute_marginal_contributions(self) -> Dict[str, float]:
        contributions = {}
        n = len(self.pool.exprs)
        if n == 0:
            return contributions
        if n == 1:
            expr_str = str(self.pool.exprs[0])
            auc = self.pool.aucs[0] if self.pool.aucs else 0.0
            contributions[expr_str] = float(auc)
            return contributions
        try:
            full_auc = self.calculator.calc_pool_auc(self.pool.exprs, use_valid=True)
        except Exception as e:
            logger.error(f'Full-pool AUC calculation failed: {e}')
            return {str(expr): 0.0 for expr in self.pool.exprs}
        for i, expr in enumerate(self.pool.exprs):
            expr_str = str(expr)
            try:
                remaining_exprs = self.pool.exprs[:i] + self.pool.exprs[i + 1:]
                leave_out_auc = self.calculator.calc_pool_auc(remaining_exprs, use_valid=True)
                delta = full_auc - leave_out_auc
                contributions[expr_str] = float(delta) if np.isfinite(delta) else 0.0
            except Exception as e:
                logger.warning(f'Marginal contribution calculation failed [{expr_str}]: {e}; setting to 0.0')
                contributions[expr_str] = 0.0
        return contributions

    def compute_all_stats(self) -> List[Dict]:
        if len(self.pool.exprs) == 0:
            return []
        marginal_contributions = self.compute_marginal_contributions()
        all_stats = []
        for i, expr in enumerate(self.pool.exprs):
            expr_str = str(expr)
            if expr_str in self._stats_cache:
                stats = self._stats_cache[expr_str].copy()
            else:
                stats = self.compute_single_stats(expr, expr_str)
                self._stats_cache[expr_str] = stats.copy()
            stats['index'] = i
            stats['marginal_contribution'] = marginal_contributions.get(expr_str, 0.0)
            stats['pattern_category'] = self._classify_pattern(expr_str)
            all_stats.append(stats)
        return all_stats

    def _classify_pattern(self, expr_str: str) -> str:
        avail = set(Feature.get_available_features())
        pattern_vars = {'profitability': [v for v in ['N20', 'N21', 'N24', 'N25', 'N28'] if v in avail], 'balance_sheet': [v for v in ['N3', 'N4', 'N5', 'N7', 'N10', 'N12', 'N26'] if v in avail], 'financing': [v for v in ['N11', 'N15', 'N16'] if v in avail], 'governance': [v for v in ['C1', 'C3', 'C4', 'N27', 'N31', 'N34', 'N35', 'N36'] if v in avail]}
        temporal_ops = ['Ref(', 'Delta(', 'Mean(', 'Std(', 'Sum(', 'EMA(', 'WMA(']
        has_temporal = any((op in expr_str for op in temporal_ops))
        best_pattern = 'other'
        best_count = 0
        for pattern, vars_list in pattern_vars.items():
            count = sum((1 for v in vars_list if v in expr_str))
            if count > best_count:
                best_count = count
                best_pattern = pattern
        if has_temporal and best_count == 0:
            return 'temporal'
        return best_pattern if best_count > 0 else 'temporal' if has_temporal else 'other'

    def invalidate_cache(self, expr_str: str=None):
        if expr_str is None:
            self._stats_cache.clear()
            logger.debug('Cleared all factor statistics caches')
        elif expr_str in self._stats_cache:
            del self._stats_cache[expr_str]
            logger.debug(f'Cleared factor statistics cache: {expr_str}')

    def compute_pool_correlation_matrix(self) -> List[Dict]:
        pairs = []
        n = len(self.pool.exprs)
        if n < 2:
            return pairs
        for i in range(n):
            for j in range(i + 1, n):
                corr = self.calculator.calc_mutual_corr(self.pool.exprs[i], self.pool.exprs[j])
                if corr > 0.3:
                    pairs.append({'expr1': str(self.pool.exprs[i]), 'expr2': str(self.pool.exprs[j]), 'corr': float(corr)})
        pairs.sort(key=lambda x: x['corr'], reverse=True)
        return pairs

    def compute_success_pattern_analysis(self) -> Dict:
        import re as _re
        op_pattern = _re.compile(build_operator_regex(require_call=True, include_legacy_aliases=True))
        op_counter: Dict[str, int] = {}
        var_counter: Dict[str, int] = {}
        depths = []
        aucs = []
        for i, expr in enumerate(self.pool.exprs):
            expr_str = str(expr)
            for m in op_pattern.findall(expr_str):
                op_name = normalize_operator_name(m)
                op_counter[op_name] = op_counter.get(op_name, 0) + 1
            for v in _re.findall('\\b(N\\d+|C\\d)\\b', expr_str):
                var_counter[v] = var_counter.get(v, 0) + 1
            depth = 0
            max_d = 0
            for ch in expr_str:
                if ch == '(':
                    depth += 1
                    max_d = max(max_d, depth)
                elif ch == ')':
                    depth -= 1
            depths.append(max_d)
            if i < len(self.pool.aucs):
                aucs.append(self.pool.aucs[i])
        return {'top_ops': sorted(op_counter.items(), key=lambda x: x[1], reverse=True)[:10], 'top_vars': sorted(var_counter.items(), key=lambda x: x[1], reverse=True)[:15], 'avg_depth': float(np.mean(depths)) if depths else 0.0, 'avg_auc': float(np.mean(aucs)) if aucs else 0.0}

    def compute_coverage_stats(self) -> Dict:
        import re as _re
        all_vars = Feature.get_available_features()
        all_ops = get_documented_operator_names()
        op_pattern = _re.compile(build_operator_regex(require_call=True, include_legacy_aliases=True))
        var_counter: Dict[str, int] = {}
        op_counter: Dict[str, int] = {}
        for expr in self.pool.exprs:
            expr_str = str(expr)
            for var in all_vars:
                if _re.search('\\b' + _re.escape(var) + '\\b', expr_str):
                    var_counter[var] = var_counter.get(var, 0) + 1
            for m in op_pattern.findall(expr_str):
                op_name = normalize_operator_name(m)
                op_counter[op_name] = op_counter.get(op_name, 0) + 1
        used_vars = sorted([(v, c) for v, c in var_counter.items()], key=lambda x: x[1], reverse=True)
        unused_vars = [v for v in all_vars if v not in var_counter]
        used_ops = sorted([(o, c) for o, c in op_counter.items()], key=lambda x: x[1], reverse=True)
        unused_ops = [o for o in all_ops if o not in op_counter]
        avail_set = set(all_vars)
        pattern_vars = {'profitability': [v for v in ['N20', 'N21', 'N24', 'N25', 'N28'] if v in avail_set], 'balance_sheet': [v for v in ['N3', 'N4', 'N5', 'N7', 'N10', 'N12', 'N26'] if v in avail_set], 'financing': [v for v in ['N11', 'N15', 'N16'] if v in avail_set], 'governance': [v for v in ['C1', 'C3', 'C4', 'N27', 'N31', 'N34', 'N35', 'N36'] if v in avail_set]}
        pattern_coverage: Dict[str, int] = {}
        for pattern, pvars in pattern_vars.items():
            count = sum((1 for v in pvars if v in var_counter))
            pattern_coverage[pattern] = count
        return {'used_vars': used_vars, 'unused_vars': unused_vars, 'var_coverage_rate': len(used_vars) / len(all_vars) if all_vars else 0.0, 'used_ops': used_ops, 'unused_ops': unused_ops, 'op_coverage_rate': len(used_ops) / len(all_ops) if all_ops else 0.0, 'pattern_coverage': pattern_coverage}

    def compute_correlation_clusters(self, threshold: float=0.6) -> List[List[str]]:
        n = len(self.pool.exprs)
        if n < 2:
            return []
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = (find(a), find(b))
            if ra != rb:
                parent[ra] = rb
        for i in range(n):
            for j in range(i + 1, n):
                corr = self.calculator.calc_mutual_corr(self.pool.exprs[i], self.pool.exprs[j])
                if corr > threshold:
                    union(i, j)
        clusters_map: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            clusters_map.setdefault(root, []).append(i)
        clusters = []
        for indices in clusters_map.values():
            if len(indices) >= 2:
                clusters.append([str(self.pool.exprs[i]) for i in indices])
        return clusters

    def get_factor_stats(self, expr_str: str) -> Optional[Dict]:
        return self._stats_cache.get(expr_str)
