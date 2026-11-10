"""Describe development misclassifications and SHAP associations without causal claims."""
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)
DIAGNOSTICS_SYSTEM_PROMPT = "You are a feature-engineering expert in financial fraud detection.\nYou will receive statistical analyses of validation misclassifications: profiles and SHAP differences for missed fraud (FN) versus detected fraud (TP), a lightweight comparison of false alarms (FP) versus correctly classified normal cases (TN), and transitions from the previous round.\n\nProvide a structured analysis within {max_chars} characters:\n- **Missed-fraud patterns**: common characteristics of persistent_fn samples and variable patterns associated with detection difficulty.\n- **Factor design suggestions**: 2-3 concrete operator/variable combinations to capture these patterns.\n- **Side-effect warnings**: regressions (previously correct, now incorrect) or false-alarm risks associated with this round's factors, and how to avoid them.\n\nOutput only these three points, without expression code or unrelated content."

class SampleDiagnostics:

    def __init__(self, calculator, config: Dict[str, Any]):
        self.calculator = calculator
        self.max_shap_samples = config.get('max_shap_samples', 200)
        self.max_analysis_chars = config.get('max_analysis_chars', 500)
        self._prev_y_true: Optional[np.ndarray] = None
        self._prev_y_pred: Optional[np.ndarray] = None
        self._prev_y_cls: Optional[np.ndarray] = None

    def compute_statistics(self, model, feature_names: List[str], X_valid: np.ndarray, y_valid: np.ndarray, pool_exprs: Optional[List]=None, iteration: int=0) -> Dict[str, Any]:
        X_valid = np.asarray(X_valid)
        y_valid = np.asarray(y_valid)
        try:
            y_prob = model.predict_proba(X_valid)[:, 1]
        except Exception as e:
            logger.warning(f"Sample diagnosis: prediction failed ({e}), skipping this round's diagnosis")
            return self._empty_stats(iteration)
        threshold = self._compute_adaptive_threshold(y_prob, y_valid)
        y_pred_cls = (y_prob >= threshold).astype(int)
        cls = np.zeros(len(y_valid), dtype=int)
        cls[(y_valid == 1) & (y_pred_cls == 1)] = 1
        cls[(y_valid == 0) & (y_pred_cls == 1)] = 2
        cls[(y_valid == 1) & (y_pred_cls == 0)] = 3
        confusion = {'TP': int((cls == 1).sum()), 'TN': int((cls == 0).sum()), 'FP': int((cls == 2).sum()), 'FN': int((cls == 3).sum())}
        transitions = self._compute_transitions(cls)
        fn_vs_tp_diff = self._compute_fn_tp_diff(X_valid, cls, feature_names)
        fp_tn_diff = self._compute_fp_tn_diff(X_valid, cls, feature_names)
        explainer = None
        try:
            import shap
            explainer = shap.TreeExplainer(model)
        except Exception as e:
            logger.warning(f'Sample diagnosis SHAP initialization failed: {e}')
        fn_tp_shap_diff = self._compute_fn_tp_shap_diff(explainer, X_valid, cls, feature_names)
        fp_tn_shap_diff = self._compute_fp_tn_shap_diff(explainer, X_valid, cls, feature_names)
        rescued_factors = []
        regressed_factors = []
        if explainer is not None and pool_exprs and (transitions.get('rescued', 0) > 0):
            rescued_factors = self._analyze_transition_factors(model, X_valid, cls, feature_names, pool_exprs, transition_type='rescued', explainer=explainer)
        if explainer is not None and pool_exprs and (transitions.get('regressed', 0) > 0):
            regressed_factors = self._analyze_transition_factors(model, X_valid, cls, feature_names, pool_exprs, transition_type='regressed', explainer=explainer)
        fn_mask = cls == 3
        fn_probs = y_prob[fn_mask]
        prob_bins = sorted(set([0.0, 0.1, 0.2, float(threshold), 1.0]))
        fn_prob_hist = np.histogram(fn_probs, bins=prob_bins)[0]
        fn_prob_distribution = {'bins': prob_bins, 'counts': fn_prob_hist.tolist(), 'total_fn': int(fn_mask.sum())}
        factor_fn_effectiveness = []
        if pool_exprs:
            fn_tp_mask = (cls == 3) | (cls == 1)
            y_fn_tp = np.where(cls[fn_tp_mask] == 1, 1, 0)
            n_base = X_valid.shape[1] - len(pool_exprs)
            for j, expr in enumerate(pool_exprs):
                col_idx = n_base + j
                if col_idx >= X_valid.shape[1]:
                    continue
                feat_vals = X_valid[fn_tp_mask, col_idx]
                valid_mask = np.isfinite(feat_vals)
                if valid_mask.sum() < 10:
                    continue
                from sklearn.metrics import roc_auc_score
                try:
                    fn_tp_auc = roc_auc_score(y_fn_tp[valid_mask], feat_vals[valid_mask])
                except ValueError:
                    continue
                fn_vals = feat_vals[y_fn_tp == 0]
                tp_vals = feat_vals[y_fn_tp == 1]
                fn_mean = float(np.nanmean(fn_vals)) if len(fn_vals) > 0 else 0.0
                tp_mean = float(np.nanmean(tp_vals)) if len(tp_vals) > 0 else 0.0
                fn_std = float(np.nanstd(fn_vals, ddof=1)) if len(fn_vals) > 1 else 1e-06
                tp_std = float(np.nanstd(tp_vals, ddof=1)) if len(tp_vals) > 1 else 1e-06
                pooled = np.sqrt((fn_std ** 2 + tp_std ** 2) / 2)
                es = (fn_mean - tp_mean) / max(pooled, 1e-06)
                factor_fn_effectiveness.append({'factor_expr': str(expr)[:60], 'fn_tp_auc': round(fn_tp_auc, 4), 'fn_mean': round(fn_mean, 4), 'tp_mean': round(tp_mean, 4), 'effect_size': round(float(es), 4)})
        self._prev_y_true = y_valid
        self._prev_y_pred = y_prob
        self._prev_y_cls = cls
        return {'iteration': iteration, 'threshold': round(float(threshold), 6), 'confusion': confusion, 'transitions': transitions, 'fn_vs_tp_diff': fn_vs_tp_diff, 'fp_tn_diff': fp_tn_diff, 'fn_tp_shap_diff': fn_tp_shap_diff, 'fp_tn_shap_diff': fp_tn_shap_diff, 'rescued_factors': rescued_factors, 'regressed_factors': regressed_factors, 'fn_prob_distribution': fn_prob_distribution, 'factor_fn_effectiveness': factor_fn_effectiveness}

    def _empty_stats(self, iteration: int) -> Dict[str, Any]:
        return {'iteration': iteration, 'threshold': None, 'confusion': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}, 'transitions': {}, 'fn_vs_tp_diff': [], 'fp_tn_diff': [], 'fn_tp_shap_diff': [], 'fp_tn_shap_diff': [], 'rescued_factors': [], 'regressed_factors': [], 'fn_prob_distribution': {}, 'factor_fn_effectiveness': []}

    def _compute_adaptive_threshold(self, y_prob: np.ndarray, y_true: np.ndarray) -> float:
        if len(y_prob) == 0:
            return 0.5
        fraud_ratio = float(np.mean(y_true))
        if fraud_ratio <= 0.0:
            return float(np.max(y_prob) + 1e-12)
        if fraud_ratio >= 1.0:
            return float(np.min(y_prob) - 1e-12)
        threshold = float(np.percentile(y_prob, 100.0 * (1.0 - fraud_ratio)))
        return min(max(threshold, 0.0), 1.0)

    def _compute_transitions(self, cls: np.ndarray) -> Dict[str, int]:
        if self._prev_y_cls is None or len(self._prev_y_cls) != len(cls):
            return {}
        prev = self._prev_y_cls
        transitions = {'rescued': int(((prev == 3) & (cls == 1)).sum()), 'regressed': int(((prev == 1) & (cls == 3)).sum()), 'persistent_fn': int(((prev == 3) & (cls == 3)).sum()), 'persistent_fp': int(((prev == 2) & (cls == 2)).sum())}
        return transitions

    def _finite_values(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return values[np.isfinite(values)]

    def _compute_iqr(self, values: np.ndarray) -> float:
        if len(values) == 0:
            return 0.0
        return float(np.percentile(values, 75) - np.percentile(values, 25))

    def _describe_shift(self, lhs_label: str, lhs_median: float, rhs_median: float) -> str:
        if not np.isfinite(lhs_median) or not np.isfinite(rhs_median):
            return 'Difference unknown'
        if abs(lhs_median - rhs_median) < 1e-12:
            return 'Groups are similar'
        if np.sign(lhs_median) == np.sign(rhs_median) and abs(lhs_median) < abs(rhs_median):
            return f'{lhs_label} closer to zero'
        if lhs_median > rhs_median:
            return f'{lhs_label} higher'
        return f'{lhs_label} lower'

    def _compute_group_diff(self, X: np.ndarray, lhs_mask: np.ndarray, rhs_mask: np.ndarray, feature_names: List[str], lhs_label: str, rhs_label: str, top_k: int) -> List[Dict[str, Any]]:
        n_lhs = int(lhs_mask.sum())
        n_rhs = int(rhs_mask.sum())
        if n_lhs < 2 or n_rhs < 2:
            return []
        X_lhs = X[lhs_mask]
        X_rhs = X[rhs_mask]
        diffs = []
        for i in range(X.shape[1]):
            lhs_vals = self._finite_values(X_lhs[:, i])
            rhs_vals = self._finite_values(X_rhs[:, i])
            if len(lhs_vals) < 2 or len(rhs_vals) < 2:
                continue
            lhs_mean = float(np.mean(lhs_vals))
            rhs_mean = float(np.mean(rhs_vals))
            lhs_median = float(np.median(lhs_vals))
            rhs_median = float(np.median(rhs_vals))
            lhs_iqr = self._compute_iqr(lhs_vals)
            rhs_iqr = self._compute_iqr(rhs_vals)
            diff = lhs_mean - rhs_mean
            lhs_std = float(np.std(lhs_vals, ddof=1))
            rhs_std = float(np.std(rhs_vals, ddof=1))
            pooled_std = np.sqrt(((len(lhs_vals) - 1) * lhs_std ** 2 + (len(rhs_vals) - 1) * rhs_std ** 2) / (len(lhs_vals) + len(rhs_vals) - 2))
            cohens_d = diff / max(float(pooled_std), 1e-06)
            var_name = feature_names[i] if i < len(feature_names) else f'feat_{i}'
            diffs.append({'var': var_name, 'lhs_label': lhs_label, 'rhs_label': rhs_label, 'lhs_mean': round(lhs_mean, 6), 'rhs_mean': round(rhs_mean, 6), 'lhs_median': round(lhs_median, 6), 'rhs_median': round(rhs_median, 6), 'lhs_iqr': round(lhs_iqr, 6), 'rhs_iqr': round(rhs_iqr, 6), 'diff': round(diff, 6), 'cohens_d': round(float(cohens_d), 4), 'direction': self._describe_shift(lhs_label, lhs_median, rhs_median)})
        diffs.sort(key=lambda x: abs(x['cohens_d']), reverse=True)
        return diffs[:top_k]

    def _compute_fn_tp_diff(self, X: np.ndarray, cls: np.ndarray, feature_names: List[str]) -> List[Dict[str, Any]]:
        return self._compute_group_diff(X, cls == 3, cls == 1, feature_names, 'FN', 'TP', 15)

    def _compute_fp_tn_diff(self, X: np.ndarray, cls: np.ndarray, feature_names: List[str]) -> List[Dict[str, Any]]:
        return self._compute_group_diff(X, cls == 2, cls == 0, feature_names, 'FP', 'TN', 5)

    def _sample_indices(self, indices: np.ndarray) -> np.ndarray:
        if len(indices) <= self.max_shap_samples:
            return indices
        rng = np.random.RandomState(42)
        return rng.choice(indices, self.max_shap_samples, replace=False)

    def _normalize_shap_values(self, shap_values: Any) -> np.ndarray:
        if isinstance(shap_values, list):
            if len(shap_values) == 1:
                return np.asarray(shap_values[0])
            return np.asarray(shap_values[1])
        values = np.asarray(shap_values)
        if values.ndim == 3 and values.shape[-1] >= 2:
            values = values[:, :, 1]
        return values

    def _compute_mean_abs_shap_by_indices(self, explainer, X: np.ndarray, indices: np.ndarray) -> Optional[np.ndarray]:
        if explainer is None or len(indices) == 0:
            return None
        indices = self._sample_indices(indices)
        shap_values = self._normalize_shap_values(explainer.shap_values(X[indices]))
        if shap_values.size == 0:
            return None
        return np.abs(shap_values).mean(axis=0)

    def _describe_shap_shift(self, lhs_label: str, rhs_label: str, diff: float) -> str:
        if abs(diff) < 1e-12:
            return 'Similar contributions'
        if diff > 0:
            return f'{lhs_label} more active'
        return f'{rhs_label} more active'

    def _compute_shap_diff(self, explainer, X: np.ndarray, lhs_mask: np.ndarray, rhs_mask: np.ndarray, feature_names: List[str], lhs_label: str, rhs_label: str, top_k: int) -> List[Dict[str, Any]]:
        lhs_mean_abs = self._compute_mean_abs_shap_by_indices(explainer, X, np.where(lhs_mask)[0])
        rhs_mean_abs = self._compute_mean_abs_shap_by_indices(explainer, X, np.where(rhs_mask)[0])
        if lhs_mean_abs is None or rhs_mean_abs is None:
            return []
        n_features = min(len(lhs_mean_abs), len(rhs_mean_abs), X.shape[1])
        results = []
        for i in range(n_features):
            diff = float(lhs_mean_abs[i] - rhs_mean_abs[i])
            var_name = feature_names[i] if i < len(feature_names) else f'feat_{i}'
            results.append({'var': var_name, 'lhs_label': lhs_label, 'rhs_label': rhs_label, 'lhs_mean_abs_shap': round(float(lhs_mean_abs[i]), 6), 'rhs_mean_abs_shap': round(float(rhs_mean_abs[i]), 6), 'diff': round(diff, 6), 'interpretation': self._describe_shap_shift(lhs_label, rhs_label, diff)})
        results.sort(key=lambda x: abs(x['diff']), reverse=True)
        return results[:top_k]

    def _compute_fn_tp_shap_diff(self, explainer, X: np.ndarray, cls: np.ndarray, feature_names: List[str]) -> List[Dict[str, Any]]:
        return self._compute_shap_diff(explainer, X, cls == 3, cls == 1, feature_names, 'FN', 'TP', 10)

    def _compute_fp_tn_shap_diff(self, explainer, X: np.ndarray, cls: np.ndarray, feature_names: List[str]) -> List[Dict[str, Any]]:
        return self._compute_shap_diff(explainer, X, cls == 2, cls == 0, feature_names, 'FP', 'TN', 5)

    def _analyze_transition_factors(self, model, X: np.ndarray, cls: np.ndarray, feature_names: List[str], pool_exprs: List, transition_type: str='rescued', explainer=None) -> List[Dict[str, Any]]:
        if self._prev_y_cls is None or explainer is None:
            return []
        prev = self._prev_y_cls
        if transition_type == 'rescued':
            mask = (prev == 3) & (cls == 1)
        else:
            mask = (prev == 1) & (cls == 3)
        n_trans = mask.sum()
        if n_trans == 0:
            return []
        try:
            trans_indices = np.where(mask)[0]
            mean_abs = self._compute_mean_abs_shap_by_indices(explainer, X, trans_indices)
            if mean_abs is None:
                return []
            n_base = max(0, len(feature_names) - len(pool_exprs))
            results = []
            for j, expr in enumerate(pool_exprs):
                col_idx = n_base + j
                if col_idx < len(mean_abs):
                    results.append({'factor_expr': str(expr), 'mean_shap_contrib': round(float(mean_abs[col_idx]), 6)})
            results.sort(key=lambda x: x['mean_shap_contrib'], reverse=True)
            return results[:5]
        except Exception as e:
            logger.warning(f'Sample diagnosis transition-factor analysis failed: {e}')
            return []

    def format_statistics_for_llm(self, stats: Dict[str, Any], iteration: int) -> str:
        parts = []
        threshold = stats.get('threshold')
        if threshold is not None:
            parts.append(f'# Round {iteration} sample-level misclassification diagnosis (validation only, adaptive threshold={threshold:.4f}）\n')
        else:
            parts.append(f'# Round {iteration} sample-level misclassification diagnosis (validation only)\n')
        c = stats.get('confusion', {})
        parts.append(f'## Confusion matrix')
        parts.append(f"- TP (correctly detected fraud): {c.get('TP', 0)}")
        parts.append(f"- TN (correctly classified normal): {c.get('TN', 0)}")
        parts.append(f"- FP (false alarms): {c.get('FP', 0)}")
        parts.append(f"- FN (missed fraud): {c.get('FN', 0)}")
        total_pred_pos = c.get('TP', 0) + c.get('FP', 0)
        if total_pred_pos > 0:
            precision = c.get('TP', 0) / total_pred_pos
            parts.append(f'- Fraud precision: {precision:.2%}')
        total_fraud = c.get('TP', 0) + c.get('FN', 0)
        if total_fraud > 0:
            recall = c.get('TP', 0) / total_fraud
            parts.append(f'- Fraud recall: {recall:.2%}')
        parts.append('')
        fn_tp = stats.get('fn_vs_tp_diff', [])
        if fn_tp:
            parts.append("## Typical FN profiles vs TP (data-level, descending |Cohen's d|)")
            parts.append("| Variable | FN median | TP median | FN IQR | TP IQR | Cohen's d | Difference direction |")
            parts.append('|------|----------|----------|--------|--------|-----------|----------|')
            for item in fn_tp[:15]:
                parts.append(f"| {item['var']} | {item['lhs_median']:.4f} | {item['rhs_median']:.4f} | {item['lhs_iqr']:.4f} | {item['rhs_iqr']:.4f} | {item['cohens_d']:+.3f} | {item['direction']} |")
            parts.append('')
        fn_tp_shap = stats.get('fn_tp_shap_diff', [])
        if fn_tp_shap:
            parts.append('## Attribution differences (FN vs TP SHAP, descending absolute difference)')
            parts.append('| Feature | TP mean absolute SHAP | FN mean absolute SHAP | Difference (FN-TP) | Interpretation |')
            parts.append('|------|-------------|-------------|-------------|------|')
            for item in fn_tp_shap[:10]:
                parts.append(f"| {item['var']} | {item['rhs_mean_abs_shap']:.4f} | {item['lhs_mean_abs_shap']:.4f} | {item['diff']:+.4f} | {item['interpretation']} |")
            parts.append('')
        fp_tn = stats.get('fp_tn_diff', [])
        if fp_tn:
            parts.append('## FP profiles vs TN (characteristics of normal firms classified as fraud; top-5)')
            parts.append("| Variable | FP median | TN median | Cohen's d | Interpretation |")
            parts.append('|------|----------|----------|-----------|------|')
            for item in fp_tn[:5]:
                parts.append(f"| {item['var']} | {item['lhs_median']:.4f} | {item['rhs_median']:.4f} | {item['cohens_d']:+.3f} | {item['direction']} |")
            parts.append('')
        fp_tn_shap = stats.get('fp_tn_shap_diff', [])
        if fp_tn_shap:
            parts.append('## FP vs TN SHAP differences (features associated with false alarms; top-5)')
            parts.append('| Feature | FP mean absolute SHAP | TN mean absolute SHAP | Difference (FP-TN) | Interpretation |')
            parts.append('|------|-------------|-------------|-------------|------|')
            for item in fp_tn_shap[:5]:
                parts.append(f"| {item['var']} | {item['lhs_mean_abs_shap']:.4f} | {item['rhs_mean_abs_shap']:.4f} | {item['diff']:+.4f} | {item['interpretation']} |")
            parts.append('')
        trans = stats.get('transitions', {})
        if trans:
            parts.append('## Comparison with the previous round')
            parts.append(f"- rescued (previously missed, now detected): {trans.get('rescued', 0)}")
            parts.append(f"- regressed (previously detected, now missed): {trans.get('regressed', 0)}")
            parts.append(f"- persistent_fn (persistently missed fraud): {trans.get('persistent_fn', 0)}")
            parts.append(f"- persistent_fp (persistent false alarms): {trans.get('persistent_fp', 0)}")
            parts.append('')
        rescued = stats.get('rescued_factors', [])
        if rescued:
            parts.append('## Generated factors with the largest contributions in rescued samples')
            for item in rescued[:3]:
                parts.append(f"- `{item['factor_expr']}` (|SHAP|={item['mean_shap_contrib']:.4f})")
            parts.append('')
        regressed = stats.get('regressed_factors', [])
        if regressed:
            parts.append('## Generated factors with the largest contributions in regressed samples (possible side effects)')
            for item in regressed[:3]:
                parts.append(f"- `{item['factor_expr']}` (|SHAP|={item['mean_shap_contrib']:.4f})")
            parts.append('')
        return '\n'.join(parts)

    def build_diagnostics_prompt(self, stats_markdown: str, iteration: int) -> str:
        return f"{stats_markdown}\nBased on the misclassification statistics for round {iteration}, provide a structured analysis.\nFocus on:\n1. Common characteristics of persistent_fn samples\n2. Operator/variable combinations to capture these missed-fraud patterns\n3. Patterns associated with false alarms among normal samples and ways to avoid them\n4. Regressions or false-alarm risks associated with this round's factor changes and possible improvements\n"

    def get_system_prompt(self) -> str:
        return DIAGNOSTICS_SYSTEM_PROMPT.format(max_chars=self.max_analysis_chars)

    @staticmethod
    def format_analysis_for_prompt(analysis: str, stats_summary: Dict[str, Any]) -> str:
        if not analysis or not analysis.strip():
            return ''
        c = stats_summary.get('confusion', {})
        trans = stats_summary.get('transitions', {})
        header_parts = [f"FN={c.get('FN', '?')}", f"FP={c.get('FP', '?')}"]
        threshold = stats_summary.get('threshold')
        if threshold is not None:
            header_parts.append(f'threshold={float(threshold):.4f}')
        if trans:
            header_parts.append(f"rescued={trans.get('rescued', 0)}")
            header_parts.append(f"regressed={trans.get('regressed', 0)}")
            header_parts.append(f"persistent_fn={trans.get('persistent_fn', 0)}")
        header = ', '.join(header_parts)
        return f'## Sample-level misclassification diagnosis (previous round)\n\n**Misclassification overview**: {header}\n\n{analysis.strip()}\n'
