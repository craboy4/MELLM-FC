"""Bounded factor pool with numerical screening, admission, and replacement."""
from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np
import shap
from fraud_alpha.expression import Expression, Feature
from fraud_alpha.calculator import FraudCalculator
from fraud_alpha.operator_catalog import build_operator_regex, normalize_operator_name
_PROFITABILITY_VARS_ALL = {'N20', 'N21', 'N24', 'N25', 'N28'}
_BALANCE_SHEET_VARS_ALL = {'N3', 'N4', 'N5', 'N7', 'N10', 'N12', 'N26'}
_FINANCING_VARS_ALL = {'N11', 'N15', 'N16'}
_GOVERNANCE_VARS_ALL = {'C1', 'C3', 'C4', 'N27', 'N34', 'N35'}

def _get_pattern_vars():
    avail = set(Feature.get_available_features())
    return (_PROFITABILITY_VARS_ALL & avail, _BALANCE_SHEET_VARS_ALL & avail, _FINANCING_VARS_ALL & avail, _GOVERNANCE_VARS_ALL & avail)
_OP_PATTERN = re.compile(build_operator_regex(require_call=False, include_legacy_aliases=True))
_VAR_PATTERN = re.compile('\\b(N\\d+|C\\d)\\b')

def extract_structural_fingerprint(expr_str: str) -> dict:
    ops = tuple(sorted({normalize_operator_name(op) for op in _OP_PATTERN.findall(expr_str)}))
    vars_ = tuple(sorted(set(_VAR_PATTERN.findall(expr_str))))
    var_set = set(vars_)
    categories = set()
    prof, bs, fin, gov = _get_pattern_vars()
    if var_set & prof:
        categories.add('P')
    if var_set & bs:
        categories.add('B')
    if var_set & fin:
        categories.add('F')
    if var_set & gov:
        categories.add('G')
    root_match = re.match('(\\w+)\\(', expr_str)
    root_op = normalize_operator_name(root_match.group(1)) if root_match else 'UNKNOWN'
    max_depth, depth = (0, 0)
    for ch in expr_str:
        if ch == '(':
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ')':
            depth -= 1
    return {'ops': ops, 'vars': vars_, 'categories': tuple(sorted(categories)), 'root_op': root_op, 'depth': max_depth}

def structural_similarity(fp1: dict, fp2: dict) -> float:

    def jaccard(a, b):
        s1, s2 = (set(a), set(b))
        if not s1 and (not s2):
            return 1.0
        return len(s1 & s2) / len(s1 | s2)
    sim_ops = jaccard(fp1['ops'], fp2['ops'])
    sim_vars = jaccard(fp1['vars'], fp2['vars'])
    sim_cat = jaccard(fp1['categories'], fp2['categories'])
    sim_root = 1.0 if fp1['root_op'] == fp2['root_op'] else 0.0
    return 0.35 * sim_ops + 0.25 * sim_vars + 0.25 * sim_cat + 0.15 * sim_root

class AlphaPool:

    def __init__(self, capacity: int, calculator: FraudCalculator, auc_threshold: float=0.5, corr_threshold: float=0.7, struct_sim_threshold: float=0.8, novelty_weight: float=0.0, rarity_weight: float=0.0, novelty_auc_margin: float=0.0001, shap_sample_size: int=500, weight_optimization_batch: int=5, raw_auc_threshold: float=0.52, competition_start_ratio: float=1.0, enable_similarity_challenge: bool=False, similarity_challenge_top_k: int=3):
        self.capacity = capacity
        self.calculator = calculator
        self.auc_threshold = auc_threshold
        self.corr_threshold = corr_threshold
        self.struct_sim_threshold = struct_sim_threshold
        self.novelty_weight = novelty_weight
        self.rarity_weight = rarity_weight
        self.novelty_auc_margin = novelty_auc_margin
        self.shap_sample_size = shap_sample_size
        self.weight_optimization_batch = weight_optimization_batch
        self.raw_auc_threshold = raw_auc_threshold
        self.competition_start_ratio = float(np.clip(competition_start_ratio, 0.0, 1.0))
        self.enable_similarity_challenge = bool(enable_similarity_challenge)
        self.similarity_challenge_top_k = max(1, int(similarity_challenge_top_k))
        self.replace_candidates_k = 5
        self.exprs: List[Expression] = []
        self.weights: List[float] = []
        self.aucs: List[float] = []
        self.selected_base_features: List[str] = calculator.base_features.copy()
        self.base_feature_weights: List[float] = []
        self.base_feature_importances: List[float] = []
        self.last_model = None
        self.last_feature_names: List[str] = []
        self.min_feature_importance: float = 0.001
        self.enable_base_feature_selection: bool = True
        self.excluded_base_features: List[str] = []
        self.eval_cnt = 0
        self.best_auc = 0.0
        self.best_auc_std = 0.0
        self.fold_aucs: Dict[str, float] = {}
        self.cv_mean_auc = 0.0
        self.temporal_cv_enabled = False
        self.temporal_folds = None
        self.temporal_robust_std_penalty = 0.0
        self.temporal_min_positive_folds = 0
        self.auc_history: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self.last_max_ast_similarity: float = 0.0
        self.last_novelty_score: float = 0.0
        self.last_rarity_score: float = 0.0
        self.last_low_coverage_vars: List[str] = []
        self.last_selected_by_novelty: bool = False
        self.last_similarity_challenge: bool = False
        self.last_similarity_challenge_target_count: int = 0
        self.last_similarity_challenge_non_bottom_k: bool = False
        self.selected_by_novelty_count = 0
        self.similarity_challenge_count = 0
        self.accepted_by_similarity_challenge_count = 0
        self.similarity_challenge_target_count = 0
        self.similarity_challenge_non_bottom_k_accept_count = 0
        self.accepted_expr_count = 0
        self.accepted_by_append_count = 0
        self.accepted_by_replace_count = 0
        self.accepted_novelty_sum = 0.0
        self.accepted_max_ast_similarity_sum = 0.0
        self.accepted_rarity_sum = 0.0
        self.accepted_low_coverage_var_set = set()
        self.current_iteration: Optional[int] = None
        self.competition_mode_enter_iter: Optional[int] = None
        self._competition_mode_iterations = set()
        self._force_competition: Optional[bool] = None

    @property
    def size(self) -> int:
        return len(self.exprs)

    @property
    def vacancy(self) -> int:
        return self.capacity - self.size

    def set_iteration_context(self, iteration: Optional[int]) -> None:
        self.current_iteration = iteration

    def _competition_start_size(self) -> int:
        if self.capacity <= 0:
            return 0
        return min(self.capacity, max(1, int(np.ceil(self.capacity * self.competition_start_ratio))))

    def _is_competition_mode(self) -> bool:
        if self._force_competition is not None:
            if self._force_competition:
                return self.size > 0
            else:
                return self.size > 0 and self.size >= self.capacity
        return self.size > 0 and self.size >= self._competition_start_size()

    def set_dynamic_mode(self, force_competition: Optional[bool]) -> None:
        self._force_competition = force_competition

    def _cv_mean_from_folds(self, fold_aucs: Dict[str, float], fallback: float) -> float:
        if not fold_aucs:
            return float(fallback)
        vals = [float(v) for v in fold_aucs.values()]
        return float(np.mean(vals)) if vals else float(fallback)

    def _set_temporal_metrics(self, score_auc: float, std_auc: float, fold_aucs: Dict[str, float]) -> None:
        self.best_auc = float(score_auc)
        self.best_auc_std = float(std_auc)
        self.fold_aucs = dict(fold_aucs or {})
        self.cv_mean_auc = self._cv_mean_from_folds(self.fold_aucs, self.best_auc)

    def _positive_fold_count(self, trial_fold_aucs: Dict[str, float]) -> int:
        if not self.fold_aucs or not trial_fold_aucs:
            return len(trial_fold_aucs or {})
        count = 0
        for key, value in trial_fold_aucs.items():
            old_value = self.fold_aucs.get(key)
            if old_value is None or float(value) >= float(old_value) - 1e-08:
                count += 1
        return count

    def _record_competition_mode(self) -> None:
        if not self._is_competition_mode():
            return
        if self.current_iteration is not None:
            if self.competition_mode_enter_iter is None:
                self.competition_mode_enter_iter = int(self.current_iteration)
            self._competition_mode_iterations.add(int(self.current_iteration))

    def _select_similarity_challenge_targets(self, similarity_target_scores: Dict[int, float]) -> List[int]:
        if not similarity_target_scores:
            return []
        ranked = sorted(similarity_target_scores.items(), key=lambda item: (-float(item[1]), int(item[0])))
        return [idx for idx, _ in ranked[:self.similarity_challenge_top_k]]

    def try_new_expr(self, expr: Expression) -> Tuple[float, bool, str]:
        self.eval_cnt += 1
        self.last_max_ast_similarity = 0.0
        self.last_novelty_score = 0.0
        self.last_rarity_score = 0.0
        self.last_low_coverage_vars = []
        self.last_selected_by_novelty = False
        self.last_similarity_challenge = False
        self.last_similarity_challenge_target_count = 0
        self.last_similarity_challenge_non_bottom_k = False
        try:
            competition_mode = self._is_competition_mode()
            if competition_mode:
                self._record_competition_mode()
            expr_str = str(expr)
            similarity_target_scores: Dict[int, float] = {}
            similarity_target_indices: List[int] = []
            similarity_challenge_active = False
            if self.size > 0:
                for existing_expr in self.exprs:
                    if expr_str == str(existing_expr):
                        return (0.0, False, 'Duplicate expression')
            if self.struct_sim_threshold < 1.0 and self.size > 0:
                new_fp = extract_structural_fingerprint(expr_str)
                for idx, existing_expr in enumerate(self.exprs):
                    sim = structural_similarity(new_fp, extract_structural_fingerprint(str(existing_expr)))
                    if sim >= self.struct_sim_threshold:
                        if competition_mode and self.enable_similarity_challenge:
                            similarity_target_scores[idx] = max(similarity_target_scores.get(idx, 0.0), float(sim))
                        else:
                            return (0.0, False, f'Structural similarity too high ({sim:.2f}>={self.struct_sim_threshold})')
            raw_auc = self.calculator.calc_raw_auc(expr, use_valid=True)
            if raw_auc < self.raw_auc_threshold:
                return (0.0, False, f'Raw AUC too low ({raw_auc:.4f}<{self.raw_auc_threshold}), skipping model evaluation')
            for idx, existing_expr in enumerate(self.exprs):
                corr = self.calculator.calc_mutual_corr(expr, existing_expr)
                if corr > self.corr_threshold:
                    if competition_mode and self.enable_similarity_challenge:
                        similarity_target_scores[idx] = max(similarity_target_scores.get(idx, 0.0), float(corr))
                    else:
                        return (0.0, False, f'Correlation too high ({corr:.2f}>{self.corr_threshold})')
            if competition_mode and self.enable_similarity_challenge and similarity_target_scores:
                similarity_target_indices = self._select_similarity_challenge_targets(similarity_target_scores)
                if similarity_target_indices:
                    similarity_challenge_active = True
                    self.last_similarity_challenge = True
                    self.last_similarity_challenge_target_count = len(similarity_target_indices)
                    self.similarity_challenge_count += 1
                    self.similarity_challenge_target_count += len(similarity_target_indices)
            fast_auc = self.calculator.calc_single_auc_fast(expr, use_valid=True)
            if fast_auc < self.auc_threshold:
                return (0.0, False, f'Fast AUC too low ({fast_auc:.4f}<{self.auc_threshold}), skipping LightGBM evaluation')
            new_auc = self.calculator.calc_single_auc(expr, use_valid=True)
            if new_auc < self.auc_threshold:
                return (0.0, False, f'AUC too low ({new_auc:.4f}<{self.auc_threshold})')
            max_sim, novelty_score, rarity_score, low_cov_vars = self._compute_novelty_metrics(str(expr))
            self.last_max_ast_similarity = float(max_sim)
            self.last_novelty_score = float(novelty_score)
            self.last_rarity_score = float(rarity_score)
            self.last_low_coverage_vars = low_cov_vars
            novelty_bonus = 0.0
            if self.novelty_weight > 0:
                novelty_bonus += self.novelty_weight * float(self.last_novelty_score)
            if self.rarity_weight > 0:
                novelty_bonus += self.rarity_weight * float(self.last_rarity_score)
            temporal_penalty = 0.0
            try:
                stability = self.calculator.compute_temporal_stability(expr)
                if not stability.get('stable', True):
                    temporal_penalty = stability.get('auc_gap', 0.0) * 0.5
            except Exception:
                pass
            old_pool_auc = self.best_auc

            def _eval_pool_auc():
                if self.temporal_cv_enabled:
                    mean_auc, std_auc, fold_aucs_dict = self.calculator.calc_pool_auc_temporal_cv(self.exprs, selected_base_features=self.selected_base_features, temporal_folds=self.temporal_folds, robust_std_penalty=self.temporal_robust_std_penalty, return_model=False)
                    return (mean_auc, std_auc, fold_aucs_dict)
                else:
                    auc, _ = self.calculator.calc_pool_auc_with_lgbm(self.exprs, selected_base_features=self.selected_base_features, return_model=False)
                    return (auc, 0.0, {})
            competition_mode = self._is_competition_mode()
            if competition_mode:
                self._record_competition_mode()
            if self.size < self.capacity and (not competition_mode):
                self.exprs.append(expr)
                self.weights.append(1.0)
                self.aucs.append(new_auc)
                trial_auc, trial_std, trial_fold_aucs = _eval_pool_auc()
                auc_tolerance = 0.001 if self.size < self.capacity * 0.8 else 0.0005
                auc_tolerance = max(0, auc_tolerance - temporal_penalty)
                positive_folds = self._positive_fold_count(trial_fold_aucs)
                if self.temporal_cv_enabled and self.fold_aucs and (self.temporal_min_positive_folds > 0) and (positive_folds < self.temporal_min_positive_folds):
                    self.exprs.pop()
                    self.weights.pop()
                    self.aucs.pop()
                    return (0.0, False, f'Insufficient cross-year robustness (positive_folds={positive_folds}<{self.temporal_min_positive_folds})')
                if trial_auc < old_pool_auc - auc_tolerance and self.size > self.capacity * 0.8:
                    self.exprs.pop()
                    self.weights.pop()
                    self.aucs.pop()
                    return (0.0, False, f'No incremental AUC gain ({trial_auc:.4f}<={old_pool_auc:.4f})')
                self._set_temporal_metrics(trial_auc, trial_std, trial_fold_aucs)
                if self._is_competition_mode():
                    self._record_competition_mode()
                fold_info = ''
                if trial_fold_aucs:
                    fold_strs = [f'{k}:{v:.4f}' for k, v in trial_fold_aucs.items()]
                    fold_info = f" [{', '.join(fold_strs)}]"
                add_reason = f'Vacancy available, single-factor AUC={new_auc:.4f}, combined AUC={trial_auc:.4f}(+{trial_auc - old_pool_auc:.4f}){fold_info}'
                accept_mode = 'append'
            else:
                abs_weights = [abs(w) for w in self.weights]
                sorted_indices = sorted(range(len(abs_weights)), key=lambda i: abs_weights[i])
                base_candidates = sorted_indices[:self.replace_candidates_k]
                base_candidate_set = set(base_candidates)
                candidates = list(base_candidates)
                if similarity_challenge_active:
                    candidates = list(dict.fromkeys(base_candidates + similarity_target_indices))
                worst_candidate_auc = min((self.aucs[i] for i in candidates))
                if new_auc < worst_candidate_auc - 0.01:
                    return (0.0, False, f'Insufficient replacement quality (AUC={new_auc:.4f}<worst={worst_candidate_auc:.4f}-0.01)')
                replaced = False
                replace_tolerance = 0.0002
                best_candidate_idx = None
                best_trial_auc = None
                best_trial_std = 0.0
                best_trial_fold_aucs = {}
                best_score = None
                best_auc_among_success = None
                best_selected_by_novelty = False
                for candidate_idx in candidates:
                    old_expr = self.exprs[candidate_idx]
                    old_weight = self.weights[candidate_idx]
                    old_auc_val = self.aucs[candidate_idx]
                    self.calculator.remove_feature_cache(str(old_expr))
                    self.exprs[candidate_idx] = expr
                    self.weights[candidate_idx] = 1.0
                    self.aucs[candidate_idx] = new_auc
                    trial_auc, trial_std, trial_fold_aucs = _eval_pool_auc()
                    auc_val = float(trial_auc)
                    positive_folds = self._positive_fold_count(trial_fold_aucs)
                    fold_success = not self.temporal_cv_enabled or not self.fold_aucs or self.temporal_min_positive_folds <= 0 or (positive_folds >= self.temporal_min_positive_folds)
                    hard_success = auc_val >= old_pool_auc - replace_tolerance and fold_success
                    novelty_success = not hard_success and fold_success and (novelty_bonus > 0.0) and (auc_val >= old_pool_auc - replace_tolerance - self.novelty_auc_margin) and (auc_val + novelty_bonus >= old_pool_auc - replace_tolerance)
                    if hard_success or novelty_success:
                        if hard_success:
                            best_auc_among_success = auc_val if best_auc_among_success is None else max(best_auc_among_success, auc_val)
                        score = auc_val + novelty_bonus
                        if best_score is None or score > best_score:
                            best_score = score
                            best_candidate_idx = candidate_idx
                            best_trial_auc = auc_val
                            best_trial_std = float(trial_std)
                            best_trial_fold_aucs = trial_fold_aucs
                            best_selected_by_novelty = novelty_success
                    self.exprs[candidate_idx] = old_expr
                    self.weights[candidate_idx] = old_weight
                    self.aucs[candidate_idx] = old_auc_val
                if best_candidate_idx is None:
                    last_trial = float(trial_auc) if 'trial_auc' in locals() else float('nan')
                    candidate_label = f'Candidate set: {len(candidates)} items' if similarity_challenge_active else f'Bottom-{self.replace_candidates_k}'
                    return (0.0, False, f'replacement {candidate_label}all reduced AUC; best attempt={last_trial:.4f}<{old_pool_auc:.4f}')
                old_expr = self.exprs[best_candidate_idx]
                self.calculator.remove_feature_cache(str(old_expr))
                self.exprs[best_candidate_idx] = expr
                self.weights[best_candidate_idx] = 1.0
                self.aucs[best_candidate_idx] = new_auc
                trial_auc, trial_std, trial_fold_aucs = _eval_pool_auc()
                self._set_temporal_metrics(trial_auc, trial_std, trial_fold_aucs)
                if best_selected_by_novelty:
                    self.last_selected_by_novelty = True
                if similarity_challenge_active and best_candidate_idx not in base_candidate_set:
                    self.last_similarity_challenge_non_bottom_k = True
                fold_info = ''
                if trial_fold_aucs:
                    fold_strs = [f'{k}:{v:.4f}' for k, v in trial_fold_aucs.items()]
                    fold_info = f" [{', '.join(fold_strs)}]"
                mode_label = 'Full-pool competition' if self.size >= self.capacity else 'Near-capacity competition'
                route_info = ''
                if similarity_challenge_active:
                    route_info = f', candidates={len(candidates)}(Bottom-{self.replace_candidates_k}+SimTop-{len(similarity_target_indices)})'
                add_reason = f'{mode_label}Replace factor {best_candidate_idx} (K={self.replace_candidates_k}scan {route_info}), single-factor AUC={new_auc:.4f}, combined AUC={trial_auc:.4f}({trial_auc - old_pool_auc:+.4f}){fold_info}'
                replaced = True
                accept_mode = 'replace'
            self._pending_optimization = getattr(self, '_pending_optimization', 0) + 1
            if self._pending_optimization >= self.weight_optimization_batch or self.size >= self.capacity or accept_mode == 'replace':
                self._optimize_weights()
                self._pending_optimization = 0
            reward = max(0.0, trial_auc - old_pool_auc)
            self.history.append({'expr': str(expr), 'auc': new_auc, 'pool_auc': self.best_auc, 'reward': reward, 'fold_aucs': trial_fold_aucs if self.temporal_cv_enabled else {}})
            self.accepted_expr_count += 1
            if accept_mode == 'append':
                self.accepted_by_append_count += 1
            elif accept_mode == 'replace':
                self.accepted_by_replace_count += 1
            if self.last_similarity_challenge:
                self.accepted_by_similarity_challenge_count += 1
                if self.last_similarity_challenge_non_bottom_k:
                    self.similarity_challenge_non_bottom_k_accept_count += 1
            self.accepted_novelty_sum += float(self.last_novelty_score)
            self.accepted_max_ast_similarity_sum += float(self.last_max_ast_similarity)
            self.accepted_rarity_sum += float(self.last_rarity_score)
            if self.last_selected_by_novelty:
                self.selected_by_novelty_count += 1
            self.accepted_low_coverage_var_set.update(self.last_low_coverage_vars)
            return (reward, True, add_reason)
        except Exception as e:
            print(f'Failed to add expression: {e}')
            return (0.0, False, f'Calculation error: {e}')

    def _compute_novelty_metrics(self, expr_str: str) -> Tuple[float, float, float, List[str]]:
        if not self.exprs:
            fp = extract_structural_fingerprint(expr_str)
            return (0.0, 1.0, 1.0, list(fp.get('vars', ())))
        new_fp = extract_structural_fingerprint(expr_str)
        max_sim = 0.0
        for existing_expr in self.exprs:
            sim = structural_similarity(new_fp, extract_structural_fingerprint(str(existing_expr)))
            if sim > max_sim:
                max_sim = sim
        novelty_score = max(0.0, 1.0 - max_sim)
        var_counts: Dict[str, int] = {}
        op_counts: Dict[str, int] = {}
        for e in self.exprs:
            fp = extract_structural_fingerprint(str(e))
            for v in fp.get('vars', ()):
                var_counts[v] = var_counts.get(v, 0) + 1
            for op in fp.get('ops', ()):
                op_counts[op] = op_counts.get(op, 0) + 1
        vars_ = list(new_fp.get('vars', ()))
        ops_ = list(new_fp.get('ops', ()))
        rarity_terms: List[float] = []
        for v in vars_:
            rarity_terms.append(1.0 / (1.0 + var_counts.get(v, 0)))
        for op in ops_:
            rarity_terms.append(1.0 / (1.0 + op_counts.get(op, 0)))
        rarity_score = float(np.mean(rarity_terms)) if rarity_terms else 0.0
        low_cov_vars = [v for v in vars_ if var_counts.get(v, 0) <= 1]
        return (float(max_sim), float(novelty_score), float(rarity_score), low_cov_vars)

    def _optimize_weights(self, use_cv: bool=False) -> None:
        if self.size == 0:
            return
        try:
            if self.temporal_cv_enabled:
                mean_auc, std_auc, fold_aucs_dict, model = self.calculator.calc_pool_auc_temporal_cv(self.exprs, selected_base_features=self.selected_base_features, temporal_folds=self.temporal_folds, robust_std_penalty=self.temporal_robust_std_penalty, return_model=True)
                auc = mean_auc
                old_auc = self.best_auc
                self._set_temporal_metrics(auc, std_auc, fold_aucs_dict)
            else:
                auc, model = self.calculator.calc_pool_auc_with_lgbm(self.exprs, selected_base_features=self.selected_base_features, return_model=True, use_cv=use_cv)
                old_auc = self.best_auc
                self._set_temporal_metrics(auc, 0.0, {})
            self.last_model = model
            self.last_feature_names = self.selected_base_features.copy() + [str(expr) for expr in self.exprs]
            if auc < old_auc:
                self.auc_history.append({'event': 'auc_decrease', 'old_auc': old_auc, 'new_auc': auc, 'pool_size': self.size, 'reason': 'weight_optimization_after_removal'})
            if model is not None:
                n_base = len(self.selected_base_features)
                try:
                    train_features = []
                    for expr in self.exprs:
                        feat = self.calculator.get_cached_train_feature(expr)
                        if feat is not None:
                            train_features.append(feat)
                    if len(train_features) == self.size:
                        X_train_base = self.calculator.train_data[self.selected_base_features].values
                        X_train_gen = np.column_stack(train_features)
                        X_train_combined = np.hstack([X_train_base, X_train_gen])
                        n_sample = min(self.shap_sample_size, len(X_train_combined))
                        idx = np.random.choice(len(X_train_combined), n_sample, replace=False)
                        X_sample = X_train_combined[idx]
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_sample)
                        if isinstance(shap_values, list):
                            shap_values = shap_values[1]
                        importances = np.abs(shap_values).mean(axis=0)
                    else:
                        importances = model.feature_importances_.astype(float)
                except Exception:
                    importances = model.feature_importances_.astype(float)
                self.base_feature_importances = importances[:n_base].tolist()
                gen_importances = importances[n_base:]
                if len(self.base_feature_importances) > 0:
                    total_base = np.sum(self.base_feature_importances) + 1e-08
                    self.base_feature_weights = (np.array(self.base_feature_importances) / total_base).tolist()
                else:
                    self.base_feature_weights = []
                if len(gen_importances) == self.size:
                    total_gen = np.sum(gen_importances) + 1e-08
                    self.weights = (gen_importances / total_gen).tolist()
                else:
                    self.weights = [1.0 / self.size] * self.size
            else:
                self.weights = [1.0 / self.size] * self.size
                if len(self.selected_base_features) > 0:
                    self.base_feature_weights = [1.0 / len(self.selected_base_features)] * len(self.selected_base_features)
        except Exception as e:
            print(f'Weight optimization failed: {e}')
            self.last_model = None
            self.last_feature_names = []
            self.weights = [1.0 / self.size] * self.size
            if len(self.selected_base_features) > 0:
                self.base_feature_weights = [1.0 / len(self.selected_base_features)] * len(self.selected_base_features)

    def flush_optimization(self, use_cv: bool=False) -> None:
        self._pending_optimization = 0
        self._optimize_weights(use_cv=use_cv)

    def select_top_k_features(self, k: int=15) -> Dict[str, Any]:
        if self.size <= k:
            return {'n_before': self.size, 'n_after': self.size, 'removed': [], 'auc_before': self.best_auc, 'auc_after': self.best_auc}
        auc_before = self.best_auc
        n_before = self.size
        abs_weights = np.abs(self.weights)
        sorted_indices = np.argsort(abs_weights)[::-1]
        keep_indices = set(sorted_indices[:k].tolist())
        drop_indices = [i for i in range(self.size) if i not in keep_indices]
        removed = []
        for idx in sorted(drop_indices, reverse=True):
            removed.append(str(self.exprs[idx]))
            self.calculator.remove_feature_cache(str(self.exprs[idx]))
            del self.exprs[idx]
            del self.weights[idx]
            del self.aucs[idx]
        self._optimize_weights()
        return {'n_before': n_before, 'n_after': self.size, 'removed': removed, 'auc_before': auc_before, 'auc_after': self.best_auc}

    def periodic_purge(self, drop_ratio: float=0.2, min_pool_size: int=20) -> dict:
        if self.size == 0:
            return {'n_purged': 0, 'purged_exprs': [], 'old_size': 0, 'new_size': 0, 'removed_base': []}
        old_size = self.size
        if self.size <= min_pool_size:
            base_result = self.remove_low_weight_features()
            removed_base = base_result.get('removed_base', [])
            return {'n_purged': 0, 'purged_exprs': [], 'old_size': old_size, 'new_size': self.size, 'removed_base': removed_base}
        n_to_drop = max(1, int(self.size * drop_ratio))
        max_droppable = self.size - min_pool_size
        n_to_drop = min(n_to_drop, max_droppable)
        if n_to_drop <= 0:
            base_result = self.remove_low_weight_features()
            removed_base = base_result.get('removed_base', [])
            return {'n_purged': 0, 'purged_exprs': [], 'old_size': old_size, 'new_size': self.size, 'removed_base': removed_base}
        abs_weights = [abs(w) for w in self.weights]
        sorted_indices = sorted(range(len(abs_weights)), key=lambda i: abs_weights[i])
        indices_to_drop = sorted_indices[:n_to_drop]
        purged_exprs = []
        for idx in sorted(indices_to_drop, reverse=True):
            purged_exprs.append(str(self.exprs[idx]))
            self.calculator.remove_feature_cache(str(self.exprs[idx]))
            del self.exprs[idx]
            del self.weights[idx]
            del self.aucs[idx]
        if purged_exprs and self.size > 0:
            self.flush_optimization()
        base_result = self.remove_low_weight_features()
        removed_base = base_result.get('removed_base', [])
        return {'n_purged': len(purged_exprs), 'purged_exprs': purged_exprs, 'old_size': old_size, 'new_size': self.size, 'removed_base': removed_base}

    def remove_least_significant(self, n: int) -> List[Expression]:
        if n <= 0 or self.size == 0:
            return []
        n = min(n, self.size)
        abs_weights = np.abs(self.weights)
        indices = np.argsort(abs_weights)[:n]
        removed = []
        for idx in sorted(indices, reverse=True):
            removed.append(self.exprs[idx])
            del self.exprs[idx]
            del self.weights[idx]
            del self.aucs[idx]
        return removed

    def remove_low_weight_features(self) -> Dict[str, Any]:
        removed_base = []
        if self.enable_base_feature_selection and len(self.base_feature_importances) > 0:
            low_weight_indices = [i for i, imp in enumerate(self.base_feature_importances) if imp < self.min_feature_importance]
            for idx in sorted(low_weight_indices, reverse=True):
                removed_base.append(self.selected_base_features[idx])
                del self.selected_base_features[idx]
                del self.base_feature_weights[idx]
                del self.base_feature_importances[idx]
        return {'removed_base': removed_base, 'n_removed_base': len(removed_base)}

    def select_crossover_parents(self, n_parents: int=5, min_diversity: float=0.5) -> List[Dict]:
        if self.size == 0:
            return []
        ranked_indices = sorted(range(self.size), key=lambda i: abs(self.weights[i]), reverse=True)
        fingerprints = {}
        for i in ranked_indices:
            fingerprints[i] = extract_structural_fingerprint(str(self.exprs[i]))
        max_sim_threshold = 1.0 - min_diversity
        selected = []
        for i in ranked_indices:
            if len(selected) >= n_parents:
                break
            fp_i = fingerprints[i]
            too_similar = False
            for sel in selected:
                sim = structural_similarity(fp_i, sel['fingerprint'])
                if sim > max_sim_threshold:
                    too_similar = True
                    break
            if not too_similar:
                selected.append({'expr_str': str(self.exprs[i]), 'auc': self.aucs[i], 'weight': self.weights[i], 'rank': len(selected) + 1, 'fingerprint': fp_i})
        if len(selected) < min(2, self.size):
            for i in ranked_indices:
                if len(selected) >= n_parents:
                    break
                expr_str = str(self.exprs[i])
                if any((s['expr_str'] == expr_str for s in selected)):
                    continue
                selected.append({'expr_str': expr_str, 'auc': self.aucs[i], 'weight': self.weights[i], 'rank': len(selected) + 1, 'fingerprint': fingerprints[i]})
        return selected

    def get_underexplored_operators(self, top_n: int=8) -> Dict[str, Any]:
        from collections import Counter
        all_ops = Counter()
        all_vars = Counter()
        for expr in self.exprs:
            expr_str = str(expr)
            ops = _OP_PATTERN.findall(expr_str)
            vars_ = _VAR_PATTERN.findall(expr_str)
            all_ops.update((normalize_operator_name(op) for op in ops))
            all_vars.update(vars_)
        known_ops = {'Add', 'Sub', 'Mul', 'Div', 'Abs', 'Log', 'Sqrt', 'Sign', 'Inv', 'Power', 'Clip', 'Sigmoid', 'Tanh', 'Mean', 'Std', 'Var', 'Med', 'Mad', 'Max', 'Min', 'Range', 'Sum', 'Quantile', 'EMA', 'WMA', 'DecayLinear', 'Delta', 'ROC', 'Ref', 'TSRank', 'Corr', 'Cov', 'Greater', 'Less', 'CSRank', 'ZScore', 'If'}
        available_vars = set(Feature.get_available_features())
        used_ops = set(all_ops.keys())
        used_vars = set(all_vars.keys())
        underexplored_ops = sorted(known_ops - used_ops)
        underexplored_vars = sorted(available_vars - used_vars)
        low_freq_ops = [op for op, cnt in all_ops.items() if cnt <= 2 and op not in underexplored_ops]
        low_freq_vars = [v for v, cnt in all_vars.items() if cnt <= 2 and v not in underexplored_vars]
        overused_ops = [op for op, _ in all_ops.most_common(3)]
        overused_vars = [v for v, _ in all_vars.most_common(5)]
        return {'underexplored_ops': underexplored_ops + sorted(low_freq_ops), 'underexplored_vars': underexplored_vars + sorted(low_freq_vars), 'overused_ops': overused_ops, 'overused_vars': overused_vars}

    def get_state(self) -> Dict[str, Any]:
        return {'size': self.size, 'capacity': self.capacity, 'best_auc': self.best_auc, 'best_auc_std': self.best_auc_std, 'fold_aucs': dict(self.fold_aucs), 'cv_mean_auc': self.cv_mean_auc, 'robust_score': self.best_auc, 'exprs': [str(expr) for expr in self.exprs], 'weights': self.weights.copy(), 'aucs': self.aucs.copy(), 'n_base_features': len(self.selected_base_features), 'selected_base_features': self.selected_base_features.copy(), 'base_feature_weights': self.base_feature_weights.copy(), 'base_feature_importances': self.base_feature_importances.copy()}

    def get_novelty_stats(self) -> Dict[str, Any]:
        accepted_low_coverage_vars = sorted(self.accepted_low_coverage_var_set)
        base_stats = {'accepted_count': self.accepted_expr_count, 'accepted_by_append_count': self.accepted_by_append_count, 'accepted_by_replace_count': self.accepted_by_replace_count, 'competition_start_ratio': self.competition_start_ratio, 'competition_start_size': self._competition_start_size(), 'competition_mode_enter_iter': self.competition_mode_enter_iter, 'competition_mode_rounds': len(self._competition_mode_iterations), 'enable_similarity_challenge': self.enable_similarity_challenge, 'similarity_challenge_top_k': self.similarity_challenge_top_k, 'similarity_challenge_count': self.similarity_challenge_count, 'similarity_challenge_target_count': self.similarity_challenge_target_count, 'similarity_challenge_avg_target_count': self.similarity_challenge_target_count / max(self.similarity_challenge_count, 1), 'accepted_by_similarity_challenge_count': self.accepted_by_similarity_challenge_count, 'accepted_by_similarity_challenge_rate': self.accepted_by_similarity_challenge_count / max(self.accepted_expr_count, 1), 'similarity_challenge_non_bottom_k_accept_count': self.similarity_challenge_non_bottom_k_accept_count, 'selected_by_novelty_count': self.selected_by_novelty_count, 'selected_by_novelty_rate': self.selected_by_novelty_count / max(self.accepted_expr_count, 1)}
        if self.accepted_expr_count == 0:
            return {**base_stats, 'accepted_avg_novelty': 0.0, 'accepted_avg_max_sim': 0.0, 'accepted_avg_rarity': 0.0, 'accepted_low_coverage_var_count': 0, 'accepted_low_coverage_vars': []}
        return {**base_stats, 'accepted_avg_novelty': self.accepted_novelty_sum / self.accepted_expr_count, 'accepted_avg_max_sim': self.accepted_max_ast_similarity_sum / self.accepted_expr_count, 'accepted_avg_rarity': self.accepted_rarity_sum / self.accepted_expr_count, 'accepted_low_coverage_var_count': len(accepted_low_coverage_vars), 'accepted_low_coverage_vars': accepted_low_coverage_vars}

    def to_dict(self) -> Dict[str, Any]:
        return {'capacity': self.capacity, 'size': self.size, 'best_auc': self.best_auc, 'best_auc_std': self.best_auc_std, 'fold_aucs': dict(self.fold_aucs), 'cv_mean_auc': self.cv_mean_auc, 'robust_score': self.best_auc, 'eval_cnt': self.eval_cnt, 'exprs': [str(expr) for expr in self.exprs], 'weights': self.weights, 'aucs': self.aucs, 'history': self.history[-100:], 'novelty_stats': self.get_novelty_stats(), 'n_base_features': len(self.selected_base_features), 'selected_base_features': self.selected_base_features, 'base_feature_weights': self.base_feature_weights, 'base_feature_importances': self.base_feature_importances, 'excluded_base_features': self.excluded_base_features}

    def test_ensemble(self, calculator: Optional[FraudCalculator]=None) -> Tuple[float, float]:
        calc = calculator if calculator is not None else self.calculator
        if self.size == 0:
            return (0.0, 0.0)
        try:
            base_features_to_use = self.selected_base_features
            train_features = []
            valid_features = []
            for expr in self.exprs:
                try:
                    train_feat = calc.get_cached_train_feature(expr)
                    valid_feat = calc.get_cached_valid_feature(expr)
                    if train_feat is not None and valid_feat is not None:
                        train_features.append(train_feat)
                        valid_features.append(valid_feat)
                except:
                    continue
            if len(train_features) == 0:
                return (0.0, 0.0)
            X_train = calc.train_data[base_features_to_use].values
            X_train_generated = np.column_stack(train_features)
            X_train_combined = np.hstack([X_train, X_train_generated])
            X_valid = calc.valid_data[base_features_to_use].values
            X_valid_generated = np.column_stack(valid_features)
            X_valid_combined = np.hstack([X_valid, X_valid_generated])
            import lightgbm as lgb
            model = lgb.LGBMClassifier(objective='binary', n_estimators=calc.ensemble_params['n_estimators'], max_depth=calc.ensemble_params['max_depth'], num_leaves=calc.ensemble_params['num_leaves'], learning_rate=calc.ensemble_params['learning_rate'], colsample_bytree=calc.ensemble_params['colsample_bytree'], subsample=calc.ensemble_params['subsample'], reg_alpha=calc.ensemble_params['reg_alpha'], reg_lambda=calc.ensemble_params['reg_lambda'], min_child_samples=calc.ensemble_params.get('min_child_samples', 50), random_state=42, verbose=-1)
            model.fit(X_train_combined, calc.y_train, eval_set=[(X_valid_combined, calc.y_valid)], eval_metric='auc', callbacks=[lgb.early_stopping(stopping_rounds=calc.early_stopping_rounds, verbose=False)])
            from sklearn.metrics import roc_auc_score
            y_pred_proba = model.predict_proba(X_valid_combined)[:, 1]
            auc = roc_auc_score(calc.y_valid, y_pred_proba)
            y_pred = model.predict(X_valid_combined)
            accuracy = float(np.mean(y_pred == calc.y_valid))
            return (auc, accuracy)
        except Exception as e:
            print(f'Ensemble test failed: {e}')
            return (0.0, 0.0)

class LinearAlphaPool(AlphaPool):

    def __init__(self, capacity: int, calculator: FraudCalculator, auc_threshold: float=0.5, corr_threshold: float=0.7, struct_sim_threshold: float=0.8, novelty_weight: float=0.0, rarity_weight: float=0.0, novelty_auc_margin: float=0.0001, ic_lower_bound: Optional[float]=None, shap_sample_size: int=500, weight_optimization_batch: int=5, raw_auc_threshold: float=0.52, competition_start_ratio: float=1.0, enable_similarity_challenge: bool=False, similarity_challenge_top_k: int=3):
        super().__init__(capacity, calculator, auc_threshold, corr_threshold, struct_sim_threshold=struct_sim_threshold, novelty_weight=novelty_weight, rarity_weight=rarity_weight, novelty_auc_margin=novelty_auc_margin, shap_sample_size=shap_sample_size, weight_optimization_batch=weight_optimization_batch, raw_auc_threshold=raw_auc_threshold, competition_start_ratio=competition_start_ratio, enable_similarity_challenge=enable_similarity_challenge, similarity_challenge_top_k=similarity_challenge_top_k)
        self.ic_lower_bound = auc_threshold if ic_lower_bound is None else ic_lower_bound
