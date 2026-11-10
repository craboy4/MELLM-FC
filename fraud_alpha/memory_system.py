"""Persist variable semantics, development summaries, and short- and long-term memory."""
import os
import json
import logging
import shutil
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional
from fraud_alpha.feedback_memory import format_feedback_summary, summarize_feedback
logger = logging.getLogger(__name__)
VARIABLE_SEMANTICS: Dict[str, Dict[str, str]] = {'N1': {'name': 'Depreciation and amortization', 'desc': 'Total depreciation of fixed assets, amortization of intangible assets, and related charges', 'fraud_hint': ''}, 'N2': {'name': 'Cash and cash equivalents', 'desc': 'Total cash on hand, bank deposits, and other monetary assets', 'fraud_hint': 'Investigate large cash balances that diverge substantially from cash flows'}, 'N3': {'name': 'Net accounts receivable', 'desc': 'Accounts receivable less allowance for doubtful accounts', 'fraud_hint': 'Growth far above revenue growth may suggest inflated revenue'}, 'N4': {'name': 'Net inventory', 'desc': 'Inventory net of impairment allowances', 'fraud_hint': 'Abnormal inventory accumulation inconsistent with revenue growth'}, 'N5': {'name': 'Total current assets', 'desc': 'Assets expected to be realized within one year', 'fraud_hint': ''}, 'N6': {'name': 'Net fixed assets', 'desc': 'Fixed assets net of accumulated depreciation', 'fraud_hint': ''}, 'N7': {'name': 'Total assets', 'desc': 'Total assets of the company', 'fraud_hint': 'Asset inflation is a common fraud mechanism'}, 'N8': {'name': 'Accounts payable', 'desc': 'Amounts owed to suppliers and not yet paid', 'fraud_hint': ''}, 'N9': {'name': 'Taxes payable', 'desc': 'Taxes and levies payable under applicable tax rules', 'fraud_hint': 'Lower-than-expected taxes may suggest misstated profits'}, 'N10': {'name': 'Total current liabilities', 'desc': 'Total liabilities due within one year', 'fraud_hint': ''}, 'N11': {'name': 'Total noncurrent liabilities', 'desc': 'Long-term liabilities due after more than one year', 'fraud_hint': ''}, 'N12': {'name': 'Total liabilities', 'desc': 'Total liabilities of the company', 'fraud_hint': ''}, 'N13': {'name': 'Paid-in capital', 'desc': 'Total capital actually contributed by shareholders', 'fraud_hint': ''}, 'N14': {'name': 'Retained earnings', 'desc': 'Total surplus reserves and undistributed profits', 'fraud_hint': 'A sudden decline may suggest earnings manipulation'}, 'N15': {'name': 'Long-term debt issuance', 'desc': 'Total long-term borrowings, bonds payable, and related items', 'fraud_hint': ''}, 'N16': {'name': 'Short-term debt', 'desc': 'Short-term borrowings and the current portion of noncurrent liabilities', 'fraud_hint': ''}, 'N17': {'name': 'Net long-term investments', 'desc': 'Total long-term equity and debt investments', 'fraud_hint': ''}, 'N18': {'name': 'Tradable shares', 'desc': 'Total shares tradable on the secondary market', 'fraud_hint': ''}, 'N19': {'name': 'Annual closing stock price', 'desc': 'Closing price on the last trading day of each year', 'fraud_hint': ''}, 'N20': {'name': 'Operating revenue', 'desc': 'Operating revenue recognized by the company', 'fraud_hint': 'Revenue inflation is a common form of fraud'}, 'N21': {'name': 'Operating cost', 'desc': 'Operating cost recognized by the company', 'fraud_hint': 'Divergent revenue and cost growth may suggest revenue inflation'}, 'N22': {'name': 'Finance expenses', 'desc': 'Interest and other financing expenses', 'fraud_hint': ''}, 'N23': {'name': 'Income tax expense', 'desc': 'Current income taxes payable', 'fraud_hint': ''}, 'N24': {'name': 'Net profit', 'desc': 'Net profit earned by the company', 'fraud_hint': 'A core indicator of earnings quality'}, 'N25': {'name': 'Gross profit', 'desc': 'Operating revenue less operating cost', 'fraud_hint': 'Investigate gross margins unusually high relative to peers'}, 'N26': {'name': 'Current ratio', 'desc': 'Current assets divided by current liabilities', 'fraud_hint': ''}, 'N27': {'name': 'Audit fees', 'desc': 'Total audit-related fees', 'fraud_hint': 'Abnormal audit fees may reflect audit risk'}, 'N28': {'name': 'Net profit excluding nonrecurring items', 'desc': 'Net profit after excluding nonrecurring gains and losses', 'fraud_hint': 'A highly predictive indicator; investigate large gaps from net profit'}, 'N29': {'name': 'Working capital turnover', 'desc': 'Operating revenue divided by working capital', 'fraud_hint': ''}, 'N30': {'name': 'Price-to-dividend ratio', 'desc': 'Closing price divided by after-tax dividend per share', 'fraud_hint': ''}, 'N31': {'name': 'Chairperson ownership ratio', 'desc': 'Proportion of company shares held by the chairperson', 'fraud_hint': ''}, 'N32': {'name': 'CEO age', 'desc': 'CEO age in the year; average if the CEO changes within the year', 'fraud_hint': ''}, 'N33': {'name': 'CEO compensation', 'desc': 'Total annual CEO compensation', 'fraud_hint': 'Investigate a substantial disconnect between compensation and performance'}, 'N34': {'name': 'Number of executives', 'desc': 'Total senior executives disclosed in the annual report', 'fraud_hint': ''}, 'N35': {'name': 'Management and supervisory shareholdings', 'desc': 'Shares held by directors, supervisors, and senior executives', 'fraud_hint': ''}, 'N36': {'name': 'Largest shareholder holdings', 'desc': 'Number of company shares held by the largest shareholder', 'fraud_hint': ''}, 'C1': {'name': 'Audit opinion', 'desc': '1=unqualified; 2=unqualified with explanatory paragraph; 3=qualified; 4=disclaimer; 5=adverse', 'fraud_hint': 'Nonstandard audit opinions are important warning signals'}, 'C2': {'name': 'CEO gender', 'desc': '1=male; 0=female', 'fraud_hint': ''}, 'C3': {'name': 'Internal control effectiveness', 'desc': '1=effective; 2=ineffective', 'fraud_hint': 'Ineffective internal controls are strongly associated with fraud'}, 'C4': {'name': 'Presence of internal control deficiencies', 'desc': '1=deficiencies present; 2=no deficiencies', 'fraud_hint': 'Internal control deficiencies can precede fraud'}}

def build_variable_semantics(available_variables: Optional[List[str]]=None, variable_semantics_override: Optional[Dict[str, Dict[str, str]]]=None) -> Dict[str, Dict[str, str]]:
    variables = available_variables or list(VARIABLE_SEMANTICS.keys())
    override = variable_semantics_override or {}
    semantics: Dict[str, Dict[str, str]] = {}
    for var in variables:
        if var in VARIABLE_SEMANTICS:
            base_info = dict(VARIABLE_SEMANTICS[var])
        else:
            prefix = str(var)[:1].upper()
            if prefix == 'N':
                name = f'Numerical metrics{str(var)[1:]}'
            elif prefix == 'C':
                name = f'Classification metrics{str(var)[1:]}'
            else:
                name = str(var)
            base_info = {'name': name, 'desc': 'Base feature in the dataset', 'fraud_hint': ''}
        merged = dict(base_info)
        extra = override.get(str(var), {})
        for field in ('name', 'desc', 'fraud_hint'):
            value = extra.get(field)
            if value is not None and str(value).strip():
                merged[field] = str(value).strip()
        semantics[str(var)] = merged
    return semantics

class MemorySystem:

    def __init__(self, memory_dir: str, stm_window: int=3, snapshot_memory_dir: Optional[str]=None, available_variables: Optional[List[str]]=None, variable_semantics: Optional[Dict[str, Dict[str, str]]]=None):
        self.memory_dir = memory_dir
        self.stm_window = stm_window
        self.ltm_file = os.path.join(memory_dir, 'long_term_memory.json')
        self.stm_dir = os.path.join(memory_dir, 'short_term')
        self.snapshot_memory_dir: Optional[str] = None
        self.snapshot_ltm_file: Optional[str] = None
        self.snapshot_stm_dir: Optional[str] = None
        if snapshot_memory_dir is not None:
            primary_abs = os.path.abspath(memory_dir)
            snapshot_abs = os.path.abspath(snapshot_memory_dir)
            if snapshot_abs != primary_abs:
                self.snapshot_memory_dir = snapshot_memory_dir
                self.snapshot_ltm_file = os.path.join(snapshot_memory_dir, 'long_term_memory.json')
                self.snapshot_stm_dir = os.path.join(snapshot_memory_dir, 'short_term')
                os.makedirs(self.snapshot_stm_dir, exist_ok=True)
        os.makedirs(self.stm_dir, exist_ok=True)
        self._ltm: Dict[str, Any] = {}
        self.available_variables = list(available_variables or VARIABLE_SEMANTICS.keys())
        self.variable_semantics_override = dict(variable_semantics or {})

    def _build_initial_ltm(self, baseline_importance: Optional[Dict]=None) -> Dict[str, Any]:
        return {'variable_semantics': build_variable_semantics(self.available_variables, self.variable_semantics_override), 'baseline_importance': baseline_importance or {}, 'experience_summaries': [], 'created_at': datetime.now().isoformat()}

    def _reset_memory_dir(self, target_dir: str) -> None:
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
            logger.info(f'Cleared memory directory: {target_dir}')

    def _copy_memory_tree(self, source_dir: str, target_dir: str) -> None:
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f'Memory snapshot directory does not exist: {source_dir}')
        if os.path.abspath(source_dir) == os.path.abspath(target_dir):
            return
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)

    def _normalize_loaded_ltm(self, baseline_importance: Optional[Dict]=None) -> None:
        old_vars = set(self._ltm.get('variable_semantics', {}).keys())
        new_vars = set(self.available_variables)
        if old_vars and old_vars != new_vars:
            removed = old_vars - new_vars
            added = new_vars - old_vars
            logger.warning(f"Dataset change detected: added variables {(sorted(added) if added else 'none')}, removed variables {(sorted(removed) if removed else 'none')}")
            old_summaries = self._ltm.get('experience_summaries', [])
            if old_summaries:
                logger.warning(f'Automatically cleared {len(old_summaries)} old experience summaries referencing unavailable variables')
                self._ltm['experience_summaries'] = []
            if 'shap_interactions' in self._ltm:
                logger.warning('Automatically cleared old SHAP interaction data')
                del self._ltm['shap_interactions']
            if 'sample_diagnostics' in self._ltm:
                logger.warning('Automatically cleared old sample diagnoses')
                del self._ltm['sample_diagnostics']
        self._ltm['variable_semantics'] = build_variable_semantics(self.available_variables, self.variable_semantics_override)
        if baseline_importance is not None:
            self._ltm['baseline_importance'] = baseline_importance
        self._ltm.setdefault('experience_summaries', [])
        self._ltm.setdefault('created_at', datetime.now().isoformat())

    def _ensure_snapshot_dirs(self) -> None:
        if self.snapshot_stm_dir is None:
            return
        os.makedirs(self.snapshot_stm_dir, exist_ok=True)

    def _sync_ltm_snapshot(self) -> None:
        if self.snapshot_ltm_file is None or not os.path.exists(self.ltm_file):
            return
        self._ensure_snapshot_dirs()
        shutil.copy2(self.ltm_file, self.snapshot_ltm_file)

    def _sync_stm_snapshot(self, round_file: str) -> None:
        if self.snapshot_stm_dir is None or not os.path.exists(round_file):
            return
        self._ensure_snapshot_dirs()
        snapshot_round_file = os.path.join(self.snapshot_stm_dir, os.path.basename(round_file))
        shutil.copy2(round_file, snapshot_round_file)

    def _sync_full_snapshot(self) -> None:
        if self.snapshot_memory_dir is None or not os.path.exists(self.memory_dir):
            return
        if os.path.exists(self.snapshot_memory_dir):
            shutil.rmtree(self.snapshot_memory_dir)
        shutil.copytree(self.memory_dir, self.snapshot_memory_dir)

    def _save_ltm(self) -> None:
        os.makedirs(os.path.dirname(self.ltm_file) or '.', exist_ok=True)
        with open(self.ltm_file, 'w', encoding='utf-8') as f:
            json.dump(self._ltm, f, indent=2, ensure_ascii=False)
        self._sync_ltm_snapshot()

    def _load_ltm(self) -> None:
        with open(self.ltm_file, 'r', encoding='utf-8') as f:
            self._ltm = json.load(f)

    def initialize_or_load(self, baseline_importance: Optional[Dict]=None, mode: str='fresh', bootstrap_memory_dir: Optional[str]=None) -> None:
        if mode not in {'fresh', 'continue_global', 'bootstrap_from_snapshot'}:
            raise ValueError(f'Unsupported memory initialization mode: {mode}')
        if mode == 'fresh':
            self._reset_memory_dir(self.memory_dir)
            os.makedirs(self.stm_dir, exist_ok=True)
            logger.info(f'Initializing new long-term memory: {self.ltm_file}')
            self._ltm = self._build_initial_ltm(baseline_importance=baseline_importance)
            self._save_ltm()
            return
        if mode == 'bootstrap_from_snapshot':
            if not bootstrap_memory_dir:
                raise ValueError('bootstrap_from_snapshot mode requires bootstrap_memory_dir')
            self._copy_memory_tree(bootstrap_memory_dir, self.memory_dir)
            os.makedirs(self.stm_dir, exist_ok=True)
            if os.path.exists(self.ltm_file):
                self._load_ltm()
                self._normalize_loaded_ltm(baseline_importance=baseline_importance)
                self._save_ltm()
            else:
                self._ltm = self._build_initial_ltm(baseline_importance=baseline_importance)
                self._save_ltm()
            self._sync_full_snapshot()
            return
        os.makedirs(self.stm_dir, exist_ok=True)
        if os.path.exists(self.ltm_file):
            logger.info(f'Loading existing long-term memory: {self.ltm_file}')
            self._load_ltm()
            self._normalize_loaded_ltm(baseline_importance=baseline_importance)
            self._save_ltm()
        else:
            logger.info(f'No existing long-term memory found; initializing: {self.ltm_file}')
            self._ltm = self._build_initial_ltm(baseline_importance=baseline_importance)
            self._save_ltm()
        self._sync_full_snapshot()

    def save_round(self, iteration: int, generated: List[Dict], deleted: List[Dict], pool_snapshot: Dict, parallel_info: Optional[Dict]=None) -> None:
        round_data = {'iteration': iteration, 'timestamp': datetime.now().isoformat(), 'generated': generated, 'deleted': deleted, 'pool_snapshot': pool_snapshot}
        if parallel_info is not None:
            round_data['parallel_info'] = parallel_info
        round_file = os.path.join(self.stm_dir, f'round_{iteration:04d}.json')
        with open(round_file, 'w', encoding='utf-8') as f:
            json.dump(round_data, f, indent=2, ensure_ascii=False)
        self._sync_stm_snapshot(round_file)
        return round_data

    def append_experience_summary(self, rounds_str: str, content: str) -> None:
        self._ltm.setdefault('experience_summaries', []).append({'rounds': rounds_str, 'timestamp': datetime.now().isoformat(), 'content': content})
        self._save_ltm()
        logger.info(f'Experience summary appended to long-term memory: round={rounds_str}')

    def load_recent_rounds(self, n: Optional[int]=None) -> List[Dict]:
        if n is None:
            n = self.stm_window
        if n <= 0:
            return []
        if not os.path.exists(self.stm_dir):
            return []
        files = sorted((f for f in os.listdir(self.stm_dir) if f.startswith('round_') and f.endswith('.json')))
        recent = files[-n:]
        rounds = []
        for fname in recent:
            fpath = os.path.join(self.stm_dir, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    rounds.append(json.load(f))
            except Exception as e:
                logger.warning(f'Failed to read short-term memory: {fpath} — {e}')
        return rounds

    def get_experience_summaries(self) -> List[Dict]:
        return self._ltm.get('experience_summaries', [])

    def format_variable_semantics(self) -> str:
        semantics: Dict = self._ltm.get('variable_semantics', build_variable_semantics(self.available_variables, self.variable_semantics_override))
        baseline: Dict = self._ltm.get('baseline_importance', {})
        all_vars = list(self.available_variables)

        def rank_key(var: str) -> float:
            return baseline.get(var, {}).get('rank', 9999)
        sorted_vars = sorted(all_vars, key=rank_key)
        lines = ['## Variable semantics (ranked by baseline predictive contribution)\n']
        lines.append('| Variable | Name | Description | Fraud warning cue | Baseline SHAP rank |')
        lines.append('|------|------|------|------------|------------|')
        for var in sorted_vars:
            info = semantics.get(var, {})
            name = info.get('name', '-')
            desc = info.get('desc', '-')
            hint = info.get('fraud_hint', '') or '-'
            b = baseline.get(var, {})
            r = b.get('rank', None)
            shap = b.get('shap_mean', None)
            if r is not None and shap is not None:
                rank_str = f'#{r} ({shap:.4f})'
            elif r is not None:
                rank_str = f'#{r}'
            else:
                rank_str = '-'
            lines.append(f'| {var} | {name} | {desc} | {hint} | {rank_str} |')
        return '\n'.join(lines)

    def format_baseline_importance(self) -> str:
        baseline: Dict = self._ltm.get('baseline_importance', {})
        if not baseline:
            return '## Baseline feature contributions\n(Not yet computed; populated after the first run)'
        semantics: Dict = self._ltm.get('variable_semantics', build_variable_semantics(self.available_variables, self.variable_semantics_override))
        sorted_items = sorted(baseline.items(), key=lambda x: x[1].get('rank', 9999))
        lines = ['## Baseline feature contributions (LightGBM SHAP, all variables in descending order)\n']
        lines.append('Prioritize highly ranked variables when constructing combinations; they have stronger baseline fraud discrimination.\n')
        lines.append('| Rank | Variable | Name | Mean SHAP | Single-factor AUC |')
        lines.append('|------|------|------|---------|----------|')
        for var, info in sorted_items:
            name = semantics.get(var, {}).get('name', '-')
            shap = info.get('shap_mean', 0)
            auc = info.get('auc_solo', 0)
            rank = info.get('rank', '?')
            lines.append(f'| #{rank} | {var} | {name} | {shap:.4f} | {auc:.4f} |')
        interactions = self._ltm.get('shap_interactions', [])
        if interactions:
            lines.append('\n### Top-20 SHAP interactions (joint predictive contribution of variable pairs)\n')
            lines.append('These variable pairs contribute additional predictive information jointly; prioritize them as inspiration for interaction features.\n')
            lines.append('| Variable 1 | Variable 2 | Interaction strength |')
            lines.append('|-------|-------|---------|')
            for item in interactions[:20]:
                lines.append(f"| {item['var1']} | {item['var2']} | {item['strength']:.4f} |")
        return '\n'.join(lines)

    def format_experience_summaries(self) -> str:
        summaries = self._ltm.get('experience_summaries', [])
        if not summaries:
            return '## Historical experience summary\n(First round; no prior experience)'
        lines = ['## Historical experience summary (accumulated across rounds)\n']
        for s in summaries:
            rnd = s.get('rounds', '?')
            ts = s.get('timestamp', '')[:10]
            content = s.get('content', '').strip()
            lines.append(f'### Round {rnd} experience ({ts}）')
            lines.append(content)
            lines.append('')
        return '\n'.join(lines)

    def format_dataset_profile(self, profile: dict) -> str:
        if not profile:
            return '## Dataset overview\n(Not yet computed)'
        lines = ['## Dataset overview\n']
        lines.append(f"- Training set: {profile['n_train']} samples, fraud rate={profile['fraud_rate_train']:.4f}")
        lines.append(f"- Validation set: {profile['n_valid']} samples, fraud rate={profile['fraud_rate_valid']:.4f}")
        sig_vars = profile.get('significant_vars', [])
        if sig_vars:
            lines.append(f'\n### Variables with significant fraud/nonfraud differences (Mann-Whitney U, p<0.01; total {len(sig_vars)})\n')
            lines.append("| Variable | Cohen's d | p-value | Fraud-group mean | Nonfraud-group mean |")
            lines.append('|------|----------|-----|----------|------------|')
            for sv in sig_vars[:30]:
                var = sv['var']
                vs = profile['variable_stats'].get(var, {})
                lines.append(f"| {var} | {sv['d']:.3f} | {sv['p']:.2e} | {vs.get('fraud_mean', 0):.4f} | {vs.get('normal_mean', 0):.4f} |")
        corr_pairs = profile.get('high_corr_pairs', [])
        if corr_pairs:
            lines.append(f'\n### Highly correlated variable pairs (|Spearman| > 0.7; total {len(corr_pairs)} pairs)\n')
            for cp in corr_pairs[:15]:
                lines.append(f"- {cp['var1']} ↔ {cp['var2']}: r={cp['corr']:.3f}")
        var_stats = profile.get('variable_stats', {})
        skewed = [(v, s) for v, s in var_stats.items() if 'skew' in s and abs(s.get('skew', 0)) > 2]
        if skewed:
            skewed.sort(key=lambda x: abs(x[1]['skew']), reverse=True)
            lines.append(f'\n### Highly skewed variables (|skew|>2; consider Log/Sqrt transforms)\n')
            for v, s in skewed[:10]:
                lines.append(f"- {v}: skew={s['skew']:.2f}, kurtosis={s.get('kurtosis', 0):.2f}")
        return '\n'.join(lines)

    def format_recent_rounds(self, n: Optional[int]=None) -> str:
        rounds = self.load_recent_rounds(n)
        if not rounds:
            return '## Recent iteration records (short-term memory)\n(No history; this is the first round)'
        lines = ['## Recent iteration records (short-term memory)\n']
        for rd in rounds:
            iteration = rd.get('iteration', '?')
            snapshot = rd.get('pool_snapshot', {})
            generated = rd.get('generated', [])
            deleted = rd.get('deleted', [])
            pool_size = snapshot.get('size', '?')
            best_auc = snapshot.get('best_auc', 0.0)
            best_auc_std = snapshot.get('best_auc_std', 0.0)
            fold_aucs = snapshot.get('fold_aucs', {})
            accepted = [g for g in generated if g.get('accepted', False)]
            rejected = [g for g in generated if not g.get('accepted', False)]
            lines.append(f'### Iteration {iteration}')
            lines.append(f'Generated {len(generated)} candidates -> accepted {len(accepted)}, rejected {len(rejected)} items')
            for g in accepted:
                contrib = g.get('auc_contribution', 0.0)
                reason = g.get('reason', '')
                fs = g.get('factor_stats', {})
                if fs:
                    lines.append(f"  ✓ {g['expr_str']}")
                    lines.append(f"    | AUC contribution: {contrib:+.4f} | Single-factor AUC: {fs.get('auc', 0.0):.4f} | Weight: {fs.get('weight', 0.0):.3f}")
                    lines.append(f"    | Effect Size: {fs.get('effect_size', 0.0):.4f} | t-test p: {fs.get('ttest_p', 1.0):.2e}")
                    lines.append(f"    | Fraud-group mean: {fs.get('fraud_mean', 0.0):.4f} | Nonfraud-group mean: {fs.get('non_fraud_mean', 0.0):.4f}")
                    lines.append(f"    | Skewness: {fs.get('skewness', 0.0):.2f} | Maximum within-pool correlation: {fs.get('max_corr', 0.0):.3f} | Missing rate: {fs.get('missing_rate', 0.0):.2%}")
                    lines.append(f'    | {reason}')
                else:
                    fold_contribs = g.get('fold_contributions', {})
                    if fold_contribs:
                        fc_strs = [f'F{i + 1}:{v:+.4f}' for i, (k, v) in enumerate(fold_contribs.items())]
                        stability = g.get('temporal_stability', 0.0)
                        lines.append(f"  ✓ {g['expr_str']} | CV mean {contrib:+.4f} ({', '.join(fc_strs)}) Stability={stability:.4f}")
                    else:
                        lines.append(f"  ✓ {g['expr_str']} | AUC contribution {contrib:+.4f} | {reason}")
            show_rejected = rejected[:5]
            for g in show_rejected:
                lines.append(f"  ✗ {g['expr_str']} | {g.get('reason', 'Unknown reason')}")
            if len(rejected) > 5:
                lines.append(f'  ... plus {len(rejected) - 5} rejected candidates (omitted)')
            if deleted:
                lines.append(f'Deleted: {len(deleted)} factors')
                for d in deleted:
                    delta = d.get('auc_delta', 0.0)
                    lines.append(f"  - {d['expr_str']} (AUC change {delta:+.4f})")
            top5 = snapshot.get('top5_factors', [])
            if top5:
                lines.append('Current Top-5 factors:')
                for f in top5:
                    expr = f.get('expr_str', '?')
                    w = f.get('weight', 0.0)
                    a = f.get('auc_solo', 0.0)
                    lines.append(f"  #{f.get('rank', '?')} {expr} | Weight={w:.3f} | Single-factor AUC={a:.4f}")
            cv_mean_auc = snapshot.get('cv_mean_auc', best_auc)
            robust_score = snapshot.get('robust_score', best_auc)
            if fold_aucs:
                fc_strs = [f'{k}:{v:.4f}' for k, v in fold_aucs.items()]
                lines.append(f"Pool status: {pool_size} factors, Robust={robust_score:.4f}, CV mean={cv_mean_auc:.4f}±{best_auc_std:.4f} ({', '.join(fc_strs)})")
            else:
                lines.append(f'Pool status: {pool_size} factors, AUC={best_auc:.4f}')
            lines.append('')
        return '\n'.join(lines)

    def format_structured_feedback(self, n: Optional[int]=None) -> str:
        rounds = self.load_recent_rounds(n)
        rows: List[Dict[str, Any]] = []
        for rd in rounds:
            for item in rd.get('generated', []):
                if 'evaluation_status' not in item:
                    continue
                rows.append(item)
        if not rows:
            return '## Structured candidate feedback\n(No candidate records with evaluation status yet)'
        return format_feedback_summary(summarize_feedback(rows))

    def save_ucb_stats(self, ucb_data: dict) -> None:
        self._ltm['ucb_stats'] = ucb_data
        self._save_ltm()

    def load_ucb_stats(self) -> Optional[dict]:
        return self._ltm.get('ucb_stats')

    def save_sample_diagnostics(self, iteration: int, llm_analysis: str, stats_summary: Dict[str, Any]) -> None:
        diag = self._ltm.setdefault('sample_diagnostics', {'latest_analysis': '', 'latest_stats_summary': {}, 'iteration': 0, 'history': []})
        diag['latest_analysis'] = llm_analysis
        diag['latest_stats_summary'] = stats_summary
        diag['iteration'] = iteration
        diag['history'].append({'iteration': iteration, 'analysis': llm_analysis, 'stats_summary': stats_summary, 'timestamp': datetime.now().isoformat()})
        if len(diag['history']) > 20:
            diag['history'] = diag['history'][-20:]
        self._save_ltm()
        logger.info(f'Sample diagnosis saved to long-term memory: iteration={iteration}')

    def get_latest_diagnostics_analysis(self) -> str:
        diag = self._ltm.get('sample_diagnostics', {})
        return diag.get('latest_analysis', '')

    def _describe_recent_trend(self, values: List[int], label: str) -> str:
        if len(values) < 2:
            return ''
        start = values[0]
        end = values[-1]
        if start == 0:
            direction = 'approximately unchanged' if end == 0 else 'increased'
        else:
            ratio = end / max(start, 1)
            if ratio >= 1.05:
                direction = 'increased'
            elif ratio <= 0.95:
                direction = 'decreased'
            else:
                direction = 'approximately unchanged'
        return f'- {label}: {start} → {end}，{direction}'

    def _format_sample_diagnostics_trend(self, history: List[Dict[str, Any]]) -> str:
        if len(history) < 5:
            return ''
        recent = history[-5:]
        lines: List[str] = []
        persistent_fn_values = [int(item.get('stats_summary', {}).get('transitions', {}).get('persistent_fn', 0)) for item in recent]
        if any((value > 0 for value in persistent_fn_values)):
            trend_line = self._describe_recent_trend(persistent_fn_values, 'persistent_fn over the last 5 rounds')
            if trend_line:
                lines.append(trend_line)
        fn_vars = [var for item in recent for var in item.get('stats_summary', {}).get('fn_vs_tp_top_vars', []) if var]
        if fn_vars:
            common_fn = ', '.join((name for name, _ in Counter(fn_vars).most_common(3)))
            lines.append(f'- Frequent FN difference variables: {common_fn}')
        fp_vars = [var for item in recent for var in item.get('stats_summary', {}).get('fp_tn_top_vars', []) if var]
        if fp_vars:
            common_fp = ', '.join((name for name, _ in Counter(fp_vars).most_common(3)))
            lines.append(f'- Frequent FP risk variables: {common_fp}')
        rescued_total = sum((int(item.get('stats_summary', {}).get('transitions', {}).get('rescued', 0)) for item in recent))
        regressed_total = sum((int(item.get('stats_summary', {}).get('transitions', {}).get('regressed', 0)) for item in recent))
        if rescued_total > 0 or regressed_total > 0:
            lines.append(f'- Cumulative rescued/regressed: {rescued_total}/{regressed_total} (net {rescued_total - regressed_total:+d}）')
        if not lines:
            return ''
        return '### Trend over the last 5 rounds\n' + '\n'.join(lines)

    def format_sample_diagnostics(self) -> str:
        diag = self._ltm.get('sample_diagnostics', {})
        analysis = diag.get('latest_analysis', '')
        stats = diag.get('latest_stats_summary', {})
        iteration = diag.get('iteration', '?')
        if not stats and (not analysis or not analysis.strip()):
            return ''
        lines = [f'## Sample-level misclassification diagnosis (round {iteration})']
        confusion = stats.get('confusion', {})
        tp = confusion.get('TP', 0)
        tn = confusion.get('TN', 0)
        fp = confusion.get('FP', 0)
        fn = confusion.get('FN', 0)
        total = tp + tn + fp + fn
        threshold = stats.get('threshold')
        lines.append('\n### Misclassification overview\n')
        lines.append('| Metric | Value |')
        lines.append('|------|----|')
        if threshold is not None:
            lines.append(f'| Threshold | {float(threshold):.4f} |')
        lines.append(f'| TP (true positive) | {tp} |')
        lines.append(f'| TN (true negative) | {tn} |')
        lines.append(f'| FP (false positive) | {fp} |')
        lines.append(f'| FN (false negative) | {fn} |')
        if tp + fp > 0:
            precision = tp / (tp + fp)
            lines.append(f'| Precision | {precision:.4f} |')
        if tp + fn > 0:
            recall = tp / (tp + fn)
            lines.append(f'| Recall | {recall:.4f} |')
        transitions = stats.get('transitions', {})
        if transitions:
            lines.append('\n### Comparison with the previous round\n')
            lines.append(f"- rescued (FN→TP): {transitions.get('rescued', 0)}")
            lines.append(f"- regressed (TP→FN): {transitions.get('regressed', 0)}")
            lines.append(f"- persistent_fn: {transitions.get('persistent_fn', 0)}")
            lines.append(f"- persistent_fp: {transitions.get('persistent_fp', 0)}")
        trend_text = self._format_sample_diagnostics_trend(diag.get('history', []))
        if trend_text:
            lines.append(f'\n{trend_text}')
        fn_prob = stats.get('fn_prob_distribution', {})
        if fn_prob and fn_prob.get('counts'):
            bins = fn_prob.get('bins', [])
            counts = fn_prob.get('counts', [])
            total_fn = fn_prob.get('total_fn', 0)
            lines.append('\n### FN predicted probability distribution\n')
            lines.append('| Probability interval | Count | Proportion |')
            lines.append('|----------|------|------|')
            for i in range(len(counts)):
                lo = bins[i] if i < len(bins) else 0
                hi = bins[i + 1] if i + 1 < len(bins) else 1
                c = counts[i]
                pct = f'{c / total_fn:.0%}' if total_fn > 0 else '0%'
                lines.append(f'| [{lo:.2f}, {hi:.2f}) | {c} | {pct} |')
        fn_tp_diff = stats.get('fn_vs_tp_diff', [])
        if fn_tp_diff:
            lines.append('\n### Top-10 variable differences: FN vs TP\n')
            lines.append('| Variable | FN mean | TP mean | Difference | Effect size |')
            lines.append('|------|--------|--------|------|-------------|')
            for item in fn_tp_diff[:10]:
                var = item.get('var', '?')
                fn_m = item.get('fn_mean', 0)
                tp_m = item.get('tp_mean', 0)
                diff = item.get('diff', fn_m - tp_m)
                es = item.get('effect_size', 0)
                lines.append(f'| {var} | {fn_m:.4f} | {tp_m:.4f} | {diff:+.4f} | {es:.4f} |')
        fn_tp_shap = stats.get('fn_tp_shap_diff', [])
        if fn_tp_shap:
            lines.append('\n### Top-10 SHAP attribution differences: FN vs TP\n')
            lines.append('| Variable | FN mean SHAP | TP mean SHAP | Difference |')
            lines.append('|------|-------------|-------------|------|')
            for item in fn_tp_shap[:10]:
                var = item.get('var', '?')
                fn_s = item.get('fn_shap_mean', 0)
                tp_s = item.get('tp_shap_mean', 0)
                diff = item.get('diff', fn_s - tp_s)
                lines.append(f'| {var} | {fn_s:.4f} | {tp_s:.4f} | {diff:+.4f} |')
        factor_eff = stats.get('factor_fn_effectiveness', [])
        if factor_eff:
            lines.append('\n### Discrimination of existing factors on FN samples\n')
            lines.append('| Factor | FN/TP AUC | FN mean | TP mean | Effect size |')
            lines.append('|------|-----------|--------|--------|-------------|')
            for item in factor_eff:
                expr = item.get('factor_expr', '?')
                auc = item.get('fn_tp_auc', 0)
                fn_m = item.get('fn_mean', 0)
                tp_m = item.get('tp_mean', 0)
                es = item.get('effect_size', 0)
                lines.append(f'| `{expr}` | {auc:.4f} | {fn_m:.4f} | {tp_m:.4f} | {es:.4f} |')
        fp_tn_diff = stats.get('fp_tn_diff', [])
        if fp_tn_diff:
            lines.append('\n### Top-5 false-positive profiles: FP vs TN\n')
            lines.append('| Variable | FP mean | TN mean | Difference | Effect size |')
            lines.append('|------|--------|--------|------|-------------|')
            for item in fp_tn_diff[:5]:
                var = item.get('var', '?')
                fp_m = item.get('fp_mean', 0)
                tn_m = item.get('tn_mean', 0)
                diff = item.get('diff', fp_m - tn_m)
                es = item.get('effect_size', 0)
                lines.append(f'| {var} | {fp_m:.4f} | {tn_m:.4f} | {diff:+.4f} | {es:.4f} |')
        rescued_factors = stats.get('rescued_factors', [])
        regressed_factors = stats.get('regressed_factors', [])
        if rescued_factors or regressed_factors:
            lines.append('\n### Factors associated with rescued/regressed samples\n')
            if rescued_factors:
                lines.append('**Factors associated with rescued samples (FN -> TP)**:')
                for f in rescued_factors[:3]:
                    expr = f.get('expr', f.get('factor_expr', '?'))
                    contrib = f.get('shap_contribution', f.get('contribution', 0))
                    lines.append(f'- `{expr}` | SHAP contribution: {contrib:+.4f}')
            if regressed_factors:
                lines.append('\n**Factors associated with regressed samples (TP -> FN)**:')
                for f in regressed_factors[:3]:
                    expr = f.get('expr', f.get('factor_expr', '?'))
                    contrib = f.get('shap_contribution', f.get('contribution', 0))
                    lines.append(f'- `{expr}` | SHAP contribution: {contrib:+.4f}')
        if analysis and analysis.strip():
            lines.append('\n### Integrated LLM analysis\n')
            lines.append(analysis.strip())
        return '\n'.join(lines) + '\n'

    def save_parallel_round(self, iteration: int, strategies: list, worker_results: list, review_records: list, pool_snapshot: dict) -> None:
        generated = []
        for rec in review_records:
            item = {'expr_str': rec.expr_str, 'accepted': rec.status == 'added', 'reason': rec.reason, 'auc_contribution': rec.auc_delta, 'candidate_id': rec.candidate_id, 'formula': rec.expr_str, 'background': getattr(rec, 'background', 'main_pool'), 'pool_hash': getattr(rec, 'pool_hash', ''), 'gate_stage': getattr(rec, 'gate_stage', 'unknown'), 'status': rec.status, 'evaluation_status': getattr(rec, 'evaluation_status', 'unmeasured'), 'auc_before': getattr(rec, 'auc_before', None), 'auc_after': getattr(rec, 'auc_after', None), 'delta_signed': getattr(rec, 'delta_signed', None), 'rejection_reason': getattr(rec, 'rejection_reason', rec.reason), 'evaluated': bool(getattr(rec, 'evaluated', False)), 'source_round': getattr(rec, 'source_round', iteration), 'trial_auc': getattr(rec, 'trial_auc', None), 'agent_id': rec.agent_id, 'strategy_name': rec.strategy_name}
            if hasattr(rec, 'rationale') and rec.rationale:
                item['rationale'] = rec.rationale
            generated.append(item)
        worker_summaries = []
        for wr in worker_results:
            worker_summaries.append({'agent_id': wr.agent_id, 'strategy_name': wr.strategy.strategy_name, 'raw_generated_count': wr.raw_generated_count, 'parsed_count': wr.parsed_count, 'screened_count': wr.screened_count})
        accepted_by_agent: dict = {}
        for rec in review_records:
            if rec.status == 'added':
                key = str(rec.agent_id)
                accepted_by_agent[key] = accepted_by_agent.get(key, 0) + 1
        raw_total = sum((wr.screened_count for wr in worker_results))
        unique_count = len(set((rec.expr_str for rec in review_records)))
        dedup_summary = {'raw_candidates': raw_total, 'unique_exact': unique_count, 'unique_structural': unique_count, 'review_queue_size': len(review_records)}
        parallel_info = {'n_agents': len(strategies), 'strategies': [{'strategy_name': s.strategy_name, 'allocation_role': s.allocation_role, 'fraud_pattern': s.fraud_pattern, 'operator_style': s.operator_style, 'focus_variables': s.focus_variables, 'complexity': s.complexity, 'specific_instruction': s.specific_instruction[:100]} for s in strategies], 'worker_summaries': worker_summaries, 'dedup_summary': dedup_summary, 'main_pool_review': [{'agent_id': rec.agent_id, 'strategy_name': rec.strategy_name, 'expr_str': rec.expr_str, 'status': rec.status, 'auc_delta': rec.auc_delta, 'reason': rec.reason, 'candidate_id': rec.candidate_id, 'formula': rec.expr_str, 'background': getattr(rec, 'background', 'main_pool'), 'pool_hash': getattr(rec, 'pool_hash', ''), 'gate_stage': getattr(rec, 'gate_stage', 'unknown'), 'evaluation_status': getattr(rec, 'evaluation_status', 'unmeasured'), 'auc_before': getattr(rec, 'auc_before', None), 'auc_after': getattr(rec, 'auc_after', None), 'delta_signed': getattr(rec, 'delta_signed', None), 'rejection_reason': getattr(rec, 'rejection_reason', rec.reason), 'evaluated': bool(getattr(rec, 'evaluated', False)), 'source_round': getattr(rec, 'source_round', iteration), 'trial_auc': getattr(rec, 'trial_auc', None)} for rec in review_records], 'accepted_by_agent': accepted_by_agent}
        feedback_rows = [item for item in generated if item.get('formula') and item.get('status') in {'added', 'rejected', 'error'}]
        parallel_info['structured_feedback'] = summarize_feedback(feedback_rows)
        return self.save_round(iteration=iteration, generated=generated, deleted=[], pool_snapshot=pool_snapshot, parallel_info=parallel_info)
