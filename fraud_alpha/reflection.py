"""Analyze pool coverage and propose exploration directions from iteration records."""
import re
import time
import logging
from typing import Dict, List, Optional
from fraud_alpha.expression import Feature
from fraud_alpha.operator_catalog import build_operator_regex, normalize_operator_name
logger = logging.getLogger(__name__)
_ALL_VARIABLES_FALLBACK = [f'N{i}' for i in range(1, 37)] + [f'C{i}' for i in range(1, 5)]

class ReflectionModule:
    FRAUD_PATTERNS = {'profitability': {'name': 'Earnings quality', 'key_vars': ['N20', 'N21', 'N24', 'N25', 'N28']}, 'balance_sheet': {'name': 'Balance sheet', 'key_vars': ['N3', 'N4', 'N5', 'N7', 'N10', 'N12', 'N26']}, 'financing': {'name': 'Financing structure', 'key_vars': ['N11', 'N15', 'N16']}, 'governance': {'name': 'Corporate governance', 'key_vars': ['C1', 'C3', 'C4', 'N27', 'N31', 'N34', 'N35', 'N36']}, 'temporal': {'name': 'Temporal anomalies', 'key_vars': []}}

    def __init__(self, force_exploration_threshold: int=2, strategy_change_threshold: int=3, high_freq_threshold: int=3, available_variables: Optional[List[str]]=None):
        self.force_exploration_threshold = force_exploration_threshold
        self.strategy_change_threshold = strategy_change_threshold
        self.high_freq_threshold = high_freq_threshold
        self.available_variables = list(available_variables or Feature.get_available_features())
        self.round_history: List[Dict] = []
        self.consecutive_zero_acceptance = 0
        self.consecutive_no_auc_improvement = 0

    def _get_available_variables(self) -> List[str]:
        return list(self.available_variables or Feature.get_available_features())

    def _get_active_patterns(self) -> Dict[str, Dict]:
        available_vars = set(self._get_available_variables())
        active_patterns: Dict[str, Dict] = {}
        for key, info in self.FRAUD_PATTERNS.items():
            if key == 'temporal':
                active_patterns[key] = dict(info)
                continue
            key_vars = [var for var in info['key_vars'] if var in available_vars]
            if key_vars:
                active_patterns[key] = {'name': info['name'], 'key_vars': key_vars}
        return active_patterns

    def record_round(self, added: int, deleted: int, pool_auc: float, acceptance_rate: float):
        prev_best_auc = max((r['pool_auc'] for r in self.round_history), default=0.0)
        record = {'iteration': len(self.round_history) + 1, 'added_count': added, 'deleted_count': deleted, 'pool_auc': pool_auc, 'acceptance_rate': acceptance_rate, 'timestamp': time.time()}
        self.round_history.append(record)
        if acceptance_rate == 0:
            self.consecutive_zero_acceptance += 1
        else:
            self.consecutive_zero_acceptance = 0
        if pool_auc <= prev_best_auc:
            self.consecutive_no_auc_improvement += 1
        else:
            self.consecutive_no_auc_improvement = 0
        logger.debug(f"Round {record['iteration']} record: added={added}, deleted={deleted}, AUC={pool_auc:.4f}, admission rate={acceptance_rate:.2%}, consecutive zero-admission rounds={self.consecutive_zero_acceptance}, consecutive rounds without improvement={self.consecutive_no_auc_improvement}")

    def classify_factor_pattern(self, expr_str: str) -> str:
        temporal_ops = ['Ref(', 'Delta(', 'Mean(', 'Std(', 'Sum(', 'EMA(', 'WMA(']
        has_temporal = any((op in expr_str for op in temporal_ops))
        best_pattern = 'other'
        best_count = 0
        for pattern_key, pattern_info in self._get_active_patterns().items():
            if pattern_key == 'temporal':
                continue
            count = sum((1 for v in pattern_info['key_vars'] if v in expr_str))
            if count > best_count:
                best_count = count
                best_pattern = pattern_key
        if best_count == 0 and has_temporal:
            return 'temporal'
        return best_pattern if best_count > 0 else 'temporal' if has_temporal else 'other'

    def analyze_variable_frequency(self, exprs: List[str]) -> Dict:
        all_variables = self._get_available_variables()
        var_counts: Dict[str, int] = {v: 0 for v in all_variables}
        for expr_str in exprs:
            for var in all_variables:
                pattern = '\\b' + re.escape(var) + '\\b'
                if re.search(pattern, expr_str):
                    var_counts[var] += 1
        high_freq = [(v, c) for v, c in var_counts.items() if c >= self.high_freq_threshold]
        high_freq.sort(key=lambda x: x[1], reverse=True)
        low_freq = [(v, c) for v, c in var_counts.items() if 1 <= c < self.high_freq_threshold]
        low_freq.sort(key=lambda x: x[1], reverse=True)
        unused = [v for v, c in var_counts.items() if c == 0]
        return {'high_freq': high_freq, 'low_freq': low_freq, 'unused': unused}

    def analyze_operator_distribution(self, exprs: List[str], overuse_threshold: float=0.3, min_pool_size: int=5) -> Dict:
        if not exprs:
            return {'root_op_counts': {}, 'root_op_share': {}, 'overrepresented': [], 'all_op_counts': {}}
        op_pattern = re.compile(build_operator_regex(require_call=False, include_legacy_aliases=True))
        root_counts: Dict[str, int] = {}
        all_op_counts: Dict[str, int] = {}
        for expr_str in exprs:
            root_match = re.match('(\\w+)\\(', expr_str.strip())
            if root_match:
                root_op = normalize_operator_name(root_match.group(1))
                root_counts[root_op] = root_counts.get(root_op, 0) + 1
            for op in op_pattern.findall(expr_str):
                op_name = normalize_operator_name(op)
                all_op_counts[op_name] = all_op_counts.get(op_name, 0) + 1
        total = len(exprs)
        root_share = {op: count / total for op, count in root_counts.items()}
        overrepresented = []
        if total >= min_pool_size:
            overrepresented = [(op, share, root_counts[op]) for op, share in root_share.items() if share > overuse_threshold]
            overrepresented.sort(key=lambda x: x[1], reverse=True)
        return {'root_op_counts': root_counts, 'root_op_share': root_share, 'overrepresented': overrepresented, 'all_op_counts': all_op_counts}

    def get_saturated_patterns(self, exprs: List[str], top_n: int=3) -> List[tuple]:
        if not exprs:
            return []
        template_counts: Dict[str, int] = {}
        for expr_str in exprs:
            template = re.sub('\\bN\\d+\\b', 'Nx', expr_str)
            template = re.sub('\\bC\\d+\\b', 'Cx', template)
            template = re.sub('(?<!\\w)\\d+\\.?\\d*(?!\\w*y)', 'const', template)
            template_counts[template] = template_counts.get(template, 0) + 1
        sorted_templates = sorted(template_counts.items(), key=lambda x: x[1], reverse=True)
        return [(t, c) for t, c in sorted_templates[:top_n] if c > 1]

    def should_force_exploration(self) -> bool:
        return self.consecutive_zero_acceptance >= self.force_exploration_threshold

    def should_suggest_strategy_change(self) -> bool:
        return self.consecutive_no_auc_improvement >= self.strategy_change_threshold

    def get_strategy_suggestions(self) -> List[str]:
        return ['Increase expression complexity (try more than 3 nesting levels)', 'Explore unused variables (see the low-frequency variable list above)', 'Try new operator combinations, such as temporal and binary operators']

    def generate_reflection(self, factor_stats: List[Dict], pool_state: Dict) -> str:
        lines = []
        lines.append('=== Reflection and analysis ===\n')
        if self.round_history:
            latest = self.round_history[-1]
            lines.append(f"[Round results] Iteration {latest['iteration']}:")
            lines.append(f"  - Added factors: {latest['added_count']} items")
            lines.append(f"  - Deleted factors: {latest['deleted_count']} items")
            if len(self.round_history) >= 2:
                prev_auc = self.round_history[-2]['pool_auc']
                auc_change = latest['pool_auc'] - prev_auc
                direction = '↑' if auc_change > 0 else '↓' if auc_change < 0 else '→'
                lines.append(f"  - Combined AUC: {latest['pool_auc']:.4f} ({direction}{abs(auc_change):.4f})")
            else:
                lines.append(f"  - Combined AUC: {latest['pool_auc']:.4f}")
            lines.append(f"  - Admission rate: {latest['acceptance_rate']:.1%}")
        lines.append('')
        exprs = [s['expr_str'] for s in factor_stats if 'expr_str' in s]
        active_patterns = self._get_active_patterns()
        pattern_counts: Dict[str, int] = {k: 0 for k in active_patterns}
        pattern_counts['other'] = 0
        for expr_str in exprs:
            category = self.classify_factor_pattern(expr_str)
            if category in pattern_counts:
                pattern_counts[category] += 1
            else:
                pattern_counts['other'] = pattern_counts.get('other', 0) + 1
        lines.append('[Fraud-pattern coverage]')
        lines.append('| Pattern category | Factor count | Coverage status |')
        lines.append('|----------|----------|----------|')
        for pattern_key, pattern_info in active_patterns.items():
            count = pattern_counts.get(pattern_key, 0)
            status = 'Covered' if count > 0 else 'Not covered'
            lines.append(f"| {pattern_info['name']} | {count} | {status} |")
        if pattern_counts.get('other', 0) > 0:
            lines.append(f"| Other | {pattern_counts['other']} | - |")
        lines.append('')
        freq_analysis = self.analyze_variable_frequency(exprs)
        lines.append('[Variable usage frequency]')
        if freq_analysis['high_freq']:
            high_str = ', '.join((f'{v}({c} occurrences)' for v, c in freq_analysis['high_freq'][:10]))
            lines.append(f'  Frequent variables (>={self.high_freq_threshold} occurrences): {high_str}')
        else:
            lines.append(f'  Frequent variables (>={self.high_freq_threshold} occurrences): none')
        if freq_analysis['low_freq']:
            low_str = ', '.join((f'{v}({c} occurrences)' for v, c in freq_analysis['low_freq'][:10]))
            lines.append(f'  Low-frequency variables (1-{self.high_freq_threshold - 1} occurrences): {low_str}')
        else:
            lines.append(f'  Low-frequency variables (1-{self.high_freq_threshold - 1} occurrences): none')
        if freq_analysis['unused']:
            unused_str = ', '.join(freq_analysis['unused'][:15])
            remaining = len(freq_analysis['unused']) - 15
            if remaining > 0:
                unused_str += f" and others (total {len(freq_analysis['unused'])})"
            lines.append(f'  Unused variables: {unused_str}')
        else:
            lines.append('  Unused variables: none (all variables have been used)')
        lines.append('')
        op_dist = self.analyze_operator_distribution(exprs)
        if op_dist['root_op_counts']:
            root_sorted = sorted(op_dist['root_op_share'].items(), key=lambda x: x[1], reverse=True)
            lines.append('[Root operator distribution]')
            dist_str = ', '.join((f"{op}:{op_dist['root_op_counts'][op]}({share:.0%})" for op, share in root_sorted[:8]))
            lines.append(f'  {dist_str}')
            if op_dist['overrepresented']:
                for op, share, count in op_dist['overrepresented']:
                    lines.append(f'  Operator {op} share {share:.0%}（{count}/{len(exprs)}) is overused; new factors should avoid {op} as the root operator')
            lines.append('')
        if op_dist['all_op_counts'] and len(exprs) >= 5:
            OPERATOR_CATEGORIES = {'Temporal trend': {'ops': {'Delta', 'ROC'}, 'hint': 'Capture temporal changes in variables (year-on-year or adjacent-record changes)'}, 'Volatility': {'ops': {'Std', 'Var', 'Range'}, 'hint': 'Capture variable volatility and dispersion'}, 'Co-movement': {'ops': {'Corr', 'Cov'}, 'hint': 'Capture co-movement between variables'}, 'Conditional logic': {'ops': {'If'}, 'hint': 'Nonlinear features based on conditional branches'}, 'Standardization': {'ops': {'ZScore'}, 'hint': 'Detect standardized deviations'}, 'Cross-sectional rank': {'ops': {'CSRank'}, 'hint': 'Relative position across samples'}}
            all_ops_used = set(op_dist['all_op_counts'].keys())
            blind_spots = []
            for cat_name, cat_info in OPERATOR_CATEGORIES.items():
                if not cat_info['ops'] & all_ops_used:
                    ops_str = '/'.join(sorted(cat_info['ops']))
                    blind_spots.append(f"  - {ops_str}: {cat_info['hint']}")
            if blind_spots:
                lines.append('[Operator blind spots: categories absent from the current pool]')
                lines.append('The following operators are underused; prioritize them:')
                for bs in blind_spots:
                    lines.append(bs)
                lines.append('')
        saturated = self.get_saturated_patterns(exprs)
        if saturated:
            lines.append('[Saturated structural patterns: avoid repetition]')
            lines.append('The following structural patterns occur repeatedly in the pool; similar structures will be rejected this round:')
            for template, count in saturated:
                lines.append(f'  - {template} (already {count} variants)')
            lines.append('Use substantially different operator nesting structures.')
            lines.append('')
        if self.should_force_exploration():
            lines.append('[Forced exploration mode]')
            lines.append(f'  For {self.consecutive_zero_acceptance} consecutive rounds the admission rate was zero; entering forced exploration')
            lines.append('  Requirements:')
            lines.append('  1. Substantially change the generation strategy to avoid repeating prior patterns')
            lines.append('  2. Prioritize unused and low-frequency variables listed above')
            lines.append('  3. Try new operator combinations and expression structures')
            lines.append('')
        if self.should_suggest_strategy_change():
            lines.append(f'[Strategy suggestion] For {self.consecutive_no_auc_improvement} rounds AUC has not improved; consider the following strategies:')
            for i, suggestion in enumerate(self.get_strategy_suggestions(), 1):
                lines.append(f'  {i}. {suggestion}')
            lines.append('')
        return '\n'.join(lines)

    @staticmethod
    def get_iteration_mode(iteration: int, cold_start_rounds: int=5, explore_interval: int=4, diversify_interval: int=8, no_improve_counter: int=0) -> str:
        if iteration <= cold_start_rounds:
            return 'explore'
        if no_improve_counter >= 20:
            return 'crossover'
        if no_improve_counter >= 10:
            cycle = iteration % 4
            if cycle < 2:
                return 'explore'
            elif cycle == 2:
                return 'diversify'
            else:
                return 'exploit'
        if diversify_interval > 0 and iteration % diversify_interval == 0:
            return 'diversify'
        if explore_interval > 0 and iteration % explore_interval == 0:
            return 'explore'
        return 'exploit'

    def get_uncovered_patterns(self, factor_stats: List[Dict]) -> List[str]:
        exprs = [s.get('expr_str', '') for s in factor_stats if 'expr_str' in s]
        active_patterns = self._get_active_patterns()
        pattern_counts = {k: 0 for k in active_patterns}
        for expr_str in exprs:
            category = self.classify_factor_pattern(expr_str)
            if category in pattern_counts:
                pattern_counts[category] += 1
        uncovered = []
        for key, info in active_patterns.items():
            if pattern_counts.get(key, 0) == 0 and info['key_vars']:
                uncovered.append(info['name'])
        return uncovered

    def get_underrepresented_vars(self, factor_stats: List[Dict]) -> List[str]:
        exprs = [s.get('expr_str', '') for s in factor_stats if 'expr_str' in s]
        freq = self.analyze_variable_frequency(exprs)
        result = freq['unused'].copy()
        result.extend([v for v, _ in freq['low_freq']])
        return result

    def get_required_exploration_vars(self, factor_stats: List[Dict], n_required: int=3) -> List[str]:
        exprs = [s.get('expr_str', '') for s in factor_stats if 'expr_str' in s]
        freq = self.analyze_variable_frequency(exprs)
        candidates = freq['unused'].copy()
        candidates.extend([v for v, _ in freq['low_freq']])
        return candidates[:n_required]

    @staticmethod
    def expr_uses_any_var(expr_str: str, required_vars: List[str]) -> bool:
        for var in required_vars:
            pattern = '\\b' + re.escape(var) + '\\b'
            if re.search(pattern, expr_str):
                return True
        return False
