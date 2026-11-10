"""Summarize accepted and rejected feature-construction patterns."""
import time
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional
from fraud_alpha.expression import Expression

class FeatureDiscoverySummary:

    def __init__(self):
        self.accepted_factors: List[Dict] = []
        self.rejection_stats: Dict[str, int] = {'auc_too_low': 0, 'corr_too_high': 0, 'calc_error': 0, 'pool_full': 0, 'other': 0}
        self.variable_pair_counts: Counter = Counter()
        self.operator_pattern_counts: Counter = Counter()

    def record_accepted(self, expr: Expression, auc: float, operator_type: str, variables: List[str]) -> None:
        self.accepted_factors.append({'expr_str': str(expr), 'auc': auc, 'operator_type': operator_type, 'variables': variables, 'accepted_at': time.time()})
        sorted_vars = sorted(set(variables))
        for i in range(len(sorted_vars)):
            for j in range(i + 1, len(sorted_vars)):
                self.variable_pair_counts[sorted_vars[i], sorted_vars[j]] += 1
        self.operator_pattern_counts[operator_type] += 1

    def record_rejected(self, expr: Expression, reason: str) -> None:
        reason_lower = reason.lower()
        if 'auc' in reason_lower or 'below' in reason_lower or 'threshold' in reason_lower:
            self.rejection_stats['auc_too_low'] += 1
        elif 'corr' in reason_lower or 'correlation' in reason_lower or 'duplicate' in reason_lower:
            self.rejection_stats['corr_too_high'] += 1
        elif 'error' in reason_lower or 'failed' in reason_lower or 'error' in reason_lower:
            self.rejection_stats['calc_error'] += 1
        elif 'full' in reason_lower or 'full' in reason_lower or 'capacity' in reason_lower:
            self.rejection_stats['pool_full'] += 1
        else:
            self.rejection_stats['other'] += 1

    def get_top_variable_pairs(self, n: int=10) -> List[Tuple]:
        return self.variable_pair_counts.most_common(n)

    def get_top_operator_patterns(self, n: int=10) -> List[Tuple]:
        return self.operator_pattern_counts.most_common(n)

    def format_for_prompt(self) -> str:
        if not self.accepted_factors and sum(self.rejection_stats.values()) == 0:
            return 'No historical records'
        lines = []
        lines.append(f'Accepted factor count: {len(self.accepted_factors)}')
        total_rejected = sum(self.rejection_stats.values())
        if total_rejected > 0:
            lines.append(f'Total rejected factors: {total_rejected}')
            for reason, count in self.rejection_stats.items():
                if count > 0:
                    lines.append(f'  - {reason}: {count}')
        top_pairs = self.get_top_variable_pairs(5)
        if top_pairs:
            lines.append('Effective variable combinations (Top 5):')
            for pair, count in top_pairs:
                lines.append(f'  - {pair[0]} + {pair[1]}: {count} times')
        top_ops = self.get_top_operator_patterns(5)
        if top_ops:
            lines.append('Effective operator patterns (Top 5):')
            for op, count in top_ops:
                lines.append(f'  - {op}: {count} times')
        return '\n'.join(lines)

    def update(self) -> None:
        pass

def extract_variables(expr: Expression) -> List[str]:
    expr_str = str(expr)
    return re.findall('\\b([NC]\\d+)\\b', expr_str)

def extract_operator_type(expr: Expression) -> str:
    return type(expr).__name__
