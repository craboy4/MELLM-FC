"""English prompts for new feature-generation runs; not exact historical prompt replay."""
from typing import Dict, List, Optional, Tuple
from fraud_alpha.expression import Feature
from fraud_alpha.operator_catalog import build_operator_regex, normalize_operator_name
from fraud_alpha.prompt_assets import _COMBINATION_EXAMPLE_BLOCK, _COMPLEXITY_BLOCK, _OPERATOR_BLOCK, _OUTPUT_BLOCK, _build_example_expressions
_DYNAMIC_PRIORITY_VARIABLES = ('N24', 'N16', 'N14', 'N13', 'N3', 'N22', 'N6', 'N15', 'N1', 'N8')

def _split_available_variables(variables: List[str]) -> Tuple[List[str], List[str]]:
    numeric = [var for var in variables if str(var).upper().startswith('N')]
    categorical = [var for var in variables if var not in numeric]
    return (numeric, categorical)

def _build_priority_variable_list(available_variables: List[str], max_items: int=5) -> List[str]:
    selected: List[str] = []
    for candidate in _DYNAMIC_PRIORITY_VARIABLES:
        if candidate in available_variables and candidate not in selected:
            selected.append(candidate)
        if len(selected) >= max_items:
            return selected
    for variable in available_variables:
        if variable not in selected:
            selected.append(variable)
        if len(selected) >= max_items:
            break
    return selected

def _build_detailed_feature_hint(available_variables: Optional[List[str]]=None) -> str:
    source_variables = available_variables if available_variables is not None else Feature.get_available_features() or []
    variables = [str(var) for var in source_variables]
    priority_vars = _build_priority_variable_list(variables)
    if not priority_vars:
        return 'Prioritize informative available variables when constructing interaction, growth-rate, volatility, and correlation features'
    return f"Prioritize key available variables in combinations (such as {', '.join(priority_vars)}), emphasizing interactions, rates of change, volatility, and correlations"

def _build_runtime_system_prompt_lines(available_variables: Optional[List[str]]=None) -> List[str]:
    source_variables = available_variables if available_variables is not None else Feature.get_available_features() or []
    variables = [str(var) for var in source_variables]
    numeric_features, categorical_features = _split_available_variables(variables)
    example_expressions = _build_example_expressions(numeric_features, categorical_features)
    priority_vars = _build_priority_variable_list(variables)
    lines: List[str] = ["You are an experienced financial-fraud detection expert. Iteratively design formulaic alpha factors to improve the fraud prediction model's AUC.", '', '# Role', '', 'Each iteration provides:', '- The currently available base variables; use only these variables', '- Variable semantics and baseline predictive-contribution rankings, if supplied in the user prompt', '- Historical experience and recent iteration feedback, if supplied in the user prompt', '- Current pool analysis and reflection suggestions, if supplied in the user prompt', '', 'Use this information to propose additions (ADD) and optional deletions (DELETE).', '', '# Currently available base variables', '']
    if numeric_features:
        lines.append(f"- Numerical variables: {', '.join(numeric_features)}")
    if categorical_features:
        lines.append(f"- Categorical variables: {', '.join(categorical_features)}")
    if not variables:
        lines.append("- No base-variable list is currently supplied; use only variable names explicitly provided in this round's prompt.")
    lines.extend(['', '# Design priorities', ''])
    if priority_vars:
        lines.append(f"- Prioritize combinations of these available variables: {', '.join(priority_vars)}")
    lines.append('- Prefer structurally distinct factors rather than changing only a window or a similar variable.')
    lines.append('- Use ratios, differences, volatility, rates of change, correlations, and conditional logic to characterize anomalies.')
    lines.append('- At least half the factors should involve multivariable interactions, composite temporal patterns, or conditional logic.')
    if example_expressions:
        lines.extend(['', '# Dynamic examples', ''])
        for item in example_expressions:
            lines.append(f"- `{item['expr']}` — {item['reason']}")
    lines.extend(['', _OPERATOR_BLOCK.strip(), '', _COMBINATION_EXAMPLE_BLOCK.strip(), '', _COMPLEXITY_BLOCK.strip(), '', _OUTPUT_BLOCK.strip(), '', '# Additional constraints', ''])
    if variables:
        lines.append(f"This run may use only these base variables:{', '.join(variables)}")
        lines.append('Do not use variables outside the list above.')
    else:
        lines.append('If a subsequent prompt supplies an allowlist of base variables, use only those variables.')
    return lines
SYSTEM_PROMPT = ''
DETAILED_SYSTEM_PROMPT = ''

def format_pool_state(pool_state: dict) -> str:
    size = pool_state.get('size', 0)
    best_auc = pool_state.get('best_auc', 0.0)
    exprs = pool_state.get('exprs', [])
    weights = pool_state.get('weights', [])
    aucs = pool_state.get('aucs', [])
    lines = [f'Current factor pool status:', f'- Factor count: {size}', f'- Combined AUC: {best_auc:.4f}', '', 'Factor details:']
    for i, (expr, weight, auc) in enumerate(zip(exprs, weights, aucs), 1):
        lines.append(f'{i}. {expr}')
        lines.append(f'   AUC={auc:.4f}, weight={weight:.4f}')
    return '\n'.join(lines)

def create_generation_prompt(n_factors: int, pool_state: Optional[dict]=None, use_detailed: bool=False) -> str:
    feature_hint = _build_detailed_feature_hint()
    if pool_state is None:
        prompt = f'Generate {n_factors} factor expressions for financial-fraud detection.\n\n'
        if use_detailed:
            prompt += 'Requirements:\n'
            prompt += f'1. {feature_hint}\n'
            prompt += '2. Cover multiple fraud patterns, including earnings quality, balance sheets, financing structure, and governance.\n'
            prompt += '3. Mix basic, temporal, and two-variable operators.\n'
            prompt += '4. Include structurally distinct features at different complexity levels.\n\n'
        prompt += 'Put each factor on a separate line containing only the expression, without numbering or explanation.'
        return prompt
    prompt = format_pool_state(pool_state) + '\n\n'
    prompt += f'Based on the current factor pool, generate {n_factors} new factor expressions to improve combined AUC.\n\n'
    prompt += 'New factors should:\n'
    prompt += '1. Avoid similarity to low-weight factors.\n'
    prompt += '2. Extend variable relationships or operator combinations not covered by current high-weight factors.\n'
    prompt += '3. Consider temporal and cross-feature relationships.\n'
    if use_detailed:
        prompt += f'4. {feature_hint}\n'
        prompt += '5. Prioritize structures substantially different from frequently used templates.\n'
    prompt += '\nPut each factor on a separate line containing only its expression, without numbering or explanation.'
    return prompt

def create_initial_prompt(n_factors: int, use_detailed: bool=False) -> str:
    feature_hint = _build_detailed_feature_hint()
    if use_detailed:
        prompt = f'Generate {n_factors} formulaic alpha expressions for financial-fraud detection.\n\n'
        prompt += 'Requirements:\n'
        prompt += f'1. {feature_hint}\n'
        prompt += '2. Cover multiple fraud patterns, including earnings quality, balance sheets, financing structure, and governance.\n'
        prompt += '3. Mix basic, temporal, and two-variable operators.\n'
        prompt += '4. Avoid minor adjustments to old templates; prioritize structurally distinct combinations.\n\n'
        prompt += 'Put each factor on a separate line containing only the expression.'
        return prompt
    return f'Generate {n_factors} formulaic alpha expressions for financial-fraud detection. Put each expression on a separate line, with no other text.'

def create_feedback_prompt(n_factors: int, pool_state: dict, discovery_summary: str, recent_rejections: Optional[list]=None, top_weighted_factors: Optional[list]=None, operator_frequency: Optional[dict]=None) -> str:
    lines = [format_pool_state(pool_state), '', '## Feature discovery summary', discovery_summary]
    if top_weighted_factors:
        lines.append('')
        lines.append('## Current high-weight factors')
        for item in top_weighted_factors[:5]:
            expr_str = item.get('expr_str', '?')
            weight = item.get('weight', 0.0)
            auc = item.get('auc', 0.0)
            lines.append(f'- `{expr_str}` (weight={weight:.4f}, auc={auc:.4f})')
    if operator_frequency:
        top_ops = sorted(operator_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        if top_ops:
            lines.append('')
            lines.append('## Currently frequent operators')
            lines.append(', '.join((f'{name}:{count}' for name, count in top_ops)))
    if recent_rejections:
        lines.append('')
        lines.append('## Recently rejected factors')
        for item in recent_rejections[:10]:
            expr_str = item.get('expr_str', '?')
            reason = item.get('reason', 'Unknown reason')
            lines.append(f'- `{expr_str}`: {reason}')
    lines.append('')
    lines.append(f'Based on the information above, generate {n_factors} new factor expressions.')
    lines.append('New factors should:')
    lines.append(f'1. {_build_detailed_feature_hint()}')
    lines.append('2. Extend variable relationships or operator combinations not covered by current high-weight factors.')
    lines.append('3. Avoid recently rejected or clearly saturated structures.')
    lines.append('4. Put each expression on a separate line, without numbering or explanation.')
    return '\n'.join(lines)
_GENERIC_SYSTEM_PROMPT = '\n'.join(_build_runtime_system_prompt_lines([]))
SYSTEM_PROMPT = _GENERIC_SYSTEM_PROMPT
DETAILED_SYSTEM_PROMPT = _GENERIC_SYSTEM_PROMPT
UNIFIED_SYSTEM_PROMPT = _GENERIC_SYSTEM_PROMPT
COMPRESSION_SYSTEM_PROMPT = 'You are a rigorous experimental-analysis assistant focused on extracting reusable knowledge from feature-engineering records.'

def build_runtime_system_prompt(available_variables: Optional[List[str]]=None) -> str:
    return '\n'.join(_build_runtime_system_prompt_lines(available_variables))

def format_pool_full_analysis(factor_stats: List[Dict], pool_summary: Dict, iteration_mode: str='exploit', top_full_display: int=3, pool_correlations: Optional[List[Dict]]=None, success_patterns: Optional[Dict]=None, coverage_stats: Optional[Dict]=None, correlation_clusters: Optional[List[List[str]]]=None) -> str:
    if not factor_stats:
        return '## Current factor pool\n(The pool is empty)'
    pool_size = pool_summary.get('size', len(factor_stats))
    best_auc = pool_summary.get('best_auc', 0.0)
    _temporal_cv_on = pool_summary.get('temporal_cv_enabled', False)
    _cv_mean = pool_summary.get('cv_mean_auc', best_auc)
    _cv_std = pool_summary.get('best_auc_std', 0.0)
    _fold_aucs = pool_summary.get('fold_aucs', {})
    if _temporal_cv_on:
        _auc_label = f'Robust Score={best_auc:.4f}, CV mean={_cv_mean:.4f}±{_cv_std:.4f}'
        if _fold_aucs:
            _fold_parts = [f'{k}:{v:.4f}' for k, v in _fold_aucs.items()]
            _auc_label += f" ({', '.join(_fold_parts)})"
    else:
        _auc_label = f'Combined AUC={best_auc:.4f}'
    import re as _re
    all_vars = Feature.get_available_features()
    op_pattern = _re.compile(build_operator_regex(require_call=True, include_legacy_aliases=True))
    op_counter: Dict[str, int] = {}
    var_counter: Dict[str, int] = {}
    for f in factor_stats:
        expr_str = f.get('expr_str', '')
        for m in op_pattern.findall(expr_str):
            op_name = normalize_operator_name(m)
            op_counter[op_name] = op_counter.get(op_name, 0) + 1
        for var in all_vars:
            if _re.search('\\b' + _re.escape(var) + '\\b', expr_str):
                var_counter[var] = var_counter.get(var, 0) + 1
    used = sorted([(v, c) for v, c in var_counter.items() if c > 0], key=lambda x: x[1], reverse=True)
    unused = [v for v in all_vars if v not in var_counter]
    if iteration_mode == 'crossover':
        sorted_by_weight = sorted(factor_stats, key=lambda x: x.get('weight', 0.0), reverse=True)
        top_parents = sorted_by_weight[:5]
        lines = [f'## Current factor pool (crossover mode)\n', f'- Factor count: {pool_size}', f'- {_auc_label}\n']
        lines.append('### Crossover parents (Top-5 high-weight factors; recombine their subexpressions)\n')
        lines.append('| # | Expression | Weight | Single-factor AUC |')
        lines.append('|---|--------|------|-----------|')
        for rank, f in enumerate(top_parents, 1):
            expr = f.get('expr_str', '?')
            w = f.get('weight', 0.0)
            auc = f.get('auc', 0.0)
            lines.append(f'| {rank} | {expr} | {w:.3f} | {auc:.4f} |')
        lines.append('')
        if unused:
            lines.append(f'### Unused variables ({len(unused)}; prioritize in crossover offspring)')
            lines.append(f"**{', '.join(unused[:20])}**")
        root_op_counts_cx: Dict[str, int] = {}
        for f in factor_stats:
            root_match = _re.match('(\\w+)\\(', f.get('expr_str', '').strip())
            if root_match:
                root_op = root_match.group(1)
                root_op_counts_cx[root_op] = root_op_counts_cx.get(root_op, 0) + 1
        overused_cx = []
        if pool_size >= 5:
            for op, count in root_op_counts_cx.items():
                share = count / pool_size
                if share > 0.3:
                    overused_cx.append((op, share, count))
            overused_cx.sort(key=lambda x: x[1], reverse=True)
        if overused_cx:
            lines.append('')
            lines.append('### Overused root operators (avoid in crossover offspring)')
            for op, share, count in overused_cx:
                lines.append(f'- **{op}**: {count}/{pool_size} ({share:.0%})')
        if op_counter:
            top_ops = sorted(op_counter.items(), key=lambda x: x[1], reverse=True)[:8]
            ops_str = ', '.join((f'{k}:{v}' for k, v in top_ops))
            lines.append(f'\n- Used operator distribution: {ops_str}')
        if used:
            lines.append(f"- Used variables ({len(used)}): {', '.join((v for v, _ in used[:12]))}")
        return '\n'.join(lines)
    if iteration_mode == 'explore':
        lines = [f'## Current factor pool summary (exploration mode; expressions hidden)\n', f'- Factor count: {pool_size}', f'- {_auc_label}']
        root_op_counts: Dict[str, int] = {}
        all_exprs_str = []
        for f in factor_stats:
            expr_str = f.get('expr_str', '')
            all_exprs_str.append(expr_str)
            root_match = _re.match('(\\w+)\\(', expr_str.strip())
            if root_match:
                root_op = root_match.group(1)
                root_op_counts[root_op] = root_op_counts.get(root_op, 0) + 1
        overused_ops = []
        if pool_size >= 5:
            for op, count in root_op_counts.items():
                share = count / pool_size
                if share > 0.3:
                    overused_ops.append((op, share, count))
            overused_ops.sort(key=lambda x: x[1], reverse=True)
        if op_counter:
            overused_set = {op for op, _, _ in overused_ops}
            total_ops = sum(op_counter.values())
            top_ops = sorted(op_counter.items(), key=lambda x: x[1], reverse=True)[:8]
            parts = []
            for k, v in top_ops:
                marker = '[!]' if k in overused_set else ''
                parts.append(f'{marker}{k}:{v}')
            lines.append(f"- Used operator distribution: {', '.join(parts)}")
        if overused_ops:
            lines.append('')
            lines.append('**The following operators are overused (more than 30% of root operators) and are strictly prohibited as outermost operators this round**:')
            for op, share, count in overused_ops:
                lines.append(f'  - **{op}**: {count}/{pool_size} factors ({share:.0%}); heavily saturated')
            non_overused_count = pool_size - sum((c for _, _, c in overused_ops))
            lines.append(f'  - Factors using other operators: {non_overused_count}/{pool_size} items')
        if used:
            lines.append(f"- Used variables ({len(used)}): {', '.join((v for v, _ in used[:12]))}")
        if unused:
            lines.append(f"- **Unused variables ({len(unused)})**: {', '.join(unused[:20])}")
        lines.append('')
        lines.append('**Exploration mode**: ignore existing factor directions completely and design new expressions from first principles.')
        lines.append('Prioritize the **unused variables** above and try **new operator combinations**.')
        if overused_ops:
            avoided = '、'.join((op for op, _, _ in overused_ops))
            lines.append(f'**Strictly prohibited**: using {avoided} as a root operator, even if you consider it effective.')
        return '\n'.join(lines)
    if iteration_mode == 'diversify':
        from fraud_alpha.reflection import ReflectionModule
        _rm = ReflectionModule()
        pattern_counts: Dict[str, int] = {k: 0 for k in _rm.FRAUD_PATTERNS}
        for f in factor_stats:
            cat = _rm.classify_factor_pattern(f.get('expr_str', ''))
            if cat in pattern_counts:
                pattern_counts[cat] += 1
        lines = [f'## Factor-pool pattern coverage (diversification mode)\n', f'- Factor count: {pool_size}, {_auc_label}\n']
        lines.append('| Fraud pattern | Existing factor count | Key variables | Status |')
        lines.append('|----------|-----------|---------|------|')
        uncovered_vars = []
        for key, info in _rm.FRAUD_PATTERNS.items():
            count = pattern_counts.get(key, 0)
            status = 'Covered' if count > 0 else '**Not yet covered**'
            vars_str = ', '.join(info['key_vars'][:6]) if info['key_vars'] else ' (temporal operators)'
            lines.append(f"| {info['name']} | {count} | {vars_str} | {status} |")
            if count == 0 and info['key_vars']:
                uncovered_vars.extend(info['key_vars'])
        if uncovered_vars:
            lines.append(f'\n**Diversification mode**: focus on the uncovered patterns marked above.')
            lines.append(f"Prioritize these variables: **{', '.join(sorted(set(uncovered_vars)))}**")
        return '\n'.join(lines)
    lines = [f'## Complete analysis of the current factor pool ({pool_size} factors, {_auc_label}）\n']
    lines.append('### Core factor metrics\n')
    lines.append('| # | Expression | Weight | Single-factor AUC | Marginal contribution | Effect size | Skewness | Maximum correlation | Missing rate |')
    lines.append('|---|--------|------|-----------|----------|-------------|------|----------|--------|')
    sorted_stats = sorted(factor_stats, key=lambda x: x.get('weight', 0.0), reverse=True)
    for rank, f in enumerate(sorted_stats):
        idx = f.get('index', rank)
        expr = f.get('expr_str', '?')
        w = f.get('weight', 0.0)
        auc = f.get('auc', 0.0)
        mc = f.get('marginal_contribution', 0.0)
        es = f.get('effect_size', 0.0)
        skew = f.get('skewness', 0.0)
        corr = f.get('max_corr', 0.0)
        miss = f.get('missing_rate', 0.0)
        lines.append(f'| {idx} | {expr} | {w:.3f} | {auc:.4f} | {mc:+.4f} | {es:.4f} | {skew:.2f} | {corr:.3f} | {miss:.2%} |')
    lines.append('\n### Factor discrimination details\n')
    lines.append('| # | Expression (abbreviated) | t-test p | Fraud mean | Nonfraud mean | Fraud median | Nonfraud median | Minimum | Maximum |')
    lines.append('|---|-------------|----------|---------|-----------|-----------|-------------|-------|-------|')
    for rank, f in enumerate(sorted_stats):
        idx = f.get('index', rank)
        expr = f.get('expr_str', '?')
        expr_short = expr[:40] + '...' if len(expr) > 40 else expr
        tp = f.get('ttest_p', 1.0)
        fm = f.get('fraud_mean', 0.0)
        nfm = f.get('non_fraud_mean', 0.0)
        fmed = f.get('fraud_median', 0.0)
        nfmed = f.get('non_fraud_median', 0.0)
        mn = f.get('min_val', 0.0)
        mx = f.get('max_val', 0.0)
        sig = '***' if tp < 0.001 else '**' if tp < 0.01 else '*' if tp < 0.05 else ''
        lines.append(f'| {idx} | {expr_short} | {tp:.2e}{sig} | {fm:.4f} | {nfm:.4f} | {fmed:.4f} | {nfmed:.4f} | {mn:.4f} | {mx:.4f} |')
    if pool_correlations:
        lines.append(f'\n### Highly correlated factor pairs (|corr|>0.3; total {len(pool_correlations)} pairs)')
        lines.append('Redundant factor pairs may dilute model attention; consider replacing one with a new direction.\n')
        for pc in pool_correlations[:10]:
            e1 = pc['expr1'][:50] + '...' if len(pc['expr1']) > 50 else pc['expr1']
            e2 = pc['expr2'][:50] + '...' if len(pc['expr2']) > 50 else pc['expr2']
            lines.append(f"- `{e1}` ↔ `{e2}`: r={pc['corr']:.3f}")
    if correlation_clusters:
        lines.append(f'\n### Factor correlation clusters (|corr|>0.6; total {len(correlation_clusters)} clusters)')
        lines.append('Factors in the same cluster provide redundant information; avoid new factors highly correlated with the entire cluster.\n')
        for ci, cluster in enumerate(correlation_clusters):
            short_exprs = [e[:45] + '...' if len(e) > 45 else e for e in cluster]
            lines.append(f'- **Cluster {ci + 1}**({len(cluster)}): ' + ' | '.join((f'`{e}`' for e in short_exprs[:5])))
            if len(cluster) > 5:
                lines.append(f'  ... total {len(cluster)} factors')
    if success_patterns and pool_size >= 3:
        lines.append(f'\n### Shared characteristics of successful factors')
        lines.append(f"- Mean nesting depth: {success_patterns.get('avg_depth', 0):.1f}")
        lines.append(f"- Mean single-factor AUC: {success_patterns.get('avg_auc', 0):.4f}")
        top_ops = success_patterns.get('top_ops', [])
        if top_ops:
            ops_str = ', '.join((f'{op}:{cnt}' for op, cnt in top_ops[:8]))
            lines.append(f'- Frequent operators: {ops_str}')
        top_vars = success_patterns.get('top_vars', [])
        if top_vars:
            vars_str = ', '.join((f'{v}:{c}' for v, c in top_vars[:10]))
            lines.append(f'- Frequent variables: {vars_str}')
    if coverage_stats:
        cov_used_vars = coverage_stats.get('used_vars', [])
        cov_unused_vars = coverage_stats.get('unused_vars', [])
        cov_used_ops = coverage_stats.get('used_ops', [])
        cov_unused_ops = coverage_stats.get('unused_ops', [])
        var_rate = coverage_stats.get('var_coverage_rate', 0.0)
        op_rate = coverage_stats.get('op_coverage_rate', 0.0)
        pat_cov = coverage_stats.get('pattern_coverage', {})
        lines.append(f'\n### Variable/operator coverage\n')
        lines.append(f'- **Variable coverage**: {var_rate:.0%}（{len(cov_used_vars)}/{len(cov_used_vars) + len(cov_unused_vars)})')
        if cov_used_vars:
            used_str = ', '.join((f'{v}:{c}' for v, c in cov_used_vars[:20]))
            lines.append(f'- Used variables: {used_str}')
        if cov_unused_vars:
            lines.append(f"- **Unused variables ({len(cov_unused_vars)})**: {', '.join(cov_unused_vars)}")
        lines.append(f'- **Operator coverage**: {op_rate:.0%}（{len(cov_used_ops)}/{len(cov_used_ops) + len(cov_unused_ops)})')
        if cov_used_ops:
            ops_str = ', '.join((f'{o}:{c}' for o, c in cov_used_ops[:15]))
            lines.append(f'- Used operators: {ops_str}')
        if cov_unused_ops:
            lines.append(f"- **Unused operators ({len(cov_unused_ops)})**: {', '.join(cov_unused_ops)}")
        if pat_cov:
            from fraud_alpha.reflection import ReflectionModule
            pattern_names = {'profitability': 'Earnings quality', 'balance_sheet': 'Balance sheet', 'financing': 'Financing structure', 'governance': 'Corporate governance'}
            pattern_total = {pat: len([v for v in info.get('key_vars', []) if v in all_vars]) for pat, info in ReflectionModule.FRAUD_PATTERNS.items() if pat in pat_cov}
            lines.append(f'\n### Variable coverage by fraud pattern\n')
            lines.append('| Fraud pattern | Covered variables | Total variables | Coverage |')
            lines.append('|---------|-----------|---------|-------|')
            for pat, cnt in pat_cov.items():
                total = pattern_total.get(pat, cnt)
                rate = cnt / total if total > 0 else 0
                name = pattern_names.get(pat, pat)
                lines.append(f'| {name} | {cnt} | {total} | {rate:.0%} |')
    else:
        if op_counter:
            total_ops = sum(op_counter.values())
            top_ops = sorted(op_counter.items(), key=lambda x: x[1], reverse=True)[:10]
            op_str = ' | '.join((f'{k}:{v}({v / total_ops:.0%})' for k, v in top_ops))
            lines.append(f'\n**Operator usage distribution**: {op_str}')
        if used:
            used_str = ' | '.join((f'{v}:{c}' for v, c in used[:15]))
            lines.append(f'**Variable usage (Top-15)**: {used_str}')
        if unused:
            lines.append(f"**Unused variables ({len(unused)})**: {', '.join(unused[:20])}" + (f' and others' if len(unused) > 20 else ''))
    return '\n'.join(lines)

def format_openfe_reference(openfe_features: Optional[List[str]]=None) -> str:
    if not openfe_features:
        return ''
    lines = ['## OpenFE reference features (useful combinations found by an automated feature-engineering baseline)\n']
    lines.append('The following OpenFE features may provide construction ideas:\n')
    for i, feat in enumerate(openfe_features[:20], 1):
        lines.append(f'{i}. `{feat}`')
    return '\n'.join(lines)

def _suggest_extension(expr_str: str) -> str:
    if not expr_str:
        return 'Try entirely new combinations'
    suggestions = []
    if 'Delta' not in expr_str and 'ROC' not in expr_str:
        suggestions.append('Add temporal operators (Delta/ROC)')
    if 'Std' not in expr_str and 'Range' not in expr_str:
        suggestions.append('Add volatility operators (Std/Range)')
    if 'Rank' not in expr_str:
        suggestions.append('Add cross-sectional operators (Rank)')
    if 'If' not in expr_str and 'Greater' not in expr_str:
        suggestions.append('Add conditional branches (If/Greater)')
    if not suggestions:
        suggestions.append('Change the core variables or use deeper nested combinations')
    return '；'.join(suggestions[:2])

def format_rich_rejection_analysis(recent_rejections: List[Dict], max_display: int=15) -> str:
    if not recent_rejections:
        return ''
    categories = {'auc_low': [], 'corr_high': [], 'struct_sim': [], 'no_gain': [], 'other': []}
    for r in recent_rejections:
        reason = r.get('reason', '')
        if 'Raw AUC too low' in reason or 'Fast AUC too low' in reason or 'AUC too low' in reason:
            categories['auc_low'].append(r)
        elif 'Correlation too high' in reason:
            categories['corr_high'].append(r)
        elif 'Structural similarity' in reason:
            categories['struct_sim'].append(r)
        elif any(marker in reason.lower() for marker in ('no incremental auc gain', 'no gain', 'insufficient', 'decreased', 'reduced auc')):
            categories['no_gain'].append(r)
        else:
            categories['other'].append(r)
    total = len(recent_rejections)
    lines = [f'## Detailed diagnosis of rejected factors (last {total})\n']
    lines.append('| Rejection reason | Count | Proportion |')
    lines.append('|---------|------|------|')
    labels = {'auc_low': 'Insufficient AUC (ineffective direction)', 'corr_high': 'Excessive correlation (saturated direction)', 'struct_sim': 'Structural similarity (insufficient variation)', 'no_gain': 'No incremental gain (insufficient marginal contribution)', 'other': 'Other'}
    for key, label in labels.items():
        cnt = len(categories[key])
        if cnt > 0:
            lines.append(f'| {label} | {cnt} | {cnt / total:.0%} |')
    if categories['auc_low']:
        lines.append('\n### Insufficient AUC (avoid these directions)')
        for r in categories['auc_low'][:max_display]:
            lines.append(f"  - `{r['expr_str']}` → {r['reason']}")
    if categories['corr_high']:
        lines.append('\n### Excessive correlation (effective but saturated; structural innovation needed)')
        for r in categories['corr_high'][:max_display]:
            suggestion = _suggest_extension(r.get('expr_str', ''))
            lines.append(f"  - `{r['expr_str']}` → {r['reason']} | Suggestion: {suggestion}")
    if categories['no_gain']:
        lines.append('\n### No incremental gain (effective alone but no contribution to the combination)')
        for r in categories['no_gain'][:max_display]:
            lines.append(f"  - `{r['expr_str']}` → {r['reason']}")
    return '\n'.join(lines)

def build_unified_prompt(n_generate: int, iteration: int, total_iterations: int, is_cold_start: bool, allow_delete: bool, factor_stats: List[Dict], pool_summary: Dict, reflection_text: str, s2_variable_semantics: str, s3_baseline_importance: str, s4_experience_summaries: str, s5_recent_rounds: str, context_manager, iteration_mode: str='exploit', recent_rejections: Optional[List[Dict]]=None, s0_dataset_profile: str='', s1_sample_diagnostics: str='', pool_correlations: Optional[List[Dict]]=None, success_patterns: Optional[Dict]=None, coverage_stats: Optional[Dict]=None, correlation_clusters: Optional[List[List[str]]]=None) -> str:
    if factor_stats:
        s6_pool_analysis = format_pool_full_analysis(factor_stats, pool_summary, iteration_mode=iteration_mode, pool_correlations=pool_correlations, success_patterns=success_patterns, coverage_stats=coverage_stats, correlation_clusters=correlation_clusters)
    else:
        s6_pool_analysis = '## Current factor pool\n(The pool is empty; cold start)'
    s5_rejection_diagnosis = ''
    if recent_rejections:
        s5_rejection_diagnosis = format_rich_rejection_analysis(recent_rejections)
    s7_reflection = f'## Reflection and strategy suggestions for this round\n\n{reflection_text}' if reflection_text else ''
    pool_size = pool_summary.get('size', 0)
    best_auc = pool_summary.get('best_auc', 0.0)
    MODE_LABELS = {'exploit': 'Refinement mode', 'explore': 'Exploration mode', 'diversify': 'Diversification mode', 'crossover': 'Crossover mode'}
    mode_label = MODE_LABELS.get(iteration_mode, 'Refinement mode')
    if is_cold_start:
        mode_hint = f'**Cold-start mode** (current pool: {pool_size} factors)\nPrioritize broad exploration of operator and variable combinations. Do not request DELETE.\nPrioritize variables with high baseline SHAP ranks (see S2/S3).'
    elif iteration_mode == 'crossover':
        mode_hint = f"**{mode_label}** (current pool: {pool_size} factors, AUC={best_auc:.4f})\nRegular generation has not improved AUC for several rounds. Switch to **crossover mode**.\nRecombine **subexpression fragments from the crossover parents** in S6 into new factors.\n\nCrossover strategies:\n1. **Operator substitution**: wrap one parent's core ratio/difference in another parent's temporal operator.\n2. **Variable exchange**: retain a parent's operator structure but introduce underexplored variables.\n3. **Deep composition**: nest fragments from multiple parents into a new structure.\n4. **Operator innovation**: wrap parent fragments with unused operators, such as Power, WMA, Corr, or If.\n5. **Window changes**: vary temporal windows (1y, 2y, 3y, 5y).\n\n**Constraints**:\n- Every offspring must include fragments from at least two different parents.\n- Do not simply copy a parent; changing only a window is not crossover.\n- Offspring structures must differ from every parent.\n- At least two offspring should use underexplored operators or variables."
    elif iteration_mode == 'explore':
        mode_hint = f'**{mode_label}** (current pool: {pool_size} factors, AUC={best_auc:.4f})\n**Ignore existing factor directions completely** and design new expressions from first principles.\nUse new variable combinations and operator structures. Do not imitate or slightly adjust existing factors.'
    elif iteration_mode == 'diversify':
        mode_hint = f'**{mode_label}** (current pool: {pool_size} factors, AUC={best_auc:.4f})\nFocus on **fraud patterns not yet covered** by the pool (see the coverage table in S6).\nUse new variable categories and operator combinations to fill these detection gaps.'
    else:
        mode_hint = f'**{mode_label}** (current pool: {pool_size} factors, AUC={best_auc:.4f})\nPrioritize refinement using historical experience (S4) and recent records (S5), avoiding repeated failure patterns.\nDo not merely change a window or substitute a similar variable; create structurally distinct factors.'
    delete_hint = '\n\nTo remove ineffective factors, output `DELETE: <index or expression>` first, followed by new expressions.' if allow_delete else '\n\nThis is a cold-start round. **Do not use DELETE**; output new expressions directly.'
    diversity_hint = ''
    crossover_suffix = ''
    if iteration_mode == 'crossover':
        crossover_suffix = '\n\nAppend `// rationale` to each expression, identifying the parent fragments used. Aim to improve AUC through recombination.'
    s8_request = f'## Current task (iteration {iteration}/{total_iterations}）\n\n{mode_hint}{delete_hint}\n\n{diversity_hint}\n\n**Generate {n_generate} new factor expressions** (ADD), each followed by `// rationale` with one sentence explaining the design. Aim to improve combined AUC.{crossover_suffix}'
    sections = [{'name': 's8_request', 'content': s8_request, 'priority': 1}, {'name': 's7_reflection', 'content': s7_reflection, 'priority': 2}, {'name': 's0_dataset_profile', 'content': s0_dataset_profile, 'priority': 3}, {'name': 's1_sample_diagnostics', 'content': s1_sample_diagnostics, 'priority': 3}, {'name': 's2_variable_semantics', 'content': s2_variable_semantics, 'priority': 4}, {'name': 's3_baseline_importance', 'content': s3_baseline_importance, 'priority': 5}, {'name': 's4_experience_summaries', 'content': s4_experience_summaries, 'priority': 6}, {'name': 's5_recent_rounds', 'content': s5_recent_rounds, 'priority': 8}, {'name': 's5_rejection_diagnosis', 'content': s5_rejection_diagnosis, 'priority': 8}, {'name': 's6_pool_analysis', 'content': s6_pool_analysis, 'priority': 9}]
    return context_manager.build_prompt(sections)

def build_compression_prompt(recent_rounds_text: str, rounds_str: str, rejected_with_rationale: Optional[List[Dict]]=None) -> str:
    parts = [f'Below are detailed records for round {rounds_str} of the financial-fraud factor-engineering experiment:\n\n{recent_rounds_text}\n\n']
    if rejected_with_rationale:
        cat_counts = {'Insufficient AUC': 0, 'Correlation too high': 0, 'Structural similarity': 0, 'No incremental gain': 0, 'Other': 0}
        for r in rejected_with_rationale:
            reason = r.get('reason', '')
            if 'AUC too low' in reason or 'Raw AUC too low' in reason or 'Fast AUC too low' in reason:
                cat_counts['Insufficient AUC'] += 1
            elif 'Correlation too high' in reason:
                cat_counts['Correlation too high'] += 1
            elif 'Structural similarity' in reason:
                cat_counts['Structural similarity'] += 1
            elif any(marker in reason.lower() for marker in ('no incremental auc gain', 'no gain', 'insufficient', 'decreased', 'reduced auc')):
                cat_counts['No incremental gain'] += 1
            else:
                cat_counts['Other'] += 1
        total_rej = sum(cat_counts.values())
        if total_rej > 0:
            parts.append(f'### Rejection-reason category statistics (total {total_rej} rejected factors)\n')
            for cat, cnt in cat_counts.items():
                if cnt > 0:
                    parts.append(f'- {cat}: {cnt} items ({cnt / total_rej:.0%})')
            parts.append('')
        rationale_items = [r for r in rejected_with_rationale if r.get('rationale')]
        if rationale_items:
            parts.append('The following rejected factors include their original design rationales. Analyze weaknesses in these rationales:\n')
            for r in rationale_items[:15]:
                parts.append(f"- {r.get('expr_str', '?')}\n  Original rationale: {r.get('rationale', '')}\n  Rejection reason: {r.get('reason', 'Unknown')}\n")
            parts.append('')
    parts.append('Summarize the experimental experience concisely, within 300 words:\n- **Successful patterns**: repeatedly accepted operator/variable combinations and their shared characteristics.\n- **Failure patterns**: repeatedly rejected patterns and reasons (insufficient AUC, high correlation, or calculation errors). If original rationales are supplied, analyze their weaknesses. Use the rejection-category statistics above to explain the root causes of each category.\n- **Directions to explore**: underexplored variables or operator types to prioritize next.\n\nOutput only these three points, without expressions or unrelated content.')
    return '\n'.join(parts)
MASTER_SYSTEM_PROMPT = 'You are an experimental-design expert. Design differentiated exploration strategies for K parallel factor-generation agents.\n\nSpecify exploration directions only; do not generate factor expressions.\n\n# Output rules\n\nOutput a strict JSON array of length K, specified in the request.\n\nEach element must include:\n- strategy_name: a short descriptive name\n- allocation_role: exploit|explore|coverage|diversify\n- rationale: 1-2 sentences explaining the strategy\n- focus_variables: target variables, such as ["N20", "N14"]\n- avoid_variables: variables to avoid\n- operator_style: ratio|temporal|interaction|volatility|composite\n- complexity: simple|medium|complex\n- fraud_pattern: profitability|balance_sheet|financing|governance|cross_domain\n- specific_instruction: exploration directions, without expression code\n\n# Constraints\n\n1. Include at least one exploit strategy.\n2. Include at least one explore strategy.\n3. Include at least one coverage strategy targeting undercovered variables or fraud patterns.\n4. Strategies must differ structurally in focus_variables and operator_style.\n5. specific_instruction must contain directional constraints only, not expression code.\n6. Output only the strict JSON array.\n'

def build_master_prompt(pool_state: dict, factor_stats: List[Dict], ucb_stats: Dict[str, List[Dict]], last_round_summary: str, experience_summaries: str, n_agents: int=3, recent_strategies: Optional[List[List[Dict]]]=None) -> str:
    import re as _re
    sections = []
    sections.append(f'Generate {n_agents} differentiated strategies as a strict JSON array of length {n_agents}.\nEach element must contain strategy_name, allocation_role, rationale, focus_variables, avoid_variables, operator_style, complexity, fraud_pattern, and specific_instruction.\n\n**Key requirements**:\n- strategy_name must be specific and actionable, e.g. "N9_N13 ratio temporal differences", not "robust earnings".\n- specific_instruction must give concrete construction directions in at least two sentences, specifying operators and target signals.\n- focus_variables must explicitly list 3-5 available variable names and must not be empty.\n- Any two strategies may overlap in at most one focus variable.')
    pool_size = pool_state.get('size', 0)
    best_auc = pool_state.get('best_auc', 0.0)
    exprs = pool_state.get('exprs', [])
    all_vars = Feature.get_available_features()
    used_vars = set()
    for expr_str in exprs:
        for var in all_vars:
            if _re.search('\\b' + _re.escape(var) + '\\b', expr_str):
                used_vars.add(var)
    low_coverage_vars = [v for v in all_vars if v not in used_vars]
    from fraud_alpha.reflection import ReflectionModule
    _rm = ReflectionModule(available_variables=all_vars)
    active_patterns = _rm._get_active_patterns()
    pattern_counts: Dict[str, int] = {k: 0 for k in active_patterns}
    for expr_str in exprs:
        cat = _rm.classify_factor_pattern(expr_str)
        if cat in pattern_counts:
            pattern_counts[cat] += 1
    pattern_str = ', '.join((f"{active_patterns[k]['name']}:{v}" for k, v in pattern_counts.items()))
    saturated = _rm.get_saturated_patterns(exprs, top_n=3)
    saturated_str = ', '.join((f'{t}({c} occurrences)' for t, c in saturated)) if saturated else 'none'
    sections.append(f"## Current main-pool status\n- Pool size: {pool_size}\n- Current best AUC: {best_auc:.4f}\n- Covered variables ({len(used_vars)}): {', '.join(sorted(used_vars)[:15])}\n- Low-coverage variables ({len(low_coverage_vars)}): {', '.join(low_coverage_vars[:20])}\n- Covered fraud-pattern proportions: {pattern_str}\n- Frequent structural templates: {saturated_str}")
    ucb_lines = ['## Strategy performance history (UCB statistics)']
    for dim, items in ucb_stats.items():
        if items:
            dim_str = ', '.join((f"{it['value']}(n={it['count']},avg={it['avg_reward']:.4f})" for it in items[:6]))
            ucb_lines.append(f'- {dim}: {dim_str}')
        else:
            ucb_lines.append(f'- {dim}: (no data yet)')
    sections.append('\n'.join(ucb_lines))
    sections.append(f'## Previous-round review summary\n{last_round_summary}')
    if experience_summaries and 'None yet' not in experience_summaries:
        sections.append(f'## Long-term experience summary\n{experience_summaries}')
    if recent_strategies:
        history_lines = ['## Recent strategy history (do not repeat)\n']
        history_lines.append("These strategies were used in recent rounds; this round's strategies must differ structurally:\n")
        for round_idx, round_strats in enumerate(recent_strategies):
            round_label = f'Round {len(recent_strategies) - round_idx} rounds ago'
            for s in round_strats:
                name = s.get('strategy_name', '?')
                role = s.get('allocation_role', '?')
                pattern = s.get('fraud_pattern', '?')
                style = s.get('operator_style', '?')
                focus = s.get('focus_variables', [])
                accepted = s.get('accepted_count', 0)
                focus_str = ', '.join(focus) if focus else 'Unspecified'
                status = f'Admitted{accepted} items' if accepted > 0 else 'Zero admissions'
                history_lines.append(f'- [{round_label}] {name} | Role={role} | Pattern={pattern} | Operators={style} | Variables=[{focus_str}] | {status}')
        all_names = [s.get('strategy_name', '') for rs in recent_strategies for s in rs]
        from collections import Counter
        name_counts = Counter(all_names)
        repeated = [f'{n}({c} occurrences)' for n, c in name_counts.most_common(3) if c >= 2]
        if repeated:
            history_lines.append(f"\nFrequently repeated strategy names: {', '.join(repeated)}; do not reuse these names this round")
        all_focus = [v for rs in recent_strategies for s in rs for v in s.get('focus_variables', [])]
        var_counts = Counter(all_focus)
        overused = [f'{v}({c} occurrences)' for v, c in var_counts.most_common(5) if c >= 3]
        if overused:
            history_lines.append(f"Frequent focus variables: {', '.join(overused)}; at least two strategies must avoid these variables this round")
        sections.append('\n'.join(history_lines))
    sections.append(f'## Current task\nGenerate {n_agents} differentiated strategies satisfying:\n1. At least one exploit strategy refining effective directions.\n2. At least one explore strategy using new variable combinations and operator structures.\n3. At least one coverage strategy addressing undercovered variables or fraud patterns.\n4. Any two strategies overlap in at most one focus variable.\n5. All strategies have distinct operator_style values.\n6. strategy_name is concrete, e.g. "N9_N13 ratio temporal differences"; avoid generic names such as "robust earnings" or "temporal exploration".\n7. specific_instruction gives concrete construction directions in at least two sentences, specifying operator combinations and target anomalies.\n8. Do not closely duplicate strategy_name or focus_variables from recent strategy history.\n9. Output only a strict JSON array.')
    return '\n\n'.join(sections)

def build_worker_prompt(strategy, n_factors: int, iteration: int, total_iterations: int, is_cold_start: bool, allow_delete: bool, factor_stats: List[Dict], pool_summary: Dict, reflection_text: str, s2_variable_semantics: str, s3_baseline_importance: str, s4_experience_summaries: str, s5_recent_rounds: str, context_manager, iteration_mode: str='exploit', recent_rejections: Optional[List[Dict]]=None) -> str:
    if factor_stats:
        s6_pool_analysis = format_pool_full_analysis(factor_stats, pool_summary, iteration_mode=iteration_mode)
    else:
        s6_pool_analysis = '## Current factor pool\n(The pool is empty; cold start)'
    s6_rejections = ''
    if recent_rejections:
        s6_rejections = format_categorized_rejections(recent_rejections)
    s7_reflection = f'## Reflection and strategy suggestions for this round\n\n{reflection_text}' if reflection_text else ''
    pool_size = pool_summary.get('size', 0)
    best_auc = pool_summary.get('best_auc', 0.0)
    focus_str = ', '.join(strategy.focus_variables) if strategy.focus_variables else 'No specific preference'
    avoid_str = ', '.join(strategy.avoid_variables) if strategy.avoid_variables else 'none'
    strategy_prefix = f'## Strategy instructions for this round (assigned by the coordinator)\n\nStrategy name: {strategy.strategy_name}\nAllocation role: {strategy.allocation_role}\nExploration rationale: {strategy.rationale}\n\nFocus variables: {focus_str}\nVariables to avoid: {avoid_str}\n\nOperator style: {strategy.operator_style}\nExpression complexity: {strategy.complexity}\nTarget fraud pattern: {strategy.fraud_pattern}\nTemporal granularity: {strategy.temporal_granularity}\n\nSpecific directions:\n{strategy.specific_instruction}\n\nConstraints:\n1. Every expression must include at least one focus variable.\n2. Avoid the listed avoid_variables where possible.\n3. Expression style must match the requested operator style.\n4. Do not repeat frequently used historical templates.\n'
    MODE_LABELS = {'exploit': 'Refinement mode', 'explore': 'Exploration mode', 'diversify': 'Diversification mode'}
    mode_label = MODE_LABELS.get(iteration_mode, 'Refinement mode')
    if is_cold_start:
        mode_hint = f'**Cold-start mode** (current pool: {pool_size} factors)\nPrioritize broad exploration of operator and variable combinations. Do not request DELETE.\nPrioritize variables with high baseline SHAP ranks (see S2/S3).'
    elif iteration_mode == 'explore':
        mode_hint = f'**{mode_label}** (current pool: {pool_size} factors, AUC={best_auc:.4f})\n**Ignore existing factor directions completely** and design new expressions from first principles.\nUse new variable combinations and operator structures. Do not imitate or slightly adjust existing factors.'
    elif iteration_mode == 'diversify':
        mode_hint = f'**{mode_label}** (current pool: {pool_size} factors, AUC={best_auc:.4f})\nFocus on **fraud patterns not yet covered** by the pool (see the coverage table in S6).\nUse new variable categories and operator combinations to fill these detection gaps.'
    else:
        mode_hint = f'**{mode_label}** (current pool: {pool_size} factors, AUC={best_auc:.4f})\nPrioritize refinement using historical experience (S4) and recent records (S5), avoiding repeated failure patterns.\nDo not merely change a window or substitute a similar variable; create structurally distinct factors.'
    delete_hint = '\n\nTo remove ineffective factors, output `DELETE: <index or expression>` first, followed by new expressions.' if allow_delete else '\n\nThis is a cold-start round. **Do not use DELETE**; output new expressions directly.'
    s8_request = f'{strategy_prefix}\n## Current task (iteration {iteration}/{total_iterations}）\n\n{mode_hint}{delete_hint}\n\n**Follow the strategy instructions above to generate {n_factors} new factor expressions**, each followed by `// rationale` with one sentence explaining the design. Aim to improve combined AUC.'
    sections = [{'name': 's8_request', 'content': s8_request, 'priority': 1}, {'name': 's7_reflection', 'content': s7_reflection, 'priority': 2}, {'name': 's2_variable_semantics', 'content': s2_variable_semantics, 'priority': 3}, {'name': 's3_baseline_importance', 'content': s3_baseline_importance, 'priority': 4}, {'name': 's4_experience_summaries', 'content': s4_experience_summaries, 'priority': 5}, {'name': 's5_recent_rounds', 'content': s5_recent_rounds, 'priority': 6}, {'name': 's6_rejections', 'content': s6_rejections, 'priority': 6}, {'name': 's6_pool_analysis', 'content': s6_pool_analysis, 'priority': 7}]
    return context_manager.build_prompt(sections)
