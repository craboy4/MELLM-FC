"""Only the unchanged prompt assets used by the main experiment; no enrichment client."""
from typing import List, Dict, Iterable
from fraud_alpha.operator_catalog import build_combination_example_block, build_operator_reference_block
_OPERATOR_BLOCK = build_operator_reference_block()
_COMBINATION_EXAMPLE_BLOCK = build_combination_example_block()
_COMPLEXITY_BLOCK = "# Factor complexity\n\n**Prefer complex factors with 3-4 nesting levels**. Simple factors with 1-2 levels have been extensively explored; more complex structures may capture nonlinear fraud patterns.\n\n**Suggested constructions**:\n- Nested transforms: construct a ratio or difference, then apply means, volatility, ranges, or logarithms.\n- Conditional logic: use If(cond,x,y) for branches or Greater/Less for piecewise signals.\n- Multivariable interactions: combine variables from different themes instead of repeatedly adjusting one variable.\n- Composite temporal patterns: explore changes in changes, volatility of volatility, and synchronized cross-variable anomalies.\n- Cross-sectional comparison: use CSRank(x) for a firm's relative position within the same year.\n- Standardization and compression: use ZScore(x,t) for rolling standardization and Tanh/Sigmoid for bounded transforms."
_OUTPUT_BLOCK = '# Output rules\n\n1. Output any optional DELETE instructions first: `DELETE: <index or expression>`.\n2. Output each new expression on a separate line: `expression // one-sentence rationale`.\n3. Output only DELETE lines and expression lines; no additional explanation.\n4. Time parameters must be integers followed by y, such as `3y`.\n5. Use at most 5 nesting levels; 3-4 are recommended.\n6. Every expression must include `//` followed by a one-sentence rationale.\n7. Use only base variables available in the current dataset.'

def _pick_first(candidates: List[str], fallback: str='N1') -> str:
    return candidates[0] if candidates else fallback

def _pick_next(candidates: List[str], used: Iterable[str], fallback: str) -> str:
    used_set = set(used)
    for candidate in candidates:
        if candidate not in used_set:
            return candidate
    return fallback

def _build_example_expressions(numeric_features: List[str], categorical_features: List[str]) -> List[Dict[str, str]]:
    if not numeric_features:
        return []
    v1 = _pick_first(numeric_features)
    v2 = _pick_next(numeric_features, [v1], v1)
    v3 = _pick_next(numeric_features, [v1, v2], v2)
    v4 = _pick_next(numeric_features, [v1, v2, v3], v3)
    examples = [{'expr': f'Div(Mean({v1},3y),Add(Std({v2},2y),0.1))', 'reason': 'Normalize trends by means and volatility to detect sudden instability'}, {'expr': f'Delta(Div({v1},Add(Abs({v2}),0.1)),1y)', 'reason': 'Monitor one-year discontinuities in key ratios'}, {'expr': f'Corr(Div({v1},Add(Abs({v2}),0.1)),Div({v3},Add(Abs({v4}),0.1)),3y)', 'reason': 'Track co-movement of financial structure variables to detect breaks'}]
    if categorical_features:
        cat = categorical_features[0]
        examples.append({'expr': f'Mul(Greater({cat},1.0),Abs(Delta({v1},1y)))', 'reason': 'Amplify changes in key indicators under abnormal categorical states'})
    return examples
