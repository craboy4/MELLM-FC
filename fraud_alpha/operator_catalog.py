from __future__ import annotations
import os
import re
from functools import lru_cache
from typing import Dict, List
_F2_OPERATORS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'document', 'prompt', 'F2_Operators.md'))
_FALLBACK_OPERATOR_LINES = ['Abs(x): absolute value', 'Log(x): natural logarithm; add 1 and use absolute values to avoid nonpositive inputs, e.g. Log(Abs(x)+1)', 'Log10(x): base-10 logarithm; add 1 and use absolute values to avoid nonpositive inputs', 'Add(x,y): addition', 'Sub(x,y): subtraction', 'Mul(x,y): multiplication', 'Div(x,y): division; add a small constant 1e-6 to the denominator to avoid division by zero', 'Sqrt(x): square root of a nonnegative input, e.g. Sqrt(Abs(x))', 'Square(x): square', 'Cube(x): cube', 'Pow(x,n): n-th power', 'Exp(x): exponential, e raised to x', 'Sign(x): sign function returning -1, 0, or 1', 'Sin(x): sine', 'Cos(x): cosine', 'Tan(x): tangent', 'Greater(x,y): the larger of x and y', 'Less(x,y): the smaller of x and y', 'Ref(x,t): x lagged by t annual records', 'Delta(x,t): current value minus its t-record lag, x - Ref(x,t)', 'Mean(x,t): mean over the last t annual records', 'Sum(x,t): sum over the last t annual records', 'Std(x,t): standard deviation over the last t annual records', 'Var(x,t): variance over the last t annual records', 'Max(x,t): maximum over the last t annual records', 'Min(x,t): minimum over the last t annual records', 'Med(x,t): median over the last t annual records', 'Mad(x,t): mean absolute deviation over the last t annual records', 'Skew(x,t): skewness over the last t annual records', 'Kurt(x,t): kurtosis over the last t annual records', 'Quantile(x,t,q): q-quantile over the last t annual records', 'Range(x,t): range over the last t annual records, Max(x,t) - Min(x,t)', 'WMA(x,t): weighted moving average over the last t annual records', 'EMA(x,t): exponential moving average over the last t annual records', 'DecayLinear(x,t): linearly decaying weights over the last t annual records, with larger weights on recent records', 'Cov(x,y,t): covariance of two series over the last t annual records', 'Corr(x,y,t): correlation of two series over the last t annual records', 'Slope(x,t): linear regression slope over the last t annual records', 'TSRank(x,t): time-series rank of x within the last t annual records', 'ROC(x,t): rate of change, (x - Ref(x,t)) / Ref(x,t)', 'Momentum(x,t): momentum, x - Ref(x,t), equivalent to Delta(x,t)', 'Neg(x): negation, -x', 'Inv(x): reciprocal, 1/x, with safe handling of zero', 'Tanh(x): hyperbolic tangent, compressing values into (-1,1)', 'Sigmoid(x): sigmoid, compressing values into (0,1)', 'If(cond,x,y): return x when cond>0, otherwise y', 'Clip(x,lo,hi): clip x to [lo,hi]', 'ZScore(x,t): rolling standardization, (x-Mean(x,t))/Std(x,t)', 'ArgMax(x,t): position of the maximum in the last t annual records, normalized to [0,1]', 'ArgMin(x,t): position of the minimum in the last t annual records, normalized to [0,1]', 'CSRank(x): within-year cross-company percentile rank, output in [0,1]']
_FALLBACK_COMBINATION_LINES = ['Growth rate: Div(Sub(x, Ref(x,1)), Add(Abs(Ref(x,1)), 1e-6))', 'Volatility: Div(Abs(Sub(x, Ref(x,1))), Add(Abs(Ref(x,1)), 1e-6))', 'Log ratio: Log(Div(Add(x,1), Add(y,1)))', 'Normalized ratio: Div(x, Add(Add(x,y), 1e-6))', 'Change acceleration: Sub(Delta(x,1), Delta(x,2))', 'Moving average: Mean(x,3)', 'Deviation: Abs(Sub(x, Mean(x,3)))', 'Relative deviation: Div(Abs(Sub(x, Mean(x,3))), Add(Abs(Mean(x,3)), 1e-6))', 'Extreme-value ratio: Div(Greater(x,y), Add(Less(x,y), 1e-6))', 'Consistency measure: Div(Abs(Sub(x,y)), Add(Add(Abs(x), Abs(y)), 1e-6))', 'Trend consistency: Mul(Delta(x,1), Delta(x,2))', 'Ratio change: Sub(Div(x, Add(y,1e-6)), Div(Ref(x,1), Add(Ref(y,1), 1e-6)))', 'Leverage interaction: Div(Mul(x,y), Add(z,1e-6))', 'Composite measure: Div(Mul(Add(x,y), z), Add(w,1e-6))', 'Weighted combination: Add(Add(Mul(0.5,x), Mul(0.3,y)), Mul(0.2,z))', 'Cross-period difference: Sub(Ref(x,1), y)', 'Cross-variable lagged ratio: Div(Ref(x,1), Add(Ref(y,2), 1e-6))', 'Coefficient of variation: Div(Std(x,3), Add(Abs(Mean(x,3)), 1e-6))', 'Stability measure: Div(Mean(x,3), Add(Std(x,3), 1e-6))', 'Relative growth rate: Div(Delta(x,1), Add(Delta(y,1), 1e-6))', 'Trend reversal: Mul(Delta(x,1), Delta(x,2))', 'Short/long volatility ratio: Div(Std(x,1), Add(Std(x,3), 1e-6))', 'Growth-rate difference: Sub(ROC(x,1), ROC(y,1))', 'Normalized temporal rank: Div(TSRank(x,5), 5)']
_OPERATOR_CATEGORIES = {'Abs': 'Unary operators', 'Log': 'Unary operators', 'Log10': 'Unary operators', 'Sqrt': 'Unary operators', 'Square': 'Unary operators', 'Cube': 'Unary operators', 'Exp': 'Unary operators', 'Sign': 'Unary operators', 'Sin': 'Unary operators', 'Cos': 'Unary operators', 'Tan': 'Unary operators', 'Neg': 'Unary operators', 'Inv': 'Unary operators', 'Tanh': 'Unary operators', 'Sigmoid': 'Unary operators', 'Add': 'Binary operators', 'Sub': 'Binary operators', 'Mul': 'Binary operators', 'Div': 'Binary operators', 'Pow': 'Binary operators', 'Greater': 'Binary operators', 'Less': 'Binary operators', 'If': 'Ternary operators', 'Clip': 'Ternary operators', 'Ref': 'Single-variable temporal operators', 'Delta': 'Single-variable temporal operators', 'Mean': 'Single-variable temporal operators', 'Sum': 'Single-variable temporal operators', 'Std': 'Single-variable temporal operators', 'Var': 'Single-variable temporal operators', 'Max': 'Single-variable temporal operators', 'Min': 'Single-variable temporal operators', 'Med': 'Single-variable temporal operators', 'Mad': 'Single-variable temporal operators', 'Skew': 'Single-variable temporal operators', 'Kurt': 'Single-variable temporal operators', 'Quantile': 'Single-variable temporal operators', 'Range': 'Single-variable temporal operators', 'WMA': 'Single-variable temporal operators', 'EMA': 'Single-variable temporal operators', 'DecayLinear': 'Single-variable temporal operators', 'Slope': 'Single-variable temporal operators', 'TSRank': 'Single-variable temporal operators', 'ROC': 'Single-variable temporal operators', 'Momentum': 'Single-variable temporal operators', 'ZScore': 'Single-variable temporal operators', 'ArgMax': 'Single-variable temporal operators', 'ArgMin': 'Single-variable temporal operators', 'Cov': 'Two-variable temporal operators', 'Corr': 'Two-variable temporal operators', 'CSRank': 'Cross-sectional operators'}
_CATEGORY_ORDER = ['Unary operators', 'Binary operators', 'Ternary operators', 'Single-variable temporal operators', 'Two-variable temporal operators', 'Cross-sectional operators']
_LEGACY_OPERATOR_ALIASES = {'Rank': 'TSRank', 'Lag': 'Ref', 'Where': 'If', 'Cond': 'If', 'Inverse': 'Inv', 'Negate': 'Neg', 'CrossRank': 'CSRank', 'Clamp': 'Clip'}
_OPERATOR_LINE_RE = re.compile('^([A-Za-z][A-Za-z0-9_]*)\\(([^)]*)\\):\\s*(.+)$')
_COMBINATION_LINE_RE = re.compile('^([^:：]+)\\s*[:：]\\s*(.+)$')

def _parse_operator_line(line: str) -> Dict[str, str] | None:
    matched = _OPERATOR_LINE_RE.match(line.strip())
    if matched is None:
        return None
    name = matched.group(1)
    signature = f'{name}({matched.group(2)})'
    return {'name': name, 'signature': signature, 'description': matched.group(3).strip(), 'category': _OPERATOR_CATEGORIES.get(name, 'Single-variable temporal operators')}

def _parse_combination_line(line: str) -> Dict[str, str] | None:
    matched = _COMBINATION_LINE_RE.match(line.strip())
    if matched is None:
        return None
    return {'label': matched.group(1).strip(), 'expression': matched.group(2).strip()}

def _fallback_operator_entries() -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    for line in _FALLBACK_OPERATOR_LINES:
        entry = _parse_operator_line(line)
        if entry is not None:
            entries.append(entry)
    return entries

def _fallback_combination_entries() -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    for line in _FALLBACK_COMBINATION_LINES:
        entry = _parse_combination_line(line)
        if entry is not None:
            entries.append(entry)
    return entries

@lru_cache(maxsize=1)
def _load_f2_sections() -> Dict[str, List[Dict[str, str]]]:
    operators: List[Dict[str, str]] = []
    combinations: List[Dict[str, str]] = []
    current_section = ''
    try:
        with open(_F2_OPERATORS_PATH, 'r', encoding='utf-8') as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith('## '):
                    current_section = line[3:].strip()
                    continue
                if current_section == 'Operator list':
                    entry = _parse_operator_line(line)
                    if entry is not None:
                        operators.append(entry)
                elif current_section == 'Composite operator examples':
                    entry = _parse_combination_line(line)
                    if entry is not None:
                        combinations.append(entry)
    except OSError:
        pass
    if not operators:
        operators = _fallback_operator_entries()
    if not combinations:
        combinations = _fallback_combination_entries()
    return {'operators': operators, 'combination_examples': combinations}

def get_f2_operator_entries() -> List[Dict[str, str]]:
    return [dict(item) for item in _load_f2_sections()['operators']]

def get_f2_combination_examples() -> List[Dict[str, str]]:
    return [dict(item) for item in _load_f2_sections()['combination_examples']]

def get_documented_operator_names() -> List[str]:
    return [item['name'] for item in _load_f2_sections()['operators']]

def get_known_operator_names(include_legacy_aliases: bool=True) -> List[str]:
    names = list(dict.fromkeys(get_documented_operator_names()))
    if include_legacy_aliases:
        for alias in _LEGACY_OPERATOR_ALIASES:
            if alias not in names:
                names.append(alias)
    return names

def get_legacy_operator_aliases() -> Dict[str, str]:
    return dict(_LEGACY_OPERATOR_ALIASES)

def normalize_operator_name(name: str) -> str:
    return _LEGACY_OPERATOR_ALIASES.get(name, name)

def build_operator_regex(require_call: bool=False, include_legacy_aliases: bool=True) -> str:
    names = get_known_operator_names(include_legacy_aliases=include_legacy_aliases)
    suffix = '\\s*\\(' if require_call else '\\b'
    return '\\b(' + '|'.join((re.escape(name) for name in names)) + ')' + suffix

def build_operator_reference_block() -> str:
    grouped: Dict[str, List[Dict[str, str]]] = {category: [] for category in _CATEGORY_ORDER}
    for item in get_f2_operator_entries():
        grouped.setdefault(item['category'], []).append(item)
    lines: List[str] = ['# Operator reference', '']
    for category in _CATEGORY_ORDER:
        items = grouped.get(category, [])
        if not items:
            continue
        lines.append(f'**{category}**:')
        for item in items:
            lines.append(f"- {item['signature']}: {item['description']}")
        lines.append('')
    return '\n'.join(lines).strip()

def build_combination_example_block() -> str:
    lines: List[str] = ['# Composite operator examples', '']
    for item in get_f2_combination_examples():
        lines.append(f"- {item['label']}: `{item['expression']}`")
    return '\n'.join(lines).strip()
