"""Parse the formula DSL into expression trees without executing arbitrary code."""
import re
from typing import Type, List, Dict, Set, Union, Optional, cast
from fraud_alpha.expression import *
from fraud_alpha.operator_catalog import get_legacy_operator_aliases
_PATTERN = re.compile('([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?|\\W|\\w+)')
_NUMERIC = re.compile('[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?')
_StackItem = Union[List[Type[Operator]], Expression]
_OpMap = Dict[str, List[Type[Operator]]]

def _expression_start_names() -> List[str]:
    names = {op.__name__ for op in Operators}
    names.update({'Max', 'Min', 'Delta', 'Where', 'Cond', 'Inverse', 'Negate', 'CrossRank', 'Clamp'})
    names.update(get_legacy_operator_aliases().keys())
    names.update(Feature.FEATURES)
    return sorted((name for name in names if name), key=len, reverse=True)

def extract_expression_candidates(line: str, ignore_case: bool=True) -> List[str]:
    text = line.strip().strip('`').strip('"').strip("'")
    if not text:
        return []
    names = _expression_start_names()
    if not names:
        return [text]
    flags = re.IGNORECASE if ignore_case else 0
    pattern = re.compile('(?<![A-Za-z0-9_])(' + '|'.join((re.escape(name) for name in names)) + ')(?=\\s*(?:\\(|$))', flags)
    candidates = []
    seen = set()
    for match in pattern.finditer(text):
        start = match.start()
        end = match.end()
        while end < len(text) and text[end].isspace():
            end += 1
        if end < len(text) and text[end] == '(':
            depth = 0
            balanced_end = None
            for idx in range(end, len(text)):
                ch = text[idx]
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0:
                        balanced_end = idx + 1
                        break
            if balanced_end is None:
                continue
            end = balanced_end
        candidate = text[start:end].strip().strip('`').strip('"').strip("'")
        if not candidate or candidate in seen:
            continue
        suffix = text[end:]
        terminal = suffix.strip(' \t\r\n`\'"，。,.;；:：!?！？') == ''
        candidates.append((candidate, terminal, end, len(candidate)))
        seen.add(candidate)
    candidates.sort(key=lambda item: (0 if item[1] else 1, -item[2], -item[3], item[0]))
    results = [item[0] for item in candidates]
    if text not in seen:
        results.append(text)
    return results

class ExpressionParsingError(Exception):
    pass

class ExpressionParser:

    def __init__(self, operators: List[Type[Operator]], ignore_case: bool=True, time_deltas_need_suffix: bool=False, additional_operator_mapping: Optional[_OpMap]=None):
        self._ignore_case = ignore_case
        self._suffix_needed = time_deltas_need_suffix
        self._feature_lookup = {f.lower() if ignore_case else f: f for f in Feature.FEATURES}
        self._features = set(self._feature_lookup.keys())
        self._operators: _OpMap = {op.__name__: [op] for op in operators}
        if additional_operator_mapping is not None:
            self._merge_op_mapping(additional_operator_mapping)
        if ignore_case:
            self._operators = {k.lower(): v for k, v in self._operators.items()}
        self._stack: List[_StackItem] = []
        self._tokens: List[str] = []

    def parse(self, expr: str) -> Expression:
        self._stack = []
        self._tokens = [t for t in _PATTERN.findall(expr) if not t.isspace()]
        self._tokens.reverse()
        while len(self._tokens) > 0:
            self._stack.append(self._get_next_item())
            self._process_punctuation()
        if len(self._stack) != 1:
            raise ExpressionParsingError(f'Multiple items remain in the stack: {self._stack}')
        if len(self._stack) == 0:
            raise ExpressionParsingError('Nothing was parsed')
        if isinstance(self._stack[0], Expression):
            return self._stack[0]
        raise ExpressionParsingError(f'{self._stack[0]} is not a valid expression')

    def _merge_op_mapping(self, map: _OpMap) -> None:
        for name, ops in map.items():
            if (old_ops := self._operators.get(name)) is not None:
                self._operators[name] = list(dict.fromkeys(old_ops + ops))
            else:
                self._operators[name] = ops

    def _get_next_item(self) -> _StackItem:
        top = self._pop_token()
        if _NUMERIC.fullmatch(top) is not None:
            value = self._to_float(top)
            next_token = self._peek_token()
            if next_token and next_token.lower() == 'y':
                self._pop_token()
                return self._as_delta_time(value)
            else:
                return Constant(value)
        top_check = top.lower() if self._ignore_case else top
        if top_check in self._features:
            return Feature(self._feature_lookup[top_check])
        op_key = top.lower() if self._ignore_case else top
        if (ops := self._operators.get(op_key)) is not None:
            return ops
        raise ExpressionParsingError(f'Cannot find the operator/feature name: {top}')

    def _process_punctuation(self) -> None:
        if len(self._tokens) == 0:
            return
        top = self._pop_token()
        stack_top_is_ops = len(self._stack) != 0 and (not isinstance(self._stack[-1], Expression))
        if (top == '(') != stack_top_is_ops:
            raise ExpressionParsingError('A left parenthesis should follow an operator name')
        if top == '(' or top == ',':
            return
        elif top == ')':
            self._build_one_subexpr()
            self._process_punctuation()
        elif len(top) > 1 and top[0] in ('+', '-') and (_NUMERIC.fullmatch(top) is not None):
            sign = top[0]
            num_part = top[1:]
            op_cls = Add if sign == '+' else Sub
            if not self._stack or not isinstance(self._stack[-1], Expression):
                raise ExpressionParsingError(f"Infix '{sign}' requires a left-hand expression")
            left = self._stack.pop()
            result = op_cls(left, Constant(self._to_float(num_part)))
            self._stack.append(result)
            self._process_punctuation()
        else:
            raise ExpressionParsingError(f'Unexpected token: {top}')

    def _build_one_subexpr(self) -> None:
        op_idx = -1
        for i in range(len(self._stack) - 1, -1, -1):
            if isinstance(self._stack[i], list):
                op_idx = i
                break
        if op_idx == -1:
            raise ExpressionParsingError('Unmatched right parenthesis')
        ops = cast(List[Type[Operator]], self._stack[op_idx])
        operands = self._stack[op_idx + 1:]
        self._stack = self._stack[:op_idx]
        if any((not isinstance(item, Expression) for item in operands)):
            raise ExpressionParsingError('An operator name cannot be used as an operand')
        operands = cast(List[Expression], operands)
        msgs: Set[str] = set()
        for op in ops:
            used_operands = operands
            if not self._suffix_needed and op.delta_time_arg_indices():
                converted = operands.copy()
                for idx in op.delta_time_arg_indices():
                    if idx >= len(converted):
                        continue
                    if isinstance(converted[idx], Constant):
                        dt = self._as_delta_time(converted[idx], noexcept=True)
                        if dt is not None:
                            converted[idx] = dt
                used_operands = converted
            try:
                result = op(*used_operands)
                self._stack.append(result)
                return
            except Exception as e:
                msgs.add(str(e))
        raise ExpressionParsingError('; '.join(msgs))

    def _to_float(self, token: str) -> float:
        try:
            return float(token)
        except:
            raise ExpressionParsingError(f"{token} can't be converted to float")

    def _pop_token(self) -> str:
        if len(self._tokens) == 0:
            raise ExpressionParsingError('No more tokens left')
        return self._tokens.pop()

    def _peek_token(self) -> Optional[str]:
        if len(self._tokens) == 0:
            return None
        return self._tokens[-1]

    def _as_delta_time(self, value: Union[float, Constant], noexcept: bool=False) -> Optional[DeltaTime]:
        if isinstance(value, Constant):
            value = value.value
        if value <= 0 or value != int(value):
            if noexcept:
                return None
            raise ExpressionParsingError(f'Time delta must be a positive integer, got {value}')
        try:
            return DeltaTime(int(value))
        except Exception as e:
            if noexcept:
                return None
            raise ExpressionParsingError(f'Failed to create DeltaTime: {e}')

def build_parser() -> ExpressionParser:
    alias_map: _OpMap = {'Max': [Greater], 'Min': [Less], 'Delta': [Sub], 'Where': [If], 'Cond': [If], 'Inverse': [Inv], 'Negate': [Neg], 'CrossRank': [CSRank], 'Clamp': [Clip]}
    for alias, target_name in get_legacy_operator_aliases().items():
        target_cls = globals().get(target_name)
        if isinstance(target_cls, type) and issubclass(target_cls, Operator):
            alias_map.setdefault(alias, []).append(target_cls)
    return ExpressionParser(Operators, ignore_case=True, time_deltas_need_suffix=False, additional_operator_mapping=alias_map)
