"""Expression tree and panel operators for firm-grouped historical features."""
from abc import ABCMeta, abstractmethod
from typing import List, Type, Union, Tuple, Iterable
import numpy as np
import pandas as pd
import re
_ExprOrFloat = Union['Expression', float]
_DTimeOrInt = Union['DeltaTime', int]

def _feature_sort_key(column: str) -> tuple:
    match = re.match('^([NC])(\\d+)(?:_lag(\\d+))?$', column)
    if match is None:
        return (2, column, -1)
    prefix, feature_idx, lag_idx = match.groups()
    prefix_order = 0 if prefix == 'N' else 1
    lag_order = int(lag_idx) if lag_idx is not None else -1
    return (prefix_order, int(feature_idx), lag_order)

class OutOfDataRangeError(IndexError):
    pass

class Expression(metaclass=ABCMeta):

    @abstractmethod
    def evaluate(self, data: pd.DataFrame) -> np.ndarray:
        pass

    def __repr__(self) -> str:
        return str(self)

    def __add__(self, other: _ExprOrFloat) -> 'Add':
        return Add(self, other)

    def __radd__(self, other: float) -> 'Add':
        return Add(other, self)

    def __sub__(self, other: _ExprOrFloat) -> 'Sub':
        return Sub(self, other)

    def __rsub__(self, other: float) -> 'Sub':
        return Sub(other, self)

    def __mul__(self, other: _ExprOrFloat) -> 'Mul':
        return Mul(self, other)

    def __rmul__(self, other: float) -> 'Mul':
        return Mul(other, self)

    def __truediv__(self, other: _ExprOrFloat) -> 'Div':
        return Div(self, other)

    def __rtruediv__(self, other: float) -> 'Div':
        return Div(other, self)

    def __pow__(self, other: _ExprOrFloat) -> 'Pow':
        return Pow(self, other)

    def __rpow__(self, other: float) -> 'Pow':
        return Pow(other, self)

    def __pos__(self) -> 'Expression':
        return self

    def __neg__(self) -> 'Sub':
        return Sub(0.0, self)

    def __abs__(self) -> 'Abs':
        return Abs(self)

    @property
    @abstractmethod
    def is_featured(self) -> bool:
        pass

class Feature(Expression):
    FEATURES = [f'N{i}' for i in range(1, 37)] + [f'C{i}' for i in range(1, 5)]

    @classmethod
    def set_available_features(cls, features: Iterable[str]) -> None:
        cls.FEATURES = sorted(list(dict.fromkeys((str(feature) for feature in features))), key=_feature_sort_key)

    @classmethod
    def get_available_features(cls) -> List[str]:
        return list(cls.FEATURES)

    def __init__(self, feature: str) -> None:
        if feature not in self.FEATURES:
            raise ValueError(f'Unknown feature: {feature}. Must be one of {self.FEATURES}')
        self._feature = feature

    def evaluate(self, data: pd.DataFrame) -> np.ndarray:
        if self._feature not in data.columns:
            raise ValueError(f'Feature {self._feature} not found in data')
        return data[self._feature].values

    def __str__(self) -> str:
        return self._feature

    @property
    def is_featured(self) -> bool:
        return True

class Constant(Expression):

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def evaluate(self, data: pd.DataFrame) -> np.ndarray:
        return np.full(len(data), self.value, dtype=np.float64)

    def __str__(self) -> str:
        return str(self.value)

    @property
    def is_featured(self) -> bool:
        return False

class DeltaTime(Expression):

    def __init__(self, delta_time: int) -> None:
        if not isinstance(delta_time, int) or delta_time < 1:
            raise ValueError(f'DeltaTime must be a positive integer, got {delta_time}')
        self._delta_time = delta_time

    def evaluate(self, data: pd.DataFrame) -> np.ndarray:
        raise AssertionError('Should not call evaluate on DeltaTime')

    def __str__(self) -> str:
        return f'{self._delta_time}y'

    @property
    def is_featured(self) -> bool:
        return False

def _into_expr(value: _ExprOrFloat) -> Expression:
    return value if isinstance(value, Expression) else Constant(value)

def _into_delta_time(value: Union[int, DeltaTime]) -> DeltaTime:
    return value if isinstance(value, DeltaTime) else DeltaTime(value)

class Operator(Expression):

    @classmethod
    @abstractmethod
    def n_args(cls) -> int:
        pass

    @classmethod
    @abstractmethod
    def category_type(cls) -> Type['Operator']:
        pass

    @classmethod
    def delta_time_arg_indices(cls) -> Tuple[int, ...]:
        return ()

    @property
    @abstractmethod
    def operands(self) -> Tuple[Expression, ...]:
        pass

    def __str__(self) -> str:
        return f"{type(self).__name__}({','.join((str(op) for op in self.operands))})"

class UnaryOperator(Operator):

    def __init__(self, operand: _ExprOrFloat) -> None:
        self._operand = _into_expr(operand)

    @classmethod
    def n_args(cls) -> int:
        return 1

    @classmethod
    def category_type(cls):
        return UnaryOperator

    @abstractmethod
    def _apply(self, operand: np.ndarray) -> np.ndarray:
        pass

    def evaluate(self, data: pd.DataFrame) -> np.ndarray:
        return self._apply(self._operand.evaluate(data))

    @property
    def operands(self):
        return (self._operand,)

    @property
    def is_featured(self):
        return self._operand.is_featured

class BinaryOperator(Operator):

    def __init__(self, lhs: _ExprOrFloat, rhs: _ExprOrFloat) -> None:
        self._lhs = _into_expr(lhs)
        self._rhs = _into_expr(rhs)

    @classmethod
    def n_args(cls) -> int:
        return 2

    @classmethod
    def category_type(cls):
        return BinaryOperator

    @abstractmethod
    def _apply(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        pass

    def evaluate(self, data: pd.DataFrame) -> np.ndarray:
        return self._apply(self._lhs.evaluate(data), self._rhs.evaluate(data))

    @property
    def operands(self):
        return (self._lhs, self._rhs)

    @property
    def is_featured(self):
        return self._lhs.is_featured or self._rhs.is_featured

class RollingOperator(Operator):

    def __init__(self, operand: _ExprOrFloat, delta_time: _DTimeOrInt) -> None:
        self._operand = _into_expr(operand)
        self._delta_time = _into_delta_time(delta_time)._delta_time

    @classmethod
    def n_args(cls) -> int:
        return 2

    @classmethod
    def category_type(cls):
        return RollingOperator

    @classmethod
    def delta_time_arg_indices(cls) -> Tuple[int, ...]:
        return (1,)

    @abstractmethod
    def _apply(self, data: pd.DataFrame, operand_values: np.ndarray, window: int) -> np.ndarray:
        pass

    def evaluate(self, data: pd.DataFrame) -> np.ndarray:
        operand_values = self._operand.evaluate(data)
        return self._apply(data, operand_values, self._delta_time)

    @property
    def operands(self):
        return (self._operand, DeltaTime(self._delta_time))

    @property
    def is_featured(self):
        return self._operand.is_featured

class PairRollingOperator(Operator):

    def __init__(self, lhs: _ExprOrFloat, rhs: _ExprOrFloat, delta_time: _DTimeOrInt) -> None:
        self._lhs = _into_expr(lhs)
        self._rhs = _into_expr(rhs)
        self._delta_time = _into_delta_time(delta_time)._delta_time

    @classmethod
    def n_args(cls) -> int:
        return 3

    @classmethod
    def category_type(cls):
        return PairRollingOperator

    @classmethod
    def delta_time_arg_indices(cls) -> Tuple[int, ...]:
        return (2,)

    @abstractmethod
    def _apply(self, data: pd.DataFrame, lhs_values: np.ndarray, rhs_values: np.ndarray, window: int) -> np.ndarray:
        pass

    def evaluate(self, data: pd.DataFrame) -> np.ndarray:
        lhs_values = self._lhs.evaluate(data)
        rhs_values = self._rhs.evaluate(data)
        return self._apply(data, lhs_values, rhs_values, self._delta_time)

    @property
    def operands(self):
        return (self._lhs, self._rhs, DeltaTime(self._delta_time))

    @property
    def is_featured(self):
        return self._lhs.is_featured or self._rhs.is_featured

class TernaryOperator(Operator):

    def __init__(self, first: _ExprOrFloat, second: _ExprOrFloat, third: _ExprOrFloat) -> None:
        self._first = _into_expr(first)
        self._second = _into_expr(second)
        self._third = _into_expr(third)

    @classmethod
    def n_args(cls) -> int:
        return 3

    @classmethod
    def category_type(cls):
        return TernaryOperator

    @abstractmethod
    def _apply(self, first: np.ndarray, second: np.ndarray, third: np.ndarray) -> np.ndarray:
        pass

    def evaluate(self, data: pd.DataFrame) -> np.ndarray:
        return self._apply(self._first.evaluate(data), self._second.evaluate(data), self._third.evaluate(data))

    @property
    def operands(self):
        return (self._first, self._second, self._third)

    @property
    def is_featured(self):
        return self._first.is_featured or self._second.is_featured or self._third.is_featured

class CrossSectionalOperator(Operator):

    def __init__(self, operand: _ExprOrFloat) -> None:
        self._operand = _into_expr(operand)

    @classmethod
    def n_args(cls) -> int:
        return 1

    @classmethod
    def category_type(cls):
        return CrossSectionalOperator

    @abstractmethod
    def _apply(self, data: pd.DataFrame, operand_values: np.ndarray) -> np.ndarray:
        pass

    def evaluate(self, data: pd.DataFrame) -> np.ndarray:
        return self._apply(data, self._operand.evaluate(data))

    @property
    def operands(self):
        return (self._operand,)

    @property
    def is_featured(self):
        return self._operand.is_featured

class Abs(UnaryOperator):

    def _apply(self, operand: np.ndarray) -> np.ndarray:
        return np.abs(operand)

class Sign(UnaryOperator):

    def _apply(self, operand: np.ndarray) -> np.ndarray:
        return np.sign(operand)

class Log(UnaryOperator):

    def _apply(self, operand: np.ndarray) -> np.ndarray:
        return np.log(np.abs(operand) + 1e-08)

class Sqrt(UnaryOperator):

    def _apply(self, operand: np.ndarray) -> np.ndarray:
        return np.sqrt(np.abs(operand))

class Square(UnaryOperator):

    def _apply(self, operand: np.ndarray) -> np.ndarray:
        return operand ** 2

class Cube(UnaryOperator):

    def _apply(self, operand: np.ndarray) -> np.ndarray:
        return operand ** 3

class Log10(UnaryOperator):

    def _apply(self, operand: np.ndarray) -> np.ndarray:
        return np.log10(np.abs(operand) + 1.0)

class Exp(UnaryOperator):

    def _apply(self, operand: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(operand, -50, 50))

class Sin(UnaryOperator):

    def _apply(self, operand: np.ndarray) -> np.ndarray:
        return np.sin(operand)

class Cos(UnaryOperator):

    def _apply(self, operand: np.ndarray) -> np.ndarray:
        return np.cos(operand)

class Tan(UnaryOperator):

    def _apply(self, operand: np.ndarray) -> np.ndarray:
        clipped = np.clip(operand, -1.55, 1.55)
        result = np.tan(clipped)
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

class Neg(UnaryOperator):

    def _apply(self, operand: np.ndarray) -> np.ndarray:
        return -operand

class Inv(UnaryOperator):

    def _apply(self, operand: np.ndarray) -> np.ndarray:
        return 1.0 / (operand + np.where(operand >= 0, 1e-08, -1e-08))

class Tanh(UnaryOperator):

    def _apply(self, operand: np.ndarray) -> np.ndarray:
        return np.tanh(operand)

class Sigmoid(UnaryOperator):

    def _apply(self, operand: np.ndarray) -> np.ndarray:
        clipped = np.clip(operand, -50, 50)
        return 1.0 / (1.0 + np.exp(-clipped))

class Add(BinaryOperator):

    def _apply(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return lhs + rhs

class Sub(BinaryOperator):

    def _apply(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return lhs - rhs

class Mul(BinaryOperator):

    def _apply(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return lhs * rhs

class Div(BinaryOperator):

    def _apply(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return lhs / (rhs + 1e-08)

class Pow(BinaryOperator):

    def _apply(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return np.power(np.abs(lhs), rhs)

class Greater(BinaryOperator):

    def _apply(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return np.maximum(lhs, rhs)

class Less(BinaryOperator):

    def _apply(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return np.minimum(lhs, rhs)

class If(TernaryOperator):

    def _apply(self, first: np.ndarray, second: np.ndarray, third: np.ndarray) -> np.ndarray:
        return np.where(first > 0, second, third)

class Clip(TernaryOperator):

    def _apply(self, first: np.ndarray, second: np.ndarray, third: np.ndarray) -> np.ndarray:
        return np.clip(first, second, third)

def _rolling_apply(data: pd.DataFrame, values: np.ndarray, window: int, func, *args) -> np.ndarray:
    if 'year' not in data.columns:
        result = pd.Series(values).rolling(window=window, min_periods=1).apply(func, args=args, raw=True)
        return result.values
    df_temp = pd.DataFrame({'value': values}, index=data.index)
    if 'firm_id' in data.columns:
        result = df_temp.groupby(data['firm_id'])['value'].transform(lambda x: x.rolling(window=window, min_periods=1).apply(func, args=args, raw=True))
    else:
        result = df_temp['value'].rolling(window=window, min_periods=1).apply(func, args=args, raw=True)
    return result.values

class Ref(RollingOperator):

    def _apply(self, data: pd.DataFrame, operand_values: np.ndarray, window: int) -> np.ndarray:
        df_temp = pd.DataFrame({'value': operand_values}, index=data.index)
        if 'firm_id' in data.columns:
            result = df_temp.groupby(data['firm_id'])['value'].transform(lambda x: x.shift(window).fillna(x))
        else:
            result = df_temp['value'].shift(window)
            result = result.ffill().bfill()
        return result.values

class Mean(RollingOperator):

    def _apply(self, data: pd.DataFrame, operand_values: np.ndarray, window: int) -> np.ndarray:
        return _rolling_apply(data, operand_values, window, np.mean)

class Sum(RollingOperator):

    def _apply(self, data: pd.DataFrame, operand_values: np.ndarray, window: int) -> np.ndarray:
        return _rolling_apply(data, operand_values, window, np.sum)

class Std(RollingOperator):

    def _apply(self, data: pd.DataFrame, operand_values: np.ndarray, window: int) -> np.ndarray:
        return _rolling_apply(data, operand_values, window, np.std)

class Var(RollingOperator):

    def _apply(self, data: pd.DataFrame, operand_values: np.ndarray, window: int) -> np.ndarray:
        return _rolling_apply(data, operand_values, window, np.var)

class Max(RollingOperator):

    def _apply(self, data: pd.DataFrame, operand_values: np.ndarray, window: int) -> np.ndarray:
        return _rolling_apply(data, operand_values, window, np.max)

class Min(RollingOperator):

    def _apply(self, data: pd.DataFrame, operand_values: np.ndarray, window: int) -> np.ndarray:
        return _rolling_apply(data, operand_values, window, np.min)

class Med(RollingOperator):

    def _apply(self, data: pd.DataFrame, operand_values: np.ndarray, window: int) -> np.ndarray:
        return _rolling_apply(data, operand_values, window, np.median)

class Mad(RollingOperator):

    def _apply(self, data: pd.DataFrame, operand_values: np.ndarray, window: int) -> np.ndarray:

        def mad_func(x):
            return np.mean(np.abs(x - np.mean(x)))
        return _rolling_apply(data, operand_values, window, mad_func)

class Quantile(Operator):

    def __init__(self, operand: _ExprOrFloat, delta_time: _DTimeOrInt, quantile: _ExprOrFloat) -> None:
        self._operand = _into_expr(operand)
        self._delta_time = _into_delta_time(delta_time)._delta_time
        self._quantile = _into_expr(quantile)
        if not isinstance(self._quantile, Constant):
            raise ValueError('Quantile q must be a constant')
        if not 0.0 <= self._quantile.value <= 1.0:
            raise ValueError(f'Quantile q must be in [0, 1], got {self._quantile.value}')

    @classmethod
    def n_args(cls) -> int:
        return 3

    @classmethod
    def category_type(cls):
        return Operator

    @classmethod
    def delta_time_arg_indices(cls) -> Tuple[int, ...]:
        return (1,)

    def evaluate(self, data: pd.DataFrame) -> np.ndarray:
        operand_values = self._operand.evaluate(data)
        quantile = self._quantile.value
        return _rolling_apply(data, operand_values, self._delta_time, np.quantile, quantile)

    @property
    def operands(self):
        return (self._operand, DeltaTime(self._delta_time), self._quantile)

    @property
    def is_featured(self):
        return self._operand.is_featured

class Range(RollingOperator):

    def _apply(self, data: pd.DataFrame, operand_values: np.ndarray, window: int) -> np.ndarray:

        def range_func(x):
            return np.max(x) - np.min(x)
        return _rolling_apply(data, operand_values, window, range_func)

class Delta(RollingOperator):

    def _apply(self, data: pd.DataFrame, operand_values: np.ndarray, window: int) -> np.ndarray:
        df_temp = pd.DataFrame({'value': operand_values}, index=data.index)
        if 'firm_id' in data.columns:
            result = df_temp.groupby(data['firm_id'])['value'].transform(lambda x: x - x.shift(window).fillna(x))
        else:
            shifted = df_temp['value'].shift(window)
            shifted = shifted.ffill().bfill()
            result = df_temp['value'] - shifted
        return result.values

class WMA(RollingOperator):

    def _apply(self, data: pd.DataFrame, operand_values: np.ndarray, window: int) -> np.ndarray:

        def wma_func(x):
            weights = np.arange(1, len(x) + 1)
            return np.sum(weights * x) / np.sum(weights)
        return _rolling_apply(data, operand_values, window, wma_func)

class DecayLinear(WMA):
    pass

class EMA(RollingOperator):

    def _apply(self, data: pd.DataFrame, operand_values: np.ndarray, window: int) -> np.ndarray:
        alpha = 2 / (window + 1)
        values = pd.Series(operand_values, index=data.index)
        if 'firm_id' in data.columns:
            result = values.groupby(data['firm_id'], sort=False).transform(lambda x: x.ewm(alpha=alpha, adjust=False).mean())
        else:
            result = values.ewm(alpha=alpha, adjust=False).mean()
        return result.values

class Skew(RollingOperator):

    def _apply(self, data: pd.DataFrame, operand_values: np.ndarray, window: int) -> np.ndarray:
        from scipy import stats

        def skew_func(x):
            if len(x) < 3:
                return 0
            return stats.skew(x, nan_policy='omit')
        return _rolling_apply(data, operand_values, window, skew_func)

class Kurt(RollingOperator):

    def _apply(self, data: pd.DataFrame, operand_values: np.ndarray, window: int) -> np.ndarray:
        from scipy import stats

        def kurt_func(x):
            if len(x) < 4:
                return 0
            return stats.kurtosis(x, nan_policy='omit')
        return _rolling_apply(data, operand_values, window, kurt_func)

class Slope(RollingOperator):

    def _apply(self, data: pd.DataFrame, operand_values: np.ndarray, window: int) -> np.ndarray:

        def slope_func(x):
            if len(x) < 2:
                return 0.0
            idx = np.arange(len(x), dtype=np.float64)
            return float(np.polyfit(idx, x, 1)[0])
        return _rolling_apply(data, operand_values, window, slope_func)

class Rank(RollingOperator):

    def _apply(self, data: pd.DataFrame, operand_values: np.ndarray, window: int) -> np.ndarray:

        def rank_func(x):
            last_val = x[-1]
            return np.sum(x < last_val) / len(x)
        return _rolling_apply(data, operand_values, window, rank_func)

class TSRank(Rank):
    pass

class ROC(RollingOperator):

    def _apply(self, data: pd.DataFrame, operand_values: np.ndarray, window: int) -> np.ndarray:
        df_temp = pd.DataFrame({'value': operand_values}, index=data.index)
        if 'firm_id' in data.columns:
            ref_values = df_temp.groupby(data['firm_id'])['value'].transform(lambda x: x.shift(window).fillna(x)).values
        else:
            ref_series = df_temp['value'].shift(window)
            ref_series = ref_series.ffill().bfill()
            ref_values = ref_series.values
        safe_ref = np.where(np.abs(ref_values) < 1e-08, np.where(ref_values < 0, -1e-08, 1e-08), ref_values)
        return (operand_values - ref_values) / safe_ref

class Momentum(Delta):
    pass

class ZScore(RollingOperator):

    def _apply(self, data: pd.DataFrame, operand_values: np.ndarray, window: int) -> np.ndarray:
        mean_vals = _rolling_apply(data, operand_values, window, np.mean)
        std_vals = _rolling_apply(data, operand_values, window, np.std)
        return (operand_values - mean_vals) / (std_vals + 1e-08)

class ArgMax(RollingOperator):

    def _apply(self, data: pd.DataFrame, operand_values: np.ndarray, window: int) -> np.ndarray:

        def argmax_func(x):
            return np.argmax(x) / max(len(x) - 1, 1)
        return _rolling_apply(data, operand_values, window, argmax_func)

class ArgMin(RollingOperator):

    def _apply(self, data: pd.DataFrame, operand_values: np.ndarray, window: int) -> np.ndarray:

        def argmin_func(x):
            return np.argmin(x) / max(len(x) - 1, 1)
        return _rolling_apply(data, operand_values, window, argmin_func)

def _pair_rolling_apply(data: pd.DataFrame, lhs_values: np.ndarray, rhs_values: np.ndarray, window: int, func) -> np.ndarray:
    df_temp = pd.DataFrame({'lhs': lhs_values, 'rhs': rhs_values}, index=data.index)
    if 'firm_id' in data.columns:
        result = df_temp.groupby(data['firm_id']).apply(lambda g: g.rolling(window=window, min_periods=1).apply(lambda w: func(w['lhs'].values, w['rhs'].values), raw=False).iloc[:, 0]).reset_index(level=0, drop=True)
    else:
        result = df_temp.rolling(window=window, min_periods=1).apply(lambda w: func(w[:, 0], w[:, 1]) if len(w) > 0 else np.nan, raw=True).iloc[:, 0]
    return result.values

def _pair_rolling_by_firm(data, lhs_values, rhs_values, window, operation):
    groups = data.groupby('firm_id', sort=False).indices.values() if 'firm_id' in data.columns else [np.arange(len(data))]
    out = np.empty(len(data), dtype=float)
    for positions in groups:
        lhs = pd.Series(np.asarray(lhs_values)[positions])
        rhs = pd.Series(np.asarray(rhs_values)[positions])
        rolling = lhs.rolling(window=window, min_periods=1)
        result = getattr(rolling, operation)(rhs)
        out[positions] = result.fillna(0).to_numpy()
    return out

class Cov(PairRollingOperator):

    def _apply(self, data: pd.DataFrame, lhs_values: np.ndarray, rhs_values: np.ndarray, window: int) -> np.ndarray:
        return _pair_rolling_by_firm(data, lhs_values, rhs_values, window, 'cov')

class Corr(PairRollingOperator):

    def _apply(self, data: pd.DataFrame, lhs_values: np.ndarray, rhs_values: np.ndarray, window: int) -> np.ndarray:
        return _pair_rolling_by_firm(data, lhs_values, rhs_values, window, 'corr')

class CSRank(CrossSectionalOperator):

    def _apply(self, data: pd.DataFrame, operand_values: np.ndarray) -> np.ndarray:
        df_temp = pd.DataFrame({'value': operand_values}, index=data.index)
        if 'year' in data.columns:
            result = df_temp.groupby(data['year'])['value'].transform(lambda x: x.rank(pct=True, na_option='keep'))
        else:
            result = df_temp['value'].rank(pct=True, na_option='keep')
        return result.fillna(0.5).values
Operators: List[Type[Operator]] = [Abs, Log, Log10, Sqrt, Square, Cube, Exp, Sign, Sin, Cos, Tan, Neg, Inv, Tanh, Sigmoid, Add, Sub, Mul, Div, Pow, Greater, Less, If, Clip, Ref, Delta, Mean, Sum, Std, Var, Max, Min, Med, Mad, Skew, Kurt, Quantile, Range, WMA, EMA, DecayLinear, Slope, TSRank, Rank, ROC, Momentum, ZScore, ArgMax, ArgMin, Cov, Corr, CSRank]
