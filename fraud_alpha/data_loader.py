"""Load and preprocess numerical panel data using explicit configuration."""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Iterable, List, Optional
from sklearn.model_selection import train_test_split
import os
import re
_BASE_FEATURE_PATTERN = re.compile('^[NCB]\\d+$')
_LAG_FEATURE_PATTERN = re.compile('^[NCB]\\d+_lag\\d+$')
_TARGET_COLUMN_CANDIDATES = ('Fraud', 'fraud', 'label', 'Label', 'target', 'Target', 'is_fraud', 'isFraud', 'is_fraud', 'is_misstatement', 'fraud_label', 'misstatement_label', 'Y', 'y')
_YEAR_COLUMN_CANDIDATES = ('year', 'fiscal_year', 'report_year', 'Year', 'year', 'fyear', 'FYEAR', 'report_year', 'reporting_year')
_ID_COLUMN_CANDIDATES = ('firm_id', 'stock_code', 'company_code', 'code', 'Stkcd', 'stkcd', 'Symbol', 'symbol', 'Ticker', 'ticker', 'Code', 'code')

def _feature_sort_key(column: str) -> tuple:
    match = re.match('^([NCB])(\\d+)(?:_lag(\\d+))?$', column)
    if match is None:
        return (2, column, -1)
    prefix, feature_idx, lag_idx = match.groups()
    prefix_order = 0 if prefix == 'N' else 1 if prefix == 'C' else 2
    lag_order = int(lag_idx) if lag_idx is not None else -1
    return (prefix_order, int(feature_idx), lag_order)

def get_base_feature_columns(columns: Iterable[str]) -> List[str]:
    return sorted([str(col) for col in columns if _BASE_FEATURE_PATTERN.fullmatch(str(col))], key=_feature_sort_key)

def get_lag_feature_columns(columns: Iterable[str]) -> List[str]:
    return sorted([str(col) for col in columns if _LAG_FEATURE_PATTERN.fullmatch(str(col))], key=_feature_sort_key)

def _find_matching_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    normalized = {str(col).strip().lower(): str(col) for col in columns}
    for candidate in candidates:
        matched = normalized.get(str(candidate).strip().lower())
        if matched is not None:
            return matched
    return None

def _resolve_requested_feature_columns(columns: Iterable[str], requested_features: Iterable[str]) -> List[str]:
    resolved: List[str] = []
    seen = set()
    for feature in requested_features:
        matched = _find_matching_column(columns, [str(feature)])
        if matched is None or matched in seen:
            continue
        resolved.append(matched)
        seen.add(matched)
    return resolved

def _is_binary_series(series: pd.Series) -> bool:
    valid = series.dropna()
    if valid.empty:
        return False
    numeric = pd.to_numeric(valid, errors='coerce')
    if numeric.notna().all():
        uniq_numeric = numeric.dropna().unique().tolist()
        if len(uniq_numeric) == 2:
            return True
        rounded = numeric.round()
        if np.allclose(numeric.values, rounded.values):
            uniq = set(rounded.astype(int).tolist())
            return len(uniq) == 2
    text_values = {str(v).strip().lower() for v in valid.unique()}
    return len(text_values) == 2

def _normalize_binary_target(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return series
    numeric = pd.to_numeric(series, errors='coerce')
    valid_numeric = numeric.dropna()
    if len(valid_numeric) == len(valid):
        uniq_numeric = sorted(valid_numeric.unique().tolist())
        if len(uniq_numeric) == 2:
            mapping = {uniq_numeric[0]: 0, uniq_numeric[1]: 1}
            return valid_numeric.map(mapping).reindex(series.index)
    text_mapping = {'0': 0, 'false': 0, 'no': 0, 'n': 0, 'negative': 0, 'non-fraud': 0, 'nonfraud': 0, '1': 1, 'true': 1, 'yes': 1, 'y': 1, 'positive': 1, 'fraud': 1}
    normalized = series.map(lambda value: text_mapping.get(str(value).strip().lower()) if pd.notna(value) else np.nan)
    if normalized.dropna().shape[0] == valid.shape[0] and normalized.dropna().nunique() == 2:
        return normalized
    return series

def _is_numeric_like_series(series: pd.Series, threshold: float=0.95) -> bool:
    numeric = pd.to_numeric(series, errors='coerce')
    if len(series) == 0:
        return False
    return float(numeric.notna().mean()) >= threshold

class FraudDataLoader:

    def __init__(self, data_path: str, target_col: str='Fraud', split_mode: str='time', random_seed: int=42, feature_columns: Optional[List[str]]=None, sheet_name: Optional[str]=None):
        self.data_path = data_path
        self.target_col = target_col
        self.split_mode = split_mode
        self.random_seed = random_seed
        self.explicit_feature_columns = list(feature_columns or [])
        self.feature_columns: List[str] = []
        self.sheet_name = sheet_name
        self.year_col: Optional[str] = None
        self.id_col: Optional[str] = None
        self.df: pd.DataFrame = None
        self.train_df: pd.DataFrame = None
        self.valid_df: pd.DataFrame = None
        self.test_df: pd.DataFrame = None

    def load(self) -> None:
        print(f'Loading data: {self.data_path}')
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f'Data file does not exist: {self.data_path}')
        ext = os.path.splitext(self.data_path)[1].lower()
        if ext in ('.csv', '.tsv'):
            sep = '\t' if ext == '.tsv' else ','
            self.df = pd.read_csv(self.data_path, sep=sep, encoding='utf-8-sig')
        elif self.sheet_name:
            self.df = pd.read_excel(self.data_path, sheet_name=self.sheet_name)
        else:
            self.df = pd.read_excel(self.data_path)
        print(f'Raw data shape: {self.df.shape}')
        print(f'Original columns: {list(self.df.columns)}')
        self._normalize_schema()
        self._validate_columns()
        self._preprocess()
        self._split_data()

    def _normalize_schema(self) -> None:
        resolved_target = self._resolve_target_col()
        if resolved_target is None:
            raise ValueError('Cannot identify the target column; set data.target_col explicitly in config')
        self.target_col = resolved_target
        self.df[self.target_col] = _normalize_binary_target(self.df[self.target_col])
        resolved_year = self._resolve_year_col()
        if resolved_year is not None and resolved_year != 'year':
            self.df = self.df.rename(columns={resolved_year: 'year'})
        self.year_col = 'year' if resolved_year is not None else None
        resolved_id = self._resolve_id_col()
        if resolved_id is not None and resolved_id != 'firm_id':
            self.df = self.df.rename(columns={resolved_id: 'firm_id'})
        self.id_col = 'firm_id' if resolved_id is not None else None
        if self.explicit_feature_columns:
            self.feature_columns = _resolve_requested_feature_columns(self.df.columns, self.explicit_feature_columns)
        else:
            self.feature_columns = get_base_feature_columns(self.df.columns)
        print(f'Identified target column: {self.target_col}')
        if self.year_col is not None:
            print(f'Identified year column: {self.year_col}')
        if self.id_col is not None:
            print(f'Identified ID column: {self.id_col}')
        print(f'Identified base feature columns: {len(self.feature_columns)}')

    def _resolve_target_col(self) -> Optional[str]:
        if self.target_col:
            matched = _find_matching_column(self.df.columns, [self.target_col])
            if matched is not None:
                return matched
        matched = _find_matching_column(self.df.columns, _TARGET_COLUMN_CANDIDATES)
        if matched is not None:
            return matched
        excluded = set(get_base_feature_columns(self.df.columns))
        excluded.update(get_lag_feature_columns(self.df.columns))
        excluded.update(_resolve_requested_feature_columns(self.df.columns, self.explicit_feature_columns))
        binary_candidates = []
        for column in self.df.columns:
            column_name = str(column)
            if column_name in excluded:
                continue
            if _is_binary_series(self.df[column]):
                binary_candidates.append(column_name)
        if len(binary_candidates) == 1:
            return binary_candidates[0]
        return None

    def _resolve_year_col(self) -> Optional[str]:
        matched = _find_matching_column(self.df.columns, _YEAR_COLUMN_CANDIDATES)
        if matched is not None:
            return matched
        excluded = set(get_base_feature_columns(self.df.columns))
        excluded.update(get_lag_feature_columns(self.df.columns))
        excluded.update(_resolve_requested_feature_columns(self.df.columns, self.explicit_feature_columns))
        if self.target_col:
            excluded.add(self.target_col)
        for column in self.df.columns:
            column_name = str(column)
            if column_name in excluded:
                continue
            numeric = pd.to_numeric(self.df[column], errors='coerce')
            valid = numeric.dropna()
            if len(valid) < max(5, len(self.df) // 2):
                continue
            rounded = valid.round()
            if not np.allclose(valid.values, rounded.values):
                continue
            if valid.between(1900, 2100).all() and valid.nunique() >= 3:
                return column_name
        return None

    def _resolve_id_col(self) -> Optional[str]:
        matched = _find_matching_column(self.df.columns, _ID_COLUMN_CANDIDATES)
        if matched is not None:
            return matched
        return None

    def _validate_columns(self) -> None:
        if self.target_col not in self.df.columns:
            raise ValueError(f"Missing data column '{self.target_col}'")
        feature_cols = self.get_feature_columns()
        if not feature_cols:
            raise ValueError('No base feature columns found in data')

    def _preprocess(self) -> None:
        print('Preprocessing data...')
        feature_cols = self.get_feature_columns()
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        for col in feature_cols:
            if _is_numeric_like_series(self.df[col]):
                numeric_series = pd.to_numeric(self.df[col], errors='coerce')
                if numeric_series.isna().any():
                    self.df[col] = numeric_series.fillna(numeric_series.median())
                else:
                    self.df[col] = numeric_series
            elif self.df[col].isna().any():
                mode_val = self.df[col].mode(dropna=True)
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 0
                self.df[col] = self.df[col].fillna(fill_val)
        remaining_nan = self.df[feature_cols].isna().sum().sum()
        print(f'Preprocessed data shape: {self.df.shape}, remaining NaNs: {remaining_nan}')

    def _split_data(self) -> None:
        if self.split_mode == 'random':
            self._split_data_random()
        else:
            self._split_data_time()

    def _split_data_time(self) -> None:
        if 'year' not in self.df.columns:
            raise ValueError("Missing 'year' column; cannot split by year")
        self.train_df = self.df[self.df['year'].between(2010, 2018)].copy()
        print(f'Training set   (2010-2018): {self.train_df.shape}, {self.target_col}={self.train_df[self.target_col].sum()}')
        self.valid_df = self.df[self.df['year'].between(2019, 2021)].copy()
        print(f'Validation set (2019-2021): {self.valid_df.shape}, {self.target_col}={self.valid_df[self.target_col].sum()}')
        self.test_df = self.df[self.df['year'].between(2022, 2023)].copy()
        print(f'Test set       (2022-2023): {self.test_df.shape}, {self.target_col}={self.test_df[self.target_col].sum()}')

    def _split_data_random(self) -> None:
        y = self.df[self.target_col].values
        train_valid_df, self.test_df = train_test_split(self.df, test_size=0.3, random_state=self.random_seed, stratify=y)
        y_tv = train_valid_df[self.target_col].values
        self.train_df, self.valid_df = train_test_split(train_valid_df, test_size=0.2, random_state=self.random_seed, stratify=y_tv)
        sort_cols = []
        if 'firm_id' in self.df.columns:
            sort_cols.append('firm_id')
        if 'year' in self.df.columns:
            sort_cols.append('year')
        if sort_cols:
            self.train_df = self.train_df.sort_values(sort_cols).reset_index(drop=True)
            self.valid_df = self.valid_df.sort_values(sort_cols).reset_index(drop=True)
            self.test_df = self.test_df.sort_values(sort_cols).reset_index(drop=True)
        else:
            self.train_df = self.train_df.copy().reset_index(drop=True)
            self.valid_df = self.valid_df.reset_index(drop=True)
            self.test_df = self.test_df.copy().reset_index(drop=True)
        print(f'Training set   (56%): {self.train_df.shape}, {self.target_col}={self.train_df[self.target_col].sum()}')
        print(f'Validation set (14%): {self.valid_df.shape}, {self.target_col}={self.valid_df[self.target_col].sum()}')
        print(f'Test set       (30%): {self.test_df.shape}, {self.target_col}={self.test_df[self.target_col].sum()}')

    def get_feature_columns(self) -> list:
        if self.feature_columns:
            return list(self.feature_columns)
        return get_base_feature_columns(self.df.columns)

    def get_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.train_df is None:
            raise ValueError('Data not loaded; call load() first')
        return (self.train_df, self.valid_df, self.test_df)

    def get_full_train_df(self) -> pd.DataFrame:
        if self.train_df is None:
            raise ValueError('Data not loaded; call load() first')
        full_train = pd.concat([self.train_df, self.valid_df], ignore_index=True)
        sort_cols = [c for c in ['firm_id', 'year'] if c in full_train.columns]
        if sort_cols:
            full_train = full_train.sort_values(sort_cols).reset_index(drop=True)
        return full_train

    def get_all_data(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError('Data not loaded; call load() first')
        return self.df

    def get_stats(self) -> Dict:
        stats = {'target_col': self.target_col, 'year_col': self.year_col, 'id_col': self.id_col, 'total_samples': len(self.df), 'train_samples': len(self.train_df), 'valid_samples': len(self.valid_df), 'test_samples': len(self.test_df), 'n_features': len(self.get_feature_columns()), f'{self.target_col}_ratio': self.df[self.target_col].mean(), f'train_{self.target_col}_ratio': self.train_df[self.target_col].mean(), f'valid_{self.target_col}_ratio': self.valid_df[self.target_col].mean(), f'test_{self.target_col}_ratio': self.test_df[self.target_col].mean()}
        return stats
