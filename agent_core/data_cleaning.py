import numpy as np
import pandas as pd
import re
from typing import Optional, Dict, List, Tuple, Any


class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.cleaning_log: List[str] = []

    def _log(self, msg: str):
        self.cleaning_log.append(msg)

    def standardize_column_names(self):
        n_before = len(self.df.columns)
        rename_map = {}
        for col in self.df.columns:
            new = col.strip().lower()
            new = re.sub(r'[^\w\s]', '_', new)
            new = re.sub(r'\s+', '_', new)
            new = re.sub(r'_+', '_', new).strip('_')
            rename_map[col] = new
        self.df = self.df.rename(columns=rename_map)
        self._log(f"Standardized {n_before} column names to snake_case")
        return self

    def detect_variable_types(self) -> Dict[str, str]:
        type_map = {}
        for col in self.df.columns:
            series = self.df[col].dropna()
            if series.empty:
                type_map[col] = 'unknown'
                continue
            n_unique = series.nunique()
            dtype = self.df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                if n_unique <= 2:
                    type_map[col] = 'binary'
                elif n_unique <= 10 and n_unique <= len(series) * 0.05:
                    type_map[col] = 'ordinal'
                else:
                    type_map[col] = 'continuous'
            elif pd.api.types.is_datetime64_dtype(dtype):
                type_map[col] = 'datetime'
            else:
                if n_unique <= 2:
                    type_map[col] = 'binary'
                elif n_unique <= 20:
                    type_map[col] = 'categorical'
                else:
                    type_map[col] = 'text'
        return type_map

    def extract_numeric_values(self, columns: Optional[List[str]] = None):
        if columns is None:
            columns = self.df.select_dtypes(include='object').columns
        for col in columns:
            if col not in self.df.columns:
                continue
            cleaned = self.df[col].astype(str).str.extract(r'(-?\d+\.?\d*)', expand=False)
            cleaned = pd.to_numeric(cleaned, errors='coerce')
            if cleaned.notna().sum() > len(self.df) * 0.5:
                self.df[col] = cleaned
                self._log(f"Extracted numeric values from '{col}'")
        return self

    def impute_missing(self, strategy: str = 'median', columns: Optional[List[str]] = None):
        if columns is None:
            columns = self.df.select_dtypes(include=np.number).columns
        for col in columns:
            if col not in self.df.columns:
                continue
            n_miss = self.df[col].isna().sum()
            if n_miss == 0:
                continue
            if strategy == 'median':
                val = self.df[col].median()
            elif strategy == 'mean':
                val = self.df[col].mean()
            elif strategy == 'mode':
                val = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else 0
            else:
                val = 0
            self.df[col] = self.df[col].fillna(val)
            self._log(f"Imputed {n_miss} missing values in '{col}' with {strategy}={val:.2f}")
        return self

    def cap_outliers(self, multiplier: float = 1.5):
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        total_capped = 0
        for col in numeric_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lo = q1 - multiplier * iqr
            hi = q3 + multiplier * iqr
            n_lo = (self.df[col] < lo).sum()
            n_hi = (self.df[col] > hi).sum()
            if n_lo + n_hi > 0:
                self.df[col] = self.df[col].clip(lo, hi)
                total_capped += n_lo + n_hi
                self._log(f"Capped {n_lo + n_hi} outliers in '{col}' ({lo:.2f}, {hi:.2f})")
        self._log(f"Total outliers capped across all variables: {total_capped}")
        return self

    def standardize_categoricals(self, columns: Optional[List[str]] = None):
        if columns is None:
            columns = self.df.select_dtypes(include='object').columns
        for col in columns:
            if col not in self.df.columns:
                continue
            self.df[col] = self.df[col].astype(str).str.strip().str.lower()
            self.df[col] = self.df[col].replace({
                r'^y(es)?$': 'yes', r'^n(o)?$': 'no',
                r'^true$': 'yes', r'^false$': 'no',
                r'^1$': 'yes', r'^0$': 'no',
                r'^male$': 'male', r'^female$': 'female',
                r'^m$': 'male', r'^f$': 'female',
            }, regex=True)
        return self

    def create_derived_variables(self, definitions: Dict[str, str]):
        for name, expr in definitions.items():
            try:
                self.df[name] = self.df.eval(expr)
                self._log(f"Created derived variable '{name}' = {expr}")
            except Exception as e:
                self._log(f"Failed to create '{name}': {e}")
        return self

    def categorize_age(self, source_col: str, new_col: str = 'age_group',
                       bins: Optional[List[int]] = None, labels: Optional[List[str]] = None):
        if bins is None:
            bins = [0, 30, 40, 50, 60, 70, 200]
        if labels is None:
            labels = ['<30', '30-39', '40-49', '50-59', '60-69', '>=70']
        if source_col in self.df.columns:
            self.df[new_col] = pd.cut(self.df[source_col], bins=bins, labels=labels, right=False)
            self._log(f"Categorized '{source_col}' into '{new_col}'")
        return self

    def categorize_duration(self, source_col: str, new_col: str = 'duration_group',
                            bins: Optional[List[float]] = None, labels: Optional[List[str]] = None):
        if bins is None:
            bins = [0, 5, 10, 200]
        if labels is None:
            labels = ['<5 years', '5-9 years', '>=10 years']
        if source_col in self.df.columns:
            self.df[new_col] = pd.cut(self.df[source_col], bins=bins, labels=labels, right=False)
            self._log(f"Categorized '{source_col}' into '{new_col}'")
        return self

    def remove_empty_columns(self, threshold: float = 0.8):
        before = len(self.df.columns)
        self.df = self.df.dropna(thresh=int(len(self.df) * threshold), axis=1)
        after = len(self.df.columns)
        if before > after:
            self._log(f"Removed {before - after} columns with <{threshold*100:.0f}% non-null values")
        return self

    def remove_duplicate_rows(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        after = len(self.df)
        if before > after:
            self._log(f"Removed {before - after} duplicate rows")
        return self

    def detect_outcome_variable(self, brief_keywords: Optional[List[str]] = None) -> str:
        if brief_keywords is None:
            brief_keywords = ['outcome', 'dependent', 'result', 'status', 'group', 'diagnosis', 'disease']
        for kw in brief_keywords:
            for col in self.df.columns:
                if kw in col.lower():
                    return col
        for col in self.df.columns:
            if self.detect_variable_types().get(col) == 'binary':
                return col
        return self.df.columns[0]

    def clean_pipeline(self, impute_strategy: str = 'median', outlier_multiplier: float = 1.5,
                       outcome_col: Optional[str] = None):
        self.standardize_column_names()
        self.remove_empty_columns()
        self.remove_duplicate_rows()
        self.extract_numeric_values()
        self.standardize_categoricals()
        self.impute_missing(strategy=impute_strategy)
        self.cap_outliers(multiplier=outlier_multiplier)
        self._log(f"Cleaning complete. Final shape: {self.df.shape}")
        return self

    def get_report(self) -> str:
        return '\n'.join(f"{i+1}. {msg}" for i, msg in enumerate(self.cleaning_log))

    def get_cleaned_df(self) -> pd.DataFrame:
        return self.df
