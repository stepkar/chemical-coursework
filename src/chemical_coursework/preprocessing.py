from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class FeatureFilterReport:
    # только для  отчёта - потом всё экспортим в eda_summary.json
    dropped_constant: list[str]
    dropped_correlated: list[str]
    dropped_all_nan: list[str]
    n_before: int
    n_after: int


"""
Дроп: констант, НаН, мультиколлинеарность
Обработка выбросов
"""


def drop_constant(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    variances = df.var(numeric_only=True)
    to_drop = variances[variances <= threshold].index.tolist()
    return df.drop(columns=to_drop), to_drop


def drop_all_nan(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    to_drop = df.columns[df.isna().all()].tolist()
    return df.drop(columns=to_drop), to_drop


def drop_highly_correlated(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    corr = df.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = [c for c in upper.columns if (upper[c] > threshold).any()]
    return df.drop(columns=to_drop), to_drop


def filter_features(
    X: pd.DataFrame, cfg: config.PreprocessConfig | None = None
) -> tuple[pd.DataFrame, FeatureFilterReport]:
    cfg = cfg or config.PreprocessConfig()
    n_before = X.shape[1]
    X1, dropped_nan = drop_all_nan(X)
    X2, dropped_const = drop_constant(X1, cfg.near_zero_var_threshold)
    X3, dropped_corr = drop_highly_correlated(X2, cfg.high_corr_threshold)
    report = FeatureFilterReport(
        dropped_constant=dropped_const,
        dropped_correlated=dropped_corr,
        dropped_all_nan=dropped_nan,
        n_before=n_before,
        n_after=X3.shape[1],
    )
    return X3, report


# дырки в фичах заливаем медианой по столбцу - для деревьев норм.
def impute_features(X: pd.DataFrame) -> pd.DataFrame:
    return X.fillna(X.median(numeric_only=True))


# выбросы режем по правилу IQR (*3 - мягко, оставляем "толстые хвосты").
def remove_target_outliers(df: pd.DataFrame, target: str, factor: float = 3.0) -> pd.DataFrame:
    s = df[target]
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - factor * iqr, q3 + factor * iqr
    mask = (s >= lo) & (s <= hi)
    return df[mask].copy()


def prepare_for_task(
    df: pd.DataFrame,
    target: str,
    cfg: config.PreprocessConfig | None = None,
    drop_outliers: bool = True,
) -> tuple[pd.DataFrame, pd.Series, FeatureFilterReport]:
    cfg = cfg or config.PreprocessConfig()
    df = df.dropna(subset=[target]).copy()
    if drop_outliers:
        df = remove_target_outliers(df, target, cfg.iqr_outlier_factor)
    y = df[target].copy()
    X = df.drop(columns=cfg.drop_targets_for_features)
    X, report = filter_features(X, cfg)
    X = impute_features(X)
    return X, y, report
