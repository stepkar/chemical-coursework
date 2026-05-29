from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config

"""
цсвшка после конвертации из xlsx получилась с разделителем;  и с , у float
"""


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else config.DATA_PATH
    df = pd.read_csv(path, sep=";", decimal=",", index_col=0)
    df.columns = [c.strip() for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def split_features_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df[config.TARGETS].copy()
    X = df.drop(columns=config.TARGETS)
    return X, y
