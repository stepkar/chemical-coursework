from __future__ import annotations

import pandas as pd

from .. import config
from ..base import BaseClassificationTask
from ..model_zoo import classification_models


class SIGreaterThan8ClassificationTask(BaseClassificationTask):
    task_name = "classification_si_gt8"
    target = config.TARGET_SI

    def _postprocess_target(self, y: pd.Series) -> pd.Series:
        self._threshold = config.SI_BUSINESS_THRESHOLD
        return (y > config.SI_BUSINESS_THRESHOLD).astype(int)

    def _candidates(self):
        pos = self._cached_y.sum() if hasattr(self, "_cached_y") else 1
        neg = max(len(self._cached_y) - pos, 1) if hasattr(self, "_cached_y") else 1
        spw = float(neg / max(pos, 1))
        return classification_models(scale_pos_weight=spw)

    def _prepare_xy(self, df):
        X, y = super()._prepare_xy(df)
        self._cached_y = y
        return X, y


if __name__ == "__main__":
    SIGreaterThan8ClassificationTask().run()
