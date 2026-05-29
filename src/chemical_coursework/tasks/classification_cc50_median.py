from __future__ import annotations

import pandas as pd

from .. import config
from ..base import BaseClassificationTask
from ..model_zoo import classification_models


class CC50MedianClassificationTask(BaseClassificationTask):
    task_name = "classification_cc50_median"
    target = config.TARGET_CC50

    def _postprocess_target(self, y: pd.Series) -> pd.Series:
        threshold = y.median()
        self._threshold = threshold
        return (y > threshold).astype(int)

    def _candidates(self):
        return classification_models(scale_pos_weight=1.0)


if __name__ == "__main__":
    CC50MedianClassificationTask().run()
