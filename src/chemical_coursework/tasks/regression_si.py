from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..base import BaseRegressionTask
from ..model_zoo import regression_models


class SIRegressionTask(BaseRegressionTask):
    task_name = "regression_si"
    target = config.TARGET_SI

    def _postprocess_target(self, y: pd.Series) -> pd.Series:
        return np.log1p(y.clip(lower=0))

    def _candidates(self):
        return regression_models()


if __name__ == "__main__":
    SIRegressionTask().run()
