from __future__ import annotations

import json
import os
import time

import pandas as pd

from . import config
from .logging_setup import format_eta, get_logger
from .tasks.classification_cc50_median import CC50MedianClassificationTask
from .tasks.classification_ic50_median import IC50MedianClassificationTask
from .tasks.classification_si_gt8 import SIGreaterThan8ClassificationTask
from .tasks.classification_si_median import SIMedianClassificationTask
from .tasks.regression_cc50 import CC50RegressionTask
from .tasks.regression_ic50 import IC50RegressionTask
from .tasks.regression_si import SIRegressionTask

log = get_logger(__name__)

TaskClass = (
    type[IC50RegressionTask]
    | type[CC50RegressionTask]
    | type[SIRegressionTask]
    | type[IC50MedianClassificationTask]
    | type[CC50MedianClassificationTask]
    | type[SIMedianClassificationTask]
    | type[SIGreaterThan8ClassificationTask]
)

ALL_TASKS: list[TaskClass] = [
    IC50RegressionTask,
    CC50RegressionTask,
    SIRegressionTask,
    IC50MedianClassificationTask,
    CC50MedianClassificationTask,
    SIMedianClassificationTask,
    SIGreaterThan8ClassificationTask,
]

"""
Общая запускалка
"""


def main() -> None:
    n_trials_override = os.environ.get("N_OPTUNA_TRIALS")
    total_tasks = len(ALL_TASKS)
    n_trials_effective = (
        int(n_trials_override) if n_trials_override is not None else config.N_OPTUNA_TRIALS
    )
    log.info("###############################")
    log.info("RUN_ALL:%d задач, n_trials=%d", total_tasks, n_trials_effective)
    log.info("###############################")

    t_global = time.time()
    elapsed_per_task: list[float] = []
    summary_rows = []
    for i, cls in enumerate(ALL_TASKS, 1):
        if elapsed_per_task:
            avg = sum(elapsed_per_task) / len(elapsed_per_task)
            eta_remaining = avg * (total_tasks - i + 1)
        else:
            eta_remaining = float("nan")
        log.info(
            ">>> ЗАДАЧА %d/%d: %s (общее время: %s, ETA до конца: %s)",
            i,
            total_tasks,
            cls.__name__,
            format_eta(time.time() - t_global),
            format_eta(eta_remaining),
        )

        t0 = time.time()
        task = cls()
        if n_trials_override is not None:
            task.n_trials = int(n_trials_override)
        result = task.run()
        elapsed = time.time() - t0
        elapsed_per_task.append(elapsed)
        log.info("<<< задача %d/%d закончена за %s", i, total_tasks, format_eta(elapsed))

        row = {
            "task": result.task_name,
            "target": result.target,
            "best_model": result.best_model_name,
            "elapsed_sec": round(elapsed, 1),
        }
        row.update({f"holdout_{k}": v for k, v in result.holdout_metrics.items()})
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(config.METRICS_DIR / "all_tasks_summary.csv", index=False)
    with open(config.METRICS_DIR / "all_tasks_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)
    log.info("##############################################################")
    log.info("ЗАКОНЧИЛИ за %s сек", format_eta(time.time() - t_global))
    log.info("##############################################################")
    print("\n========== ИТОГ ПО ВСЕМ ТАСКАМ ==========")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
