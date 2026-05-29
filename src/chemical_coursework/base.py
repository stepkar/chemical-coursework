from __future__ import annotations

import json
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config, data_io, preprocessing
from .logging_setup import format_eta, get_logger

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

log = get_logger(__name__)



"""
колбэк для Optuna - без неё 50 мин обучения вообще тишна - почти без логов.
"""
class _OptunaProgressLogger:
    def __init__(self, total: int, model_name: str, every: int = 5) -> None:
        self.total = total
        self.model_name = model_name
        self.every = max(1, every)
        self._t0 = time.time()

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        done = trial.number + 1
        if done != self.total and done % self.every != 0:
            return
        elapsed = time.time() - self._t0
        per_trial = elapsed / done if done else 0
        eta = per_trial * (self.total - done)
        try:
            best = study.best_value
            best_str = f"{best:.4f}"
        except ValueError:
            best_str = "n/a"
        log.info(
            "      [%s] trial %d/%d (%3d%%) best=%s elapsed=%s eta=%s",
            self.model_name,
            done,
            self.total,
            int(100 * done / self.total),
            best_str,
            format_eta(elapsed),
            format_eta(eta),
        )


"""
Базовые модели +абстрактные классы для задач
"""
@dataclass
class CandidateModel:
    name: str
    estimator_factory: Callable[[dict[str, Any]], Any]
    param_space: Callable[[optuna.Trial], dict[str, Any]]
    needs_scaling: bool = False


@dataclass
class TaskResult:
    task_name: str
    target: str
    leaderboard: pd.DataFrame
    best_model_name: str
    best_params: dict[str, Any]
    holdout_metrics: dict[str, float]
    feature_importance: pd.DataFrame | None = None
    feature_filter: dict[str, Any] = field(default_factory=dict)


class BaseTask(ABC):
    task_name: str = "base"
    target: str = ""
    direction: str = "maximize"
    n_trials: int = config.N_OPTUNA_TRIALS

    def __init__(self) -> None:
        config.ensure_dirs()
        self.rng = config.RANDOM_STATE

    def run(self) -> TaskResult:
        log.info("=" * 70)
        log.info("задача '%s' (target=%s) — старт", self.task_name, self.target)
        log.info("=" * 70)

        t = time.time()
        log.info("[1/5] загрузка сета...")
        df = self._load_dataframe()
        log.info(
            " загружено %d строк × %d колонок за %.1fs", df.shape[0], df.shape[1], time.time() - t
        )

        t = time.time()
        log.info("[2/5] препроцессинг (фильтрация фич, обработка таргета)...")
        X, y = self._prepare_xy(df)
        log.info(
            "  после препроцессинга: X=%s, y=%d (фильтр: %s) за %.1fs",
            X.shape,
            len(y),
            self._last_filter_report,
            time.time() - t,
        )

        log.info("[3/5] train/test split (test_size=%.2f)...", config.TEST_SIZE)
        X_train, X_test, y_train, y_test = self._split(X, y)
        log.info("  train=%d, test=%d", len(y_train), len(y_test))

        candidates = self._candidates()
        log.info(
            "[4/5] подбор моделей: %d кандидатов × %d trials × %d-fold CV",
            len(candidates),
            self.n_trials,
            config.CV_SPLITS,
        )

        leaderboard_rows = []
        best_overall: tuple[float, str, dict, Any] | None = None
        for i, cand in enumerate(candidates, 1):
            log.info("  ── модель %d/%d: %s ──", i, len(candidates), cand.name)
            t_model = time.time()
            study = self._tune(cand, X_train, y_train)
            tune_dt = time.time() - t_model
            best_score = study.best_value
            best_params = study.best_params
            log.info("    тюнинг закончен за %.1fs, лучший CV=%.4f", tune_dt, best_score)

            estimator = self._build_estimator(cand, best_params)
            estimator.fit(X_train, y_train)
            holdout = self._evaluate(estimator, X_test, y_test)
            holdout_str = ", ".join(f"{k}={v:.4f}" for k, v in holdout.items())
            log.info("    holdout: %s (всего %.1fs на модель)", holdout_str, time.time() - t_model)

            leaderboard_rows.append(
                {
                    "model": cand.name,
                    "cv_score": best_score,
                    **{f"holdout_{k}": v for k, v in holdout.items()},
                    "best_params": json.dumps(best_params, ensure_ascii=False),
                }
            )
            score_for_pick = best_score if self.direction == "maximize" else -best_score
            if best_overall is None or score_for_pick > best_overall[0]:
                best_overall = (score_for_pick, cand.name, best_params, estimator)

        leaderboard = (
            pd.DataFrame(leaderboard_rows)
            .sort_values("cv_score", ascending=(self.direction == "minimize"))
            .reset_index(drop=True)
        )

        assert best_overall is not None
        _, best_name, best_params, best_estimator = best_overall
        holdout = self._evaluate(best_estimator, X_test, y_test)
        importance = self._extract_importance(best_estimator, X.columns)
        result = TaskResult(
            task_name=self.task_name,
            target=self.target,
            leaderboard=leaderboard,
            best_model_name=best_name,
            best_params=best_params,
            holdout_metrics=holdout,
            feature_importance=importance,
            feature_filter=self._last_filter_report,
        )
        log.info("[5/5] сохранение артефактов и саммари...")
        self._save_artifacts(result, best_estimator)
        self._print_summary(result)
        log.info("задача '%s' завершена", self.task_name)
        return result

    def _load_dataframe(self) -> pd.DataFrame:
        return data_io.load_raw()

    def _prepare_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        X, y, report = preprocessing.prepare_for_task(
            df, target=self.target, drop_outliers=self._drop_target_outliers()
        )
        self._last_filter_report = {
            "n_before": report.n_before,
            "n_after": report.n_after,
            "dropped_constant": len(report.dropped_constant),
            "dropped_correlated": len(report.dropped_correlated),
            "dropped_all_nan": len(report.dropped_all_nan),
        }
        y = self._postprocess_target(y)
        return X, y

    def _drop_target_outliers(self) -> bool:
        return True

    def _postprocess_target(self, y: pd.Series) -> pd.Series:
        return y

    @abstractmethod
    def _split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: ...

    @abstractmethod
    def _candidates(self) -> list[CandidateModel]: ...

    @abstractmethod
    def _scoring_for_optuna(self) -> str: ...

    @abstractmethod
    def _evaluate(
        self, estimator: Any, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[str, float]: ...

    def _build_estimator(self, cand: CandidateModel, params: dict[str, Any]) -> Any:
        est = cand.estimator_factory(params)
        if cand.needs_scaling:
            return Pipeline([("scaler", StandardScaler()), ("est", est)])
        return est

    def _tune(self, cand: CandidateModel, X: pd.DataFrame, y: pd.Series) -> optuna.Study:
        cv = self._cv_strategy(y)
        scoring = self._scoring_for_optuna()

        def objective(trial: optuna.Trial) -> float:
            params = cand.param_space(trial)
            estimator = self._build_estimator(cand, params)
            scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=config.N_JOBS)
            return float(np.mean(scores))

        study = optuna.create_study(
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(seed=self.rng),
        )

        progress = _OptunaProgressLogger(
            total=self.n_trials, model_name=cand.name, every=max(1, self.n_trials // 10)
        )
        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=False,
            callbacks=[progress],
        )
        return study

    @abstractmethod
    def _cv_strategy(self, y: pd.Series): ...

    def _extract_importance(self, estimator: Any, feature_names: pd.Index) -> pd.DataFrame | None:
        underlying = estimator
        if isinstance(estimator, Pipeline):
            underlying = estimator.named_steps["est"]
        importances = None
        if hasattr(underlying, "feature_importances_"):
            importances = underlying.feature_importances_
        elif hasattr(underlying, "coef_"):
            coef = underlying.coef_
            importances = np.abs(coef).ravel() if coef.ndim > 1 else np.abs(coef)
        if importances is None:
            return None
        df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
            "importance", ascending=False
        )
        return df.reset_index(drop=True)

    def _save_artifacts(self, result: TaskResult, model: Any) -> None:
        slug = self.task_name
        result.leaderboard.to_csv(config.METRICS_DIR / f"{slug}_leaderboard.csv", index=False)
        with open(config.METRICS_DIR / f"{slug}_summary.json", "w") as f:
            json.dump(
                {
                    "task": self.task_name,
                    "target": self.target,
                    "best_model": result.best_model_name,
                    "best_params": result.best_params,
                    "holdout": result.holdout_metrics,
                    "feature_filter": result.feature_filter,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        if result.feature_importance is not None:
            result.feature_importance.head(30).to_csv(
                config.METRICS_DIR / f"{slug}_top_features.csv", index=False
            )
        joblib.dump(model, config.MODELS_DIR / f"{slug}.joblib")

    def _print_summary(self, result: TaskResult) -> None:
        print(f"\n=== {self.task_name} | target={self.target} ===")
        print(result.leaderboard.to_string(index=False))
        print(f"лучшая модель: {result.best_model_name}")
        print(f"holdout: {result.holdout_metrics}")


class BaseRegressionTask(BaseTask):
    direction = "maximize"

    def _split(self, X, y):
        return train_test_split(X, y, test_size=config.TEST_SIZE, random_state=self.rng)

    def _scoring_for_optuna(self) -> str:
        return "neg_root_mean_squared_error"

    def _cv_strategy(self, y):
        return KFold(n_splits=config.CV_SPLITS, shuffle=True, random_state=self.rng)

    def _evaluate(self, estimator, X_test, y_test) -> dict[str, float]:
        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )

        pred = estimator.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        return {
            "MAE": float(mean_absolute_error(y_test, pred)),
            "RMSE": rmse,
            "R2": float(r2_score(y_test, pred)),
        }


class BaseClassificationTask(BaseTask):
    direction = "maximize"
    positive_label: int = 1

    def _drop_target_outliers(self) -> bool:
        return False

    def _split(self, X, y):
        return train_test_split(
            X,
            y,
            test_size=config.TEST_SIZE,
            random_state=self.rng,
            stratify=y,
        )

    def _scoring_for_optuna(self) -> str:
        return "roc_auc"

    def _cv_strategy(self, y):
        return StratifiedKFold(n_splits=config.CV_SPLITS, shuffle=True, random_state=self.rng)

    def _evaluate(self, estimator, X_test, y_test) -> dict[str, float]:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        pred = estimator.predict(X_test)
        try:
            proba = estimator.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, proba))
        except Exception:
            auc = float("nan")
        return {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": auc,
        }
