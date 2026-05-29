from __future__ import annotations

from typing import Any

import optuna
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from . import config
from .base import CandidateModel

"""
Зоопарк моделей:
регрессия: Ridge,ElasticNet,KNN, RF, GB, XGB, LGBM
классификация: LogReg, KNN, RF, GB, XGB, LGBM

На каждую модель:
*_factory(p)- отдельный estimator
*_space(t) - словарь параметров для Optuna
отдельно две сборки для бейз: регрессия/классификация
"""


def _ridge_factory(p: dict[str, Any]) -> Any:
    return Ridge(alpha=p["alpha"], random_state=config.RANDOM_STATE)


def _ridge_space(t: optuna.Trial) -> dict[str, Any]:
    return {"alpha": t.suggest_float("alpha", 1e-3, 1e3, log=True)}


def _enet_factory(p: dict[str, Any]) -> Any:
    return ElasticNet(
        alpha=p["alpha"],
        l1_ratio=p["l1_ratio"],
        max_iter=10000,
        random_state=config.RANDOM_STATE,
    )


def _enet_space(t: optuna.Trial) -> dict[str, Any]:
    return {
        "alpha": t.suggest_float("alpha", 1e-4, 10, log=True),
        "l1_ratio": t.suggest_float("l1_ratio", 0.0, 1.0),
    }


def _knn_reg_factory(p: dict[str, Any]) -> Any:
    return KNeighborsRegressor(n_neighbors=p["n_neighbors"], weights=p["weights"], p=p["p"])


def _knn_clf_factory(p: dict[str, Any]) -> Any:
    return KNeighborsClassifier(n_neighbors=p["n_neighbors"], weights=p["weights"], p=p["p"])


def _knn_space(t: optuna.Trial) -> dict[str, Any]:
    return {
        "n_neighbors": t.suggest_int("n_neighbors", 3, 30),
        "weights": t.suggest_categorical("weights", ["uniform", "distance"]),
        "p": t.suggest_int("p", 1, 2),  # 1 = Манхэттен, 2 = Евклид
    }


def _rf_reg_factory(p: dict[str, Any]) -> Any:
    return RandomForestRegressor(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        min_samples_split=p["min_samples_split"],
        min_samples_leaf=p["min_samples_leaf"],
        max_features=p["max_features"],
        random_state=config.RANDOM_STATE,
        n_jobs=config.N_JOBS,
    )


def _rf_clf_factory(p: dict[str, Any]) -> Any:
    return RandomForestClassifier(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        min_samples_split=p["min_samples_split"],
        min_samples_leaf=p["min_samples_leaf"],
        max_features=p["max_features"],
        class_weight="balanced",
        random_state=config.RANDOM_STATE,
        n_jobs=config.N_JOBS,
    )


def _rf_space(t: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("n_estimators", 200, 700),
        "max_depth": t.suggest_int("max_depth", 4, 25),
        "min_samples_split": t.suggest_int("min_samples_split", 2, 15),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10),
        "max_features": t.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
    }

def _gb_reg_factory(p: dict[str, Any]) -> Any:
    return GradientBoostingRegressor(
        n_estimators=p["n_estimators"],
        learning_rate=p["learning_rate"],
        max_depth=p["max_depth"],
        subsample=p["subsample"],
        random_state=config.RANDOM_STATE,
    )


def _gb_clf_factory(p: dict[str, Any]) -> Any:
    return GradientBoostingClassifier(
        n_estimators=p["n_estimators"],
        learning_rate=p["learning_rate"],
        max_depth=p["max_depth"],
        subsample=p["subsample"],
        random_state=config.RANDOM_STATE,
    )


def _gb_space(t: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("n_estimators", 200, 700),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": t.suggest_int("max_depth", 2, 8),
        "subsample": t.suggest_float("subsample", 0.6, 1.0),
    }

def _logreg_factory(p: dict[str, Any]) -> Any:
    return LogisticRegression(
        C=p["C"],
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        random_state=config.RANDOM_STATE,
    )


def _logreg_space(t: optuna.Trial) -> dict[str, Any]:
    return {"C": t.suggest_float("C", 1e-3, 1e2, log=True)}


def _xgb_reg_factory(p: dict[str, Any]) -> Any:
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=p["n_estimators"],
        learning_rate=p["learning_rate"],
        max_depth=p["max_depth"],
        subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        reg_lambda=p["reg_lambda"],
        random_state=config.RANDOM_STATE,
        n_jobs=config.N_JOBS,
        verbosity=0,
        tree_method="hist",
    )


def _xgb_clf_factory(p: dict[str, Any]) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=p["n_estimators"],
        learning_rate=p["learning_rate"],
        max_depth=p["max_depth"],
        subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        reg_lambda=p["reg_lambda"],
        scale_pos_weight=p.get("scale_pos_weight", 1.0),
        random_state=config.RANDOM_STATE,
        n_jobs=config.N_JOBS,
        verbosity=0,
        eval_metric="logloss",
        tree_method="hist",
    )


def _xgb_space(t: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("n_estimators", 200, 800),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": t.suggest_int("max_depth", 3, 10),
        "subsample": t.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }

def _xgb_clf_space_factory(scale_pos_weight: float):
    def space(t: optuna.Trial) -> dict[str, Any]:
        params = _xgb_space(t)
        params["scale_pos_weight"] = scale_pos_weight
        return params

    return space


def _lgbm_reg_factory(p: dict[str, Any]) -> Any:
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=p["n_estimators"],
        learning_rate=p["learning_rate"],
        num_leaves=p["num_leaves"],
        max_depth=p["max_depth"],
        subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        reg_lambda=p["reg_lambda"],
        random_state=config.RANDOM_STATE,
        n_jobs=config.N_JOBS,
        verbosity=-1,
    )


def _lgbm_clf_factory(p: dict[str, Any]) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=p["n_estimators"],
        learning_rate=p["learning_rate"],
        num_leaves=p["num_leaves"],
        max_depth=p["max_depth"],
        subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        reg_lambda=p["reg_lambda"],
        class_weight="balanced",
        random_state=config.RANDOM_STATE,
        n_jobs=config.N_JOBS,
        verbosity=-1,
    )


def _lgbm_space(t: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("n_estimators", 200, 800),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": t.suggest_int("num_leaves", 15, 127),
        "max_depth": t.suggest_int("max_depth", -1, 12),
        "subsample": t.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }


# needs_scaling=True - в бейз используем StandardScaler
# лин и кнн скейлим обязательно, деревьям - пофиг
def regression_models() -> list[CandidateModel]:
    return [
        CandidateModel("ridge", _ridge_factory, _ridge_space, needs_scaling=True),
        CandidateModel("elastic_net", _enet_factory, _enet_space, needs_scaling=True),
        CandidateModel("knn", _knn_reg_factory, _knn_space, needs_scaling=True),
        CandidateModel("random_forest", _rf_reg_factory, _rf_space),
        CandidateModel("grad_boosting", _gb_reg_factory, _gb_space),
        CandidateModel("xgboost", _xgb_reg_factory, _xgb_space),
        CandidateModel("lightgbm", _lgbm_reg_factory, _lgbm_space),
    ]


def classification_models(scale_pos_weight: float = 1.0) -> list[CandidateModel]:
    return [
        CandidateModel("logreg", _logreg_factory, _logreg_space, needs_scaling=True),
        CandidateModel("knn", _knn_clf_factory, _knn_space, needs_scaling=True),
        CandidateModel("random_forest", _rf_clf_factory, _rf_space),
        CandidateModel("grad_boosting", _gb_clf_factory, _gb_space),
        CandidateModel(
            "xgboost",
            _xgb_clf_factory,
            _xgb_clf_space_factory(scale_pos_weight),
        ),
        CandidateModel("lightgbm", _lgbm_clf_factory, _lgbm_space),
    ]
