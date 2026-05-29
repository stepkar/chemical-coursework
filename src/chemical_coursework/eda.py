from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config, data_io, preprocessing
from .logging_setup import get_logger

sns.set_theme(style="whitegrid")
log = get_logger(__name__)


"""
EDA по стандарту:
метрики: кол-во строк/стлбцов, float/int, describe, NaN
графики боксплоты, heatmap и т.п.
саммари с json
"""


# утилитка - чтоб не таскать одни и те же 3 строчки по всем графикам
def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _shape_and_dtypes(df: pd.DataFrame) -> dict:
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "n_targets": len(config.TARGETS),
        "n_features_raw": int(df.shape[1] - len(config.TARGETS)),
        "dtype_counts": {str(k): int(v) for k, v in df.dtypes.value_counts().items()},
    }


def _missing_report(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    return miss.to_frame("n_missing").assign(
        pct=lambda x: (x["n_missing"] / len(df) * 100).round(2)
    )


def _target_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df[config.TARGETS].describe().T


def _plot_target_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, target in enumerate(config.TARGETS):
        s = df[target].dropna()
        axes[0, i].hist(s, bins=60, color="blue", alpha=0.8)
        axes[0, i].set_title(f"{target} — исходное")
        axes[0, i].set_yscale("log")
        log_s = np.log1p(s.clip(lower=0))
        axes[1, i].hist(log_s, bins=60, color="green", alpha=0.8)
        axes[1, i].set_title(f"{target} — log1p")
    _save(fig, "01_target_distributions.png")


def _plot_target_boxplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for i, target in enumerate(config.TARGETS):
        axes[i].boxplot(df[target].dropna(), vert=True)
        axes[i].set_title(f"{target}: бокс с выбросами")
        axes[i].set_yscale("log")
    _save(fig, "02_target_boxplots.png")


# pairplot 3х3 в лог-шкале - проверяем связь si-cc50-ic50
def _plot_target_pairplot(df: pd.DataFrame) -> None:
    sub = df[config.TARGETS].dropna().apply(np.log1p)
    g = sns.pairplot(sub, plot_kws={"s": 8, "alpha": 0.4, "color": "blue"})
    g.fig.suptitle("парные графики таргетов в log1p", y=1.02)
    g.fig.savefig(config.FIGURES_DIR / "03_targets_pairplot.png", dpi=130)
    plt.close(g.fig)


# перепроверяем на всякий si=cc50/ic50 - вдруг расходится.
def _verify_si_definition(df: pd.DataFrame) -> dict:
    sub = df[config.TARGETS].dropna()
    derived = sub[config.TARGET_CC50] / sub[config.TARGET_IC50]
    diff = (sub[config.TARGET_SI] - derived).abs()
    rel = diff / sub[config.TARGET_SI].abs().clip(lower=1e-9)  # защита от деления на 0
    return {
        "max_abs_diff": float(diff.max()),
        "median_rel_diff": float(rel.median()),
        "share_close_1pct": float((rel < 0.01).mean()),
    }


"""
обёртка чисто для json (модели дёргают filter_features напрямую)
"""


def _feature_filter_report(df: pd.DataFrame) -> dict:
    X = df.drop(columns=config.TARGETS)
    _, report = preprocessing.filter_features(X)
    return {
        "n_before": report.n_before,
        "n_after": report.n_after,
        "n_dropped_constant": len(report.dropped_constant),
        "n_dropped_correlated": len(report.dropped_correlated),
        "n_dropped_all_nan": len(report.dropped_all_nan),
        "examples_constant": report.dropped_constant[:10],
        "examples_correlated": report.dropped_correlated[:10],
    }


def _top_target_correlations(df: pd.DataFrame, k: int = 15) -> pd.DataFrame:
    X = df.drop(columns=config.TARGETS)
    rows = []
    for tgt in config.TARGETS:
        y = np.log1p(df[tgt].clip(lower=0))
        corr = X.apply(lambda c, y=y: c.corr(y))
        top = corr.abs().sort_values(ascending=False).head(k)
        for feat in top.index:
            rows.append(
                {
                    "target": tgt,
                    "feature": feat,
                    "pearson": float(corr[feat]),
                    "abs": float(top[feat]),
                }
            )
    return pd.DataFrame(rows)


# хитмап - топ30
# отбор фич - по макс связи хотя бы с одним из трёх таргетов
def _plot_correlation_heatmap_targets(df: pd.DataFrame) -> None:
    X = df.drop(columns=config.TARGETS)
    target_log = df[config.TARGETS].apply(lambda c: np.log1p(c.clip(lower=0)))
    cor = pd.concat([X, target_log], axis=1).corr().loc[X.columns, config.TARGETS]
    top_features = cor.abs().max(axis=1).sort_values(ascending=False).head(30).index
    fig, ax = plt.subplots(figsize=(7, 9))
    sns.heatmap(cor.loc[top_features], cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("топ30 признаков по корреляции с таргетами")
    _save(fig, "04_top_corr_with_targets.png")


# 4 будущие бин задачи: "выше медианы" по каждому из 3 таргетов + отдельно si>8
def _class_balance(df: pd.DataFrame) -> dict:
    out = {}
    for tgt in config.TARGETS:
        med = float(df[tgt].median())
        out[f"{tgt} > median"] = float((df[tgt] > med).mean())
    out[f"SI > {config.SI_BUSINESS_THRESHOLD}"] = float(
        (df[config.TARGET_SI] > config.SI_BUSINESS_THRESHOLD).mean()
    )
    return out


def _plot_class_balance(df: pd.DataFrame) -> None:
    labels = list(_class_balance(df).keys())
    values = list(_class_balance(df).values())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color="orange")
    ax.set_ylim(0, 1)
    ax.set_ylabel("доля положительного класса")
    ax.set_title("баланс классов для классификационных задач")
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.xticks(rotation=15, ha="right")
    _save(fig, "05_class_balance.png")


# область применимости:
# z-нормируем фичи считаем ср.квадрат -смотрим у кого он сильно больше 1.
def _applicability_domain(df: pd.DataFrame) -> dict:
    X = df.drop(columns=config.TARGETS)
    X_filt, _ = preprocessing.filter_features(X)
    X_filt = preprocessing.impute_features(X_filt)
    z = (X_filt - X_filt.mean()) / X_filt.std(ddof=0).replace(0, 1)
    leverage = (z**2).sum(axis=1) / X_filt.shape[1]
    return {
        "leverage_mean": float(leverage.mean()),
        "leverage_p95": float(leverage.quantile(0.95)),
        "n_potential_outliers_AD": int((leverage > leverage.quantile(0.95)).sum()),
    }


# для логирования ЕДА,
def _step(name: str):
    class _Ctx:
        def __enter__(self_inner):
            log.info(" %s", name)
            self_inner.t0 = time.time()
            return self_inner

        def __exit__(self_inner, *exc):
            log.info("  %s за %.1fs", name, time.time() - self_inner.t0)
            return False

    return _Ctx()


# точка вход --- chemcw-eda из pyproject.toml.
def run_eda() -> dict:
    log.info("=" * 70)
    log.info("EDA — старт")
    log.info("=" * 70)
    config.ensure_dirs()

    with _step("загрузка датасета"):
        df = data_io.load_raw()
        log.info("  shape=%s", df.shape)

    summary: dict = {}
    with _step("базовая статистика (shape, dtypes, missing, target_stats)"):
        summary["shape"] = _shape_and_dtypes(df)
        miss = _missing_report(df)
        miss.to_csv(config.METRICS_DIR / "eda_missing.csv")
        summary["missing_features_count"] = int(len(miss))
        summary["missing_total_cells"] = int(miss["n_missing"].sum() if len(miss) else 0)
        target_stats = _target_stats(df)
        target_stats.to_csv(config.METRICS_DIR / "eda_target_stats.csv")
        summary["target_stats"] = json.loads(target_stats.to_json(orient="index"))

    with _step("проверка определения SI"):
        summary["si_definition_check"] = _verify_si_definition(df)

    with _step("отчёт по фильтрации фич (constant/correlated/all-nan)"):
        summary["feature_filter_report"] = _feature_filter_report(df)
        log.info(
            "  оставили %d из %d фич",
            summary["feature_filter_report"]["n_after"],
            summary["feature_filter_report"]["n_before"],
        )

    with _step("баланс классов для классификационных задач"):
        summary["class_balance"] = _class_balance(df)

    with _step("applicability domain (leverage)"):
        summary["applicability_domain"] = _applicability_domain(df)

    with _step("топ корреляций фич с таргетами"):
        top_corr = _top_target_correlations(df)
        top_corr.to_csv(config.METRICS_DIR / "eda_top_corr_with_targets.csv", index=False)

    with _step("график 01:распределения таргетов"):
        _plot_target_distributions(df)
    with _step("график 02:бокс-плоты таргетов"):
        _plot_target_boxplots(df)
    with _step("график 03: pairplot таргетов в log1p"):
        _plot_target_pairplot(df)
    with _step("график 04: heatmap топ30 фич × таргеты"):
        _plot_correlation_heatmap_targets(df)
    with _step("график 05: баланс классов"):
        _plot_class_balance(df)

    with open(config.METRICS_DIR / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("ЕДА готов,артефакты: %s", config.ARTIFACTS_DIR)
    return summary


if __name__ == "__main__":
    run_eda()
