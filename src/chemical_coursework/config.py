from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

DATA_PATH = DATA_DIR / "Данные_для_курсовои_Классическое_МО.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
N_OPTUNA_TRIALS = 40
N_JOBS = -1

TARGET_IC50 = "IC50, mM"
TARGET_CC50 = "CC50, mM"
TARGET_SI = "SI"
TARGETS = [TARGET_IC50, TARGET_CC50, TARGET_SI]

SI_BUSINESS_THRESHOLD = 8.0


@dataclass
class PreprocessConfig:
    drop_targets_for_features: list[str] = field(default_factory=lambda: TARGETS.copy())
    near_zero_var_threshold: float = 1e-6
    high_corr_threshold: float = 0.95
    iqr_outlier_factor: float = 3.0
    outlier_target_action: str = "log1p_clip"


def ensure_dirs() -> None:
    for d in (DATA_DIR, ARTIFACTS_DIR, FIGURES_DIR, MODELS_DIR, METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)
