# Курсовая работа по ML
- МО для QSAR-данных по 1000 соединениям против вируса гриппа. Таргеты - IC50, CC50, SI; решаются 3 регрессии и 4 классификации.
- [ссылка на задание](https://apps.skillfactory.ru/learning/course/course-v1:skillfactory+MIFIML-2sem+2026/block-v1:skillfactory+MIFIML-2sem+2026+type@sequential+block@fb3b363f2b6942f0ba7f652fa41f5548/block-v1:skillfactory+MIFIML-2sem+2026+type@vertical+block@7c01879e1f6b479989db90354c5151fb)
- В рамках работы решаются следующие проблемы:
- Регрессия для IC50
- Регрессия для CC50
- Регрессия для SI
- Классификация: превышает ли значение IC50 медианное значение выборки
- Классификация: превышает ли значение CC50 медианное значение выборки
- Классификация: превышает ли значение SI медианное значение выборки
- Классификация: превышает ли значение SI значение 8

# Формат выполнения:

- [Код анализа представленных данных (EDA)](src/chemical_coursework/eda.py)
- [Код построения моделей для каждой из задач](src/chemical_coursework/tasks)
- [Аналитический отчёт (в формате .pdf или .doc) с изложением результатов проделанной работы.](docs/report_chemical_kursach.pdf)


## Структура проекта

```
(.venv) stepankarsakov@Stepans-MacBook-Pro:~/IdeaProjects/python_session/src/chemical_coursework % tree -I "__pycache__|*.png|metrics|*.joblib"
.
├── README.md
├── data
│         └── Данные_для_курсовои_Классическое_МО.csv
├── docs
│         └── ai-in-chemistry-and-materials-science-M-3.pdf   саммари конспект вместо всех видео курса МГУ. Почему его не выложили, непонятно)
├── pyproject.toml
├── src
│         └── chemical_coursework
│             ├── __init__.py
│             ├── base.py    модели, абст.классы
│             ├── config.py         константы,пути
│             ├── data_io.py       чтение цсвшки после конвертации из xlsx
│             ├── eda.py 
│             ├── logging_setup.py
│             ├── model_zoo.py     кандидаты на регрессию/классификацию
│             ├── preprocessing.py     фильтрация фич и удаление выбросов
│             ├── run_all.py      общий запуск
│             └── tasks         реализации моделек
│                 ├── __init__.py
│                 ├── classification_cc50_median.py
│                 ├── classification_ic50_median.py
│                 ├── classification_si_gt8.py
│                 ├── classification_si_median.py
│                 ├── regression_cc50.py
│                 ├── regression_ic50.py
│                 └── regression_si.py
└── uv.lock

9 directories, 23 files

```

## Данные

csv лежит в data/Данные_для_курсовои_Классическое_МО.csv. Путь зашит - config.DATA_PATH относительно корня репо. 

## Запуск (через uv)

### Шаг 1. Установка зависимостей

(Из корня - chemical_coursework/):

```bash
uv sync
```

**В pyproject.toml настройки ruff+mypy, зависимости из того же корня:**

```bash
uv sync --group dev

## прогон:
uv run ruff check src
uv run ruff format --check src
uv run mypy
```

### Шаг 2. Тест

```bash
uv run python -c "import sklearn, xgboost, lightgbm, optuna, chemical_coursework; print('arrrbaiten!')"
```

### Шаг 3. Запуск

**Через entry-points** (смотри в pyproject.toml):

```bash
uv run chemcw-eda
uv run chemcw-run-all
```

**Через пайтон:**

```bash
uv run python -m chemical_coursework.eda
uv run python -m chemical_coursework.run_all
uv run python -m chemical_coursework.tasks.regression_ic50
```


### Переменные окружения

- N_OPTUNA_TRIALS - число итераций подбора гиперпараметров для каждой модели (по дефолту 40). Для быстрой проверки:

```bash
N_OPTUNA_TRIALS=20 uv run chemcw-run-all
```

## Чуть описания - что делают задачи

Класс-таска наследуется от BaseRegressionTask/ BaseClassificationTask - переопределяет только то, что отличается (имя таргета, преобразование у, набор моделей). Общий пайплайн в base.BaseTask.run:

1. Загрузка csv --> data_io.load_raw
2. Удаление выбросов по таргету (3*IQR, только для регрессий) и фильтрация фич: дроп констант, НаН, корреляции выше 0.95
3. Стратифицированный (классификация) или случайный train/test split (80/20)
4. Для каждого кандидата из model_zoo - Optuna tpe c 5-fold CV
5. Обучение лучшего конфига на train
6. Сохранение в artifacts/: leaderboard.csv, summary.json, top_features.csv, model.joblib (результаты ЕДА там же)

## Метрики

- Регрессия - MAE, RMSE, R^2
- Классификация - accuracy, precision, recall, F1, ROC-AUC 

## Модели

- Регрессия: Ridge, ElasticNet, KNN, RandomForest, GradientBoosting, XGBoost, LightGBM
- Классификация: LogisticRegression, KNN, RandomForest, GradientBoosting, XGBoost, LightGBM. 

## Ключевые результаты

смотри в метриках /artifacts либо в [ан.отчёте](docs/report_chemical_kursach.pdf)
