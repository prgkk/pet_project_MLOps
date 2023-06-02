"""
Программа: Тренировка данных
Версия: 1.0
"""

import optuna
from lightgbm import LGBMRegressor

from optuna import Study

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from ..data.split_dataset import get_train_test_data
from ..train.metrics import save_metrics


def objective(
    trial,
    data_x: pd.DataFrame,
    data_y: pd.Series,
    n_folds: int = 5,
    random_state: int = 10
) -> np.array:
    """
    Целевая функция для поиска параметров
    :param trial: кол-во trials
    :param data_x: данные объект-признаки
    :param data_y: данные с целевой переменной
    :param n_folds: кол-во фолдов
    :param random_state: random_state
    :return: среднее значение метрики по фолдам
    """
    lgb_params = {
        "n_estimators": trial.suggest_categorical("n_estimators", [348]),
        # "n_estimators": trial.suggest_int("n_estimators", 100, 1000, log=True),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.025261718741189964]),
        # "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 500, step=10),
        "max_depth": trial.suggest_int("max_depth", 6, 10),
        "max_bin": trial.suggest_int("max_bin", 50, 300),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100, step=3),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 20.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "random_state": trial.suggest_categorical("random_state", [random_state])}

    cv = KFold(n_splits=n_folds, shuffle=True)

    cv_predicts = np.empty(n_folds)
    for idx, (train_idx, test_idx) in enumerate(cv.split(data_x, data_y)):
        X_train, X_test = data_x.iloc[train_idx], data_x.iloc[test_idx]
        y_train, y_test = data_y.iloc[train_idx], data_y.iloc[test_idx]

        pruning_callback = optuna.integration.LightGBMPruningCallback(
            trial, "l2")
        model = LGBMRegressor(**lgb_params)
        model.fit(X_train,
                  y_train,
                  eval_set=[(X_test, y_test)],
                  eval_metric='rmse',
                  early_stopping_rounds=100,
                  callbacks=[pruning_callback],
                  verbose=0)

        preds = model.predict(X_test)
        cv_predicts[idx] = np.sqrt(mean_squared_error(y_test, preds))

    return np.mean(cv_predicts)


def find_optimal_params(
    data_train: pd.DataFrame, data_test: pd.DataFrame, **kwargs
) -> Study:
    """
    Пайплайн для тренировки модели
    :param data_train: датасет train
    :param data_test: датасет test
    :return: [LGBMClassifier tuning, Study]
    """
    x_train, x_test, y_train, y_test = get_train_test_data(
        data_train=data_train, data_test=data_test, target=kwargs["target_column"]
    )

    study = optuna.create_study(direction="minimize", study_name="LGB")
    function = lambda trial: objective(
        trial, x_train, y_train, kwargs["n_folds"], kwargs["random_state"]
    )
    study.optimize(function, n_trials=kwargs["n_trials"], show_progress_bar=True)
    return study


def train_model(
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    study: Study,
    target: str,
    metric_path: str,
) -> LGBMRegressor:
    """
    Обучение модели на лучших параметрах
    :param data_train: тренировочный датасет
    :param data_test: тестовый датасет
    :param study: study optuna
    :param target: название целевой переменной
    :param metric_path: путь до папки с метриками
    :return: LGBMRegressor
    """
    # get data
    x_train, x_test, y_train, y_test = get_train_test_data(
        data_train=data_train, data_test=data_test, target=target
    )

    # training optimal params
    model = LGBMRegressor(**study.best_params, silent=True, verbose=-1)
    model.fit(x_train, y_train, verbose=-1)

    # save metrics
    save_metrics(data_x=x_test, data_y=y_test, model=model, metric_path=metric_path)
    return model
