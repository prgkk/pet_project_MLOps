"""
Программа: Тренировка модели на backend, отображение метрик и
графиков обучения на экране
Версия: 1.0
"""

import os
import json
import joblib
import requests
import streamlit as st
from optuna.visualization import plot_optimization_history
from ..data.get_data import get_dataset
from ..plotting.charts import permutation_imp


def start_training(config: dict, endpoint: object) -> None:
    """
    Тренировка модели с выводом результатов
    :param config: конфигурационный файл
    :param endpoint: endpoint
    """
    # Last metrics
    if os.path.exists(config["train"]["metrics_path"]):
        with open(config["train"]["metrics_path"]) as json_file:
            old_metrics = json.load(json_file)
    else:
        # если до этого не обучали модель и нет прошлых значений метрик
        old_metrics = {'MAE': 0, 'MSE': 0, 'RMSE': 0, 'RMSLE': 0, 'MPE_%': 0}

    # Train
    with st.spinner("Модель подбирает параметры..."):
        output = requests.post(endpoint, timeout=8000)
    st.success("Success!")

    new_metrics = output.json()["metrics"]

    # diff metrics
    mae, mse, rmse, rmsle, mpe = st.columns(5)
    mae.metric(
        "MAE",
        new_metrics["MAE"],
        f"{new_metrics['MAE']-old_metrics['MAE']:.2f}",
    )
    mse.metric(
        "MSE",
        new_metrics["MSE"],
        f"{new_metrics['MSE']-old_metrics['MSE']:.2f}",
    )
    rmse.metric(
        "RMSE",
        new_metrics["RMSE"],
        f"{new_metrics['RMSE']-old_metrics['RMSE']:.2f}",
    )
    rmsle.metric(
        "RMSLE",
        new_metrics["RMSLE"],
        f"{new_metrics['RMSLE'] - old_metrics['RMSLE']:.2f}",
    )
    mpe.metric(
        "MPE_%", new_metrics["MPE_%"], f"{new_metrics['MPE_%']-old_metrics['MPE_%']:.2f}"
    )

    # plot study
    study = joblib.load(os.path.join(config["train"]["study_path"]))
    fig_history = plot_optimization_history(study)
    st.plotly_chart(fig_history, use_container_width=True)

    # plot features importance
    model = joblib.load(os.path.join(config["train"]["model_path"]))
    data = get_dataset(dataset_path=config["preprocessing"]["train_path_proc"], cat=True)
    with st.spinner("plot features importance..."):
        st.pyplot(permutation_imp(model, data, n_rep=2))
