"""
Программа: Отрисовка слайдеров и кнопок для ввода данных
с дальнейшим получением предсказания на основании введенных значений
Версия: 1.0
"""

import json
from io import BytesIO
import pandas as pd
import requests
import streamlit as st


def evaluate_input(unique_data_path: str, endpoint: object) -> None:
    """
    Получение входных данных путем ввода в UI -> вывод результата
    :param unique_data_path: путь до уникальных значений
    :param endpoint: endpoint
    """
    with open(unique_data_path) as file:
        unique_df = json.load(file)

    # поля для вводы данных, используем уникальные значения
    child_gender = st.sidebar.selectbox("child_gender", (unique_df["child_gender"]))
    child_age = st.sidebar.slider(
        "child_age", min_value=min(unique_df["child_age"]), max_value=max(unique_df["child_age"]))
    child_grant = st.sidebar.selectbox("child_grant", (unique_df["child_grant"]))
    id_dc_best = st.sidebar.selectbox("id_dc_best", (unique_df["id_dc_best"]))
    child_years_in_programme = st.sidebar.selectbox("child_years_in_programme", (unique_df["child_years_in_programme"]))
    pra_engaged = st.sidebar.selectbox("pra_engaged", (unique_df["pra_engaged"]))
    language_child = st.sidebar.selectbox("language_child", (unique_df["language_child"]))
    pri_fees_amount = st.sidebar.number_input(
        "pri_fees_amount", min_value=min(unique_df["pri_fees_amount"]), max_value=max(unique_df["pri_fees_amount"]))
    child_observe_attentive = st.sidebar.selectbox("child_observe_attentive", (unique_df["child_observe_attentive"]))
    child_observe_concentrated = st.sidebar.selectbox("child_observe_concentrated",
                                                      (unique_df["child_observe_concentrated"]))
    child_observe_diligent = st.sidebar.selectbox("child_observe_diligent", (unique_df["child_observe_diligent"]))
    child_observe_interested = st.sidebar.selectbox("child_observe_interested", (unique_df["child_observe_interested"]))
    child_zha = st.sidebar.number_input("child_zha",
                                        min_value=min(unique_df["child_zha"]), max_value=max(unique_df["child_zha"]))

    dict_data = {
        "child_gender": child_gender,
        "child_age": child_age,
        "child_grant": child_grant,
        "id_dc_best": id_dc_best,
        "child_years_in_programme": child_years_in_programme,
        "pra_engaged": pra_engaged,
        "language_child": language_child,
        "pri_fees_amount": pri_fees_amount,
        "child_observe_attentive": child_observe_attentive,
        "child_observe_concentrated": child_observe_concentrated,
        "child_observe_diligent": child_observe_diligent,
        "child_observe_interested": child_observe_interested,
        "child_zha": child_zha,
    }

    st.write(
        f"""### Данные ребенка:\n
    1) child_gender: {dict_data['child_gender']}
    2) child_age: {dict_data['child_age']}
    3) child_grant: {dict_data['child_grant']}
    4) id_dc_best: {dict_data['id_dc_best']}
    5) child_years_in_programme: {dict_data['child_years_in_programme']}
    6) pra_engaged: {dict_data['pra_engaged']}
    7) language_child: {dict_data['language_child']}
    8) pri_fees_amount: {dict_data['pri_fees_amount']}
    9) child_observe_attentive: {dict_data['child_observe_attentive']}
    10) child_observe_concentrated: {dict_data['child_observe_concentrated']}
    11) child_observe_diligent: {dict_data['child_observe_diligent']}
    12) child_observe_interested: {dict_data['child_observe_interested']}
    13) child_zha: {dict_data['child_zha']}
    """
    )

    # evaluate and return prediction (text)
    button_ok = st.button("Predict")
    if button_ok:
        result = requests.post(endpoint, timeout=8000, json=dict_data)
        json_str = json.dumps(result.json())
        output = json.loads(json_str)
        st.write(f"## {round(output[0], 2)}")
        st.success("Success!")


def evaluate_from_file(data: pd.DataFrame, endpoint: object, files: BytesIO):
    """
    Получение входных данных в качестве файла -> вывод результата в виде таблицы
    :param data: датасет
    :param endpoint: endpoint
    :param files:
    """
    button_ok = st.button("Predict")
    if button_ok:
        # заглушка так как не выводим все предсказания
        data_ = data[:5]
        output = requests.post(endpoint, files=files, timeout=8000)
        data_["predict"] = output.json()["prediction"]
        st.write(data_[['child_id', 'predict']].head())
