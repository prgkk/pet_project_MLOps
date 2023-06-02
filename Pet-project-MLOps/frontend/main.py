"""
Программа: Frontend часть проекта
Версия: 1.0
"""

import os

import yaml
import streamlit as st
from src.data.get_data import load_data, get_dataset
from src.plotting.charts import boxplot, kdeplot
from src.train.training import start_training
from src.evaluate.evaluate import evaluate_input, evaluate_from_file

CONFIG_PATH = "../config/params.yml"


def main_page():
    """
    Страница с описанием проекта
    """
    st.image(
        "https://datadrive2030.co.za/wp-content/uploads/2022/07/DataDrive-final-logo-01.svg",
        width=600,
    )

    st.title("Описание проекта")
    st.markdown("## MLOps project:  DataDrive2030: Early learning predictors challenge")
    st.write(
        """
        Использовать методы машинного обучения для выявления факторов в программах раннего обучения,
        которые способствуют улучшению результатов обучения у детей.
        При прогнозировании оценки ELOM ребенка и 15 наиболее важных переменных для каждого ребенка."""
    )

    # name of the columns
    st.markdown(
        """
        ### Данные
        Были сопоставлены данные из нескольких программ и проектов, которые использовали инструменты ELOM,
        за период 2019-2022 годов:

            - PQA: оценка качества учебной программы ELOM

            - PRA: Интервью с практикующим (teacher interview)

            - PRI: Основное интервью

            - OBS: Сохранение окружающей среды

        Каждый из них является разными источниками данных. Отдельное описание каждого столбца можно посмотреть в VariableDescription
    """
    )


def exploratory():
    """
    Exploratory data analysis
    """
    st.markdown("# Exploratory data analysis️")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load and write dataset
    data = get_dataset(dataset_path=config["preprocessing"]["train_path"])
    st.write(data[config["preprocessing"]["explored_cols"]].head())

    # выбор типа графика
    st.session_state['type'] = st.radio("Выберите тип графика", ['Boxplot', 'Kdeplot'])
    if st.session_state['type'] == 'Boxplot':
        type_plot = boxplot
    else:
        type_plot = kdeplot

    # plotting with checkbox
    child_observe_target = st.sidebar.checkbox("Оценка ребенка/Таргет")
    child_grant_target = st.sidebar.checkbox("Пособие за ребенка/Таргет")
    pri_separate_target = st.sidebar.checkbox("Разница в возрасте в классе/Таргет")
    ses_cat_target = st.sidebar.checkbox("Спутниковые данные/Таргет")
    language_child_target = st.sidebar.checkbox("Язык ребенка/Таргет")
    region_target = st.sidebar.checkbox("Регион/Таргет")

    if child_observe_target:
        st.markdown("#### Влияние оценки ребенка на целевую переменную")
        for col in config["preprocessing"]["child_observe_cols"]:
            st.pyplot(type_plot(data=data,
                                col_object=col,
                                col_value="target"))
        st.markdown("Вывод: чем выше оценка ребенка - тем больше целевая переменная")

    if child_grant_target:
        st.markdown("#### Влияние пособия за ребенка на целевую переменную")
        st.pyplot(type_plot(data=data,
                            col_object="child_grant",
                            col_value="target"))
        st.markdown("Вывод: у детей без пособия или с отказом таргет выше. Наиболее низкий таргет у детей с ответом 'не знаю'")

    if pri_separate_target:
        st.markdown("#### Влияение разделения детей в классе по возрасту на целевую переменную")
        st.pyplot(type_plot(data=data,
                            col_object="pri_separate",
                            col_value="target"))
        st.markdown("Вывод: у детей учащихся в классе с разницей в возрасте таргет заметно ниже")

    if ses_cat_target:
        st.markdown("#### Влияение категории спутниковых данных на целевую переменную")
        st.pyplot(type_plot(data=data,
                            col_object="ses_cat",
                            col_value="target"))
        st.markdown("Вывод: чем выше категория спутниковых данных - тем выше таргет")

    if language_child_target:
        st.markdown("#### Влияение языка ребенка на целевую переменную")
        st.pyplot(type_plot(data=data,
                            col_object="language_child",
                            col_value="target"))
        st.markdown("Вывод: язык влияет на таргет. Наиболее высокий - у английского, низкий - у Sesotho, Setswana, Sepedi и Tshivenda языков")

    if region_target:
        st.markdown("#### Влияние региона на целевую переменную")
        st.pyplot(type_plot(data=data,
                            col_object="id_prov",
                            col_value="target"))
        st.markdown("Вывод: регион влияет на целевую переменную")


def training():
    """
    Тренировка модели
    """
    st.markdown("# Training model LightGBM")
    # get params
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # endpoint
    endpoint = config["endpoints"]["train"]

    if st.button("Start training"):
        start_training(config=config, endpoint=endpoint)


def prediction():
    """
    Получение предсказаний путем ввода данных
    """
    st.markdown("# Prediction")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_input"]
    unique_data_path = config["preprocessing"]["unique_values_path"]

    # проверка на наличие сохраненной модели
    if os.path.exists(config["train"]["model_path"]):
        evaluate_input(unique_data_path=unique_data_path, endpoint=endpoint)
    else:
        st.error("Сначала обучите модель")


def prediction_from_file():
    """
    Получение предсказаний из файла с данными
    """
    st.markdown("# Prediction")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_from_file"]

    upload_file = st.file_uploader(
        "", type=["csv", "xlsx"], accept_multiple_files=False
    )
    # проверка загружен ли файл
    if upload_file:
        dataset_csv_df, files = load_data(data=upload_file, type_data="Test")
        # проверка на наличие сохраненной модели
        if os.path.exists(config["train"]["model_path"]):
            evaluate_from_file(data=dataset_csv_df, endpoint=endpoint, files=files)
        else:
            st.error("Сначала обучите модель")


def main():
    """
    Сборка пайплайна в одном блоке
    """
    page_names_to_funcs = {
        "Описание проекта": main_page,
        "Exploratory data analysis": exploratory,
        "Training model": training,
        "Prediction": prediction,
        "Prediction from file": prediction_from_file,
    }
    selected_page = st.sidebar.selectbox("Выберите пункт", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == "__main__":
    main()
