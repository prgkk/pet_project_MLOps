"""
Программа: Предобработка данных
Версия: 1.0
"""

import json
import warnings
import pandas as pd
import yaml

warnings.filterwarnings("ignore")


def replace_values(data: pd.DataFrame, map_change_columns: dict) -> pd.DataFrame:
    """
    Замена значений в датасете
    :param data: датасет
    :param map_change_columns: словарь с признаками и значениями
    :return: датасет
    """
    for cols, map_values in map_change_columns.items():
        for col in cols.split(', '):
            data[col] = data[col].map(map_values)
    return data


def obs_hygiene_rang(obs_col: str) -> str:
    """
    преобразование значений в признаках гигиенты
    :obs_col: значение признака
    :return: ранговое значение признака
    """
    obs_col = str(obs_col)
    if '0' in obs_col:
        return 0
    elif '1' in obs_col or '2' in obs_col or '8' in obs_col:
        return 3
    elif '3' in obs_col or '4' in obs_col or '5' in obs_col:
        return 2
    elif '6' in obs_col or '7' in obs_col or '97' in obs_col:
        return 1


def filna(data: pd.DataFrame) -> pd.DataFrame:
    """
    заполнение пропусков в датасете
    :param data: датасет
    :return: датасет с заполненными пропусками
    """
    for col in data.columns:

        if data[col].dtype in ['int64', 'int32', 'float64', int, float]:
            data[col] = data[col].fillna(-99)

        else:
            if data[col].nunique() == 2:
                # если 2 уникальных значения выражаем их 0 и 1, пропуски заполняем -99
                data[col] = (data[col].dropna() == data[col].dropna().unique()[0]).astype(int)
                data[col] = data[col].fillna(-99)
            else:
                data[col] = data[col].fillna("No_info")
    return data


def transform_types(data: pd.DataFrame, change_type_columns: dict) -> pd.DataFrame:
    """
    Преобразование признаков в заданный тип данных
    :param data: датасет
    :param change_type_columns: словарь с признаками и типами данных
    :return:
    """
    return data.astype(change_type_columns, errors="raise")


def drop_cols(data: pd.DataFrame, drop_cols: list) -> pd.DataFrame:
    """
    удаление колонок в датасете
    :param data: датасет
    :param drop_cols: список колонок для удаления
    :return: датасет
    """
    return data.drop(columns=drop_cols)


def check_columns_evaluate(data: pd.DataFrame, unique_values_path: str) -> pd.DataFrame:
    """
    Проверка на наличие признаков из train и упорядочивание признаков согласно train
    :param data: датасет test
    :param unique_values_path: путь до списока с признаками train для сравнения
    :return: датасет test
    """
    with open(unique_values_path) as json_file:
        unique_values = json.load(json_file)

    column_sequence = unique_values.keys()

    assert set(column_sequence) == set(data.columns), "Разные признаки"
    return data[column_sequence]


def save_unique_train_data(
    data: pd.DataFrame, drop_cols: list, target_column: str, unique_values_path: str
) -> None:
    """
    Сохранение словаря с признаками и уникальными значениями
    :param drop_cols: список с признаками для удаления
    :param data: датасет
    :param target_column: целевая переменная
    :param unique_values_path: путь до файла со словарем
    :return: None
    """
    unique_df = data.drop(
        columns=drop_cols + [target_column], axis=1, errors="ignore")
    # создаем словарь с уникальными значениями для вывода в UI
    dict_unique = {key: unique_df[key].dropna().unique().tolist() for key in unique_df.columns}
    with open(unique_values_path, "w") as file:
        json.dump(dict_unique, file)


def pipeline_preprocess(data: pd.DataFrame, flg_evaluate: bool = True, **kwargs) -> pd.DataFrame:
    """
    Пайплайн по предобработке данных
    :param data: датасет
    :param flg_evaluate: флаг для evaluate
    :return: датасет
    """

    # drop columns
    data = data.drop(kwargs["drop_cols"], axis=1, errors="ignore")
    # проверка dataset на совпадение с признаками из train
    # либо сохранение уникальных данных с признаками из train
    if flg_evaluate:
        data = check_columns_evaluate(
            data=data, unique_values_path=kwargs["unique_values_path"]
        )
    else:
        save_unique_train_data(
            data=data,
            drop_cols=kwargs["drop_cols"],
            target_column=kwargs["target_column"],
            unique_values_path=kwargs["unique_values_path"],
        )

    # преобразование значений в признаках
    data['obs_toilet'] = data.apply(lambda x: obs_hygiene_rang(x['obs_toilet']), axis=1)
    data['obs_handwashing'] = data.apply(lambda x: obs_hygiene_rang(x['obs_handwashing']), axis=1)
    data['id_dc_best'] = ['Rare' if x not in kwargs["dc_freq"] else x for x in data['id_dc_best']]

    # drop columns
    data = data.drop(kwargs["drop_cols"], axis=1, errors="ignore")

    # replace values
    data = replace_values(data=data, map_change_columns=kwargs["map_change_columns"])

    # fillna
    data = filna(data)

    # change category types
    dict_category = {key: "category" for key in data.select_dtypes(["object"]).columns}
    data = transform_types(data=data, change_type_columns=dict_category)
    return data


def pipeline_input(config_path: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    Добавление вводимых признаков в датасет со всеми столбцами
    :param config_path: путь до файла с конфигурациями
    :param data: датасет с признаками ручного ввода
    :return: датасет с признаками ручного ввода, где остальные столбцы NaN
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    data_test = pd.read_csv(config['evaluate']['predict_path'])
    input_data = pd.concat([data_test, data], ignore_index=True, sort=False)
    return input_data
