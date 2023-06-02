"""
Программа: Отрисовка графиков
Версия: 1.0
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance


def boxplot(col_object: str, col_value: str, data: pd.DataFrame,
            title: str = None, hue: str = None) -> matplotlib.figure.Figure:
    """
    Отрисовка графика boxplot
    :param data: датасет
    :param col_object: признак типа object
    :param col_value: числовой признак
    :param hue: признак типа object для дополнительного разделения
    :param title: название графика
    :return: поле рисунка
    """
    fig = plt.figure(figsize=(15, 7))

    sns.boxplot(x=col_object, y=col_value, hue=hue, data=data)

    if title is None:
        title = col_object + '/' + col_value
    plt.title(title, fontsize=20)
    plt.ylabel(col_value, fontsize=14)
    plt.xlabel(col_object, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return fig


def kdeplot(col_object: str, col_value: str,
            data: pd.DataFrame, title: str = None) -> matplotlib.figure.Figure:
    """
    Отрисовка графика kdeplot
    :param data: датасет
    :param col_object: признак типа object
    :param col_value: числовой признак
    :param title: название графика
    :return: поле рисунка
    """
    fig = plt.figure(figsize=(15, 7))

    sns.kdeplot(data=data, x=col_value, hue=col_object, common_norm=False, fill=True)

    if title is None:
        title = col_object + '/' + col_value
    plt.title(title, fontsize=20)
    plt.ylabel("Percentage", fontsize=14)
    plt.xlabel(col_value, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return fig


def permutation_imp(model, data: pd.DataFrame, n_rep: int) -> matplotlib.figure.Figure:
    """
    Отрисовка важности признаков
    :param model: модель
    :param data: датасет
    :param n_rep: число итераций permutation_importance
    :return: поле рисунка
    """
    X = data.drop(columns=['target'])
    y = data['target']
    r = permutation_importance(model, X, y, n_repeats=n_rep)

    feature_imp = pd.DataFrame()
    feature_imp['column'] = X.columns
    feature_imp['value'] = r['importances_mean']
    feature_imp.sort_values(by='value', inplace=True, ascending=False)

    fig = plt.figure(figsize=(15, 7))
    sns.barplot(data=feature_imp[:15], x='value', y='column', palette="Blues")
    plt.title('features importance', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return fig
