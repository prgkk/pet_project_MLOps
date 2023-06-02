"""
Программа: Модель для выявления факторов в программах раннего обучения,
которые способствуют улучшению результатов обучения у детей
Версия: 1.0
"""

import warnings
import optuna
import pandas as pd

import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel

from src.pipelines.pipeline import pipeline_training
from src.evaluate.evaluate import pipeline_evaluate
from src.transform.transform import pipeline_input
from src.train.metrics import load_metrics

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

app = FastAPI()
CONFIG_PATH = "../config/params.yml"


class ChildFeatures(BaseModel):
    """
    Признаки для получения результатов модели
    """
    child_age: float
    child_grant: object
    child_years_in_programme: object
    child_observe_attentive: object
    child_observe_concentrated: object
    child_observe_diligent: object
    child_observe_interested: object
    child_gender: object
    child_zha: float
    pra_engaged: object
    pri_fees_amount: float
    id_dc_best: object
    language_child: object


@app.get("/hello")
def welcome() -> dict:
    """
    Hello
    :return: Hello Data Scientist
    """
    return {'message': 'Hello Data Scientist!'}


@app.post("/train")
def training() -> dict:
    """
    Обучение модели, логирование метрик
    """
    pipeline_training(config_path=CONFIG_PATH)
    metrics = load_metrics(config_path=CONFIG_PATH)

    return {"metrics": metrics}


@app.post("/predict")
def prediction(file: UploadFile = File(...)) -> dict:
    """
    Предсказание модели по данным из файла
    """
    result = pipeline_evaluate(config_path=CONFIG_PATH, data_path=file.file)
    assert isinstance(result, list), "Результат не соответствует типу list"
    # заглушка так как не выводим все предсказания, иначе зависнет
    return {"prediction": result[:5]}


@app.post("/predict_input")
def prediction_input(child: ChildFeatures):
    """
    Предсказание модели по введенным данным
    """
    features = [
        [
            child.child_age,
            child.child_grant,
            child.child_years_in_programme,
            child.child_observe_attentive,
            child.child_observe_concentrated,
            child.child_observe_diligent,
            child.child_observe_interested,
            child.child_gender,
            child.child_zha,
            child.pra_engaged,
            child.pri_fees_amount,
            child.id_dc_best,
            child.language_child,
        ]
    ]

    cols = ["child_age",
            "child_grant",
            "child_years_in_programme",
            "child_observe_attentive",
            "child_observe_concentrated",
            "child_observe_diligent",
            "child_observe_interested",
            "child_gender",
            "child_zha",
            "pra_engaged",
            "pri_fees_amount",
            "id_dc_best",
            "language_child",
            ]

    data = pd.DataFrame(features, columns=cols)
    new_data = pipeline_input(CONFIG_PATH, data)
    predictions = pipeline_evaluate(config_path=CONFIG_PATH, dataset=new_data)[-1]
    return [predictions]


if __name__ == "__main__":
    # Запустите сервер, используя заданный хост и порт
    uvicorn.run(app, host="127.0.0.1", port=80)
