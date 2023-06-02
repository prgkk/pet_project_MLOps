# Описание проекта

### MLOps pet-project на основе соревнования "DataDrive2030: Early learning predictors challenge"

Задача: Использовать методы машинного обучения для выявления факторов в программах раннего обучения,
которые способствуют улучшению результатов обучения у детей.
При прогнозировании оценки ELOM ребенка и 15 наиболее важных переменных для каждого ребенка.

Ссылка на соревнование: https://zindi.africa/competitions/datadrive2030-early-learning-predictors-challenge

# Инструкция

## Основные команды для запуска FastAPI

- Запуск приложения из папки backend, где --reload - указывает на автоматическое обновление при изменении кода

`cd backend`

`uvicorn main:app --host=0.0.0.0 --port=8000 --reload`

Доступ к сервису FastAPI, при условии, что прописан ранее 8000 порт
http://localhost:8000/docs

___

## Основные команды для запуска Streamlit

Frontend чась проекта запускаем только после запуска backend в отдельном терминале

- Команда в отдельном теминале для запуска приложения Streamlit

`cd frontend`

`streamlit run main.py`
Приложение будет доступно по адресу http://localhost:8501 

Запуск по конкретному порту, например 8080:

`streamlit run main.py --server.port 8080`

Убить процессы:

`for pid in $(ps -ef | grep "streamlit run" | awk '{print $2}'); do kill -9 $pid; done`
___

## Configuration file

Если запускаете проект не в контейнере, то необходимо поменять в конфигурационном файле endpoints:

```yaml
endpoints:
  train: 'http://localhost:8000/train'
  prediction_input: 'http://localhost:8000/predict_input'
  prediction_from_file: 'http://localhost:8000/predict'
```

Если запускаете в контенере, то вместо localhost должно быть название сервиса или контейнера
```yaml
endpoints:
  train: 'http://fastapi:8000/train'
  prediction_input: 'http://fastapi:8000/predict_input'
  prediction_from_file: 'http://fastapi:8000/predict'
```
___

## Основные команды Docker на примере backend

- Отдельный запуск образа backend из директории mlops

`docker build -t fastapi:ver1 backend -f backend/Dockerfile`

- Запуск образа из папки backend

`cd backend`

`docker build -t fastapi:ver1 .`

- Запуск и **создание** нового контейнера из образа fastapi с названием fastapi_run в автономном режиме с использованием портов

`docker run -p 8000:8000 -d --name fastapi_run fastapi:ver1`

- Остановить контейнер

`docker stop fastapi_run`

- Запустить существующий контейнер

`docker start fastapi_run`

- Удалить контейнер

`docker rm fastapi_run`

- При изменении кода необходимо сначала остановить (также можно удалить созданные контейнеры, связанные с образом)

`docker stop fastapi_run`

`docker rm fastapi_run`

- Далее создаем новый образ с новым тэгом

`docker build -t fastapi:ver2 backend -f backend/Dockerfile`

- Далее создаем новый контейнер по новому образу

`docker run -p 8000:8000 -d --name fastapi_run fastapi:ver2`
___

## Docker Compose

- Сборка сервисов из образов внутри backend/frontend и запуск контейнеров в автономном режиме

`docker compose up -d`

- При изменениях в коде необходимо заново пересобрать образы

`docker compose up -d --build`

- Остановка сервисов (чтобы сделать изменения в коде)

`docker compose stop`

- Удалить **остановленные** контейнеры

`docker compose rm`

___
## Folders
- `/backend` - Папка с проектом FastAPI
- `/frontend` - Папка с проектом Streamlit
- `/config` - Папка, содержащая конфигурационный файл
- `/data` - Папка, содержащая исходные данные, обработанные данные, уникальные значения в формате JSON, а также неразмеченный файл для подачи на вход модели
- `/models` - Папка, содержащая сохраненную модель после тренировки, а также объект study (Optuna)
- `/notebooks` - Папка, содержащая jupyter ноутбуки с предварительным анализом данных
- `/report` - Папка, содержащая информацию о лучших параметрах и метриках после обучения
