# FCA Handler

Библиотека для работы с ПФА моделью.

Основные возможности:
- Извлечение данных из NLD файлов;
- Предобработка текстов для обучения;
- Обучение и оценка качества модели ПФА.

## Установка библиотеки

### 1. Инициализировать виртуальное окружение

Создание виртуального окружения:
```
python3 -m virtualenv .venv
```

Активация окружения (для linux):
```
source .venv/bin/activate
```

Активация окружения (для windows):
```
.venv/Scripts/activate
```


### 2. Загрузить зависимости проекта
```
pip install -r ./requirements.txt 
```

## Работа с библиотекой

Начальное взаимодействие происходит в файле **prepare_data.py**. В нем производиться обработка директории с nld-файлами.

Перед запуском необходимо:
- В переменной _FOLDER_PATH_ указать абсолютный путь до директории с nld файлами.
- В переменной _TARGET_CLASSES_ указать ГРНТИ-коды классов, которые будут отобраны.
- В переменной _INDEX_OBJECTS_ указать типы объектов, по которым будут отобраны термины ("TERM", "TERMTREE", "TERMTREELOW" или все вместе).
- В переменной _WEIGHT_THRESHOLD_ указать пороговое значение веса для терминов.
- В переменной _SAVE_RESULT_PATH_ указать путь для сохранения результата обработки.

Запуск модуля: `python3 -m ./prepare_data.py`

Результатом обработки будет датафрейм с ID документа, аннотацией, кодом ГРНТИ и терминами. 
Датафрейм для удобства сохраняется в двух форматах: _".csv"_ для дальнейшей обработки и _".excel"_ для чтения.

---

Далее переходим в файл **run_training.py** для запуска обучения модели ПФА по подготовленным данным.

Перед запуском необходимо:
- В переменной _DOCS_DF_PATH_ указать путь до подготовленного ранее _".csv"_ файла (из модуля **prepare_data.py**).
- В переменной _TRAINER_MODE_ указать режим обучения модели: 
  - _"texts"_, если модель нужна на основе лемматизированных слов из текстов.
  - _"terms"_, если модель нужна на основе терминов.
  - _(В разработке)_ _"mixed"_, если модель нужна на основе текстов и терминов.
- В переменной _MAX_FEATURES_TF_IDF_ указать максимальное количества слов (или терминов при _"terms"_), которые сохранятся в матрице TF-IDF.
- В переменной _TEST_SIZE_ указать долю данных, которые будут отобраны для тестовой выборки (Например, 0.2)

Также в данном модуле в `clf = BinaryFCAClassifier` можно изменить параметры обучения ПФА модели.

Запуск модуля: `python3 -m ./run_training.py` 

_Примечение: Обучение на всех данных занимает более 1 часа. Для демо запуска раскомментируйте 25 строку кода_

По результатам обучения:
- Будет получен файл в формате excel с метриками качества модели. Файл сохранится в директории `./data/metrics` с названием, привязанным к размеру обучающей выборке и времени сохранения.
- Будут сохранены логи обучения модели  в директории `./data/logs` с названием, привязанным ко времени запуска программы.
