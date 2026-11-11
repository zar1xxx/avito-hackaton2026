# Avito ranking hackathon 2026

Ранжирование кандидатов в рекомендательной ленте. Метрика — NDCG по лентам.
Дедлайн выбора финала: 09.12.2026 до 12:00.

## Ветки
- `main` — только проверенный код (прошёл валидатор, результат на val записан в `experiments/log.md`)
- `danya` — GBDT, ансамбль
- `avgust` — нейросеть
- `dima` — данные, метрика, валидация
- `anton` — признаки из истории

Правила: в чужие ветки не пушим; чужой код забираем через `main`; без `push --force`;
данные, модели и сабмиты в git не кладём (см. `.gitignore`).

## Структура
```
src/common/metric.py     NDCG / NDCG@10 / NDCG@30, сверены с эталоном на синтетике
src/common/data.py       чтение лент, признаки сходства/истории, ранги внутри ленты
src/common/validation.py временное разбиение val и групповые OOF-фолды
src/common/preparation.py подготовка таблиц и точный EDA по всем данным
src/gbdt/train.py        CatBoost YetiRank: train / predict
eval/make_synthetic.py   крошечный синтетический датасет для проверки кода (не данные Авито)
scripts/validate_submission.py   проверка формата сабмита
scripts/check_dataset.py         проверка распакованного датасета
scripts/make_validation_split.py создание таблицы разбиения val
scripts/prepare_data.py          полный проход: Parquet + EDA
scripts/build_eda_notebook.py    исполняемый отчёт с графиками
scripts/run_baseline_check.py    небольшой CatBoost: весь val, test и валидатор
scripts/reproduce_data.py        одна команда для данных, EDA, фолдов и cosine-кеша
src/common/embedding_catalogue.py общий каталог версий эмбеддингов
src/common/similarity.py         пакетные признаки сходства
notebooks/01_data_eda.ipynb       выполненный EDA по всему датасету
tests/                          сверка с эталоном и проверки данных/сабмитов
experiments/log.md       таблица экспериментов
models/nn/, features/    зона Августа и Антона
```

## Запуск

Окружение данных и GBDT: Python 3.11, версии фиксируются в `uv.lock`.
Основные зависимости совпадают с предоставленным бейзлайном. PyTorch для
нейросети устанавливается отдельно из `competition/baseline/pyproject.toml`.

Из корня проекта в PowerShell:

```powershell
uv sync --locked --no-config --default-index https://pypi.org/simple
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Для сверки с эталоном нужен скачанный
`competition/baseline/calculate_metrics.py`; без него этот тест пропускается.
Проверки используют искусственные данные и не измеряют качество модели.

После распаковки `competition.zip` папки `train`, `val`, `test` должны
лежать непосредственно в `competition/`. Проверить структуру и создать фолды:

```powershell
.\.venv\Scripts\python.exe -m scripts.check_dataset --data competition --sample-feeds 100
.\.venv\Scripts\python.exe -m scripts.make_validation_split --data competition --out runs/val_folds.parquet
.\.venv\Scripts\python.exe scripts/validate_submission.py competition/sample_submission.parquet runs/submission.parquet
```

Проверка датасета читает схемы всех шардов, размеры memmap и все `feed_id`
для поиска дублей и null в ключах/времени. Содержимое списков проверяется
только на первых `--sample-feeds` лентах каждого split; полный EDA нужен отдельно.

Таблица `val_folds.parquet`: `feed_id`, `timestamp`, `leaderboard_split`, `fold`.
Ранние ~30% лент — псевдо-Public, поздние ~70% — псевдо-Private.
Ленты с одинаковым timestamp остаются на одной стороне, поэтому доля может
отличаться от 30%. Фолды (по умолчанию 5, seed=42) случайные и группируются
по `feed_id`; все кандидаты ленты получают одно назначение через join по ключу.
Для OOF модель обучают без фолда, на котором получают предсказания.
Фолды не разделяют пользователей и не гарантируют обучение только на прошлом;
таргетные статистики считаются отдельно по прошлому или OOF.
Подбор по псевдо-Private раскрывает этот holdout, поэтому финальную оценку
устойчивости нужно интерпретировать с учётом использованных экспериментов.

Эталон считает DCG и IDCG с одинаковым отсечением для @10/@30.
`ndcg_idcg30` сохранён для совместимости как экспериментальный вариант,
он не входит в метрики предоставленного `calculate_metrics.py`.

## Подготовленные данные и полный EDA

```powershell
$env:POLARS_MAX_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -u -m scripts.prepare_data --data competition --out data/prepared --eda-out runs/eda
.\.venv\Scripts\python.exe -m scripts.build_eda_notebook
.\.venv\Scripts\python.exe -m scripts.make_validation_split --data competition --out data/prepared/val_folds.parquet
```

Подготовка обрабатывает по одному исходному шарду. При прерывании добавьте
`--resume` к той же команде: записанные части переиспользуются только при
совпадении версии формата, отпечатков источника/выходных файлов и числа строк.
EDA при возобновлении пересчитывается по всем исходным данным. Файлы записываются
через временный файл; `manifest.json` получает `complete=true` только после
всех проверок, включая точное совпадение ключей test с шаблоном сабмита.

```text
data/prepared/
  manifest.json
  train/                       # аналогично val/ и test/
    candidates/part-*.parquet   # одна строка = кандидат
    feeds/part-*.parquet        # одна строка = лента, история хранится один раз
runs/eda/
  summary.json                 # полный EDA, без выборки
  figures/                     # графики из ноутбука
notebooks/
  01_data_eda.ipynb             # код, выполненные таблицы и встроенные графики
  eda_summary.json              # переносимый снимок агрегатов для ноутбука
```

Ключ строки кандидата: `(split, feed_id, candidate_index)`, где candidate_index —
uint32, исходный номер кандидата внутри ленты с 0. Это технический ключ, не признак
модели. В train встречаются повторные item_id в одной ленте с разными pos и f1–f5;
они сохраняются без удаления или объединения меток. Число повторов и конфликтов
меток отражено в EDA, сами строки — в `runs/eda/duplicate_candidates/`.
Признаки уровня товара можно соединять по `(split, feed_id, item_id)` только
как many-to-one; для признаков конкретного вхождения нужен candidate_index.
В test ключи `(feed_id, item_id)` дополнительно сверяются с шаблоном сабмита.

Колонки: `feed_id`, `timestamp`,
`user_id`, `item_id`, `item_id__embedding_idx` — int32; `category_id` — int8;
f1–f5 — float32; target — int8 только в train/val, pos — int16 только в train.
`split` — строка. Таблица feeds содержит `(split, feed_id)`, timestamp, user_id,
user_embedding, все поля history и feed_size. Истории присоединяют по ключу ленты.
Это исходные признаки: сходство с историей и обученные признаки сохраняются
отдельными группами. Таблицы эмбеддингов остаются в `competition/<split>/item_embeddings/`;
их относительный путь указан через `raw_root_relative` в манифесте. При передаче
команде подготовленных таблиц нужно также передать исходные таблицы эмбеддингов.

Ленивое чтение без загрузки всего train в память:

```python
from src.common.preparation import scan_candidates

val = scan_candidates("data/prepared", "val")
print(val.select("feed_id", "candidate_index", "item_id", "f1", "target").limit(5).collect())
```

EDA включает размеры лент/историй, доли позитивов и лент без позитивов, CTR по
позиции, пропуски эмбеддингов, точные пересечения user/item ID, распределения
признаков/категорий/типов событий и объём по датам UTC. Нулевые user/item ID
исключены из уникальных объектов и пересечений, их вхождения посчитаны отдельно.
Таблицы и исходные данные игнорируются Git; ноутбук содержит только агрегаты.

На полном train используйте чтение по частям или отбор целых лент: существующий
`load_split()` загружает весь split в память, даже если подготовка была потоковой.

## Обучение

### Команда воспроизведения для команды

Каждый участник скачивает `competition.zip` с платформы соревнования и
распаковывает его в `competition/`. Дополнительный уровень `competition/competition/`
определяется автоматически. Из корня репозитория:

```powershell
uv run --locked python -m scripts.reproduce_data --raw competition --resume
```

Команда проверяет и подготавливает все исходные таблицы, EDA-ноутбук в `runs/eda/`,
разбиение val, общий каталог эмбеддингов и cosine-кеш для первого train-шарда,
**всего val и всего test**. Подготовленные базовые таблицы при этом полные,
все 512 train-шардов. Проверенный кеш переиспользуется; повреждённые файлы
не считаются готовыми. Передача архивов и тяжёлых таблиц через Git не нужна.

По умолчанию cosine-кеш соответствует небольшому проверочному прогону.
Полная материализация cosine-признаков train на текущем CPU может занять
несколько часов. Она запускается явно с `--all-train-features`; промежуточный
объём — через `--feature-train-parts N`. Покрытие каждого split записано в
`data/features/cosine/manifest.json`: `complete` относится к запрошенному
покрытию, `all_train_parts` показывает наличие всех train-шардов. Потребитель
читает перечень файлов из манифеста и отклоняет незакешированные шарды.

### Каталог эмбеддингов и признаки сходства

Для одного item_id векторы разных split могут различаться. Полный побайтовый
аудит подтвердил различия у 17,30% общих товаров train/val, 27,92% train/test
и 20,60% val/test. Поэтому ключ версии — **`(split, item_id)`**. Каталог
`data/embedding_catalogue/` объединяет 83 260 664 товара, плотные индексы и
Parquet-таблицу item_id с индексами всех трёх split. Векторы хранятся в исходных
memmap-файлах, каталог ссылается на них по относительному пути; повторная
копия 11 ГБ векторов не создаётся. Внутри split проверена согласованность
всех соответствий item_id и векторов.

```python
from src.common.embedding_catalogue import EmbeddingCatalogue

store = EmbeddingCatalogue("data/embedding_catalogue")
vectors = store.gather("val", item_ids, original_missing=(embedding_indices == -1))
```

Cosine-группа `data/features/cosine/` имеет ключ `(split, feed_id, candidate_index)`
и контрольную колонку item_id. Признаки: наличие эмбеддинга кандидата,
cos(user, item), длина истории и для последних 10/50 **исходных событий** —
число доступных эмбеддингов, mean/max cosine. Пропуски исключаются внутри
выбранного окна, не заменяются более старыми событиями. При пустой истории
cosine равен 0. Математика выполняется в float32; исходный `-1` сохраняется,
версии и пропуски не подменяются данными других split. Изменённый кеш
проверяется по исходным файлам, версии расчёта и числу строк.

Самостоятельная сборка и расширение кеша:

```powershell
uv run --locked python -m scripts.build_embedding_catalogue --data competition --out data/embedding_catalogue --resume
uv run --locked python -m scripts.build_similarity_features --train-parts 1 --resume
```

### Проверка моделей

Проверка полной цепочки на подготовленных таблицах (один train-шард,
100 итераций, CPU, f1–f5 и категория):

```powershell
.\.venv\Scripts\python.exe -u -m scripts.run_baseline_check --data data/prepared --raw competition --out runs/baseline-check-001
.\.venv\Scripts\python.exe -u -m scripts.run_baseline_check --data data/prepared --raw competition --feature-root data/features/cosine --out runs/baseline-cosine-001
```

Нужны `data/prepared/val_folds.parquet` и скачанный эталон метрики.
Параметры размера прогона: `--train-parts`, `--iterations`, `--depth`,
`--seed`, `--threads`. Лучший чекпойнт выбирается только по NDCG@30
псевдо-Public; псевдо-Private не передаётся в обучение или early stopping.
Все ленты val оцениваются с float32-скорами; full/@10/@30 сверяются с эталоном
отдельно для всего val и обеих временных частей. Это проверочный прогон
на небольшой части train, не финальное обучение.

Выходы: `model.cbm`, `validation_predictions.parquet`, `submission.parquet`,
`run.json` с параметрами, коммитом/хешами кода, метриками и результатом валидатора.
Val-предсказания — holdout, не OOF: псевдо-Public участвовал в выборе итерации.
Сабмит создаётся локально. Скрипт не отправляет его на платформу и не
перезаписывает существующую папку прогона.

Обучение и инференс (bash, после активации окружения):

```bash
source .venv/bin/activate
export PYTHONPATH=.
python -m src.gbdt.train train --data /path/to/competition --out runs/gbdt-01
python -m src.gbdt.train predict --data /path/to/competition \
    --model runs/gbdt-01/model.cbm --out runs/gbdt-01/submission.parquet
python scripts/validate_submission.py /path/to/competition/sample_submission.parquet runs/gbdt-01/submission.parquet
```
Быстрая проверка на синтетике: `python eval/make_synthetic.py --out /tmp/synth` и тот же запуск с `--data /tmp/synth`.
GPU: `--task-type GPU`. Для быстрых экспериментов: `--max-parts 2`.

## Формат общих артефактов
- Предсказание модели: parquet `feed_id:int32, item_id:int32, score:float32`.
- Для ансамбля также нужны OOF-предсказания на val в том же формате (с колонкой `target`).
