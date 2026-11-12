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
src/gbdt/train.py        простой CatBoost YetiRank: train / predict (бейзлайн)
src/gbdt/run_exp.py      раннер экспериментов: CatBoost/LightGBM, позиция, режимы holdout/cv/final
src/gbdt/tune.py         Optuna по val (ES на pseudo-public, цель — pseudo-private)
src/gbdt/weights.py      IPW-веса против смещения позиции
src/gbdt/extra_features.py  подключение признаков Антона/Августа (контракт на формат — в docstring)
src/ensemble/blend.py    общий ансамбль N моделей: mean / подбор весов / стекинг по OOF
src/ensemble/select_final.py  выбор двух финальных решений
src/common/cv.py         pseudo-public/private (30/70 по времени) и временные фолды
src/nn/                  нейросетевой пайплайн Августа (docs/AVGUST_PIPELINE.md)
src/ensemble/blend_nn_gbdt.py  ранговый блендинг NN + GBDT (Август), сетка по весу
src/ensemble/stacking.py Ridge-стекинг NN + GBDT (Август)
eval/make_synthetic.py   крошечный синтетический датасет для проверки кода (не данные Авито)
scripts/validate_submission.py   проверка формата сабмита
scripts/check_dataset.py         проверка распакованного датасета
scripts/make_validation_split.py создание таблицы разбиения val
scripts/prepare_data.py          полный проход: Parquet + EDA
scripts/build_eda_notebook.py    исполняемый отчёт с графиками
scripts/run_baseline_check.py    небольшой CatBoost: весь val, test и валидатор
scripts/reproduce_data.py        одна команда для данных, EDA, фолдов и cosine-кеша
scripts/build_infeed_features.py ранги, z-score, размер и категории внутри ленты
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

Полная проверочная цепочка от сырых файлов до валидированного сабмита одной
командой, включая обучение cosine-baseline на одном train-шарде:

```powershell
uv run --locked python -m scripts.reproduce_data --raw competition --resume --baseline-out runs/reproduced-baseline-001
```

Для этого режима нужен предоставленный `competition/baseline/calculate_metrics.py`.
Используются seed=42, фиксированные версии зависимостей, CPU=2 потока, depth=6,
до 100 итераций с выбором по псевдо-Public. Выходная папка модели должна быть
новой. Псевдо-Private оценивается после fit; полноценное обучение на всём train
этот проверочный режим не подразумевает. Дополнительные группы признаков
включаются отдельным вызовом модельного скрипта после сравнения качества.

Команда проверяет и подготавливает все исходные таблицы, EDA-ноутбук в `runs/eda/`,
разбиение val, общий каталог эмбеддингов и cosine-кеш для первого train-шарда,
**всего val и всего test**. Подготовленные базовые таблицы при этом полные,
все 512 train-шардов. Статистики внутри ленты можно дополнительно построить
через `--infeed data/features/infeed`; по умолчанию необязательная группа
не вычисляется. Проверенный кеш переиспользуется; повреждённые файлы
не считаются готовыми. Передача архивов и тяжёлых таблиц через Git не нужна.

По умолчанию cosine-кеш соответствует небольшому проверочному прогону.
Полная материализация cosine-признаков train на текущем CPU может занять
несколько часов. Она запускается явно с `--all-train-features`; промежуточный
объём — через `--feature-train-parts N`. Покрытие каждого split записано в
`data/features/cosine/manifest.json`: `complete` относится к запрошенному
покрытию, `all_train_parts` показывает наличие всех train-шардов. Потребитель
читает перечень файлов из манифеста и отклоняет незакешированные шарды.

Полные базовые таблицы и обе группы признаков одной командой (общая загрузка
векторов для четырёх шардов использует больше RAM):

```powershell
uv run --locked python -m scripts.reproduce_data --raw competition --resume --all-train-features --prefetch-parts 4 --infeed data/features/infeed
```

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

При расширении cosine-кеша проверенные готовые шарды сохраняются: последующий
запуск на одном шарде не исключает остальные из манифеста. Отпечаток численной
формулы и параметров отделён от способа чтения; у каждого файла записана версия
производителя. Поддерживается миграция ранее опубликованного стандартного кеша.

Режим `--reader auto` читает большой train-memmap последовательно и удерживает
в RAM только нужные векторы кандидатов и первых 50 событий. Для небольших
val/test остаётся чтение через отображаемые окна. На одном реальном train-шарде
проверено побитовое совпадение девяти Float32-признаков с прежним кешем.
Скорость зависит от файлового кеша и носителя; замер сохраняется отдельно:

```powershell
uv run --locked python -m scripts.benchmark_similarity --out runs/cosine-reader-benchmark.json
```

Полный cosine-кеш train можно построить с общим чтением векторов для четырёх
шардов за проход. Это сокращает повторное чтение большого memmap и увеличивает
потребление RAM; `--prefetch-parts 1` оставляет обработку по одному шарду.
Прогресс сохраняется атомарно после каждого готового файла, а записи val/test
сохраняются и при прерывании во время расчёта train.

```powershell
uv run --locked python -m scripts.build_similarity_features --resume --prefetch-parts 4
```

После завершения полный кеш проверяется отдельной командой. Для всех строк
сверяются ключи, типы, конечность чисел, доступность эмбеддинга и счётчики
истории; на трёх лентах каждого шарда и всех лентах с повторными item_id
формулы независимо пересчитываются в Float64. Полнота численной проверки
явно ограничена этой выборкой.

```powershell
uv run --locked python -m scripts.audit_similarity_features --out runs/cosine_cache_audit_full.json
```

### Проверка моделей

Увеличение обучающей выборки проверяется отдельно от изменения признаков:

```powershell
uv run --locked python -m scripts.build_similarity_features --train-parts 2 --resume
uv run --locked python -m scripts.run_baseline_check --feature-root data/features/cosine --train-parts 2 --out runs/baseline-cosine-2parts-001
uv run --locked python -m scripts.compare_baselines --baseline runs/baseline-cosine-001 --candidate runs/baseline-cosine-2parts-001 --allow-train-growth --out runs/baseline-cosine-2parts-001/comparison.json
```

Для такого сравнения обязательны те же признаки, параметры, val-строки и
назначения Public/Private; train должен быть строгим расширением исходных
шардов. Без `--allow-train-growth` несовпадающий train отклоняется. В `run.json`
пишутся этап выполнения, доступная RAM и пиковый рабочий набор процесса —
это измерение, а не жёсткий лимит памяти.

Группа `data/features/infeed/` содержит 12 признаков только из текущей ленты:
размер, долю категории кандидата и для f1–f5 — нормированный ранг и z-score.
Ранги возрастающие: средний ранг при равенстве / размер ленты. Z-score использует
генеральное стандартное отклонение (`ddof=0`), для постоянной и одиночной ленты
равен 0. Target, исходная позиция, другие ленты и ID кандидатов в расчёте не
участвуют. Повторные item_id остаются отдельными кандидатами с candidate_index.
Старый общий загрузчик использует те же формулы, включая конечный z-score для
одиночных лент.

```powershell
uv run --locked python -m scripts.build_infeed_features --prepared data/prepared --out data/features/infeed --resume
```

Для быстрого сравнения на одном train-шарде, полном val и test:

```powershell
uv run --locked python -m scripts.build_infeed_features --prepared data/prepared --out data/features/infeed-check --train-parts 1 --resume
```

Параметр `--feature-root` можно повторять: группы соединяются по ключу
конкретного вхождения кандидата, дубли имён признаков отклоняются. Сравнение
cosine-модели с добавлением признаков внутри ленты:

```powershell
uv run --locked python -m scripts.run_baseline_check --feature-root data/features/cosine --feature-root data/features/infeed-check --out runs/baseline-infeed-001
uv run --locked python -m scripts.compare_baselines --baseline runs/baseline-cosine-001 --candidate runs/baseline-infeed-001 --out runs/baseline-infeed-001/comparison.json
```

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

## Запуск GBDT-раннера (Даня)
```bash
export PYTHONPATH=.
python -m src.gbdt.train train --data /path/to/competition --out runs/gbdt-01
python -m src.gbdt.train predict --data /path/to/competition \
    --model runs/gbdt-01/model.cbm --out runs/gbdt-01/submission.parquet
python scripts/validate_submission.py /path/to/competition/sample_submission.parquet runs/gbdt-01/submission.parquet
```
Синтетика для отладки (формат соревнования, смещение позиции в train, несколько частей; НЕ данные Авито):
`python eval/make_synthetic.py --out data/synth --feeds 20000 --parts 4` и тот же запуск с `--data data/synth`.
GPU: `--task-type GPU`. Для быстрых экспериментов: `--max-parts 2`.

## Формат общих артефактов
- Предсказание модели: parquet `feed_id:int32, item_id:int32, score:float32`.
- Для ансамбля также нужны OOF-предсказания на val в том же формате (с колонкой `target`).

## Эксперименты GBDT (зона Дани)
```bash
export PYTHONPATH=.
D=/path/to/competition
# 1. сравнение лоссов и режимов позиции (кэш признаков ускоряет повторные запуски; --n-jobs — процессы)
python -m src.gbdt.run_exp --data $D --out runs/cb-yeti   --cache runs/cache --n-jobs 4 --log
python -m src.gbdt.run_exp --data $D --out runs/cb-pair   --cache runs/cache --loss YetiRankPairwise --log
python -m src.gbdt.run_exp --data $D --out runs/lgb       --cache runs/cache --kind lightgbm --log
python -m src.gbdt.run_exp --data $D --out runs/lgb-ipw   --cache runs/cache --kind lightgbm --pos-mode ipw --log
python -m src.gbdt.run_exp --data $D --out runs/cb-posf   --cache runs/cache --pos-mode feature --log
python -m src.gbdt.run_exp --data $D --out runs/cb-top50  --cache runs/cache --pos-mode topn --topn 50 --log
# 2. подбор параметров, затем перепроверка лучших режимом cv
python -m src.gbdt.tune --data $D --out runs/tune-lgb --kind lightgbm --trials 40 --cache runs/cache
python -m src.gbdt.run_exp --data $D --out runs/lgb-tuned --kind lightgbm --cache runs/cache $(cat runs/tune-lgb/best_cli.txt)
# 3. OOF на val для ансамбля, затем финальное обучение на train+val
python -m src.gbdt.run_exp --data $D --out runs/lgb-cv --kind lightgbm --mode cv --iterations-from runs/lgb-tuned/metrics.json ...
python -m src.gbdt.run_exp --data $D --out runs/lgb-final --kind lightgbm --mode final --val-weight 3 --iterations-from runs/lgb-tuned/metrics.json ...
# 4. ансамбль и выбор финала
python -m src.ensemble.blend --data $D --runs runs/lgb-cv runs/cb-cv runs/nn-cv --out runs/blend
python -m src.ensemble.select_final --data $D --runs runs/lgb-cv runs/cb-cv runs/blend --safe runs/lgb-cv runs/cb-cv
python scripts/validate_submission.py $D/sample_submission.parquet runs/blend/test_pred.parquet
```
Заметки:
- CatBoost с парными лоссами (YetiRank, PairLogit) **игнорирует веса объектов**, поэтому IPW и `--val-weight` работают с LightGBM или с `--loss LambdaMart`; раннер сам остановится с ошибкой.
- `--log` дописывает строку в `experiments/log.md` — включать только на реальных данных.
- Метрики `priv_*` — на pseudo-private (поздние 70% val по времени); early stopping идёт по pseudo-public.
- Все числа на синтетике (`eval/make_synthetic.py`) проверяют только работоспособность кода.

## Установка на Windows (PowerShell)
Единое окружение команды — Python 3.11 и версии из `uv.lock` (код Димы не работает с polars 2.x и Python 3.12).
```powershell
winget install Python.Python.3.11
winget install astral-sh.uv          # затем перезапустить PowerShell
uv sync --locked --no-config --default-index https://pypi.org/simple
.\.venv\Scripts\python.exe -m unittest discover -s tests
```
Если `.venv` уже создан другой версией Python — удалить папку `.venv` перед `uv sync`.
В PowerShell команды запускаются по одной строке (без `&&`); перенос строки — обратный апостроф.

## Быстрый старт для нейросети (Август)

Подробная техническая документация архитектуры и контрактов: [docs/AVGUST_PIPELINE.md](docs/AVGUST_PIPELINE.md).

**Рекомендации по железу:**
- **CPU (Windows / локальная разработка):** добавляйте `--num-workers 0 --batch-size 32`.
- **GPU (Linux / облако):** добавляйте `--amp --num-workers 4 --batch-size 128` (включает смешанную точность fp16 и параллельную загрузку).

### 1. Проверка формата реальных данных (5 секунд)
```bash
python -u scripts/quick_check_real_data.py --data /path/to/competition
```

### 2. Полный пайплайн одной командой (проверка + smoke + train + finetune + OOF + сабмит)
```bash
# Вариант для GPU:
python -u scripts/run_full_pipeline.py \
    --data /path/to/competition \
    --amp --scheduler cosine --warmup-epochs 1 \
    --epochs 10 --finetune-epochs 3 --batch-size 128 --num-workers 4 \
    --out models/nn/pipeline_run

# Вариант для CPU:
python -u scripts/run_full_pipeline.py \
    --data /path/to/competition \
    --epochs 10 --finetune-epochs 3 --batch-size 32 --num-workers 0 \
    --out models/nn/pipeline_run
```

### 3. Экспорт нейро-фичей для GBDT Дани (`features/avgust/`)
Обучает модель K-Fold (честный OOF на `train` без утечки таргета) и сохраняет `train.parquet`, `val.parquet`, `test.parquet` с колонками `x_avgust_*`:
```bash
python -u src/nn/export_gbdt_features.py \
    --data /path/to/competition \
    --folds 4 --epochs 8 --batch-size 64 \
    --out features
```

### 4. Взвешенный сид-ансамбль нейросетей (`seed_ensemble.py`)
Обучает 3 модели (сиды 42, 43, 44), подбирает оптимальные веса по `val` и генерирует сабмит:
```bash
python -u src/nn/seed_ensemble.py \
    --data /path/to/competition \
    --seeds 42 43 44 --epochs 10 --batch-size 64 \
    --mixup-alpha 0.2 --temperature 1.5 --agg rank --predict-test \
    --out models/nn/ensemble_v1
```

### 5. Инференс в формате Дани (`val_pred.parquet` / `test_pred.parquet`)
Для подключения в `select_final.py` и `blend.py` Дани используйте флаг `--save-as-danya`:
```bash
python -u src/nn/predict.py --data /path/to/competition --split val --model models/nn/exp/best_model.pt --save-as-danya --out runs/nn-01
python -u src/nn/predict.py --data /path/to/competition --split test --model models/nn/exp/best_model.pt --save-as-danya --out runs/nn-01
```

### 6. Блендинг и стекинг с GBDT (Даня + Август)
```bash
# Ранговый блендинг (подбор весов по сетке на val + сабмит на test):
python -u src/ensemble/blend_nn_gbdt.py \
    --val-nn runs/nn-01/oof_val.parquet \
    --val-gbdt runs/gbdt-01/oof_val.parquet \
    --test-nn runs/nn-01/submission.parquet \
    --test-gbdt runs/gbdt-01/submission.parquet \
    --out runs/ensemble/submission_blend.parquet

# Ridge-стекинг на [nn_score, gbdt_score, f1..f5] с GroupKFold по feed_id:
python -u src/ensemble/stacking.py \
    --data /path/to/competition \
    --val-nn runs/nn-01/oof_val.parquet \
    --val-gbdt runs/gbdt-01/oof_val.parquet \
    --test-nn runs/nn-01/submission.parquet \
    --test-gbdt runs/gbdt-01/submission.parquet \
    --alpha 10.0 \
    --out runs/ensemble/submission_stack.parquet
```

## GBDT-раннер на подготовленных данных Димы
Сначала данные и кеши признаков (см. раздел Димы выше), для N train-шардов:
```powershell
uv run --locked python -m scripts.reproduce_data --raw competition --resume --feature-train-parts 8 --infeed data/features/infeed
```
Затем эксперименты (`--prepared` читает готовые таблицы, `--feature-root` можно повторять,
`--train-parts` — сколько train-шардов, равномерно по времени; кеш должен их покрывать):
```powershell
$F = "--data competition --prepared data/prepared --feature-root data/features/cosine --feature-root data/features/infeed --train-parts 8"
uv run --locked python -m src.gbdt.run_exp $F.Split() --out runs/cb8 --kind catboost --log
uv run --locked python -m src.gbdt.run_exp $F.Split() --out runs/lgb8 --kind lightgbm --log
uv run --locked python -m src.gbdt.run_exp $F.Split() --out runs/lgb8-ipw --kind lightgbm --pos-mode ipw --log
uv run --locked python -m src.gbdt.run_exp $F.Split() --out runs/cb8-cv --kind catboost --mode cv --log
uv run --locked python -m src.ensemble.blend --data competition --runs runs/cb8-cv runs/lgb8-cv --out runs/blend
uv run --locked python scripts/validate_submission.py competition/sample_submission.parquet runs/blend/test_pred.parquet
```
Метрика выбора везде — NDCG@30 (как в эталоне); early stopping — по pseudo-public, оценка — по pseudo-private.
Режим `cv` добавляет val в обучение: val собран так же, как test (без смещения позиции), поэтому это главный кандидат на прирост.
