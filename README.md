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
src/gbdt/train.py        CatBoost YetiRank: train / predict
eval/make_synthetic.py   крошечный синтетический датасет для проверки кода (не данные Авито)
scripts/validate_submission.py   проверка формата сабмита
scripts/check_dataset.py         проверка распакованного датасета
scripts/make_validation_split.py создание таблицы разбиения val
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
