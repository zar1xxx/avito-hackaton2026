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
src/common/metric.py     NDCG по правилам соревнования (3 варианта, сверить с calculate_metrics.py)
src/common/data.py       чтение лент, признаки сходства/истории, ранги внутри ленты
src/gbdt/train.py        CatBoost YetiRank: train / predict
eval/make_synthetic.py   крошечный синтетический датасет для проверки кода (не данные Авито)
scripts/validate_submission.py   проверка формата сабмита
experiments/log.md       таблица экспериментов
models/nn/, features/    зона Августа и Антона
```

## Запуск
```bash
pip install polars catboost lightgbm pyarrow numpy pandas
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

## Нейросеть (Август)
```bash
# Обучение модели
python -u src/nn/train.py --data /path/to/competition --epochs 10 --batch-size 64 --out runs/nn-01

# Предсказание на test (сабмит)
python -u src/nn/predict.py --data /path/to/competition --split test --model runs/nn-01/best_model.pt --out runs/nn-01/submission.parquet

# OOF-предсказания на val (для блендинга)
python -u src/nn/predict.py --data /path/to/competition --split val --model runs/nn-01/best_model.pt --out runs/nn-01/oof_val.parquet
```

## Ранговый блендинг GBDT + NN (Даня и Август)
```bash
# Подбор весов на val и генерация итогового сабмита:
python -u src/ensemble/blend.py \
    --val-nn runs/nn-01/oof_val.parquet \
    --val-gbdt runs/gbdt-01/oof_val.parquet \
    --test-nn runs/nn-01/submission.parquet \
    --test-gbdt runs/gbdt-01/submission.parquet \
    --out runs/ensemble/submission_blend.parquet
```
