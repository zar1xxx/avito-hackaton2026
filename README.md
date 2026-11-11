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
src/common/cv.py         временное деление val на pseudo-public (30%) и pseudo-private (70%)
src/gbdt/train.py        CatBoost YetiRank: train / predict
src/nn/                  нейросетевой пайплайн Августа (подробнее в docs/AVGUST_PIPELINE.md)
src/ensemble/            ранговый блендинг NN+GBDT (blend_nn_gbdt.py) и Ridge-стекинг (stacking.py)
eval/make_synthetic.py   крошечный синтетический датасет для проверки кода (не данные Авито)
scripts/validate_submission.py   проверка формата сабмита
experiments/log.md       таблица экспериментов
models/nn/, features/    зона Августа и Антона
```

## Запуск GBDT
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

---

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
