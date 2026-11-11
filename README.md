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
src/gbdt/train.py        простой CatBoost YetiRank: train / predict (бейзлайн)
src/gbdt/run_exp.py      раннер экспериментов: CatBoost/LightGBM, позиция, режимы holdout/cv/final
src/gbdt/tune.py         Optuna по val (ES на pseudo-public, цель — pseudo-private)
src/gbdt/weights.py      IPW-веса против смещения позиции
src/gbdt/extra_features.py  подключение признаков Антона/Августа (контракт на формат — в docstring)
src/ensemble/blend.py    ансамбль: mean / подбор весов / стекинг по OOF
src/ensemble/select_final.py  выбор двух финальных решений
src/common/cv.py         pseudo-public/private (30/70 по времени) и временные фолды
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
