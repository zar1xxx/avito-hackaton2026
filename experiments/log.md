# Журнал экспериментов

Все числа — только на реальных данных. Синтетика проверяет код, а не качество.
Псевдо-Public/Private — временные части локального val; колонка Public относится к платформе соревнования.

| Дата | Автор | Коммит кода | Что сделано | val NDCG (full / @10 / @30) | Псевдо-Public (full / @30) | Псевдо-Private (full / @30) | Public платформы | Примечание |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11.11.2026 | Дима | `0eb8c43` | CatBoost YetiRank; f1–f5 + категория; 1 train-шард, 476,301 кандидатов / 6,132 лент; 60 деревьев, seed=42 | 0.328563 / 0.188423 / 0.259971 | 0.331998 / 0.265654 | 0.327090 / 0.257534 | — | Весь val: 14,423 лент. Чекпойнт выбран по псевдо-Public; Private не передавался в fit. Сверка с эталоном и валидатор сабмита пройдены. |
| 11.11.2026 | Дима | `ff159ef` | Тот же CatBoost + cosine-группа (9 признаков), тот же train-шард; 34 дерева, seed=42 | 0.357691 / 0.225219 / 0.295816 | 0.363805 / 0.304976 | 0.355070 / 0.291889 | — | Весь val; выбор итерации только на псевдо-Public. Эталон и валидатор пройдены; новая группа на исходных версиях split. |
| 11.11.2026 | Дима | `da4c0a6` | Cosine + 12 признаков внутри ленты; тот же train-шард, 40 деревьев, seed=42 | 0.356414 / 0.225054 / 0.294685 | 0.363243 / 0.304885 | 0.353487 / 0.290312 | — | Эталон и валидатор пройдены. Улучшение не подтверждено, группа опциональная. |

Подробные параметры, хеши кода и данных, метрики и результат проверки: [baseline_check_001.json](baseline_check_001.json). Коммит в таблице — чистый коммит, на котором выполнялся прогон.

Воспроизведение из корня проекта после подготовки данных:

```powershell
.\.venv\Scripts\python.exe -u -m scripts.run_baseline_check --data data/prepared --raw competition --out runs/baseline-check-001 --train-parts 1 --iterations 100 --depth 6 --seed 42 --threads 2
```

Если папка прогона уже существует, задайте новое имя в `--out`. Модель и предсказания находятся локально в `runs/baseline-check-001/`. Это проверочный прогон на 0,193% кандидатов train; обучение на полном train и OOF на val здесь не выполнялись.


## Cosine-группа: сравнение с baseline

Тот же train-срез (476 301 кандидат / 6 132 ленты), параметры и разбиение val. NDCG@30 на псевдо-Private: 0.257534 → 0.291889; прирост 0.034355 (13.34% относительно baseline). Парный bootstrap по 10 095 лентам, 1000 повторов, seed=42: 95% интервал разности [0.030372, 0.038557]. Это оценка на фиксированном val для моделей с малым train, а не результат обучения на полном train.

[Параметры cosine-прогона](baseline_cosine_001.json), [парное сравнение](cosine_comparison_001.json), [полный аудит эмбеддингов](embedding_catalogue_audit.json).

```powershell
.\.venv\Scripts\python.exe -u -m scripts.run_baseline_check --data data/prepared --raw competition --feature-root data/features/cosine --out runs/baseline-cosine-001 --train-parts 1 --iterations 100 --depth 6 --seed 42 --threads 2
.\.venv\Scripts\python.exe -m scripts.compare_baselines --baseline runs/baseline-check-001 --candidate runs/baseline-cosine-001 --out runs/baseline-cosine-001/comparison.json
```

Cosine-кеш покрывает один train-шард и весь val/test. Полные исходные таблицы и общий каталог построены для всех split. Расширение train-кеша запускается отдельно; текущий эксперимент использует только заявленное покрытие.


## Признаки внутри ленты: сравнение с cosine-моделью

При совпадающих train-срезе, параметрах и val-назначениях NDCG@30 псевдо-Private: 0.291889 → 0.290312. Разность -0.001577 (-0.54%). Парный bootstrap по лентам, 1000 повторов: 95% интервал [-0.004025, 0.000615]. Улучшение на этом малом train не подтверждено. Проверочный baseline по умолчанию остаётся cosine-моделью; группа статистик доступна отдельно для экспериментов на других объёмах train и моделях.

[Параметры прогона](baseline_infeed_001.json), [парное сравнение](infeed_comparison_001.json). Предсказания — holdout, не OOF.

```powershell
uv run --locked python -m scripts.run_baseline_check --feature-root data/features/cosine --feature-root data/features/infeed-check --out runs/baseline-infeed-001
uv run --locked python -m scripts.compare_baselines --baseline runs/baseline-cosine-001 --candidate runs/baseline-infeed-001 --out runs/baseline-infeed-001/comparison.json
```


## Полный кеш и сквозная воспроизводимость

Признаки внутри ленты материализованы для всех 640 файлов: 246 745 836 кандидатов train, 1 389 971 val, 2 904 806 test; всего 251 040 613 строк, около 7,1 ГиБ. Проверены число строк, схемы и отпечатки каждого файла; формулы дополнительно сверены независимо на 22 реальных лентах, включая ленту с повторным товаром. [Аудит полного кеша](infeed_cache_audit.json).

Единая команда `scripts.reproduce_data --baseline-out` проверена на реальных данных с переиспользованием проверенных кешей. Cosine-baseline обучен заново на одном train-шарде. Схемы, ключи и все значения val/test-предсказаний совпали с исходным прогоном; Float32-скоры совпали побитово. Метрики совпали, валидатор пройден. [Проверка воспроизводимости](reproduction_check_001.json).

```powershell
uv run --locked python -m scripts.reproduce_data --raw competition --resume --baseline-out runs/reproduced-baseline-001
```

Обычная команда не материализует необязательную группу внутри ленты. Для её полного построения или проверки существующего кеша добавьте `--infeed data/features/infeed`. Default-модель остаётся cosine-baseline; негативный/неопределённый результат новой группы не скрывается и не заменяет принятый baseline.
