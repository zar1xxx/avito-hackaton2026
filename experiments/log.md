# Журнал экспериментов

Все числа — только на реальных данных. Синтетика проверяет код, а не качество.
Псевдо-Public/Private — временные части локального val; колонка Public относится к платформе соревнования.

| Дата | Автор | Коммит кода | Что сделано | val NDCG (full / @10 / @30) | Псевдо-Public (full / @30) | Псевдо-Private (full / @30) | Public платформы | Примечание |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11.11.2026 | Дима | `0eb8c43` | CatBoost YetiRank; f1–f5 + категория; 1 train-шард, 476,301 кандидатов / 6,132 лент; 60 деревьев, seed=42 | 0.328563 / 0.188423 / 0.259971 | 0.331998 / 0.265654 | 0.327090 / 0.257534 | — | Весь val: 14,423 лент. Чекпойнт выбран по псевдо-Public; Private не передавался в fit. Сверка с эталоном и валидатор сабмита пройдены. |
| 11.11.2026 | Дима | `ff159ef` | Тот же CatBoost + cosine-группа (9 признаков), тот же train-шард; 34 дерева, seed=42 | 0.357691 / 0.225219 / 0.295816 | 0.363805 / 0.304976 | 0.355070 / 0.291889 | — | Весь val; выбор итерации только на псевдо-Public. Эталон и валидатор пройдены; новая группа на исходных версиях split. |

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
