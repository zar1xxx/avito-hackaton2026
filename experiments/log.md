# Журнал экспериментов

Все числа — только на реальных данных. Синтетика проверяет код, а не качество.
Псевдо-Public/Private — временные части локального val; колонка Public относится к платформе соревнования.

| Дата | Автор | Коммит кода | Что сделано | val NDCG (full / @10 / @30) | Псевдо-Public (full / @30) | Псевдо-Private (full / @30) | Public платформы | Примечание |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11.11.2026 | Дима | `0eb8c43` | CatBoost YetiRank; f1–f5 + категория; 1 train-шард, 476,301 кандидатов / 6,132 лент; 60 деревьев, seed=42 | 0.328563 / 0.188423 / 0.259971 | 0.331998 / 0.265654 | 0.327090 / 0.257534 | — | Весь val: 14,423 лент. Чекпойнт выбран по псевдо-Public; Private не передавался в fit. Сверка с эталоном и валидатор сабмита пройдены. |
| 11.11.2026 | Дима | `ff159ef` | Тот же CatBoost + cosine-группа (9 признаков), тот же train-шард; 34 дерева, seed=42 | 0.357691 / 0.225219 / 0.295816 | 0.363805 / 0.304976 | 0.355070 / 0.291889 | — | Весь val; выбор итерации только на псевдо-Public. Эталон и валидатор пройдены; новая группа на исходных версиях split. |
| 11.11.2026 | Дима | `da4c0a6` | Cosine + 12 признаков внутри ленты; тот же train-шард, 40 деревьев, seed=42 | 0.356414 / 0.225054 / 0.294685 | 0.363243 / 0.304885 | 0.353487 / 0.290312 | — | Эталон и валидатор пройдены. Улучшение не подтверждено, группа опциональная. |

| 12.11.2026 | Дима | `f948754` | Cosine, 2 train-шарда; 958,087 кандидатов / 12,265 лент; 80 дерева, seed=42 | 0.359267 / 0.227615 / 0.297908 | 0.365115 / 0.307071 | 0.356759 / 0.293979 | — | Расширение train при тех же признаках/параметрах. Весь val, выбор чекпойнта только по псевдо-Public. Эталон и валидатор пройдены. |

| 13.11.2026 | Дима | `3f9c669` | Cosine + 32 itemstats; 1 train-шард, 42 дерева; статистики из 504 других шардов, alpha=20 | 0.360929 / 0.231232 / 0.299849 | 0.368352 / 0.310286 | 0.357746 / 0.295374 | — | Выбор чекпойнта по pseudo-Public. Эталон и валидатор пройдены. Источники статистик исключают 8 шардов, включая обучающий. |
| 13.11.2026 | Дима | `6c20d45` | Cosine + itemstats + 16 userstats; тот же 1 train-шард, 63 признака, 41 дерево; alpha=20 | 0.361030 / 0.230818 / 0.300081 | 0.368273 / 0.310417 | 0.357925 / 0.295649 | — | Выбор чекпойнта по pseudo-Public. Эталон и валидатор пройдены. Прирост от userstats не подтверждён: 95% интервал разности включает ноль. |

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


## Расширение train до двух шардов — 12.11.2026

Cosine-кеш покрывает 2/512 train-шардов (958,087 кандидатов), а также весь val/test. Проверены число строк и отпечатки всех 130 файлов. Запуск `--train-parts 1 --resume` сохраняет оба готовых train-шарда и не перезаписывает Parquet. [Аудит покрытия](cosine_cache_audit_002.json).

Чтение большого train-memmap заменено последовательным проходом с удержанием только нужных int8-векторов. На одном реальном шарде старый способ занял 164,05 с, новый — 20,75 с (наблюдаемое отношение 7,91). Порядок запуска был «старый, затем новый», поэтому файловый кеш мог повлиять на время. Все девять Float32-признаков совпали с прежним кешем побитово. После явного выравнивания окон это проверено повторно на 476 301 кандидате; нестандартный batch=3 проверен тестом с пересечением границы 128 лент. [Замер и проверка](cosine_reader_benchmark_001.json).

Это сравнение объёма обучения: 1 → 2 train-шарда, то есть 0,193% → 0,388% кандидатов train. Набор признаков, параметры, val-ключи, таргеты и временные назначения совпадают. Чекпойнты выбирались только по псевдо-Public; псевдо-Private оценён после обучения. NDCG@30 псевдо-Private: 0.291889 → 0.293979; разность +0.002090 (+0.72%). Парный bootstrap по 10 095 лентам, 1000 повторов, seed=42: 95% интервал [-0.000129, 0.004226]. Интервал включает ноль: устойчивый прирост пока не подтверждён. Это оценка на фиксированном val для малого train; она не переносится автоматически на полный train или лидерборд.

Прогон выполнен с чистого коммита `f948754`. Позднейшее исправление окон при нестандартном батче не меняет использованные в нём признаки с batch=16. Пиковый рабочий набор процесса — 1.036 ГиБ; длительность всей цепочки — 807.0 с. Полный test-сабмит на 2,904,806 строк прошёл валидатор; загрузки на платформу не было. Полная материализация cosine для train остаётся отдельным шагом.

[Параметры и метрики](baseline_cosine_2parts_001.json), [сравнение роста train](train_growth_comparison_001.json).

```powershell
uv run --locked python -m scripts.build_similarity_features --train-parts 2 --resume
uv run --locked python -m scripts.run_baseline_check --feature-root data/features/cosine --train-parts 2 --out runs/baseline-cosine-2parts-001
uv run --locked python -m scripts.compare_baselines --baseline runs/baseline-cosine-001 --candidate runs/baseline-cosine-2parts-001 --allow-train-growth --out runs/baseline-cosine-2parts-001/comparison.json
```


## Полный cosine-кеш — 12.11.2026

Завершена материализация всех 640 файлов с девятью Float32-признаками. Train содержит 246,745,836 строк в 512 шардах, val — 1,389,971 в 64, test — 2,904,806 в 64. Всего 251,040,613 вхождений кандидатов, размер кеша 4.936 ГиБ.

Полный аудит прошёл. Для каждой строки проверены ключи и порядок относительно подготовленных таблиц, схема, конечность чисел, доступность эмбеддинга и счётчики исходной истории. Девять формул независимо пересчитаны в Float64 на 1,921 лентах (153,230 кандидатов), включая 1 ленты с повторными item_id. Численная проверка охватывает эту выборку. Допуски абсолютной и относительной ошибки составляют 2e-6. Длительность аудита — 429.1 с.

Расчёт возобновлялся по сохранённому манифесту. Последний проход использовал `--prefetch-parts 16`, пиковый рабочий набор процесса составил 1.197 ГиБ. Версия численных формул сохранена, все файлы имеют сведения о производителе. Аудит хранит отпечаток итогового манифеста и версию проверяющего скрипта.

[Отчёт полного аудита](cosine_cache_audit_full.json). Полные cosine- и within-feed-признаки готовы для командных моделей. Подготовка после скачивания исходного архива воспроизводится одной командой, размер группы чтения можно выбрать по доступной RAM.

```powershell
uv run --locked python -m scripts.reproduce_data --raw competition --resume --all-train-features --prefetch-parts 4 --infeed data/features/infeed
uv run --locked python -m scripts.audit_similarity_features --out runs/cosine_cache_audit_full.json
```


## History-кеш и типы событий — 12.11.2026

Дима подключил общий код `origin/danya` (`e79674c`) в ветку `dima`. Фактическая группа history содержит 46 признаков. В прежнем описании было указано 49; README исправлен по списку колонок кода.

Построены выборки на 8 (3,850,759 кандидатов) и 32 (15,373,984 кандидатов) train-шардах, val/test покрыты полностью. Первый расчёт занял 86.91 с, расширение — 131.27 с. Медианное время расчёта train-шарда в итоговом кеше — 4.289 с. Наборы выбираются через linspace и пересекаются частично, поэтому кеш сохраняет объединение из 38 train-шардов. Файлы и метаданные проверяются по отпечаткам кандидатов и историй; прогресс сохраняется после каждого шарда.

Аудит прошёл для 22,561,433 строк. Проверены ключи, схема, отсутствие null/NaN/inf, отпечатки обоих источников и файлов признаков. Число совпадений кандидата с товаром в истории независимо проверено на всех строках. Все 46 формул независимо пересчитаны на 498 лентах (50,326 строк), допуски rtol/atol=2e-6. Повторные item_id дополнительно проверены синтетическим тестом с сохранением отдельных candidate_index.

Таблицы `h_item_evtN > 0 → доля target=1` рассчитаны для обеих выборок train и всего val. Во всех десяти группах нет наблюдений. Полная проверка исходных историй подтвердила ноль совпадений кандидата с товаром собственной истории на 38 покрытых train-шардах и всём val/test. Поэтому вероятность target по этим группам оценить невозможно. Типы 1–10 присутствуют в историях, признаки категорий и профиля событий рассчитаны. Первые и последние ленты первого файла каждого split дополнительно сверены с сырым архивом.

Загрузчик `src.gbdt.prepared.load_prepared` проверен на реальном train-шарде: 476 301 строка, 73 признака (6 базовых + 9 cosine + 12 infeed + 46 history), соединение строго по вхождению кандидата. Отдельно подтверждено покрытие нужных шардов для размеров 8 и 32.

Проверки — 62 unittest-теста инфраструктуры и history, все прошли. В общем uv-окружении отсутствуют PyTorch и pytest; нейросетевой pytest-модуль `tests/test_edge_cases.py` в этот прогон не входил.

[Время и покрытие](history_materialization_001.json), [аудит 8 шардов](history_cache_audit_008.json), [итоговый аудит](history_cache_audit_032.json), [интеграция с GBDT](history_gbdt_integration_001.json), [события на 8 шардах](history_event_target_008.md), [события на 32 шардах](history_event_target_032.md).

```powershell
uv run --locked python -m scripts.build_history_features --train-parts 8 --resume
uv run --locked python -m scripts.build_history_features --train-parts 32 --resume
uv run --locked python -m scripts.audit_history_features --out runs/history_cache_audit.json
uv run --locked python -m scripts.analyse_history_events --train-parts 32 --out experiments/history_event_target_032
```


## Itemstats — 13.11.2026

Реализованы 32 признака товаров и категорий. Для модели на восьми train-шардах агрегаты рассчитаны по остальным 504 шардам, содержащим 242,895,077 показов и 5,362,045 позитивов. Общий prior CTR равен 0.02207556. По этим же источникам рассчитана кривая CTR для 300 позиций. Target val/test не участвует в статистиках; идентификаторы кандидатов используются для ограничения размера таблицы lookup.

Для каждой сущности вычислены показы, клики, CTR со сглаживанием alpha=20, ожидаемые клики по позициям, сглаженное отношение кликов к ожидаемым, число событий в историях и частоты типов 1–10. Повторные события учитываются как отдельные вхождения. Неизвестные сущности получают нулевые счётчики, prior CTR и нейтральное отношение 1. Значение alpha выбрано до оценки модели и в этом эксперименте не подбиралось.

Кеш покрывает 8 train-шардов и весь val/test: 8,145,536 строк, 0.251 ГиБ Parquet. Исходный лимит DuckDB 1024 МиБ с двумя потоками был достигнут при агрегации историй. Расчёт возобновлён с сохранённых таблиц при лимите 1536 МиБ и одном потоке. Успешный проход занял 358.36 с, пиковый рабочий набор процесса составил 1.661 ГиБ.

Полный аудит ключей, схемы, конечности чисел, источников и раздельности шардов прошёл. Для 408 кандидатов (три строки каждого файла) все 32 значения независимо пересчитаны через Polars и Python по всем 504 исходным шардам. Допуски rtol/atol=2e-6. Повторно сверены общий prior и CTR всех позиций. Длительность аудита — 79.47 с.

В test у 53.71% строк есть показы товара в источниках, у 51.36% — события товара в исходных историях. Это доли вхождений кандидатов.

Интеграция с `src.gbdt.prepared.load_prepared` проверена на 476 301 строке со всеми четырьмя группами и 105 признаками. Попытка использовать этот itemstats-кеш для выбранных 32 обучающих шардов отклоняется, поскольку часть из них входит в источники. Для другой обучающей выборки нужен отдельный кеш с новым исключением шардов.

Проверочный CatBoost YetiRank обучен на том же одном шарде и с теми же параметрами, что исходная cosine-модель: 100 максимальных итераций, depth=6, lr=0.1, seed=42, CPU=2. Использовано 47 признаков (6 базовых, 9 cosine, 32 itemstats). Выбрано 42 дерева по pseudo-Public. Статистики исключают восемь шардов, включая этот обучающий шард. NDCG@30 pseudo-Private вырос 0.291889 → 0.295374, разность +0.003485 (+1.19%). Парный bootstrap по 10 095 лентам, 1000 повторов, seed=42 дал 95% интервал [0.000296, 0.006773]. Оценка относится к этому фиксированному val и проверочному обучению на одном шарде.

Full/@10/@30 совпали с предоставленным эталоном. Сабмит на 2 904 806 строк прошёл валидатор и сохранён локально. Общий прогон — 64 unittest-теста инфраструктуры, history и itemstats, все прошли. Нейросетевой pytest-модуль требует отдельного окружения с PyTorch и pytest.

[Материализация и покрытие](itemstats_materialization_008.json), [аудит](itemstats_008_audit.json), [интеграция](itemstats_gbdt_integration_001.json), [модель](baseline_itemstats_001.json), [парное сравнение](itemstats_comparison_001.json).

```powershell
uv run --locked python -m scripts.build_itemstats_features --train-parts 8 --out data/features/itemstats-008 --memory-mb 1536 --threads 1 --resume
uv run --locked python -m scripts.audit_itemstats_features --cache data/features/itemstats-008 --out runs/itemstats_008_audit.json
uv run --locked python -m scripts.run_baseline_check --feature-root data/features/cosine --feature-root data/features/itemstats-008 --train-parts 1 --out runs/baseline-itemstats-001
uv run --locked python -m scripts.compare_baselines --baseline runs/baseline-cosine-001 --candidate runs/baseline-itemstats-001 --out runs/baseline-itemstats-001/comparison.json
```


## Itemstats для 32 train-шардов — 13.11.2026

Построен отдельный кеш `data/features/itemstats-032` с 32 Float32-признаками товаров и категорий. Из источников исключены все 32 выбранных обучающих шарда. Статистики рассчитаны по остальным 480 train-шардам: 231,371,852 показов и 5,106,943 позитивов. Prior CTR и кривая CTR позиций рассчитаны по тем же источникам, alpha=20. Кеш `itemstats-008` сохранён с прежним отпечатком манифеста.

Покрытие нового кеша — 15,373,984 кандидатов train в 32 шардах, 1,389,971 val и 2,904,806 test. Всего 19,668,761 строк, 0.631 ГиБ Parquet. Расчёт занял 506.45 с при лимите DuckDB 2048 МиБ и 1 потоке. Пиковый рабочий набор процесса составил 2.160 ГиБ. Медианное время записи train-шарда — 1.773 с.

Аудит проверил ключи и порядок всех строк, схему, конечность чисел, отпечатки источников и раздельность обучающих и исходных шардов. Для 480 кандидатов все 32 признака независимо пересчитаны через Polars/Python по всем 480 исходным шардам, rtol/atol=2e-6. Общие числа показов и позитивов, prior CTR и CTR каждой позиции также совпали. Длительность аудита — 80.17 с.

В test у 53.37% вхождений кандидатов есть показы товара в источниках, у 51.02% — события товара в историях источников.

Проверено покрытие выбранных 32 шардов всеми четырьмя группами. GBDT-загрузчик прочитал один реальный train-шард со всеми 105 признаками; дополнительно проверено присоединение признаков для другого шарда из выбранного набора. Попытка использовать itemstats-032 для несовместимого набора из 8 шардов получила ожидаемый отказ.

[Материализация](itemstats_materialization_032.json), [полный аудит кеша](itemstats_032_audit.json), [проверка GBDT-загрузчика](itemstats_gbdt_integration_032.json).

```powershell
uv run --locked python -m scripts.build_itemstats_features --train-parts 32 --out data/features/itemstats-032 --memory-mb 2048 --threads 1 --resume
uv run --locked python -m scripts.audit_itemstats_features --cache data/features/itemstats-032 --out runs/itemstats_032_audit.json
```

## Userstats — 13.11.2026

Реализованы 16 Float32-признаков `is_user_*`: показы, клики, CTR со сглаживанием alpha=20, ожидаемые клики по позициям, сглаженное отношение кликов к ожидаемым, число событий в донорских историях и частоты типов 1–10. Использованы те же формулы, что в itemstats. Повторные события в историях разных донорских лент считаются повторно. Для неизвестных пользователей счётчики нулевые, CTR равен общему prior, отношение кликов равно 1. Alpha задана до оценки модели и не подбиралась.

Отдельные кеши для 8 и 32 выбранных train-шардов построены по остальным 504 и 480 шардам соответственно. Target исключённых обучающих шардов и val/test не участвует в расчёте. Списки источников, отпечатки кандидатов и историй, исходный код и формулы зафиксированы в manifest. Несовместимый обучающий набор отклоняется загрузчиком. Существующие itemstats-кеши сохранили прежние отпечатки.

Кеш `userstats-008` содержит 8 145 536 строк, 35 988 477 байт Parquet; расчёт занял 30.95 с, пиковый рабочий набор — 526 802 944 байта. Кеш `userstats-032` содержит 19 668 761 строку, 86 074 994 байта; расчёт занял 39.58 с, пик — 557 281 280 байт. DuckDB: лимит 512 МиБ, один поток. Весь val/test покрыт в обоих кешах. В test показы пользователя известны для 33.65% / 32.80% вхождений кандидатов соответственно; доли относятся к строкам, а не уникальным пользователям.

Оба аудита прошли: ключи, типы, конечность значений, источники и исключения проверены на всех строках. Формулы независимо пересчитаны по всем донорским шардам для трёх кандидатов каждого выходного файла: 408 строк / 375 пользователей и 480 строк / 446 пользователей. Общие показы, позитивы, prior и CTR всех позиций также сверены. Соответствие пользователя между кандидатами и метаданными истории проверено для каждой донорской ленты. Аудиты заняли 20.66 и 23.86 с, допуски rtol/atol=2e-6.

Проверено покрытие обоих обучающих наборов всеми пятью группами. `src.gbdt.prepared.load_prepared` прочитал реальный train-шард (476 301 строка, 6 132 ленты) со 121 признаком. Для обоих кешей проверен отказ на несовместимой обучающей выборке.

Выполнен один проверочный CatBoost-прогон с добавлением userstats к cosine + itemstats-008: тот же обучающий шард, параметры и val, 63 признака, 41 дерево. Чекпойнт выбран только по pseudo-Public. NDCG@30 pseudo-Private: 0.295374 → 0.295649, разность +0.000275 (+0.093%). Парный bootstrap по 10 095 лентам, 1000 повторов, seed=42: 95% интервал [-0.001836, 0.002119]. Прирост не подтверждён; группа остаётся опциональной до отдельной проверки на большем train. Private не использовался для выбора параметров. Метрики full/@10/@30 сверены с эталоном; сабмит на 2 904 806 строк прошёл валидатор после загрузки сохранённой модели. На платформу он не отправлялся.

66 unittest-тестов общей инфраструктуры, history, itemstats и userstats прошли. Проверены исключение обучающих target, неизвестный пользователь, повторные кандидаты, нулевой prior, восстановление после прерывания и обнаружение повреждённых чисел. Модуль `tests/test_edge_cases.py` с необязательными PyTorch/pytest в этот прогон не входил.

[Материализация 8](userstats_materialization_008.json), [материализация 32](userstats_materialization_032.json), [аудит 8](userstats_008_audit.json), [аудит 32](userstats_032_audit.json), [интеграция GBDT](userstats_gbdt_integration_001.json), [прогон модели](baseline_userstats_001.json), [парное сравнение](userstats_comparison_001.json).

```powershell
uv run --locked python -m scripts.build_userstats_features --train-parts 8 --out data/features/userstats-008 --memory-mb 512 --threads 1 --resume
uv run --locked python -m scripts.build_userstats_features --train-parts 32 --out data/features/userstats-032 --memory-mb 512 --threads 1 --resume
uv run --locked python -m scripts.audit_userstats_features --cache data/features/userstats-008 --out runs/userstats_008_audit.json
uv run --locked python -m scripts.audit_userstats_features --cache data/features/userstats-032 --out runs/userstats_032_audit.json
uv run --locked python -m scripts.run_baseline_check --data data/prepared --raw competition --feature-root data/features/cosine --feature-root data/features/itemstats-008 --feature-root data/features/userstats-008 --out runs/baseline-userstats-001 --train-parts 1 --iterations 100 --depth 6 --seed 42 --threads 2
uv run --locked python -m scripts.compare_baselines --baseline runs/baseline-itemstats-001 --candidate runs/baseline-userstats-001 --out runs/baseline-userstats-001/comparison.json
```

## GBDT на 8 шардах (Даня) — 12–13.11.2026

| дата | ветка | - | модель | pseudo-private NDCG full / idcg30 / @30 | Public | примечание |
|---|---|---|---|---|---|---|
| 2026-11-12 | danya | - | catboost/default pos=none mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3596/0.3596/0.2973 | - | out=runs/cb8; feats=$B (f+cosine+infeed), 8 parts |
| 2026-11-12 | danya | - | catboost/default pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3600/0.3601/0.2986 | - | out=runs/cb8-cv-B; feats=$B (f+cosine+infeed), 8 parts; iters 165 from cb8; Public submission 2026-11-12 |
| 2026-11-12 | danya | - | catboost/default pos=none mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3602/0.3602/0.2984 | - | out=runs/cb8-hist; feats=$H (+history), 8 parts |
| 2026-11-12 | danya | - | lightgbm/default pos=none mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3500/0.3501/0.2859 | - | out=runs/lgb8-hist; feats=$H (+history), 8 parts |
| 2026-11-12 | danya | - | lightgbm/default pos=ipw mode=holdout extra=- depth=8 lr=0.05 | priv: 0.2778/0.2779/0.1960 | - | out=runs/lgb8-ipw; feats=$H (+history), 8 parts; ipw-max 10, rejected (non-monotone CTR by pos) |
| 2026-11-12 | danya | - | lightgbm/default pos=feature mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3496/0.3497/0.2851 | - | out=runs/lgb8-posf; feats=$H (+history), 8 parts |
| 2026-11-12 | danya | - | lightgbm/default pos=topn mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3497/0.3498/0.2858 | - | out=runs/lgb8-top50; feats=$H (+history), 8 parts |
| 2026-11-12 | danya | - | catboost/YetiRankPairwise pos=none mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3604/0.3605/0.2985 | - | out=runs/cb8-pair; feats=$H (+history), 8 parts; 14 min vs 8.7 |
| 2026-11-13 | danya | - | catboost/default pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3619/0.3620/0.3005 | - | out=runs/cb8-cv; feats=$H (+history), 8 parts; iters 323 from cb8-hist |
| 2026-11-13 | danya | - | lightgbm/default pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3565/0.3566/0.2953 | - | out=runs/lgb8-cv; feats=$H (+history), 8 parts; iters 126 from lgb8-hist |
| 2026-11-13 | danya | - | lightgbm/default pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3562/0.3563/0.2948 | - | out=runs/lgb8-cv-w3; feats=$H (+history), 8 parts; val-weight 3 |
| 2026-11-13 | danya | - | catboost/default pos=none mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3595/0.3595/0.2976 | - | out=runs/cb8-is; feats=$H + itemstats-008, 8 parts; vs cb8-hist priv -0.0008 [-0.0041, +0.0026], rejected |
