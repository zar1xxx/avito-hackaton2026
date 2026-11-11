# Техническая документация нейросетевого пайплайна (Август)

Документ описывает архитектуру данных, нейросетевую модель ранжирования, систему валидации и контракты интеграции с GBDT-пайплайном команды.

---

## 1. Структура модулей

```text
src/nn/
  ├── dataset.py               # FeedRankingDataset (одна лента = пример), memmap, 15 фичей, collate_fn с масками
  ├── model.py                 # FeedRankingModel (History Transformer + Cross-Attn + Set-Attn + Debias)
  ├── train.py                 # Цикл обучения (ListNet, MixUp, Temperature, AMP, Cosine/Plateau, CV Danya)
  ├── finetune.py              # Дообучение предобученной модели на (train + val) с весом для val
  ├── predict.py               # Инференс одиночной модели или Weighted Seed Ensemble + --save-as-danya
  ├── seed_ensemble.py         # Оркестратор сид-ансамбля с подбором весов по Val NDCG
  ├── export_gbdt_features.py  # K-Fold OOF экспорт признаков x_avgust_* для CatBoost/LightGBM Дани
  └── profile_speed.py         # Профайлер пропускной способности (лент/сек) и пиковой VRAM GPU

src/ensemble/
  ├── blend_nn_gbdt.py         # Ранговый блендинг (подбор весов NN + GBDT по сетке)
  └── stacking.py              # Ridge-стекинг на [nn_score, gbdt_score, f1..f5] с GroupKFold по feed_id

scripts/
  ├── quick_check_real_data.py # Экспресс-проверка совместимости реального датасета (5 секунд)
  ├── run_full_pipeline.py     # Оркестратор полного цикла (check -> smoke -> train -> finetune -> OOF -> sub)
  └── validate_submission.py   # Валидатор формата сабмита

tests/
  └── test_edge_cases.py       # Unit-тесты краевых случаев (H=0, M=1, missing=-1, Set-Attn padding isolation)
```

---

## 2. Чтение данных и инженерия признаков (`dataset.py`)

### Работа с памятью (`np.memmap`)
- Табличные данные читаются лениво по parquet-партам (`FeedRankingDataset`): одна запись датасета = одна рекомендательная лента (`feed_id`). Кандидаты **не** разворачиваются в плоскую таблицу в RAM.
- Эмбеддинги товаров (`item_embeddings/data.memmap`) открываются в режиме `np.memmap(..., mode="r")` внутри каждого процесса воркера (lazy init).
- Чтение выполняется индексацией по `item_id__embedding_idx`: значения `int8` конвертируются в `float32` с нормировкой `/ 128.0` на лету.
- Для индекса `-1` (отсутствие эмбеддинга) выделяется нулевой вектор и формируются маски `cand_has_emb_mask` и `hist_has_emb_mask`.

### Признаки кандидатов (`NUM_CAND_FEATS = 15`)
Помимо сырых признаков `f1..f5`, на лету для каждой ленты вычисляются:
1. `f1..f5` *(5)* — исходные числовые признаки.
2. `f1_z..f5_z` *(5)* — внутрилентовая z-нормализация `(x - mean) / (std + 1e-5)` внутри текущей ленты.
3. `item_in_hist` *(1)* — бинарный флаг: встречался ли данный `item_id` ранее в истории пользователя.
4. `cat_share_hist` *(1)* — доля событий этой же `category_id` в истории пользователя.
5. `feed_size` *(1)* — нормализованный размер выдачи (`M / 40.0`).
6. `hist_len` *(1)* — нормализованная длина истории (`H / 256.0`).
7. `cand_rel_pos` *(1)* — относительная позиция кандидата в списке (`idx / M`).

### Динамический паддинг (`collate_ranking_batch`)
Внутри батча ленты выравниваются по максимальной длине кандидатов $M_{\max}$ и истории $H_{\max} \le 256$ в текущем батче. Формируются булевы маски:
- `candidate_mask`: `[B, M_max]`
- `cand_has_emb_mask`: `[B, M_max]`
- `history_mask`: `[B, H_max]`
- `hist_has_emb_mask`: `[B, H_max]`

---

## 3. Архитектура нейросети (`FeedRankingModel`)

```text
  История пользователя [B, H, 128] + cat + event_type + event_source + pos
         │
         ▼
  HistoryTransformerEncoder (Self-Attention + защита от H=0) ──► hist_encoded [B, H, d]
                                                                       │
  Кандидаты [B, M, 128] + cat + 15 фичей ──► CandidateEncoder          │ (Key, Value)
         │                                       │                     ▼
         │                                    Query ──────────► Cross-Attention
         │                                                             │
         │                                      cand_hist_ctx = LayerNorm(cand + attn)
         │                                                             │
         │                                                             ▼
         │                           (Опц.) Listwise Set-Attention (Self-Attn по M кандидатам)
         │                                                             │
         ▼                                                             ▼
  User Tower [B, d] ──► Косинусные сходства (u·c, c·h, u·h) ──► Main Head (MLP + LayerNorm)
                                                                       │
                                                                       ▼
                                                                main_scores [B, M]
                                                                       │
  (Только train: + PosBiasHead(pos)) ─────────────────────────► total_scores [B, M]
```

1. **History Transformer Encoder**: кодирует до 256 событий истории (эмбеддинг товара, категория, тип и источник события, позиция давности) через `TransformerEncoderLayer` с `src_key_padding_mask=~safe_mask`. Если у пользователя история пуста ($H=0$), первый токен временно открывается для предотвращения `NaN` в softmax, а выход зануляется.
2. **Candidate-to-History Cross-Attention**: позволяет каждому кандидату найти в истории пользователя наиболее релевантные прошлые события.
3. **Listwise Set-Attention (`--use_set_attention`, вкл. по умолчанию)**: слой `MultiheadAttention` между кандидатами внутри одной ленты с `key_padding_mask=~safe_cand_mask` и без позиционных эмбеддингов (перестановочно-инвариантен). Позволяет оценивать кандидата в контексте конкурентов ленты. Отключается флагом `--no_set_attention`.
4. **Two-Tower Positional Debias**: в обучающем сплите (`train`) присутствует позиционное смещение (`pos`). Отдельная башня `pos_bias_head` моделирует эффект позиции, суммируясь с `main_scores` только на этапе `train`. На валидации (`val`) и тесте (`test`) башня позиции отключается (`use_pos_bias=False`), и ранжирование опирается только на несмещенный контентный скор `main_scores`.

---

## 4. Механизмы обучения и валидации

### Валидация в стиле Дани (`--cv-mode danya`, по умолчанию)
Для полной сопоставимости с GBDT Дани используется [`src/common/cv.py`](file:///d:/avito_hackathon/avito-hackaton2026/src/common/cv.py):
- Сплит `val` делится по `timestamp` на ранние 30% (**`pseudo-public`**) и оставшиеся 70% (**`pseudo-private`**), имитируя публичный и приватный лидерборды соревнования.
- Чекпоинт `best_model.pt`, `EarlyStopping` и `ReduceLROnPlateau` ориентируются на **`val_ndcg_public`**, а **`val_ndcg_private`** только пишется в `history.csv` и выводится в отчет для честного сравнения с GBDT без утечек.
- После каждой эпохи все метрики пишутся в `history.csv`:
  `epoch, train_loss, val_ndcg_full, val_ndcg_public, val_ndcg_private, val_ndcg_idcg30, val_ndcg_at30, lr, epoch_time_sec, best_ndcg`.

### Регуляризация и ускорение
- **ListNet Loss с температурой (`--temperature`)**: $\text{log\_softmax}(\text{logits} / T)$ и мягкое распределение по релевантности внутри ленты.
- **Внутрилентовый MixUp (`--mixup-alpha`)**: перемешивает эмбеддинги, табличные фичи и таргеты исключительно между валидными кандидатами внутри одной ленты с $\lambda = \max(\lambda_0, 1 - \lambda_0)$, $\lambda_0 \sim \text{Beta}(\alpha, \alpha)$.
- **LR Scheduler (`--scheduler cosine|plateau`, `--warmup-epochs`)**: поддерживаются пошаговый линейный Warmup + косинусное затухание и `ReduceLROnPlateau`.
- **AMP (`--amp`)**: смешанная точность `fp16` на CUDA (финальное маскирование `-1e9` и softmax-лосс принудительно переводятся в `float32`, чтобы избежать переполнения половинной точности). На CPU флаг `--amp` безопасно игнорируется.
- **Fine-Tuning (`src/nn/finetune.py`)**: дообучает `best_model.pt` на объединенной выборке `train + val` с повышенным весом `val` (`--val-weight 2.0`) и пониженным LR (`2e-4`).

---

## 5. Контракт интеграции с GBDT Дани

### Вариант А: Внешние нейро-фичи для GBDT (`features/avgust/`)
Скрипт `src/nn/export_gbdt_features.py` полностью соблюдает контракт [`extra_features.py`](file:///d:/avito_hackathon/avito-hackaton2026/src/gbdt/train.py) Дани:
- `train.parquet` формируется строго в режиме **K-Fold Out-Of-Fold** (нейросеть не видит ленту, для которой предсказывает признаки — утечка таргета исключена).
- `val.parquet` и `test.parquet` — усреднение предсказаний всех $K$ фолдов.
- Ключи: `feed_id: Int32`, `item_id: Int32` (без дублей и пропусков).
- Экспортируемые колонки (`Float32`):
  - `x_avgust_nn_score` — чистый контентный скор нейросети (без position bias)
  - `x_avgust_nn_rank` — нормализованный ранг внутри ленты $[0, 1]$
  - `x_avgust_nn_z` — z-score внутри ленты
  - `x_avgust_dot_user` — косинус `cand_repr · user_repr`
  - `x_avgust_dot_hist` — косинус `cand_repr · cand_hist_ctx`
  - `x_avgust_dot_uh` — косинус `user_repr · cand_hist_ctx`

**Команда подключения к GBDT Дани:**
```bash
PYTHONPATH=. python -m src.gbdt.run_exp --data /path/to/data --features-dir features --extra avgust --out runs/cb-avgust
```

### Вариант Б: Ансамблирование и отбор финала
1. **Формат `select_final.py` / `blend.py` Дани:** флаг `--save-as-danya` в `src/nn/predict.py` сохраняет файлы с точными именами `val_pred.parquet` и `test_pred.parquet`.
   *(Важно: для `val_pred.parquet` всегда используется базовая модель **до** `finetune.py` на `val`, а для `test_pred.parquet` — дообученная модель! Оркестратор `run_full_pipeline.py` соблюдает это автоматически.)*
2. **Ранговый блендинг (`src/ensemble/blend_nn_gbdt.py`):** переводит скоры обеих моделей внутри каждой ленты в ранги $[0, 1]$ и подбирает оптимальные веса `w_nn` / `w_gbdt` на `val`.
3. **Ridge-стекинг (`src/ensemble/stacking.py`):** обучает `StandardScaler + Ridge` на `[nn_score, gbdt_score, f1..f5]` через `GroupKFold` по `feed_id` и генерирует итоговый сабмит.
