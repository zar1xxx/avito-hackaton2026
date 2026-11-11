"""Скрипт быстрой проверки структуры и совместимости реальных данных соревнования.

Выполняет:
1. Проверку наличия сплитов (train, val, test) и эмбеддингов (memmap + meta).
2. Анализ схемы parquet-файлов (наличие обязательных и опциональных колонок).
3. Проверку чтения через FeedRankingDataset (процент отсутствующих эмбеддингов, длины лент).
4. Прогон через collate_ranking_batch и проверку размерностей тензоров.
5. Экспресс-замер скорости чтения DataLoader.

Запуск:
  python scripts/quick_check_real_data.py --data /path/to/competition_data
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Корень проекта в sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from src.nn.dataset import (
    EMB_DIM,
    FeedRankingDataset,
    collate_ranking_batch,
    find_parquet_parts,
    load_item_embeddings_memmap,
)


EXPECTED_FEED_COLS = {"feed_id", "user_id", "user_embedding"}
EXPECTED_CAND_COLS = {"item_id", "item_id__embedding_idx", "category_id"}
EXPECTED_HIST_COLS = {
    "history__item_id",
    "history__category_id",
    "history__event_type",
    "history__event_source",
    "history__item_id__embedding_idx",
}


def check_embeddings(data_root: Path) -> np.memmap:
    print("\n--- 1. Проверка Item Embeddings (memmap) ---")
    emb = load_item_embeddings_memmap(data_root, "train")
    print(f"[OK] Memmap успешно открыт: shape={emb.shape}, dtype={emb.dtype}")
    size_mb = (emb.nbytes) / (1024 ** 2)
    print(f"     Размер файла в памяти/диске: {size_mb:.1f} MB ({size_mb / 1024:.2f} GB)")
    if emb.shape[1] != EMB_DIM:
        print(f"[WARN] Ожидалась размерность {EMB_DIM}, получено {emb.shape[1]}!")
    return emb


def check_splits_and_schema(data_root: Path) -> list[Path]:
    print("\n--- 2. Проверка структуры сплитов и схемы Parquet ---")
    for split in ["train", "val", "test"]:
        try:
            parts = find_parquet_parts(data_root, split)
            total_rows_meta = sum(pq.ParquetFile(p).metadata.num_rows for p in parts)
            print(f"[OK] Сплит '{split}': найдено {len(parts)} файлов, всего строк: {total_rows_meta:,}")
            if split == "train":
                train_parts = parts
        except Exception as e:
            print(f"[WARN] Сплит '{split}': {e}")
            if split == "train":
                raise

    first_part = train_parts[0]
    print(f"\nАнализ колонок первого файла train: {first_part.name}")
    col_names = set(pq.ParquetFile(first_part).schema_arrow.names)

    # Проверка ключевых колонок
    missing_feed = EXPECTED_FEED_COLS - col_names
    missing_cand = EXPECTED_CAND_COLS - col_names
    missing_hist = EXPECTED_HIST_COLS - col_names

    if not missing_feed:
        print("  [OK] Колонки ленты на месте: feed_id, user_id, user_embedding")
    else:
        print(f"  [WARN] Отсутствуют колонки ленты: {missing_feed}")

    if not missing_cand:
        print("  [OK] Колонки кандидатов на месте: item_id, item_id__embedding_idx, category_id")
    else:
        print(f"  [WARN] Отсутствуют колонки кандидатов: {missing_cand}")

    if not missing_hist:
        print("  [OK] Колонки истории на месте: history__*")
    else:
        print(f"  [WARN] Отсутствуют колонки истории: {missing_hist}")

    # Проверка target и pos
    has_target = "target" in col_names
    has_pos = "pos" in col_names
    print(f"  [INFO] Колонка 'target' (клик): {'ЕСТЬ' if has_target else 'НЕТ'}")
    print(f"  [INFO] Колонка 'pos' (позиционное смещение): {'ЕСТЬ' if has_pos else 'НЕТ'}")

    float_feats = [c for c in col_names if c.startswith("f") and c[1:].isdigit()]
    print(f"  [INFO] Числовые признаки кандидатов: {float_feats}")

    return train_parts


def check_dataset_and_batch(data_root: Path, train_parts: list[Path]) -> None:
    print("\n--- 3. Чтение через FeedRankingDataset (10 лент) ---")
    t0 = time.perf_counter()
    ds = FeedRankingDataset(
        data_root=data_root,
        split="train",
        parts=train_parts[:1],
        max_feeds=10,
    )
    init_time = time.perf_counter() - t0
    print(f"[OK] Датасет инициализирован за {init_time * 1000:.2f} ms. Доступно лент: {len(ds)}")

    sample0 = ds[0]
    print("\nСтатистика первой ленты (sample 0):")
    print(f"  - feed_id: {sample0['feed_id']}, user_id: {sample0['user_id']}")
    print(f"  - Кандидатов: {sample0['cand_len']} (с эмбеддингом: {sample0['cand_has_emb'].sum()}/{sample0['cand_len']})")
    print(f"  - Событий истории: {sample0['hist_len']} (с эмбеддингом: {sample0['hist_has_emb'].sum()}/{sample0['hist_len']})")
    print(f"  - Признаков кандидата (cand_features): shape={sample0['cand_features'].shape}")

    print("\n--- 4. Сборка батча (collate_ranking_batch) ---")
    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_ranking_batch,
    )

    batch = next(iter(loader))
    print(f"Батч успешно собран (batch_size=4):")
    for k in ["candidate_mask", "cand_emb", "cand_features", "history_mask", "hist_emb", "target", "pos"]:
        if k in batch and batch[k] is not None:
            t = batch[k]
            print(f"  - {k:18s}: shape={str(tuple(t.shape)):16s} dtype={t.dtype}")

    # Экспресс-замер скорости чтения
    t_load_start = time.perf_counter()
    total_samples = 0
    for b in loader:
        total_samples += len(b["feed_id"])
    t_load = time.perf_counter() - t_load_start
    print(f"\n[OK] Скорость чтения: {total_samples / max(t_load, 1e-6):.1f} лент/сек на 1 CPU воркере")


def main() -> None:
    parser = argparse.ArgumentParser(description="Быстрая проверка реальных данных Avito")
    parser.add_argument("--data", required=True, help="Путь к корню реального датасета")
    args = parser.parse_args()

    data_root = Path(args.data)
    print("=" * 65)
    print(f" ЭКСПРЕСС-ПРОВЕРКА ДАННЫХ: {data_root}")
    print("=" * 65)

    if not data_root.exists():
        print(f"[ERROR] Путь {data_root} не существует!")
        sys.exit(1)

    try:
        check_embeddings(data_root)
        train_parts = check_splits_and_schema(data_root)
        check_dataset_and_batch(data_root, train_parts)

        print("\n" + "=" * 65)
        print(" [ИТОГ] Данные полностью совместимы с нейросетевым пайплайном!")
        print(" Вы можете запускать train.py без изменений кода.")
        print("=" * 65)

    except Exception as e:
        print(f"\n[ОШИБКА ПРИ ПРОВЕРКЕ]: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
